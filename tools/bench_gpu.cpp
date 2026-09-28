// ============================================================================
//  tools/bench_gpu.cpp — benchmarks de la iGPU (Vulkan de cómputo propio).
//
//    bench_gpu --bw               ancho de banda de flujo (copia u32, lectura, patrón
//                                 LBM de 19 corrientes f16 in-place) por tipo de memoria,
//                                 tamaño de subgrupo y de grupo de trabajo
//    bench_gpu --lbm [opciones]   MLUPS del solver LBM en la GPU frente a la CPU
//        --cells N   --fp32   --sg 8|16|32   --wg N   --steps N   --reps N
//    bench_gpu --cpu-read         lectura de la CPU de memoria escrita por la GPU (WC vs cacheada)
//
//  Tiempos de GPU con marcas de tiempo (vkCmdWriteTimestamp); medianas de varias rondas
//  (hay carga pesada de fondo en esta máquina y la iGPU comparte la LPDDR5x con la CPU).
// ============================================================================
#include "../src/app/app.hpp"
#include "../src/core/threadpool.hpp"
#include "../src/gpu/lbm_gpu.hpp"
#include "../src/gpu/spirv.hpp"
#include "../src/models/model.hpp"
#include "../src/gpu/vk.hpp"
#include "../src/core/util.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::gpu;
using namespace cfd::gpu::spv;

namespace {

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Ejecuta `reps` despachos de `groups` grupos con marcas de tiempo alrededor de cada uno; mediana en ms.
double time_dispatch(vk::Context& ctx, const vk::Context::Pipeline& p, vk::VkDescriptorSet set, u32 groups, int reps,
                     const void* push = nullptr, u32 push_bytes = 0, u32 gy = 1) {
    const auto& f = ctx.fn();
    vk::VkQueryPool qp = ctx.timestamps(2 * static_cast<u32>(reps) + 2);
    vk::VkCommandBuffer c = ctx.alloc_cmd();
    ctx.begin(c, true);
    f.CmdResetQueryPool(c, qp, 0, 2 * static_cast<u32>(reps));
    f.CmdBindPipeline(c, vk::PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    f.CmdBindDescriptorSets(c, vk::PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    if (push_bytes) f.CmdPushConstants(c, p.layout, vk::SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    for (int r = 0; r < reps; ++r) {
        f.CmdWriteTimestamp(c, vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 2 * static_cast<u32>(r));
        f.CmdDispatch(c, groups, gy, 1);
        f.CmdWriteTimestamp(c, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2 * static_cast<u32>(r) + 1);
        ctx.barrier_compute(c);
    }
    ctx.end(c);
    ctx.wait(ctx.submit(c));
    ctx.free_cmd(c);
    std::vector<double> ns(2 * static_cast<usize>(reps));
    ctx.read_timestamps(0, 2 * static_cast<u32>(reps), ns.data());
    std::vector<double> ms;
    for (int r = 1; r < reps; ++r) ms.push_back((ns[2 * static_cast<usize>(r) + 1] - ns[2 * static_cast<usize>(r)]) * 1e-6);   // la 1.ª calienta
    return median(ms);
}

// ---- Ancho de banda --------------------------------------------------------------------------------
int bench_bw() {
    vk::Context ctx;
    std::string err;
    if (!ctx.init(&err)) { std::printf("sin GPU: %s\n", err.c_str()); return 0; }
    std::printf("%s", vk::describe(ctx.info()).c_str());
    const u32 cells = 6u << 20;   // 6.3 M celdas (≈ Media)
    for (int mt = 0; mt < 2; ++mt) {
        const vk::Mem mem = mt ? vk::Mem::Cached : vk::Mem::Coherent;
        const char* mname = mt ? "CACHED (tipo 1)" : "COHERENT (tipo 0)";
        // (a) copia u32: 2 corrientes de 4 B.
        for (u32 sg : {8u, 16u, 32u}) {
            if (sg < ctx.info().sg_min || sg > ctx.info().sg_max) continue;
            for (u32 wg : {64u, 256u}) {
                KernelOptions o; o.local_size = wg;
                std::vector<u32> code;
                {
                    Kernel k(o);
                    Buf a = k.buffer(0, Elem::U32, true), b = k.buffer(1, Elem::U32, false, true);
                    const U i = k.global_id(0);
                    k.stu(b, i, k.ldu(a, i) + U(1u));
                    code = k.finish();
                }
                vk::Context::Pipeline p;
                if (!ctx.create_pipeline(p, code, 2, 0, sg, &err)) { std::printf("pipeline: %s\n", err.c_str()); return 1; }
                const u32 n = 64u << 20;   // 256 MB por búfer
                vk::Buffer A, B;
                ctx.create_buffer(A, u64(n) * 4, mem); ctx.create_buffer(B, u64(n) * 4, mem);
                vk::Context::Bind bs[2] = {{&A, 0, 0}, {&B, 0, 0}};
                vk::VkDescriptorSet s = ctx.make_set(p, bs, 2);
                const double ms = time_dispatch(ctx, p, s, n / wg, 12);
                std::printf("copia u32   %-18s sg %2u wg %3u: %7.3f ms → %6.1f GB/s (L+E)\n", mname, sg, wg, ms, 8.0 * n / (ms * 1e6));
                ctx.destroy_buffer(A); ctx.destroy_buffer(B); ctx.destroy_pipeline(p);
            }
        }
        // (b) patrón LBM: 19 corrientes f16 in-place (carga 19 + escritura 19 por celda), 2 de ellas desplazadas ±nx.
        for (u32 sg : {8u, 16u, 32u}) {
            if (sg < ctx.info().sg_min || sg > ctx.info().sg_max) continue;
            for (u32 wg : {64u, 128u, 256u}) {
                KernelOptions o; o.local_size = wg;
                std::vector<u32> code;
                {
                    Kernel k(o);
                    std::vector<Buf> d;
                    for (u32 q = 0; q < 19; ++q) d.push_back(k.buffer(q, Elem::F16));
                    const U n = k.global_id(0) + U(4096u);
                    F v[19];
                    for (u32 q = 0; q < 19; ++q) v[q] = k.ldf(d[q], q & 1 ? n + U(1u) : n);
                    F s = v[0];
                    for (u32 q = 1; q < 19; ++q) s = s + v[q];
                    for (u32 q = 0; q < 19; ++q) k.stf(d[q], (q & 1) ? n + U(1u) : n, v[q] + s * F(1e-3f));
                    code = k.finish();
                }
                vk::Context::Pipeline p;
                if (!ctx.create_pipeline(p, code, 19, 0, sg, &err)) { std::printf("pipeline: %s\n", err.c_str()); return 1; }
                const u64 S = cells + 8192;
                vk::Buffer D;
                ctx.create_buffer(D, S * 2 * 19, mem);
                std::memset(D.map, 0, D.size);
                ctx.flush(D);
                std::vector<vk::Context::Bind> bs;
                for (u32 q = 0; q < 19; ++q) bs.push_back({&D, q * S * 2, S * 2});
                vk::VkDescriptorSet s = ctx.make_set(p, bs.data(), 19);
                const double ms = time_dispatch(ctx, p, s, cells / wg, 12);
                std::printf("LBM 19×f16  %-18s sg %2u wg %3u: %7.3f ms → %6.1f GB/s · %6.0f M celdas/s\n", mname, sg, wg, ms,
                            76.0 * cells / (ms * 1e6), cells / (ms * 1e3));
                ctx.destroy_buffer(D); ctx.destroy_pipeline(p);
            }
        }
    }
    return 0;
}

// ---- Lectura de la CPU de memoria escrita por la GPU ------------------------------------------------
int bench_cpu_read() {
    vk::Context ctx;
    std::string err;
    if (!ctx.init(&err)) { std::printf("sin GPU: %s\n", err.c_str()); return 0; }
    const u64 bytes = 96ull << 20;   // ≈ 4 campos macro de 6 M celdas
    std::vector<float> dst(bytes / 4);
    for (int mt = 0; mt < 2; ++mt) {
        vk::Buffer B;
        ctx.create_buffer(B, bytes, mt ? vk::Mem::Cached : vk::Mem::Coherent);
        std::memset(B.map, 1, bytes);
        ctx.flush(B);
        std::vector<double> tinv, tcpy;
        for (int r = 0; r < 5; ++r) {
            const double t0 = now_sec();
            ctx.invalidate(B);
            const double t1 = now_sec();
            std::memcpy(dst.data(), B.map, mt ? bytes : bytes / 16);   // WC: sólo 1/16 (lentísimo)
            const double t2 = now_sec();
            tinv.push_back(t1 - t0);
            tcpy.push_back((t2 - t1) * (mt ? 1.0 : 16.0));
        }
        std::printf("CPU lee %s: invalidate %.2f ms · memcpy %.1f ms (%.2f GB/s) para %.0f MB\n", mt ? "CACHED" : "COHERENT (WC)",
                    median(tinv) * 1e3, median(tcpy) * 1e3, bytes / median(tcpy) * 1e-9, bytes / 1048576.0);
        ctx.destroy_buffer(B);
    }
    return 0;
}

// ---- LBM: MLUPS GPU frente a CPU en un modelo real -----------------------------------------------------
int bench_lbm(int argc, char** argv) {
    using namespace cfd::app;
    std::string model = "f1_2022";
    Preset pr = Preset::Media;
    bool fp32 = false, cpu = true;
    int steps = 100, reps = 5;
    gpu::LbmGpuTuning tun;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto nxt = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = nxt();
        else if (a == "--res") parse_preset(nxt(), pr);
        else if (a == "--fp32") fp32 = true;
        else if (a == "--nocpu") cpu = false;
        else if (a == "--steps") steps = std::atoi(nxt());
        else if (a == "--reps") reps = std::atoi(nxt());
        else if (a == "--sg") tun.sg = static_cast<u32>(std::atoi(nxt()));
        else if (a == "--wg") tun.wg = static_cast<u32>(std::atoi(nxt()));
        else if (a == "--wgn") tun.wg_nodes = static_cast<u32>(std::atoi(nxt()));
        else if (a == "--coherent") tun.ddf_mem = vk::Mem::Coherent;
        else if (a == "--rerecord") tun.reuse_cmd = false;
        else if (a == "--single") tun.pair = false;
    }
    pool().start();
    Sim sim;
    sim.cfg.model = models::find(model);
    if (sim.cfg.model < 0) { std::printf("modelo desconocido %s\n", model.c_str()); return 1; }
    sim.cfg.preset = pr;
    sim.cfg.fp32 = fp32;
    sim.cfg.ground = models::info(sim.cfg.model).needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None;
    sim.init();
    const double N = static_cast<double>(sim.dom.cells());
    std::printf("[bench] %s · %s · %d×%d×%d = %.2f M celdas · %s · %d hilos\n", model.c_str(), preset_name(pr), sim.dom.nx, sim.dom.ny,
                sim.dom.nz, N * 1e-6, fp32 ? "FP32" : "FP16S", pool().size());
    if (cpu) {
        sim.solver.step(steps);   // calentamiento
        std::vector<double> ml, kt, ft;
        for (int r = 0; r < reps; ++r) {
            sim.solver.step(steps);
            ml.push_back(sim.solver.last_mlups());
            kt.push_back(sim.solver.last_kernel_seconds() * 1e3 / steps);
            ft.push_back(sim.solver.last_force_seconds() * 1e3 / steps);
        }
        std::printf("[bench] CPU: %.0f MLUPS (mediana de %d × %d pasos; kernel %.2f ms/paso, contorno+fuerzas %.3f ms/paso)\n", median(ml), reps,
                    steps, median(kt), median(ft));
    }
    gpu::LbmGpu G;
    std::string err;
    if (!G.init(&err)) { std::printf("sin GPU: %s\n", err.c_str()); return 0; }
    G.set_tuning(tun);
    const double ta = now_sec();
    if (!G.attach(sim.solver, &err)) { std::printf("attach: %s\n", err.c_str()); return 1; }
    std::printf("[bench] attach (generar + compilar shaders + subir estado): %.0f ms (subida %.0f ms)\n", (now_sec() - ta) * 1e3,
                G.stats().upload_ms);
    const auto& st = G.stats();
    std::printf("[bench] nodos de pared %zu · registros %zu · trozos %zu · nodos multi-id %zu · >4 ids %zu · grupos puros %.1f %%\n", st.nodes,
                st.records, st.chunks, st.multi_id_nodes, st.overflow_nodes, st.pure_groups * 100.0);
    G.step(steps);   // calentamiento (la iGPU baja de frecuencia en reposo)
    std::vector<double> ml, gb, wall, rec, pub;
    for (int r = 0; r < reps; ++r) {
        G.step(steps);
        ml.push_back(st.mlups); gb.push_back(st.gbs); wall.push_back(st.wall_ms); rec.push_back(st.record_us); pub.push_back(st.publish_ms);
    }
    std::printf("[bench] GPU: %.0f MLUPS (tiempo de GPU) · %.1f GB/s de poblaciones · lote de %d pasos: pared %.1f ms, publicar %.2f ms, grabar %.0f µs\n",
                median(ml), median(gb), steps, median(wall), median(pub), median(rec));
    gpu::LbmGpuTuning t2 = tun;
    t2.profile = true;
    G.set_tuning(t2);
    G.step(steps);
    std::vector<double> a, b, c;
    for (int r = 0; r < reps; ++r) { G.step(steps); a.push_back(st.k_step_ms); b.push_back(st.k_boundary_ms); c.push_back(st.k_reduce_ms); }
    const double ks = median(a);
    std::printf("[bench] perfil por paso: celdas %.3f ms (%.0f MLUPS, %.1f GB/s) · contorno %.3f ms · reducción %.3f ms\n", ks, N / (ks * 1e3),
                N * 2 * 19 * (fp32 ? 4 : 2) / (ks * 1e6), median(b), median(c));
    // Relevo CPU ↔ GPU en caliente (lo que cuesta mover un deslizador de geometría): bajada + subida.
    std::vector<double> dl, ul, rb;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_sec();
        sim.solver.set_viscosity(sim.nu);   // gancho: termina, baja poblaciones; revisión nueva → subida al siguiente lote
        const double t1 = now_sec();
        G.step(1);
        rb.push_back((now_sec() - t1) * 1e3);
        dl.push_back(st.download_ms); ul.push_back(st.upload_ms);
        (void)t0;
    }
    std::printf("[bench] relevo en caliente: bajada %.1f ms · subida %.1f ms (siguiente lote de 1 paso %.1f ms)\n", median(dl), median(ul), median(rb));
    G.detach();
    std::printf("[bench] divergencia: %d\n", sim.solver.diverged());
    return 0;
}

} // namespace


int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "--bw";
    if (mode == "--bw") return bench_bw();
    if (mode == "--cpu-read") return bench_cpu_read();
    if (mode == "--lbm") return bench_lbm(argc, argv);
    std::printf("uso: %s --bw | --cpu-read | --lbm [opciones]\n", argv[0]);
    return 0;
}
