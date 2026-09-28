// ============================================================================
//  tests/test_gpu_spirv.cpp — el emisor de SPIR-V propio contra la iGPU real.
//
//  Sin spirv-val: cada módulo se valida creando el pipeline en el controlador
//  (Mesa ANV rechaza SPIR-V mal formado) y ejecutándolo; los resultados se
//  comparan con la CPU. Si no hay dispositivo Vulkan, PASA con una nota.
//   [1] suma de vectores f32 (índices, constantes de empuje, límites)
//   [2] f32 → f16 → f32: redondeo RTE y subnormales idénticos a F16C
//   [3] búfer u8 → u32 y control de flujo if/else con variables locales
//   [4] reducción: subgrupo (FAdd) + memoria compartida + barrera
//   [5] desplazamiento de subgrupo (ShuffleUp) y elección (Elect) + atómicos
//   [6] funciones GLSL.std.450 (sqrt, fma, max, clamp) y comparaciones no ordenadas
// ============================================================================
#include "../src/gpu/spirv.hpp"
#include "../src/gpu/vk.hpp"
#include "../src/core/util.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cfd;
using namespace cfd::gpu;
using namespace cfd::gpu::spv;

namespace {

int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FALLO: " __VA_ARGS__); std::printf("\n"); } } while (0)

struct Run {
    vk::Context& ctx;
    vk::Context::Pipeline p;
    bool ok = false;
    Run(vk::Context& c, const std::vector<u32>& spirv, u32 nbind, u32 push, u32 sg = 0) : ctx(c) {
        std::string e;
        ok = ctx.create_pipeline(p, spirv, nbind, push, sg, &e);
        if (!ok) std::printf("  pipeline: %s\n", e.c_str());
    }
    ~Run() { ctx.destroy_pipeline(p); }
    void go(std::vector<const vk::Buffer*> bufs, u32 groups, const void* push = nullptr) {
        std::vector<vk::Context::Bind> b;
        for (auto* x : bufs) b.push_back({x, 0, 0});
        vk::VkDescriptorSet s = ctx.make_set(p, b.data(), static_cast<u32>(b.size()));
        vk::VkCommandBuffer c = ctx.alloc_cmd();
        ctx.begin(c, true);
        const auto& f = ctx.fn();
        f.CmdBindPipeline(c, vk::PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        f.CmdBindDescriptorSets(c, vk::PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &s, 0, nullptr);
        if (push && p.push) f.CmdPushConstants(c, p.layout, vk::SHADER_STAGE_COMPUTE_BIT, 0, p.push, push);
        f.CmdDispatch(c, groups, 1, 1);
        ctx.barrier_host(c);
        ctx.end(c);
        const u64 v = ctx.submit(c);
        ctx.wait(v);
        ctx.free_cmd(c);
        for (auto* x : bufs) ctx.invalidate(*x);
    }
};

void t_vadd(vk::Context& ctx) {
    std::printf("[1] suma de vectores f32\n");
    KernelOptions o; o.local_size = 64;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf a = k.buffer(0, Elem::F32, true), b = k.buffer(1, Elem::F32, true), c = k.buffer(2, Elem::F32, false, true);
        k.push_constants(2, 0b10);   // [0] n (u32), [1] escala (f32)
        const U i = k.global_id(0);
        k.if_(i < k.pc_u(0), [&] { k.stf(c, i, (k.ldf(a, i) + k.ldf(b, i)) * k.pc_f(1)); });
        code = k.finish();
    }
    Run r(ctx, code, 3, 8);
    CHECK(r.ok, "pipeline de suma");
    if (!r.ok) return;
    const u32 n = 1000;
    vk::Buffer A, B, C;
    ctx.create_buffer(A, n * 4, vk::Mem::Cached); ctx.create_buffer(B, n * 4, vk::Mem::Coherent); ctx.create_buffer(C, n * 4 + 64, vk::Mem::Cached);
    for (u32 i = 0; i < n; ++i) { A.as<float>()[i] = float(i); B.as<float>()[i] = 0.5f * float(i); }
    C.as<float>()[n] = -7.0f;
    ctx.flush(A); ctx.flush(C);
    struct { u32 n; float s; } pc{n, 2.0f};
    r.go({&A, &B, &C}, (n + 63) / 64, &pc);
    int bad = 0;
    for (u32 i = 0; i < n; ++i) bad += C.as<float>()[i] != 3.0f * float(i);
    CHECK(bad == 0, "suma: %d errores", bad);
    CHECK(C.as<float>()[n] == -7.0f, "escritura fuera de rango");
    ctx.destroy_buffer(A); ctx.destroy_buffer(B); ctx.destroy_buffer(C);
}

void t_f16(vk::Context& ctx) {
    std::printf("[2] f32 → f16 (RTE) → f32 frente a F16C\n");
    KernelOptions o; o.local_size = 64;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf a = k.buffer(0, Elem::F32, true), h = k.buffer(1, Elem::F16);
        (void)k.buffer(2, Elem::F32, false, true);
        const U i = k.global_id(0);
        k.stf(h, i, k.ldf(a, i));
        code = k.finish();
    }
    std::vector<u32> code2;
    {
        Kernel k(o);
        Buf h = k.buffer(1, Elem::F16, true), back = k.buffer(2, Elem::F32, false, true);
        (void)k.buffer(0, Elem::F32, true);
        const U i = k.global_id(0);
        k.stf(back, i, k.ldf(h, i));
        code2 = k.finish();
    }
    Run r1(ctx, code, 3, 0), r2(ctx, code2, 3, 0);
    CHECK(r1.ok && r2.ok, "pipelines f16");
    if (!r1.ok || !r2.ok) return;
    const u32 n = 1u << 16;
    vk::Buffer A, H, O;
    ctx.create_buffer(A, n * 4, vk::Mem::Cached); ctx.create_buffer(H, n * 2, vk::Mem::Cached); ctx.create_buffer(O, n * 4, vk::Mem::Cached);
    WyRand rng(7);
    float* a = A.as<float>();
    for (u32 i = 0; i < n; ++i) {
        // Valores en todo el rango de f16 (incluidos subnormales y empates de redondeo).
        const u32 h = static_cast<u32>(rng.next()) & 0x7FFF;
        float v = f16_to_f32(static_cast<u16>(h)) * (i & 1 ? -1.0f : 1.0f);
        if (i % 3 == 0) {   // entre dos f16: punto medio exacto (empate) o perturbado
            const float v2 = f16_to_f32(static_cast<u16>((h + 1) & 0x7FFF));
            v = 0.5f * (v + v2 * (i & 1 ? -1.0f : 1.0f));
            if (i % 9 == 0) v = std::nextafter(v, 0.0f);
        }
        if (!std::isfinite(v)) v = 1.0f;
        a[i] = v;
    }
    a[0] = 3.0e-8f; a[1] = 6.0e-8f; a[2] = -2.9802322e-8f; a[3] = 65504.0f; a[4] = 1e-10f;
    ctx.flush(A);
    r1.go({&A, &H, &O}, n / 64);
    r2.go({&A, &H, &O}, n / 64);
    int bad = 0, badh = 0;
    for (u32 i = 0; i < n; ++i) {
        const u16 ref = f32_to_f16(a[i]);
        const u16 got = H.as<u16>()[i];
        if (ref != got && !((ref & 0x7FFF) == 0 && (got & 0x7FFF) == 0)) { if (badh < 5) std::printf("    %.9g: F16C %04x, GPU %04x\n", double(a[i]), ref, got); ++badh; }
        if (O.as<float>()[i] != f16_to_f32(got)) ++bad;
    }
    CHECK(badh == 0, "conversión a f16: %d diferencias con F16C", badh);
    CHECK(bad == 0, "conversión a f32: %d diferencias", bad);
    ctx.destroy_buffer(A); ctx.destroy_buffer(H); ctx.destroy_buffer(O);
}

void t_u8_if(vk::Context& ctx) {
    std::printf("[3] u8 + if/else + variables locales\n");
    KernelOptions o; o.local_size = 32;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf fl = k.buffer(0, Elem::U8, true), out = k.buffer(1, Elem::U32, false, true);
        const U i = k.global_id(0);
        const U f = k.ldu(fl, i);
        VarU acc = k.varu(U(0u));
        k.if_else(bit(f, 0), [&] { acc.set(f * U(3u)); }, [&] {
            k.if_(any_bits(f, 0x6), [&] { acc.set(f + U(1000u)); });
        });
        k.stu(out, i, acc.get() + ext(f, 4, 3));
        code = k.finish();
    }
    Run r(ctx, code, 2, 0);
    CHECK(r.ok, "pipeline u8");
    if (!r.ok) return;
    const u32 n = 256;
    vk::Buffer F8, O;
    ctx.create_buffer(F8, n, vk::Mem::Coherent); ctx.create_buffer(O, n * 4, vk::Mem::Cached);
    for (u32 i = 0; i < n; ++i) F8.as<u8>()[i] = static_cast<u8>(i);
    r.go({&F8, &O}, n / 32);
    int bad = 0;
    for (u32 i = 0; i < n; ++i) {
        u32 ref = (i & 1) ? i * 3 : ((i & 6) ? i + 1000 : 0);
        ref += (i >> 4) & 7;
        bad += O.as<u32>()[i] != ref;
    }
    CHECK(bad == 0, "u8/if: %d errores", bad);
    ctx.destroy_buffer(F8); ctx.destroy_buffer(O);
}

void t_reduce(vk::Context& ctx, u32 sg) {
    std::printf("[4] reducción subgrupo + compartida (subgrupo %u)\n", sg);
    const u32 wg = 256;
    KernelOptions o; o.local_size = wg;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf in = k.buffer(0, Elem::F32, true), out = k.buffer(1, Elem::F32, false, true);
        Shared sh = k.shared(wg / 8, true);
        const U li = k.local_index();
        const U i = k.global_id(0);
        const F s = k.sg_add(k.ldf(in, i));
        const U sid = k.subgroup_id();
        k.if_(k.sg_elect(), [&] { k.stf(sh, sid, s); });
        k.barrier();
        k.if_(li == U(0u), [&] {
            F t = k.ldf(sh, U(0u));
            for (u32 j = 1; j < wg / sg; ++j) t = t + k.ldf(sh, U(j));
            k.stf(out, k.workgroup_id(0), t);
        });
        code = k.finish();
    }
    Run r(ctx, code, 2, 0, sg);
    CHECK(r.ok, "pipeline de reducción");
    if (!r.ok) return;
    const u32 ng = 64, n = ng * wg;
    vk::Buffer I, O;
    ctx.create_buffer(I, n * 4, vk::Mem::Cached); ctx.create_buffer(O, ng * 4, vk::Mem::Cached);
    for (u32 i = 0; i < n; ++i) I.as<float>()[i] = float(i % 97) * 0.25f;
    ctx.flush(I);
    r.go({&I, &O}, ng);
    int bad = 0;
    for (u32 g = 0; g < ng; ++g) {
        double ref = 0;
        for (u32 j = 0; j < wg; ++j) ref += double((g * wg + j) % 97) * 0.25;
        bad += std::fabs(O.as<float>()[g] - ref) > 1e-3 * std::fabs(ref);
    }
    CHECK(bad == 0, "reducción: %d grupos mal", bad);
    ctx.destroy_buffer(I); ctx.destroy_buffer(O);
}

void t_shuffle(vk::Context& ctx, u32 sg) {
    std::printf("[5] ShuffleUp + Elect + atómicos (subgrupo %u)\n", sg);
    KernelOptions o; o.local_size = 64;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf in = k.buffer(0, Elem::F32, true), out = k.buffer(1, Elem::F32, false, true), cnt = k.buffer(2, Elem::U32);
        const U i = k.global_id(0);
        const F v = k.ldf(in, i);
        const F up = k.sg_shuffle_up(v, 1);
        k.stf(out, i, select(k.subgroup_local_id() == U(0u), F(-1.0f), up));
        k.if_(k.sg_elect(), [&] { k.atomic_add(cnt, U(0u), U(1u)); });
        k.if_(k.sg_any(v > F(100.0f)), [&] { k.atomic_or(cnt, U(1u), U(1u) << (i & U(31u))); });
        code = k.finish();
    }
    Run r(ctx, code, 3, 0, sg);
    CHECK(r.ok, "pipeline shuffle");
    if (!r.ok) return;
    const u32 n = 512;
    vk::Buffer I, O, C;
    ctx.create_buffer(I, n * 4, vk::Mem::Cached); ctx.create_buffer(O, n * 4, vk::Mem::Cached); ctx.create_buffer(C, 64, vk::Mem::Cached);
    for (u32 i = 0; i < n; ++i) I.as<float>()[i] = float(i) * 0.5f;
    std::memset(C.map, 0, 64);
    ctx.flush(I); ctx.flush(C);
    r.go({&I, &O, &C}, n / 64);
    int bad = 0;
    for (u32 i = 0; i < n; ++i) {
        const float ref = (i % sg == 0) ? -1.0f : float(i - 1) * 0.5f;
        bad += O.as<float>()[i] != ref;
    }
    CHECK(bad == 0, "shuffle: %d errores", bad);
    CHECK(C.as<u32>()[0] == n / sg, "elect: %u subgrupos (esperado %u)", C.as<u32>()[0], n / sg);
    CHECK(C.as<u32>()[1] != 0, "any/atomic_or");
    ctx.destroy_buffer(I); ctx.destroy_buffer(O); ctx.destroy_buffer(C);
}

void t_glsl(vk::Context& ctx) {
    std::printf("[6] GLSL.std.450 y comparaciones no ordenadas\n");
    KernelOptions o; o.local_size = 64;
    std::vector<u32> code;
    {
        Kernel k(o);
        Buf in = k.buffer(0, Elem::F32, true), out = k.buffer(1, Elem::F32, false, true);
        const U i = k.global_id(0);
        const F x = k.ldf(in, i);
        const F y = sqrt(fabs(x)) + fma(x, F(2.0f), F(1.0f)) + fmax(x, F(0.5f)) + clamp(x, F(-1.0f), F(1.0f)) - fmin(x, F(0.0f));
        const F flag = select(ngt(x, F(0.2f)) || nlt(x, F(5.0f)), F(1000.0f), F(0.0f));
        k.stf(out, i, y + flag + to_f(to_u(fabs(x))));
        code = k.finish();
    }
    Run r(ctx, code, 2, 0);
    CHECK(r.ok, "pipeline glsl");
    if (!r.ok) return;
    const u32 n = 256;
    vk::Buffer I, O;
    ctx.create_buffer(I, n * 4, vk::Mem::Cached); ctx.create_buffer(O, n * 4, vk::Mem::Cached);
    for (u32 i = 0; i < n; ++i) I.as<float>()[i] = (float(i) - 100.0f) * 0.07f;
    I.as<float>()[5] = std::nanf("");
    ctx.flush(I);
    r.go({&I, &O}, n / 64);
    int bad = 0;
    for (u32 i = 0; i < n; ++i) {
        const float x = I.as<float>()[i];
        const float g = O.as<float>()[i];
        if (i == 5) { bad += !(std::isnan(g) || g >= 1000.0f); continue; }
        const float ref = std::sqrt(std::fabs(x)) + (2 * x + 1) + std::fmax(x, 0.5f) + std::fmin(std::fmax(x, -1.0f), 1.0f) - std::fmin(x, 0.0f) +
                          ((!(x > 0.2f) || !(x < 5.0f)) ? 1000.0f : 0.0f) + float(u32(std::fabs(x)));
        bad += std::fabs(g - ref) > 1e-4f * (1 + std::fabs(ref));
    }
    CHECK(bad == 0, "glsl: %d errores", bad);
    ctx.destroy_buffer(I); ctx.destroy_buffer(O);
}

} // namespace

int main() {
    std::printf("== test_gpu_spirv: emisor SPIR-V propio en la iGPU\n");
    vk::Context ctx;
    std::string err;
    if (!ctx.init(&err)) {
        std::printf("PASA (sin dispositivo Vulkan utilizable: %s)\n", err.c_str());
        return 0;
    }
    std::printf("%s", vk::describe(ctx.info()).c_str());
    t_vadd(ctx);
    t_f16(ctx);
    t_u8_if(ctx);
    for (u32 sg : {8u, 16u, 32u})
        if (sg >= ctx.info().sg_min && sg <= ctx.info().sg_max) { t_reduce(ctx, sg); t_shuffle(ctx, sg); }
    t_glsl(ctx);
    std::printf(g_fail ? "FALLA (%d)\n" : "PASA\n", g_fail);
    return g_fail ? 1 : 0;
}
