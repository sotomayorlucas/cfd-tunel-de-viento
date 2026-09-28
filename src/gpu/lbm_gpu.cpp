// ============================================================================
//  gpu/lbm_gpu.cpp — backend iGPU del LBM: memoria, pipelines, lotes de pasos,
//  publicación de fuerzas y campos (ver lbm_gpu.hpp y docs/GPU.md).
// ============================================================================
#include "lbm_gpu.hpp"
#include "lbm_kernels.hpp"
#include "../core/threadpool.hpp"
#include "../core/util.hpp"
#include "../lbm/lattice.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cfd::gpu {

namespace L = lbm::d3q19;
using namespace lbmk;

namespace {
constexpr u32 kPipeBindings = kNumBindings;
constexpr u32 kRecsPerChunk = 256;   // = grupo de trabajo del kernel de reducción
u32 fbits(float f) { return std::bit_cast<u32>(f); }
}

struct LbmGpu::Impl {
    vk::Context ctx;
    LbmGpuTuning tun;
    LbmGpuStats st;
    std::string err;
    lbm::Solver* s = nullptr;

    Spec spec;
    bool have_pipes = false;
    vk::Context::Pipeline p_step[2][2];   // [paridad][macro]
    vk::Context::Pipeline p_bnd[2];       // [paridad]
    vk::Context::Pipeline p_red;
    vk::Buffer ddf, flags, sid, tau, macro, motion, params, bad, nodes, recs, chunks, fout;
    vk::VkDescriptorSet set[2] = {0, 0};   // por paridad: las vistas de ubicación (kBindLoc) cambian

    u64 rev = 0;
    bool uploaded = false;
    bool gpu_newer = false;         // la GPU tiene pasos que la CPU no tiene
    u64 N = 0, Npad = 0;
    u32 gx = 1, gy = 1;
    usize n_nodes = 0, n_recs = 0, n_chunks = 0;
    std::vector<u8> chunk_id;
    u32 fout_cap = 0;               // pasos que caben en fout
    int cur_half = 0;               // mitad de los campos macro con el último lote publicado

    // Lote en vuelo y lote terminado pendiente de publicar (cycle: el siguiente se encola ANTES de publicar)
    u64 inflight = 0;
    int inflight_n = 0;
    int inflight_parity = 0;
    int inflight_half = 0;          // mitad de campos macro que escribe
    int inflight_region = 0;        // región de fout (alterna por lote)
    double t_submit = 0;
    u64 batch_seq = 0;
    int last_written_half = 0;      // mitad escrita por el último lote encolado (o la vigente tras subir)
    struct Done { bool valid = false; int n = 0, half = 0, region = 0; bool bad = false; double t_submit = 0, t_done = 0; };
    Done done;

    struct Cmd { int n = 0, parity = 0; vk::VkCommandBuffer cb = nullptr; };
    std::vector<Cmd> cmds;
    vk::VkCommandBuffer scratch = nullptr;

    ~Impl() { release_all(); }

    void release_all() {
        if (!ctx.ok()) return;
        ctx.wait_idle();
        for (auto& c : cmds) ctx.free_cmd(c.cb);
        cmds.clear();
        if (scratch) { ctx.free_cmd(scratch); scratch = nullptr; }
        destroy_pipes();
        for (vk::Buffer* b : {&ddf, &flags, &sid, &tau, &macro, &motion, &params, &bad, &nodes, &recs, &chunks, &fout}) ctx.destroy_buffer(*b);
        ctx.reset_sets();
        set[0] = set[1] = 0;
    }
    void destroy_pipes() {
        for (auto& a : p_step) for (auto& p : a) ctx.destroy_pipeline(p);
        for (auto& p : p_bnd) ctx.destroy_pipeline(p);
        ctx.destroy_pipeline(p_red);
        have_pipes = false;
    }
    void clear_cmds() {
        for (auto& c : cmds) ctx.free_cmd(c.cb);
        cmds.clear();
    }

    static bool same(const Spec& a, const Spec& b) {
        if (a.nx != b.nx || a.ny != b.ny || a.nz != b.nz || a.N != b.N || a.S != b.S || a.P != b.P) return false;
        if (a.fp16 != b.fp16 || a.regularized != b.regularized || a.wall != b.wall || a.interp != b.interp || a.slip != b.slip) return false;
        if (a.gauge != b.gauge || a.galilean != b.galilean || a.wg != b.wg || a.sg != b.sg || a.wg_nodes != b.wg_nodes || a.gx != b.gx) return false;
        if (a.pair != b.pair || a.ground != b.ground || a.classify != b.classify) return false;
        return a.rte16 == b.rte16 && a.denorm16 == b.denorm16;
    }

    bool build_pipes(const Spec& sp) {
        destroy_pipes();
        clear_cmds();
        const double t0 = now_sec();
        std::string e;
        for (int p = 0; p < 2; ++p)
            for (int m = 0; m < 2; ++m)
                if (!ctx.create_pipeline(p_step[p][m], gen_step(sp, p, m != 0), kPipeBindings, kPushBytes, sp.sg, &e)) {
                    err = "shader de celdas: " + e;
                    return false;
                }
        for (int p = 0; p < 2; ++p)
            if (!ctx.create_pipeline(p_bnd[p], gen_boundary(sp, p), kPipeBindings, kPushBytes, 0, &e)) { err = "shader de contorno: " + e; return false; }
        if (!ctx.create_pipeline(p_red, gen_reduce(sp), kPipeBindings, kPushBytes, sp.sg, &e)) { err = "shader de reducción: " + e; return false; }
        spec = sp;
        have_pipes = true;
        if (std::getenv("CFD_GPU_VERBOSE")) std::fprintf(stderr, "[gpu] 7 pipelines generados y compilados en %.0f ms\n", (now_sec() - t0) * 1e3);
        return true;
    }

    // (Re)asigna un búfer si su tamaño cambia. true si se reasignó.
    bool ensure(vk::Buffer& b, u64 bytes, vk::Mem m) {
        bytes = std::max<u64>(bytes, 64);
        const u64 want = (bytes + 63) & ~u64(63);
        if (b.buf && b.size == want && b.type == static_cast<u32>(ctx.mem_type(m))) return false;
        ctx.destroy_buffer(b);
        if (!ctx.create_buffer(b, bytes, m)) { err = "sin memoria para un búfer de la GPU"; return true; }
        return true;
    }

    void make_set() {
        ctx.reset_sets();
        const u64 es = spec.fp16 ? 2 : 4;
        const u64 total = 19 * spec.S * es;
        for (int par = 0; par < 2; ++par) {
            vk::Context::Bind b[kNumBindings];
            for (u32 q = 0; q < kNumBindings; ++q) b[q] = {&flags, 0, 0};   // relleno para bindings sin uso
            for (u32 i = 0; i < 19; ++i) {
                int slot = 0;
                u64 elem = 0;
                location(spec, par, static_cast<int>(i), &slot, &elem);
                const u64 off = (static_cast<u64>(slot) * spec.S + elem) * es;   // múltiplo de 4 B (ver location)
                const u64 range = std::min<u64>((spec.N + 64) * es, total - off);
                b[kBindLoc + i] = b[kBindLocAlt + i] = {&ddf, off, range};
                b[kBindSlot + i] = {&ddf, i * spec.S * es, spec.S * es};
            }
            b[kBindFlags] = {&flags, 0, 0}; b[kBindSid] = {&sid, 0, 0}; b[kBindClass] = {&tau, 0, 0}; b[kBindMacro] = {&macro, 0, 0};
            b[kBindMotion] = {&motion, 0, 0}; b[kBindParams] = {&params, 0, 0}; b[kBindBad] = {&bad, 0, 0}; b[kBindNodes] = {&nodes, 0, 0};
            b[kBindRecs] = {&recs, 0, 0}; b[kBindChunks] = {&chunks, 0, 0}; b[kBindForces] = {&fout, 0, 0};
            b[kBindFlags32] = {&flags, 0, 0};
            set[par] = ctx.make_set(p_step[0][0], b, kNumBindings);
        }
        clear_cmds();
    }

    bool upload();
    void download();
    void ladd_fixup(void* base, float sign);
    void write_params(int n, u64 t0, int half, int region);
    vk::VkCommandBuffer command_buffer(int n, int parity);
    void record(vk::VkCommandBuffer cb, int n, int parity);
    bool submit(int n);
    void complete();
    int publish();
    int wait() { complete(); return publish(); }
    int cycle(int n);
};

// ---- Subida completa del estado de la CPU ----------------------------------------------------------
bool LbmGpu::Impl::upload() {
    const double t0 = now_sec();
    const lbm::Config& c = s->config();
    const lbm::ExternalView v = s->external_view();
    Spec sp;
    sp.nx = c.nx; sp.ny = c.ny; sp.nz = c.nz;
    sp.N = static_cast<u64>(v.N); sp.S = static_cast<u64>(v.S); sp.P = static_cast<u64>(v.P);
    for (int k = 0; k < 19; ++k) sp.off[k] = v.off[k];
    sp.fp16 = c.precision == lbm::Precision::FP16S;
    sp.regularized = c.collision == lbm::Collision::Regularized;
    sp.wall = c.wall_model != lbm::WallModel::None;
    sp.interp = c.bounce == lbm::BounceBack::Interpolated;
    sp.slip = sp.interp && c.wall_model == lbm::WallModel::Slip;
    sp.ground = c.ground == lbm::GroundMode::None ? 0 : (c.ground == lbm::GroundMode::Static ? 1 : 2);
    sp.gauge = c.force_gauge;
    sp.galilean = c.force_galilean;
    const vk::DeviceInfo& di = ctx.info();
    sp.sg = std::clamp(tun.sg, std::max(di.sg_min, 8u), std::max(di.sg_max, 8u));
    sp.wg = std::clamp(tun.wg, sp.sg, di.max_wg_invocations);
    sp.wg -= sp.wg % sp.sg;
    sp.wg_nodes = std::clamp(tun.wg_nodes, 8u, di.max_wg_invocations);
    sp.rte16 = di.rte16;
    sp.denorm16 = di.denorm_preserve16;
    sp.pair = tun.pair;
    sp.classify = std::getenv("CFD_GPU_CLASSIFY") != nullptr;
    N = sp.N;
    static const int exp_mode = std::getenv("CFD_GPU_EXP") ? std::atoi(std::getenv("CFD_GPU_EXP")) : 0;
    if (exp_mode >= 4) sp.pair = true;       // experimentos empaquetados (docs/opt/gpu.md)
    else if (exp_mode) sp.pair = false;
    const u64 lanes = sp.pair ? N / 2 : N;   // modo par: un hilo por par de celdas contiguas en x
    const u64 groups = (lanes + sp.wg - 1) / sp.wg;
    gy = static_cast<u32>((groups + 65534) / 65535);
    gx = static_cast<u32>((groups + gy - 1) / gy);
    sp.gx = gx;
    Npad = static_cast<u64>(gx) * gy * sp.wg * (sp.pair ? 2 : 1);
    Npad = (Npad + 3) & ~u64(3);   // vista u32 de los flags
    if (sp.S * (sp.fp16 ? 2 : 4) > di.max_storage_range) { err = "dominio demasiado grande para un descriptor"; return false; }
    if (!have_pipes || !same(sp, spec)) {
        if (!build_pipes(sp)) return false;
    }
    const u64 es = sp.fp16 ? 2 : 4;
    bool re = false;
    re |= ensure(ddf, 19 * sp.S * es, tun.ddf_mem);
    re |= ensure(flags, Npad, vk::Mem::Cached);
    re |= ensure(sid, Npad, vk::Mem::Cached);
    re |= ensure(tau, static_cast<u64>(gx) * gy * 4, vk::Mem::Cached);   // clase por grupo de trabajo (kBindClass)
    const bool mac_re = ensure(macro, 8 * N * 4, vk::Mem::Cached);   // la CPU lee: CACHED (WC ≈ 0.25 GB/s)
    re |= mac_re;
    re |= ensure(motion, 256 * kMotionWords * 4, vk::Mem::Cached);
    re |= ensure(params, (kParamHeader + kParamPerStep * kMaxBatch) * 4, vk::Mem::Cached);
    re |= ensure(bad, 64, vk::Mem::Cached);
    if (!err.empty()) return false;

    // Poblaciones (misma disposición que la CPU: dirección k en (k·S + P)·es).
    std::memcpy(ddf.map, v.ddf, 19 * sp.S * es);
    spec.fp16 = sp.fp16; spec.S = sp.S; spec.P = sp.P;   // (ya iguales si no se regeneraron los pipelines)
    ladd_fixup(ddf.map, -1.0f);
    ctx.flush(ddf);
    // Flags y sid con relleno sólido hasta Npad (hilos sobrantes del despacho).
    std::memcpy(flags.map, v.flags, N);
    std::memset(flags.as<u8>() + N, lbm::kSolid, Npad - N);
    std::memcpy(sid.map, v.sid, N);
    std::memset(sid.as<u8>() + N, 0, Npad - N);
    ctx.flush(flags); ctx.flush(sid);
    {   // Grupos "puros": todos sus flags = los deducibles de las coordenadas (el kernel no los carga).
        u32* C = tau.as<u32>();
        const u64 ngroups = static_cast<u64>(gx) * gy;
        const u64 cells_per = static_cast<u64>(sp.wg) * (sp.pair ? 2 : 1);
        std::atomic<u64> npure{0};
        parallel_for(0, static_cast<i64>(ngroups), 64, [&](i64 lo, i64 hi) {
            u64 cnt = 0;
            for (i64 w = lo; w < hi; ++w) {
                bool pure = true;
                const u64 a = static_cast<u64>(w) * cells_per, b = std::min<u64>(a + cells_per, N);
                for (u64 n = a; n < b && pure; ++n) {
                    const int x = static_cast<int>(n % static_cast<u64>(sp.nx));
                    const u64 r = n / static_cast<u64>(sp.nx);
                    const int y = static_cast<int>(r % static_cast<u64>(sp.ny)), z = static_cast<int>(r / static_cast<u64>(sp.ny));
                    pure = v.flags[n] == expected_flags(sp, x, y, z);
                }
                C[w] = pure ? 1u : 0u;
                cnt += pure;
            }
            npure.fetch_add(cnt, std::memory_order_relaxed);
        });
        st.pure_groups = static_cast<double>(npure.load()) / static_cast<double>(std::max<u64>(ngroups, 1));
        ctx.flush(tau);
    }

    // Nodos de pared: empaquetado + registros (nodo, id) ordenados por id + trozos por id.
    n_nodes = v.n_nodes;
    struct Rec { u8 id; u32 node, slot; };
    std::vector<Rec> rv;
    rv.reserve(n_nodes + n_nodes / 8);
    std::vector<u32> packed(std::max<usize>(n_nodes, 1) * kNodeWords, 0u);
    st.multi_id_nodes = st.overflow_nodes = 0;
    for (usize i = 0; i < n_nodes; ++i) {
        const lbm::WallNode& nd = v.nodes[i];
        u32* w = packed.data() + i * kNodeWords;
        w[0] = nd.n; w[1] = nd.mask;
        w[2] = u32(nd.x) | (u32(nd.y) << 16);
        w[3] = u32(nd.z) | (u32(nd.kind) << 16);
        w[4] = fbits(nd.nx); w[5] = fbits(nd.ny); w[6] = fbits(nd.nz);
        w[7] = fbits(nd.yw); w[8] = fbits(nd.yw17); w[9] = fbits(nd.sgeo);
        std::memcpy(w + 10, nd.q, 20);
        u8 ids[kSlots] = {};
        u32 nid = 0;
        bool over = false;
        for (int k = 1; k < 19; ++k) {
            if (!(nd.mask & (1u << k))) continue;
            const u8 id = v.sid[static_cast<i64>(nd.n) + v.off[k]];
            bool found = false;
            for (u32 j = 0; j < nid; ++j) found |= ids[j] == id;
            if (found) continue;
            if (nid < kSlots) ids[nid++] = id;
            else over = true;
        }
        st.multi_id_nodes += nid > 1;
        st.overflow_nodes += over;
        w[15] = u32(ids[0]) | (u32(ids[1]) << 8) | (u32(ids[2]) << 16) | (u32(ids[3]) << 24);
        for (u32 j = 0; j < kSlots; ++j) {
            w[16 + j] = 0xFFFFFFFFu;
            if (j < nid) rv.push_back({ids[j], static_cast<u32>(i), j});
        }
    }
    std::stable_sort(rv.begin(), rv.end(), [](const Rec& a, const Rec& b) { return a.id < b.id; });
    n_recs = rv.size();
    std::vector<u32> ch;
    chunk_id.clear();
    for (usize r = 0; r < n_recs;) {
        usize e = r;
        while (e < n_recs && rv[e].id == rv[r].id && e - r < kRecsPerChunk) ++e;
        ch.push_back(static_cast<u32>(r));
        ch.push_back(static_cast<u32>(e - r));
        chunk_id.push_back(rv[r].id);
        r = e;
    }
    for (usize r = 0; r < n_recs; ++r) packed[rv[r].node * kNodeWords + 16 + rv[r].slot] = static_cast<u32>(r);
    n_chunks = chunk_id.size();
    re |= ensure(nodes, packed.size() * 4, vk::Mem::Cached);
    re |= ensure(recs, std::max<usize>(n_recs, 1) * 6 * 4, vk::Mem::Cached);
    re |= ensure(chunks, std::max<usize>(ch.size(), 2) * 4, vk::Mem::Cached);
    const u32 cap = std::max<u32>(fout_cap, 64);
    re |= ensure(fout, 2ull * cap * std::max<usize>(n_chunks, 1) * 6 * 4, vk::Mem::Cached);   // 2 regiones (lotes alternos)
    fout_cap = cap;
    if (!err.empty()) return false;
    std::memcpy(nodes.map, packed.data(), packed.size() * 4);
    if (!ch.empty()) std::memcpy(chunks.map, ch.data(), ch.size() * 4);
    ctx.flush(nodes); ctx.flush(chunks);
    st.nodes = n_nodes; st.records = n_recs; st.chunks = n_chunks;

    // Campos macro: si los vigentes son los de la CPU (arranque, reset, nuevo dominio) se copian a la mitad 0.
    const bool cpu_current = s->field().rho == v.rho;
    if (mac_re || cpu_current) {
        float* M = macro.as<float>();
        std::memcpy(M, v.rho, N * 4);
        std::memcpy(M + N, v.ux, N * 4);
        std::memcpy(M + 2 * N, v.uy, N * 4);
        std::memcpy(M + 3 * N, v.uz, N * 4);
        ctx.flush(macro, 0, 4 * N * 4);
        cur_half = 0;
    }
    last_written_half = cur_half;
    const float* M = macro.as<float>() + static_cast<u64>(cur_half) * 4 * N;
    s->set_field_override(M, M + N, M + 2 * N, M + 3 * N);
    if (re || !set[0]) make_set();
    rev = s->revision();
    uploaded = true;
    gpu_newer = false;
    st.upload_ms = (now_sec() - t0) * 1e3;
    return true;
}

// ---- Convención de Ladd en los relevos CPU ↔ GPU ------------------------------------------------------
// La CPU suma el término de pared móvil implícita (Ladd) AL CARGAR la población que llega desde el sólido; la GPU
// lo deja ya sumado en memoria (kernel de nodos: in − Sc·l(t+1)). En cada relevo, las posiciones que se cargarán
// en el paso t = steps() se corrigen: subida → −Sc·l(t) (convención GPU); bajada → +Sc·l(t) (convención CPU).
// Mismas fórmulas que boundary_pass/block_scalar (corrección de masa de las paredes impermeables incluida).
void LbmGpu::Impl::ladd_fixup(void* base, float sign) {
    const lbm::Config& c = s->config();
    const lbm::ExternalView v = s->external_view();
    const bool interp = c.bounce == lbm::BounceBack::Interpolated;
    const u64 t = s->steps();
    float uc = 0, r = 1;
    s->ramp_at(t, &uc, &r);
    const float ug = c.ground == lbm::GroundMode::Moving ? uc : 0.0f;
    const int par = static_cast<int>(t & 1);
    const bool fp16 = spec.fp16;
    const float sc = fp16 ? 32768.0f : 1.0f;
    auto wall_vel = [&](int id, Vec3 p) -> Vec3 {
        if (id == lbm::k_ground_id) return {ug, 0, 0};
        const lbm::WallMotion& m = v.motion[id];
        if (p.z <= m.contact_z) return {ug, 0, 0};
        return (m.v + cross(m.omega, p - m.center)) * r;
    };
    for (usize i = 0; i < v.n_nodes; ++i) {
        const lbm::WallNode& nd = v.nodes[i];
        if (!(nd.kind & 1)) continue;
        auto implicit_moving = [&](int k, int* id) {
            const i64 sidx = static_cast<i64>(nd.n) + v.off[k];
            *id = v.sid[sidx];
            return (v.flags[sidx] & lbm::kMoving) && (!interp || *id == lbm::k_ground_id);
        };
        float Lm = 0, Wm = 0;
        for (int k = 1; k < 19; ++k) {
            int id = 0;
            if (!(nd.mask & (1u << k)) || !implicit_moving(k, &id) || !v.motion[id].impermeable || id == lbm::k_ground_id) continue;
            const Vec3 ck(float(L::c[k][0]), float(L::c[k][1]), float(L::c[k][2]));
            const Vec3 uw = wall_vel(id, Vec3(nd.x, nd.y, nd.z) + ck);
            Lm += 6.0f * L::w[k] * dot(ck, uw);
            Wm += L::w[k];
        }
        const float lcorr = Wm > 0.0f ? Lm / Wm : 0.0f;
        for (int k = 1; k < 19; ++k) {
            int id = 0;
            if (!(nd.mask & (1u << k)) || !implicit_moving(k, &id)) continue;
            const Vec3 ck(float(L::c[k][0]), float(L::c[k][1]), float(L::c[k][2]));
            const Vec3 uw = wall_vel(id, Vec3(nd.x, nd.y, nd.z) + ck);
            const bool cons = id != lbm::k_ground_id && v.motion[id].impermeable;
            const float l = 6.0f * L::w[k] * dot(ck, uw) - (cons ? lcorr * L::w[k] : 0.0f);
            int slot = 0;
            i64 off = 0;
            load_slot(spec, par, L::opp[k], &slot, &off);
            const u64 e = static_cast<u64>(slot) * spec.S + spec.P + static_cast<u64>(static_cast<i64>(nd.n) + off);
            if (fp16) {
                u16* p = static_cast<u16*>(base) + e;
                *p = f32_to_f16(f16_to_f32(*p) + sign * sc * l);
            } else {
                float* p = static_cast<float*>(base) + e;
                *p += sign * sc * l;
            }
        }
    }
}

// ---- Bajada de las poblaciones a la CPU --------------------------------------------------------------
void LbmGpu::Impl::download() {
    if (!gpu_newer || !s) return;
    const double t0 = now_sec();
    const lbm::ExternalView v = s->external_view();
    const u64 bytes = 19 * spec.S * (spec.fp16 ? 2 : 4);
    ctx.invalidate(ddf);
    std::memcpy(v.ddf, ddf.map, bytes);
    ladd_fixup(v.ddf, +1.0f);
    gpu_newer = false;
    st.download_ms = (now_sec() - t0) * 1e3;
}

// ---- Parámetros por lote (rampa, velocidades de pared, modelo de pared) ------------------------------
void LbmGpu::Impl::write_params(int n, u64 t0, int half, int region) {
    const lbm::Config& c = s->config();
    const lbm::ExternalView v = s->external_view();
    float* P = params.as<float>();
    P[kPK] = 18.0f * std::sqrt(2.0f) * c.cs_smag * c.cs_smag;
    P[kPWallC3] = v.wall_c3;
    P[kPWallFloor] = v.wall_floor;
    P[kPMacroBase] = std::bit_cast<float>(static_cast<u32>(static_cast<u64>(half) * 4 * N));
    const float nu_w = std::max(c.wall_nu > 0.0f ? c.wall_nu : c.nu, 1e-9f);
    P[kPNuW] = nu_w;
    P[kPInvNuW] = 1.0f / nu_w;
    P[kPWwA0] = 8.3f * std::pow(1.0f / nu_w, 1.0f / 7.0f);
    {   // τ0(x) de la esponja (= Solver::Impl::build_tau; kSpongeNu = 0.12)
        const float nu = std::max(c.nu, 1e-7f);
        const int xs = static_cast<int>(static_cast<float>(c.nx) * (1.0f - std::clamp(c.sponge_frac, 0.0f, 0.9f)));
        P[kPNu] = nu;
        P[kPXs] = static_cast<float>(xs);
        P[kPNuMax] = std::max(nu, 0.12f);
        P[kPSponge] = (c.sponge_frac > 0.0f && c.nx - 1 > xs) ? 1.0f : 0.0f;
    }
    const bool belt = c.ground == lbm::GroundMode::Moving;
    const usize fbase = static_cast<usize>(region) * fout_cap * n_chunks * 6;
    for (int i = 0; i < n; ++i) {
        float* q = P + kParamHeader + static_cast<usize>(i) * kParamPerStep;
        float u, r, u1, r1;
        s->ramp_at(t0 + static_cast<u64>(i), &u, &r);
        s->ramp_at(t0 + static_cast<u64>(i) + 1, &u1, &r1);
        q[kSUin] = u; q[kSR] = r; q[kSUg] = belt ? u : 0.0f;
        q[kSR1] = r1; q[kSUg1] = belt ? u1 : 0.0f;
        q[kSFout] = std::bit_cast<float>(static_cast<u32>(fbase + static_cast<usize>(i) * n_chunks * 6));
    }
    ctx.flush(params, 0, (kParamHeader + kParamPerStep * static_cast<u64>(n)) * 4);
    // Tabla de movimiento SIN rampa (el shader multiplica por r); el suelo (255): contact_z = +∞ → cinta.
    float* M = motion.as<float>();
    std::memset(M, 0, 256 * kMotionWords * 4);
    for (int id = 0; id < 255; ++id) {
        const lbm::WallMotion& m = v.motion[id];
        float* e = M + id * kMotionWords;
        e[0] = m.v.x; e[1] = m.v.y; e[2] = m.v.z;
        e[3] = m.omega.x; e[4] = m.omega.y; e[5] = m.omega.z;
        e[6] = m.center.x; e[7] = m.center.y; e[8] = m.center.z;
        e[9] = m.contact_z;
        e[10] = m.impermeable ? 1.0f : 0.0f;
    }
    M[255 * kMotionWords + 9] = INFINITY;
    ctx.flush(motion);
    bad.as<u32>()[0] = 0;
    ctx.flush(bad);
}

// ---- Búfer de comandos de un lote: n × (celdas → barrera → contorno → barrera → reducción) --------------
void LbmGpu::Impl::record(vk::VkCommandBuffer cb, int n, int parity) {
    const auto& f = ctx.fn();
    const u32 nq = tun.profile ? 2 + 3 * static_cast<u32>(n) : 2;
    vk::VkQueryPool qp = ctx.timestamps(2 + 3 * kMaxBatch);
    ctx.begin(cb, !tun.reuse_cmd);
    f.CmdResetQueryPool(cb, qp, 0, nq);
    f.CmdWriteTimestamp(cb, vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    const u32 node_groups = static_cast<u32>((n_nodes + spec.wg_nodes - 1) / spec.wg_nodes);
    for (int i = 0; i < n; ++i) {
        const int p = (parity + i) & 1;
        const u32 pc[4] = {static_cast<u32>(i), static_cast<u32>(n_nodes), 0, 0};
        f.CmdBindDescriptorSets(cb, vk::PIPELINE_BIND_POINT_COMPUTE, p_step[0][0].layout, 0, 1, &set[p], 0, nullptr);
        f.CmdPushConstants(cb, p_step[0][0].layout, vk::SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
        f.CmdBindPipeline(cb, vk::PIPELINE_BIND_POINT_COMPUTE, p_step[p][i == n - 1].pipe);
        f.CmdDispatch(cb, gx, gy, 1);
        ctx.barrier_compute(cb);
        if (tun.profile) f.CmdWriteTimestamp(cb, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1 + 3 * static_cast<u32>(i));
        if (n_nodes) {
            f.CmdBindPipeline(cb, vk::PIPELINE_BIND_POINT_COMPUTE, p_bnd[p].pipe);
            f.CmdDispatch(cb, node_groups, 1, 1);
            ctx.barrier_compute(cb);
        }
        if (tun.profile) f.CmdWriteTimestamp(cb, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2 + 3 * static_cast<u32>(i));
        if (n_chunks) {
            f.CmdBindPipeline(cb, vk::PIPELINE_BIND_POINT_COMPUTE, p_red.pipe);
            f.CmdDispatch(cb, static_cast<u32>(n_chunks), 1, 1);
            // Sin barrera: la reducción del paso i corre a la vez que el kernel de celdas del paso i+1
            // (no comparten datos); la barrera tras ese kernel ordena reducción(i) → contorno(i+1).
            if (tun.profile) ctx.barrier_compute(cb);
        }
        if (tun.profile) f.CmdWriteTimestamp(cb, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 3 + 3 * static_cast<u32>(i));
    }
    ctx.barrier_host(cb);
    f.CmdWriteTimestamp(cb, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, tun.profile ? nq - 1 : 1);
    ctx.end(cb);
}

vk::VkCommandBuffer LbmGpu::Impl::command_buffer(int n, int parity) {
    if (tun.reuse_cmd) {
        for (auto& c : cmds)
            if (c.n == n && c.parity == parity) { st.record_us = 0; return c.cb; }
        if (cmds.size() >= 16) { ctx.free_cmd(cmds.front().cb); cmds.erase(cmds.begin()); }
        Cmd c{n, parity, ctx.alloc_cmd()};
        const double t0 = now_sec();
        record(c.cb, n, parity);
        st.record_us = (now_sec() - t0) * 1e6;
        cmds.push_back(c);
        return c.cb;
    }
    if (!scratch) scratch = ctx.alloc_cmd();
    const double t0 = now_sec();
    record(scratch, n, parity);
    st.record_us = (now_sec() - t0) * 1e6;
    return scratch;
}

bool LbmGpu::Impl::submit(int n) {
    if (!s) { err = "sin solver enganchado"; return false; }
    if (inflight) complete();
    n = std::clamp(n, 1, static_cast<int>(kMaxBatch));
    if (!uploaded || s->revision() != rev) {
        publish();   // (el gancho ya lo habrá hecho; por si acaso)
        err.clear();
        if (!upload()) return false;
    }
    if (static_cast<u32>(n) > fout_cap) {   // capacidad de fuerzas por paso: crece a potencias de 2
        publish();                           // la región del lote pendiente se reasigna
        u32 cap = fout_cap;
        while (cap < static_cast<u32>(n)) cap *= 2;
        cap = std::min<u32>(cap, kMaxBatch);
        fout_cap = cap;
        ensure(fout, 2ull * cap * std::max<usize>(n_chunks, 1) * 6 * 4, vk::Mem::Cached);
        if (!err.empty()) return false;
        make_set();
    }
    // Un lote terminado sin publicar (cycle) ya avanzó el tiempo: el siguiente empieza tras él.
    const u64 t0 = s->steps() + (done.valid ? static_cast<u64>(done.n) : 0);
    const int half = 1 - last_written_half;
    const int region = static_cast<int>(batch_seq & 1);
    write_params(n, t0, half, region);
    const int parity = static_cast<int>(t0 & 1);
    vk::VkCommandBuffer cb = command_buffer(n, parity);
    t_submit = now_sec();
    inflight = ctx.submit(cb);
    if (!inflight) { err = "vkQueueSubmit falló"; return false; }
    inflight_n = n;
    inflight_parity = parity;
    inflight_half = half;
    inflight_region = region;
    last_written_half = half;
    ++batch_seq;
    return true;
}

// Espera el lote en vuelo y guarda lo que el siguiente lote pisaría (marcas de tiempo, divergencia).
void LbmGpu::Impl::complete() {
    if (!inflight) return;
    publish();   // como mucho un lote pendiente
    ctx.wait(inflight);
    done.valid = true;
    done.n = inflight_n;
    done.half = inflight_half;
    done.region = inflight_region;
    done.t_submit = t_submit;
    done.t_done = now_sec();
    inflight = 0;
    gpu_newer = true;
    const int n = done.n;
    const u32 nq = tun.profile ? 2 + 3 * static_cast<u32>(n) : 2;
    std::vector<double> ts(nq);
    if (ctx.read_timestamps(0, nq, ts.data())) {
        st.gpu_ms = (ts[nq - 1] - ts[0]) * 1e-6;
        if (tun.profile) {
            double a = 0, b = 0, c = 0;
            for (int i = 0; i < n; ++i) {
                const double prev = ts[3 * static_cast<usize>(i)];
                a += ts[1 + 3 * static_cast<usize>(i)] - prev;
                b += ts[2 + 3 * static_cast<usize>(i)] - ts[1 + 3 * static_cast<usize>(i)];
                c += ts[3 + 3 * static_cast<usize>(i)] - ts[2 + 3 * static_cast<usize>(i)];
            }
            st.k_step_ms = a * 1e-6 / n; st.k_boundary_ms = b * 1e-6 / n; st.k_reduce_ms = c * 1e-6 / n;
        }
    }
    ctx.invalidate(bad, 0, 64);
    done.bad = bad.as<u32>()[0] != 0;
}

int LbmGpu::Impl::cycle(int n) {
    complete();                   // espera el lote anterior (si lo hay)…
    const bool ok = submit(n);    // …encola el siguiente enseguida (la GPU no espera a la publicación)…
    const int k = publish();      // …y publica el anterior mientras tanto
    return ok ? k : -1;
}

// ---- Publicación de un lote terminado --------------------------------------------------------------
int LbmGpu::Impl::publish() {
    if (!done.valid) return 0;
    done.valid = false;
    const int n = done.n;
    const double t_done = done.t_done;
    const double t0 = now_sec();
    // Fuerzas por paso: trozos → ids (doble precisión, como la CPU).
    lbm::ExternalStep r;
    r.steps = n;
    const Vec3 ref = s->moment_reference();
    const double rr[3] = {ref.x, ref.y, ref.z};
    double acc_f[256][3] = {}, acc_m[256][3] = {};
    const u64 rbase = static_cast<u64>(done.region) * fout_cap * n_chunks * 6;
    if (n_chunks) ctx.invalidate(fout, rbase * 4, static_cast<u64>(n) * n_chunks * 6 * 4);
    const float* FO = fout.as<float>() + rbase;
    for (int i = 0; i < n; ++i) {
        double f[256][3] = {}, m[256][3] = {};
        const float* row = FO + static_cast<usize>(i) * n_chunks * 6;
        for (usize ci = 0; ci < n_chunks; ++ci) {
            const int id = chunk_id[ci];
            for (int a = 0; a < 3; ++a) { f[id][a] += row[ci * 6 + static_cast<usize>(a)]; m[id][a] += row[ci * 6 + 3 + static_cast<usize>(a)]; }
        }
        const bool last = i == n - 1;
        double tf[3] = {0, 0, 0}, tm[3] = {0, 0, 0};
        for (int id = 0; id < 256; ++id) {
            const double mx = m[id][0] - (rr[1] * f[id][2] - rr[2] * f[id][1]);
            const double my = m[id][1] - (rr[2] * f[id][0] - rr[0] * f[id][2]);
            const double mz = m[id][2] - (rr[0] * f[id][1] - rr[1] * f[id][0]);
            acc_f[id][0] += f[id][0]; acc_f[id][1] += f[id][1]; acc_f[id][2] += f[id][2];
            acc_m[id][0] += mx; acc_m[id][1] += my; acc_m[id][2] += mz;
            if (last) {
                r.last.force[id] = Vec3(float(f[id][0]), float(f[id][1]), float(f[id][2]));
                r.last.moment[id] = Vec3(float(mx), float(my), float(mz));
                if (id >= 1 && id <= 254) {
                    for (int a = 0; a < 3; ++a) tf[a] += f[id][a];
                    tm[0] += mx; tm[1] += my; tm[2] += mz;
                }
            }
        }
        if (last) {
            r.last.total = Vec3(float(tf[0]), float(tf[1]), float(tf[2]));
            r.last.total_moment = Vec3(float(tm[0]), float(tm[1]), float(tm[2]));
        }
    }
    const double inv = 1.0 / n;
    Vec3 tf3{0, 0, 0}, tm3{0, 0, 0};
    for (int id = 0; id < 256; ++id) {
        r.mean.force[id] = Vec3(float(acc_f[id][0] * inv), float(acc_f[id][1] * inv), float(acc_f[id][2] * inv));
        r.mean.moment[id] = Vec3(float(acc_m[id][0] * inv), float(acc_m[id][1] * inv), float(acc_m[id][2] * inv));
        if (id >= 1 && id <= 254) { tf3 += r.mean.force[id]; tm3 += r.mean.moment[id]; }
    }
    r.mean.total = tf3;
    r.mean.total_moment = tm3;
    r.diverged = done.bad;
    const double gpu_s = st.gpu_ms > 0 ? st.gpu_ms * 1e-3 : (t_done - done.t_submit);
    r.mlups = static_cast<double>(N) * n / gpu_s * 1e-6;
    r.kernel_s = gpu_s;
    r.force_s = tun.profile ? (st.k_boundary_ms + st.k_reduce_ms) * 1e-3 * n : 0.0;
    // Campos macro del último paso: la mitad que acaba de escribir la GPU (invalidar: memoria no coherente).
    cur_half = done.half;
    const u64 off = static_cast<u64>(cur_half) * 4 * N;
    ctx.invalidate(macro, off * 4, 4 * N * 4);
    const float* M = macro.as<float>() + off;
    s->set_field_override(M, M + N, M + 2 * N, M + 3 * N);
    s->external_commit(r);
    st.steps = n;
    st.wall_ms = (t_done - done.t_submit) * 1e3;
    st.mlups = r.mlups;
    st.gbs = static_cast<double>(N) * n * 2.0 * 19.0 * (spec.fp16 ? 2 : 4) / gpu_s * 1e-9;
    st.publish_ms = (now_sec() - t0) * 1e3;
    return n;
}

// ================================================================================================
namespace {
void sync_hook(void* ctx) {
    auto* I = static_cast<LbmGpu::Impl*>(ctx);
    I->wait();
    I->download();
}
} // namespace

LbmGpu::LbmGpu() : impl_(new Impl) {}
LbmGpu::~LbmGpu() {
    detach();
    delete impl_;
}

bool LbmGpu::init(std::string* err, bool verbose) {
    if (impl_->ctx.ok()) return true;
    std::string e;
    const bool ok = impl_->ctx.init(&e, verbose);
    if (!ok) impl_->err = e;
    if (err) *err = e;
    return ok;
}
bool LbmGpu::ok() const { return impl_->ctx.ok(); }
const vk::DeviceInfo& LbmGpu::device() const { return impl_->ctx.info(); }
void LbmGpu::set_tuning(const LbmGpuTuning& t) {
    Impl& I = *impl_;
    I.wait();
    const bool mem = t.ddf_mem != I.tun.ddf_mem;
    I.tun = t;
    I.clear_cmds();
    if (I.s) {   // aplicar: descargar y forzar una subida completa (pipelines y memoria nuevos si cambian)
        I.download();
        if (mem) I.ctx.destroy_buffer(I.ddf);
        I.uploaded = false;
    }
}
const LbmGpuTuning& LbmGpu::tuning() const { return impl_->tun; }

bool LbmGpu::attach(lbm::Solver& s, std::string* err) {
    Impl& I = *impl_;
    if (!I.ctx.ok() && !init(err)) return false;
    if (I.s && I.s != &s) detach();
    I.s = &s;
    I.err.clear();
    I.uploaded = false;
    if (!I.upload()) {
        if (err) *err = I.err;
        I.s->set_field_override(nullptr, nullptr, nullptr, nullptr);
        I.s = nullptr;
        return false;
    }
    s.set_external(&sync_hook, impl_);
    return true;
}

void LbmGpu::detach() {
    Impl& I = *impl_;
    if (!I.s) return;
    I.wait();
    I.download();
    // Campos macro vigentes → CPU (field() vuelve a los propios del solver).
    const lbm::ExternalView v = I.s->external_view();
    const lbm::FieldView fv = I.s->field();
    if (fv.rho && fv.rho != v.rho) {
        std::memcpy(v.rho, fv.rho, I.N * 4);
        std::memcpy(v.ux, fv.ux, I.N * 4);
        std::memcpy(v.uy, fv.uy, I.N * 4);
        std::memcpy(v.uz, fv.uz, I.N * 4);
    }
    I.s->set_field_override(nullptr, nullptr, nullptr, nullptr);
    I.s->set_external(nullptr, nullptr);
    I.s = nullptr;
    I.uploaded = false;
}

bool LbmGpu::attached() const { return impl_->s != nullptr; }
bool LbmGpu::submit(int n) { return impl_->submit(n); }
int LbmGpu::wait() { return impl_->wait(); }
int LbmGpu::cycle(int n) { return impl_->cycle(n); }
bool LbmGpu::busy() const { return impl_->inflight != 0; }
const LbmGpuStats& LbmGpu::stats() const { return impl_->st; }
const std::string& LbmGpu::error() const { return impl_->err; }

bool gpu_available(std::string* why) {
    static int state = -1;
    static std::string reason;
    if (state < 0) {
        vk::Context c;
        state = c.init(&reason) ? 1 : 0;
    }
    if (why) *why = reason;
    return state == 1;
}

} // namespace cfd::gpu
