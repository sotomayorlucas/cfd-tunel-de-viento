// ============================================================================
//  tools/drm_probe.cpp — sonda "bare metal" de la iGPU Intel por ioctl DRM crudos.
//
//  Compilar (no depende de nada más del proyecto que src/gpu/drm_i915.cpp):
//    g++ -std=c++23 -O3 -march=native -Isrc -pthread tools/drm_probe.cpp
//        src/gpu/drm_i915.cpp -o build/drm/drm_probe
//  Uso:  build/drm/drm_probe [/dev/dri/renderD128] [--rapido]
//
//  Informe: versión del driver, GETPARAM, QUERY (motores, regiones de memoria,
//  topología, HWCONFIG), buffers GEM (PAT/mmap), lectura de RING_TIMESTAMP,
//  lotes MI_* escritos a mano en rcs/bcs/ccs (verificados desde la CPU), latencia de
//  envío (ioctl → GPU → CPU) y ancho de banda de copia en el motor de copia (blitter).
//  Todos los lotes son cortos (< ~0.1 s) y cada espera tiene timeout.
// ============================================================================
#include "gpu/drm_i915.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <vector>

using namespace cfd::gpu::drm;

namespace {

// Constantes de i915_drm.h (repetidas para no arrastrar la cabecera del kernel aquí).
constexpr int P_CHIPSET_ID = 4, P_REVISION = 32, P_SUBSLICE_TOTAL = 33, P_EU_TOTAL = 34,
              P_HAS_LLC = 17, P_HAS_SOFTPIN = 37, P_HAS_EXEC_NO_RELOC = 25, P_HAS_SCHEDULER = 41,
              P_SLICE_MASK = 46, P_SUBSLICE_MASK = 47, P_CS_TS_FREQ = 51, P_MMAP_GTT_VERSION = 40,
              P_CMD_PARSER = 28, P_HAS_CTX_ISOLATION = 50, P_OA_TS_FREQ = 57,
              P_HAS_TIMELINE = 55, P_HUC_STATUS = 42, P_MMAP_VERSION = 30, P_HAS_GPU_RESET = 35;
constexpr u64 CTX_PARAM_GTT_SIZE = 0x3;
constexpr i64 kTimeout = 2'000'000'000;   // 2 s: nada de lo que se envía debería tardar tanto

u64 now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return u64(ts.tv_sec) * 1'000'000'000ull + u64(ts.tv_nsec);
}
void sleep_ms(int ms) {
    timespec ts{ms / 1000, (ms % 1000) * 1'000'000L};
    nanosleep(&ts, nullptr);
}

struct Stats { double med = 0, p10 = 0, p90 = 0, min = 0; };
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, size_t(f * double(v.size() - 1) + 0.5))]; };
    s.med = q(0.5); s.p10 = q(0.1); s.p90 = q(0.9); s.min = v.front();
    return s;
}

std::string read_sysfs(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return "?";
    char buf[128] = {};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    std::string s(buf, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// Nombres de claves HWCONFIG (enum intel_hwconfig de IGT/i915, sólo las primeras y
// más estables; el resto se imprime con su número).
const char* hwconfig_name(u32 k) {
    switch (k) {
        case 1: return "MAX_SLICES_SUPPORTED";
        case 2: return "MAX_DUAL_SUBSLICES_SUPPORTED";
        case 3: return "MAX_NUM_EU_PER_DSS";
        case 4: return "NUM_PIXEL_PIPES";
        case 8: return "L3_CACHE_WAYS_SIZE_IN_BYTES";
        case 9: return "L3_CACHE_WAYS_PER_SECTOR";
        case 10: return "MAX_MEMORY_CHANNELS";
        case 11: return "MEMORY_TYPE";
        case 12: return "CACHE_TYPES";
        case 15: return "NUM_THREADS_PER_EU";
        case 23: return "MAX_RCS";
        case 24: return "MAX_CCS";
        case 25: return "MAX_VCS";
        case 26: return "MAX_VECS";
        case 27: return "MAX_COPY_CS";
        default: return nullptr;
    }
}

// ---- Estado de la sonda ----------------------------------------------------------
struct Probe {
    Device dev;
    int ts_freq = 0;
    u32 vm = 0, ctx = 0;
    std::vector<EngineInfo> map;   // mapa de motores del contexto (índice = selector)
    Bo res, batch;                 // resultados (WC) y lote (WC)
    u64 next_va = 0x0000'0001'0000'0000ull;   // direcciones softpin (4 GiB en adelante)
    bool rapido = false;
    bool gpu_ok = true;

    u64 alloc_va(u64 size) {
        const u64 va = next_va;
        next_va += (size + (2ull << 20) - 1) & ~((2ull << 20) - 1);   // alineado a 2 MiB
        next_va += 2ull << 20;                                        // hueco de guarda
        return va;
    }
    Batch begin() {
        Batch b;
        b.p = batch.u32p();
        b.cap = u32(batch.size / 4);
        return b;
    }
    // Envía el lote actual con 'res' (y objetos extra) y espera. Devuelve 0 ó -errno.
    int run(u32 eng, const Batch& b, const Bo* extra[] = nullptr, int n_extra = 0, bool wait = true) {
        ExecObj o[8];
        int n = 0;
        o[n++] = {&res, true};
        for (int i = 0; i < n_extra; ++i) o[n++] = {extra[i], true};
        o[n++] = {&batch, false};
        int r = execbuf(dev, ctx, eng, o, u32(n), b.bytes());
        if (r) return r;
        if (wait) {
            r = gem_wait(dev, batch.handle, kTimeout);
            if (r == -ETIME) {
                std::printf("  !! el lote no terminó en 2 s (¿GPU colgada?) — se abortan las pruebas GPU\n");
                gpu_ok = false;
            }
        }
        return r;
    }
    int engine_index(u16 cls) const {
        for (size_t i = 0; i < map.size(); ++i) if (map[i].cls == cls && map[i].instance == 0) return int(i);
        return -1;
    }
};

// ---- 1. Identificación ---------------------------------------------------------------
void report_params(Probe& P) {
    const Device& d = P.dev;
    struct { int id; const char* name; bool hex; } ps[] = {
        {P_CHIPSET_ID, "CHIPSET_ID (PCI device id)", true},
        {P_REVISION, "REVISION", false},
        {P_EU_TOTAL, "EU_TOTAL", false},
        {P_SUBSLICE_TOTAL, "SUBSLICE_TOTAL", false},
        {P_SLICE_MASK, "SLICE_MASK", true},
        {P_SUBSLICE_MASK, "SUBSLICE_MASK", true},
        {P_CS_TS_FREQ, "CS_TIMESTAMP_FREQUENCY (Hz)", false},
        {P_OA_TS_FREQ, "OA_TIMESTAMP_FREQUENCY (Hz)", false},
        {P_HAS_LLC, "HAS_LLC", false},
        {P_HAS_SOFTPIN, "HAS_EXEC_SOFTPIN", false},
        {P_HAS_EXEC_NO_RELOC, "HAS_EXEC_NO_RELOC", false},
        {P_HAS_SCHEDULER, "HAS_SCHEDULER (máscara)", true},
        {P_HAS_CTX_ISOLATION, "HAS_CONTEXT_ISOLATION (máscara)", true},
        {P_HAS_TIMELINE, "HAS_EXEC_TIMELINE_FENCES", false},
        {P_HAS_GPU_RESET, "HAS_GPU_RESET", false},
        {P_MMAP_VERSION, "MMAP_VERSION", false},
        {P_MMAP_GTT_VERSION, "MMAP_GTT_VERSION", false},
        {P_CMD_PARSER, "CMD_PARSER_VERSION", false},
        {P_HUC_STATUS, "HUC_STATUS", false},
    };
    std::printf("\n== I915_GETPARAM\n");
    for (auto& p : ps) {
        int v = 0;
        const int r = d.getparam(p.id, &v);
        if (r) std::printf("  %-34s  no disponible: %s\n", p.name, errstr(r).c_str());
        else if (p.hex) std::printf("  %-34s  0x%x\n", p.name, unsigned(v));
        else std::printf("  %-34s  %d\n", p.name, v);
        if (p.id == P_CS_TS_FREQ && !r) P.ts_freq = v;
    }
}

void report_queries(Probe& P) {
    const Device& d = P.dev;
    std::printf("\n== I915_QUERY: motores\n");
    std::vector<EngineInfo> eng;
    if (int r = d.query_engines(eng); r) std::printf("  falló: %s\n", errstr(r).c_str());
    for (auto& e : eng)
        std::printf("  %s%u  (clase %u, instancia lógica %u, capacidades 0x%llx)\n",
                    engine_class_name(e.cls), e.instance, e.cls, e.logical, (unsigned long long)e.caps);
    // Mapa de motores del contexto: rcs0, bcs0, ccs0 (los que existan).
    for (u16 cls : {u16(0), u16(1), u16(4)})
        for (auto& e : eng) if (e.cls == cls && e.instance == 0) P.map.push_back(e);

    std::printf("\n== I915_QUERY: regiones de memoria\n");
    std::vector<MemRegion> mr;
    if (int r = d.query_memory_regions(mr); r) std::printf("  falló: %s\n", errstr(r).c_str());
    for (auto& m : mr)
        std::printf("  clase %u (%s) inst %u: sondeada %.2f GiB, libre %s\n", m.cls,
                    m.cls == 0 ? "sistema" : m.cls == 1 ? "local/VRAM" : "?", m.instance,
                    double(m.probed) / double(1ull << 30),
                    m.unallocated == ~0ull ? "desconocida (sin privilegios)"
                                           : (std::to_string(double(m.unallocated) / double(1ull << 30)) + " GiB").c_str());

    std::printf("\n== I915_QUERY: topología\n");
    Topology t;
    if (int r = d.query_topology(t); r) std::printf("  falló: %s\n", errstr(r).c_str());
    else
        std::printf("  máx: %d slices × %d subslices × %d EU  |  presentes: %d slices, %d subslices (Xe-cores/DSS), %d EU\n",
                    t.max_slices, t.max_subslices, t.max_eus_per_subslice, t.slices, t.subslices, t.eus);

    std::printf("\n== I915_QUERY: GuC / HWCONFIG\n");
    u32 br, ma, mi, pa;
    if (int r = d.query_guc_version(br, ma, mi, pa); r) std::printf("  versión de envío GuC: no disponible (%s)\n", errstr(r).c_str());
    else std::printf("  envío por GuC, interfaz %u.%u.%u (rama %u)\n", ma, mi, pa, br);
    std::vector<std::pair<u32, std::vector<u32>>> hw;
    if (int r = d.query_hwconfig(hw); r) std::printf("  HWCONFIG no disponible: %s\n", errstr(r).c_str());
    else {
        std::printf("  HWCONFIG: %zu claves.", hw.size());
        int col = 0;
        for (auto& [k, v] : hw) {
            if (const char* nm = hwconfig_name(k)) {
                std::printf("%s  %s=%u", col++ % 3 ? "" : "\n", nm, v.empty() ? 0u : v[0]);
            }
        }
        std::printf("\n  (resto, clave=valor):");
        col = 0;
        for (auto& [k, v] : hw)
            if (!hwconfig_name(k)) std::printf("%s %u=%u", col++ % 10 ? "" : "\n   ", k, v.empty() ? 0u : v[0]);
        std::printf("\n");
    }

    std::printf("\n== sysfs (GT0: render/cómputo/copia)\n");
    std::printf("  frecuencias GT0 MHz: min %s, max %s, RP0 %s, RPn %s, actual %s (0 = en RC6)\n",
                read_sysfs("/sys/class/drm/card1/gt/gt0/rps_min_freq_mhz").c_str(),
                read_sysfs("/sys/class/drm/card1/gt/gt0/rps_max_freq_mhz").c_str(),
                read_sysfs("/sys/class/drm/card1/gt/gt0/rps_RP0_freq_mhz").c_str(),
                read_sysfs("/sys/class/drm/card1/gt/gt0/rps_RPn_freq_mhz").c_str(),
                read_sysfs("/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz").c_str());
}

// ---- 2. Buffers, contexto, timestamp ----------------------------------------------------
bool setup(Probe& P) {
    Device& d = P.dev;
    std::printf("\n== I915_REG_READ (RING_TIMESTAMP del rcs, 0x2358)\n");
    for (u64 off : {u64(0x2358) | 1u, u64(0x2358)}) {
        u64 a = 0, b = 0;
        const int r1 = d.reg_read(off, &a);
        sleep_ms(1);
        const int r2 = d.reg_read(off, &b);
        if (r1 || r2) std::printf("  offset 0x%llx: no permitido: %s\n", (unsigned long long)off, errstr(r1 ? r1 : r2).c_str());
        else std::printf("  offset 0x%llx%s: %llu → %llu tras 1 ms (Δ = %lld ticks ≈ %.3f ms)\n",
                         (unsigned long long)(off & ~1ull), (off & 1) ? " | 8B_WA" : "        ",
                         (unsigned long long)a, (unsigned long long)b, (long long)(b - a),
                         P.ts_freq ? ticks_to_ns(i64(b - a), P.ts_freq) * 1e-6 : 0.0);
    }
    {   // coste de la ioctl de lectura
        std::vector<double> v;
        u64 x = 0;
        for (int i = 0; i < 200; ++i) { const u64 t0 = now_ns(); if (d.reg_read(0x2358 | 1, &x)) break; v.push_back(double(now_ns() - t0)); }
        if (!v.empty()) { auto s = stats(v); std::printf("  coste de I915_REG_READ: mediana %.2f µs (p10 %.2f, p90 %.2f)\n", s.med * 1e-3, s.p10 * 1e-3, s.p90 * 1e-3); }
    }

    std::printf("\n== VM + contexto (I915_GEM_VM_CREATE, I915_GEM_CONTEXT_CREATE_EXT con mapa de motores)\n");
    if (P.map.empty()) { std::printf("  sin motores rcs/bcs/ccs: nada que enviar\n"); return false; }
    if (int r = d.vm_create(&P.vm); r) { std::printf("  VM_CREATE falló: %s (se usa la VM por defecto)\n", errstr(r).c_str()); P.vm = 0; }
    else std::printf("  VM id %u\n", P.vm);
    if (int r = d.context_create(P.map, P.vm, &P.ctx); r) {
        std::printf("  CONTEXT_CREATE_EXT falló: %s\n", errstr(r).c_str());
        return false;
    }
    std::printf("  contexto id %u, mapa de motores:", P.ctx);
    for (size_t i = 0; i < P.map.size(); ++i) std::printf(" [%zu]=%s%u", i, engine_class_name(P.map[i].cls), P.map[i].instance);
    u64 gtt = 0;
    if (!d.context_getparam(P.ctx, CTX_PARAM_GTT_SIZE, &gtt))
        std::printf("\n  tamaño de la VM (PPGTT): %.0f TiB (%d bits)", double(gtt) / double(1ull << 40), 63 - __builtin_clzll(gtt) + 1);
    std::printf("\n");

    std::printf("\n== GEM: I915_GEM_CREATE_EXT + I915_GEM_MMAP_OFFSET\n");
    if (int r = P.res.create(d, 64 << 10); r) { std::printf("  GEM_CREATE_EXT falló: %s\n", errstr(r).c_str()); return false; }
    if (int r = P.res.mmap(MapMode::WC); r) { std::printf("  MMAP_OFFSET WC falló: %s\n", errstr(r).c_str()); return false; }
    if (int r = P.batch.create(d, 512 << 10); r) { std::printf("  GEM_CREATE_EXT (lote) falló: %s\n", errstr(r).c_str()); return false; }
    if (int r = P.batch.mmap(MapMode::WC); r) { std::printf("  MMAP_OFFSET WC (lote) falló: %s\n", errstr(r).c_str()); return false; }
    P.res.gpu_va = P.alloc_va(P.res.size);
    P.batch.gpu_va = P.alloc_va(P.batch.size);
    std::memset(P.res.map, 0, P.res.size);
    std::printf("  resultados: handle %u, %llu KiB, VA 0x%llx (WC) | lote: handle %u, %llu KiB, VA 0x%llx (WC)\n",
                P.res.handle, (unsigned long long)P.res.size >> 10, (unsigned long long)P.res.gpu_va,
                P.batch.handle, (unsigned long long)P.batch.size >> 10, (unsigned long long)P.batch.gpu_va);
    // Variantes de PAT / modo de mapeo (sólo creación + mmap; la coherencia se prueba luego).
    struct { int pat; MapMode m; const char* desc; } vs[] = {
        {-1, MapMode::WB, "PAT por defecto + mmap WB"},
        {-1, MapMode::UC, "PAT por defecto + mmap UC"},
        {0, MapMode::WC, "PAT 0 (WB, no coherente) + mmap WC"},
        {2, MapMode::WC, "PAT 2 (UC) + mmap WC"},
        {3, MapMode::WB, "PAT 3 (WB, coherente 1 vía) + mmap WB"},
        {4, MapMode::WB, "PAT 4 (WB, coherente 2 vías) + mmap WB"},
    };
    for (auto& v : vs) {
        Bo b;
        int r = b.create(d, 1 << 20, v.pat);
        if (!r) r = b.mmap(v.m);
        std::printf("  %-42s %s\n", v.desc, r ? ("falló: " + errstr(r)).c_str() : "ok");
    }
    return true;
}

// ---- 3. Lotes MI_* verificados ----------------------------------------------------------
void functional(Probe& P) {
    std::printf("\n== EXECBUFFER2: lote MI_STORE_REGISTER_MEM(ts) + MI_STORE_DATA_IMM + MI_BATCH_BUFFER_END\n");
    u32* r = P.res.u32p();
    for (size_t e = 0; e < P.map.size() && P.gpu_ok; ++e) {
        const u32 base = engine_mmio_base(P.map[e].cls, 0);
        std::memset(P.res.map, 0, 256);
        Batch b = P.begin();
        const u64 va = P.res.gpu_va;
        b.mi_store_reg(base + kRingTimestamp, va + 0);
        b.mi_store_reg(base + kRingTimestamp + 4, va + 4);   // parte alta (UDW)
        for (int i = 0; i < 16; ++i) b.mi_store_dword(va + 16 + 4 * u64(i), 0xC0DE0000u + u32(e) * 256 + u32(i));
        b.mi_store_reg(base + kRingTimestamp, va + 8);
        b.mi_store_reg(base + kRingTimestamp + 4, va + 12);
        b.end();
        const u64 t0 = now_ns();
        const int rc = P.run(u32(e), b);
        const u64 t1 = now_ns();
        const char* nm = engine_class_name(P.map[e].cls);
        if (rc) { std::printf("  %s0: execbuf/wait falló: %s\n", nm, errstr(rc).c_str()); continue; }
        int good = 0;
        for (int i = 0; i < 16; ++i) good += r[4 + i] == 0xC0DE0000u + u32(e) * 256 + u32(i);
        const u64 ts0 = u64(r[0]) | (u64(r[1]) << 32), ts1 = u64(r[2]) | (u64(r[3]) << 32);
        u64 now_ts = 0;
        const bool rr = P.dev.reg_read(0x2358 | 1, &now_ts) == 0;
        std::printf("  %s0: %d/16 dwords correctos | ts GPU %llu → %llu (16 SDI en %.0f ns) | ida y vuelta %.1f µs",
                    nm, good, (unsigned long long)ts0, (unsigned long long)ts1,
                    P.ts_freq ? ticks_to_ns(i64(ts1 - ts0), P.ts_freq) : 0.0, double(t1 - t0) * 1e-3);
        if (rr) std::printf(" | ts rcs (REG_READ) ahora %llu", (unsigned long long)now_ts);
        std::printf("\n");
    }
}

// Latencia de envío: ioctl execbuf, espera por I915_GEM_WAIT o por sondeo del mapeo WC.
// No se usa la correlación CPU↔GPU por REG_READ para partir la latencia: cada motor sólo
// puede leer su propio RING_TIMESTAMP (los de otros motores leen 0) y los contadores de
// motores distintos no son comparables entre sí (ver docs/DRM.md).
void latency(Probe& P) {
    std::printf("\n== Latencia de envío (lote trivial: SRM ts + SDI secuencia + BBE)\n");
    const int N = P.rapido ? 60 : 400;
    std::printf("  %-5s %-22s %9s %9s %9s   (µs, %d envíos, esperando cada uno)\n", "motor", "medida", "mediana", "p10", "p90", N);
    for (size_t e = 0; e < P.map.size() && P.gpu_ok; ++e) {
        const u32 base = engine_mmio_base(P.map[e].cls, 0);
        const u64 va = P.res.gpu_va;
        volatile u32* r = P.res.u32p();
        std::vector<double> t_ioctl, t_wait, t_poll;
        for (int mode = 0; mode < 2 && P.gpu_ok; ++mode) {   // 0: GEM_WAIT, 1: sondeo
            for (int it = 0; it < N + 5; ++it) {
                const u32 seq = 0x5EC00000u + u32(it) + u32(mode) * 0x10000u + u32(e) * 0x100000u;
                Batch b = P.begin();
                b.mi_store_reg(base + kRingTimestamp, va + 64);
                b.mi_store_dword(va + 68, seq);
                b.end();
                const u64 t0 = now_ns();
                ExecObj o[2] = {{&P.res, true}, {&P.batch, false}};
                int rc = execbuf(P.dev, P.ctx, u32(e), o, 2, b.bytes());
                const u64 t1 = now_ns();
                if (rc) { std::printf("  execbuf falló: %s\n", errstr(rc).c_str()); P.gpu_ok = false; break; }
                u64 t2;
                if (mode == 0) {
                    rc = gem_wait(P.dev, P.batch.handle, kTimeout);
                    t2 = now_ns();
                    if (rc) { std::printf("  gem_wait: %s\n", errstr(rc).c_str()); P.gpu_ok = false; break; }
                    if (r[17] != seq) std::printf("  valor inesperado tras GEM_WAIT\n");
                } else {
                    for (;;) {
                        if (r[17] == seq) break;
                        _mm_pause();
                        if (now_ns() - t0 > u64(kTimeout)) { std::printf("  sondeo: timeout\n"); P.gpu_ok = false; break; }
                    }
                    t2 = now_ns();
                    // El lote debe haber terminado antes de reescribirlo.
                    if (gem_wait(P.dev, P.batch.handle, kTimeout)) { P.gpu_ok = false; break; }
                }
                if (it < 5) continue;   // calentamiento
                t_ioctl.push_back(double(t1 - t0));
                (mode == 0 ? t_wait : t_poll).push_back(double(t2 - t0));
            }
        }
        const char* nm = engine_class_name(P.map[e].cls);
        auto row = [&](const char* what, const std::vector<double>& v) {
            if (v.empty()) return;
            auto s = stats(v);
            std::printf("  %s0   %-22s %9.2f %9.2f %9.2f\n", nm, what, s.med * 1e-3, s.p10 * 1e-3, s.p90 * 1e-3);
        };
        row("ioctl EXECBUFFER2", t_ioctl);
        row("envío→GEM_WAIT vuelve", t_wait);
        row("envío→CPU ve el dato", t_poll);
    }

    // Envíos encadenados: K lotes distintos (un BO, distintos batch_start_offset) enviados
    // sin esperar; cada lote guarda su RING_TIMESTAMP → separación GPU entre lotes
    // consecutivos (mismo motor ⇒ mismo reloj) y coste CPU por envío.
    const u32 K = 128, kStride = 64;   // 64 B por lote
    std::printf("  -- %u envíos encadenados sin esperar (un lote = SRM ts + BBE)\n", K);
    for (size_t e = 0; e < P.map.size() && P.gpu_ok; ++e) {
        const u32 base = engine_mmio_base(P.map[e].cls, 0);
        std::vector<double> per_sub, gap, total;
        for (int rep = 0; rep < (P.rapido ? 3 : 7) && P.gpu_ok; ++rep) {
            u32 len = 0;
            for (u32 i = 0; i < K; ++i) {
                Batch b;
                b.p = P.batch.u32p() + i * kStride / 4;
                b.cap = kStride / 4;
                b.mi_store_reg(base + kRingTimestamp, P.res.gpu_va + 1024 + 4 * u64(i));
                b.end();
                len = b.bytes();
            }
            ExecObj o[2] = {{&P.res, true}, {&P.batch, false}};
            const u64 t0 = now_ns();
            for (u32 i = 0; i < K; ++i)
                if (int rc = execbuf(P.dev, P.ctx, u32(e), o, 2, len, i * kStride); rc) {
                    std::printf("  execbuf falló: %s\n", errstr(rc).c_str()); P.gpu_ok = false; break;
                }
            const u64 t1 = now_ns();
            if (!P.gpu_ok || gem_wait(P.dev, P.batch.handle, kTimeout)) { P.gpu_ok = false; break; }
            const u64 t2 = now_ns();
            const u32* ts = P.res.u32p() + 256;
            std::vector<double> g;
            for (u32 i = 1; i < K; ++i) g.push_back(ticks_to_ns(std::int32_t(ts[i] - ts[i - 1]), P.ts_freq));
            gap.push_back(stats(g).med);
            per_sub.push_back(double(t1 - t0) / K);
            total.push_back(double(t2 - t0) / K);
        }
        if (gap.empty()) continue;
        std::printf("  %s0   CPU %.2f µs/envío | GPU: %.2f µs entre inicios de lotes consecutivos | total %.2f µs/lote\n",
                    engine_class_name(P.map[e].cls), stats(per_sub).med * 1e-3, stats(gap).med * 1e-3, stats(total).med * 1e-3);
    }

    // Arranque en frío: tras 50 ms ociosa la GT entra en RC6 y baja a 0 MHz.
    if (P.gpu_ok && !P.map.empty()) {
        std::vector<double> cold;
        const int M = P.rapido ? 4 : 12;
        for (int i = 0; i < M && P.gpu_ok; ++i) {
            sleep_ms(50);
            Batch b = P.begin();
            b.mi_store_dword(P.res.gpu_va + 72, u32(i));
            b.end();
            const u64 t0 = now_ns();
            if (P.run(0, b)) break;
            cold.push_back(double(now_ns() - t0));
        }
        auto s = stats(cold);
        std::printf("  %s0   en frío (tras 50 ms ociosa, GEM_WAIT): mediana %.1f µs [p10 %.1f, p90 %.1f], %zu muestras\n",
                    engine_class_name(P.map[0].cls), s.med * 1e-3, s.p10 * 1e-3, s.p90 * 1e-3, cold.size());
    }
}
// ---- 4. Ancho de banda ------------------------------------------------------------------
// Relleno/verificación: patrón dependiente del índice (detecta copias desplazadas o parciales).
inline u32 pattern(u64 i) { return u32(i * 0x9E3779B1u) ^ 0xA5A5A5A5u; }

void fill_wc(u32* p, u64 n) {
    const __m256i step = _mm256_set1_epi32(8);
    __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i mul = _mm256_set1_epi32(int(0x9E3779B1u)), x = _mm256_set1_epi32(int(0xA5A5A5A5u));
    for (u64 i = 0; i < n; i += 8) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(p + i), _mm256_xor_si256(_mm256_mullo_epi32(idx, mul), x));
        idx = _mm256_add_epi32(idx, step);
    }
    _mm_sfence();
}

void bandwidth(Probe& P) {
    const int bcs = P.engine_index(1);
    std::printf("\n== Ancho de banda de copia en el motor de copia (XY_FAST_COPY_BLT, lineal 32 bpp)\n");
    if (bcs < 0) { std::printf("  no hay bcs0\n"); return; }
    const u32 base = engine_mmio_base(1, 0);
    struct Cfg { u64 mib; int pat; const char* pat_desc; };
    std::vector<Cfg> cfgs = {{8, -1, "defecto"}, {16, -1, "defecto"}, {32, -1, "defecto"},
                             {64, -1, "defecto"}, {256, -1, "defecto"},
                             {256, 0, "PAT0 WB"}, {256, 3, "PAT3 WB coh."}};
    if (P.rapido) cfgs = {{64, -1, "defecto"}};
    std::printf("  %8s  %-13s %12s %12s %14s %8s\n", "tamaño", "PAT", "GPU (ms)", "copia GB/s", "lect+escr GB/s", "verif.");
    for (auto& c : cfgs) {
        if (!P.gpu_ok) break;
        const u64 bytes = c.mib << 20;
        Bo src, dst;
        int rc = src.create(P.dev, bytes, c.pat);
        if (!rc) rc = dst.create(P.dev, bytes, c.pat);
        if (!rc) rc = src.mmap(MapMode::WC);
        if (!rc) rc = dst.mmap(MapMode::WC);
        if (rc) { std::printf("  %5llu MiB  %-13s  no se pudo crear/mapear: %s\n", (unsigned long long)c.mib, c.pat_desc, errstr(rc).c_str()); continue; }
        src.gpu_va = P.alloc_va(bytes);
        dst.gpu_va = P.alloc_va(bytes);
        fill_wc(src.u32p(), bytes / 4);
        // Filas de 32 KiB (8192 píxeles de 4 B); bloques de ≤ 2048 filas (64 MiB) por blit.
        constexpr u32 kPitch = 32768, kW = kPitch / 4, kRowsPerBlit = 2048;
        const u32 rows = u32(bytes / kPitch);
        std::vector<double> gpu_ms, cpu_ms, freq;
        const int reps = P.rapido ? 3 : 9;
        for (int rep = 0; rep < reps + 1 && P.gpu_ok; ++rep) {
            Batch b = P.begin();
            b.mi_store_reg(base + kRingTimestamp, P.res.gpu_va + 128);
            for (u32 r0 = 0; r0 < rows; r0 += kRowsPerBlit) {
                const u32 h = std::min(kRowsPerBlit, rows - r0);
                b.xy_fast_copy(dst.gpu_va + u64(r0) * kPitch, src.gpu_va + u64(r0) * kPitch, kPitch, kW, h);
            }
            b.mi_flush_dw();
            b.mi_store_reg(base + kRingTimestamp, P.res.gpu_va + 132);
            b.end();
            const Bo* extra[2] = {&src, &dst};
            const u64 t0 = now_ns();
            // En copias largas se muestrea la frecuencia real de la GT a mitad del blit.
            const bool sample = bytes >= (64ull << 20);
            if (int e = P.run(u32(bcs), b, extra, 2, !sample); e) { std::printf("  envío falló: %s\n", errstr(e).c_str()); break; }
            if (sample) {
                sleep_ms(2);
                freq.push_back(std::atof(read_sysfs("/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz").c_str()));
                if (int e = gem_wait(P.dev, P.batch.handle, kTimeout); e) {
                    std::printf("  gem_wait: %s — se abortan las pruebas GPU\n", errstr(e).c_str());
                    P.gpu_ok = false;
                    break;
                }
            }
            const u64 t1 = now_ns();
            const u32* r = P.res.u32p();
            if (rep == 0) continue;   // la primera incluye poblar páginas y mapear en la VM
            gpu_ms.push_back(ticks_to_ns(i64(r[33] - r[32]), P.ts_freq) * 1e-6);
            cpu_ms.push_back(double(t1 - t0) * 1e-6);
        }
        // Verificación por muestreo (lectura WC: lenta, se muestrean 64 Ki posiciones).
        const u32* d = dst.u32p();
        u64 bad = 0, checked = 0;
        const u64 n = bytes / 4;
        for (u64 i = 0; i < n; i += 1 + (n >> 16)) { bad += d[i] != pattern(i); ++checked; }
        bad += d[n - 1] != pattern(n - 1); ++checked;
        if (gpu_ms.empty()) continue;
        const auto s = stats(gpu_ms), sc = stats(cpu_ms);
        if (!P.gpu_ok) break;
        const double gbs = double(bytes) / (s.med * 1e-3) * 1e-9;
        std::printf("  %5llu MiB  %-13s %8.3f     %9.1f     %11.1f     %s   (CPU ida y vuelta %.3f ms; GPU p10-p90 %.3f-%.3f ms)\n",
                    (unsigned long long)c.mib, c.pat_desc, s.med, gbs, 2 * gbs,
                    bad ? "MAL" : "ok", sc.med, s.p10, s.p90);
        if (!freq.empty()) std::printf("      frecuencia GT0 a mitad de copia: mediana %.0f MHz\n", stats(freq).med);
        if (bad) std::printf("      %llu/%llu muestras incorrectas\n", (unsigned long long)bad, (unsigned long long)checked);
    }
    std::printf("  frecuencia GT0 ahora: %s MHz\n", read_sysfs("/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz").c_str());
}

// Coherencia CPU↔GPU según PAT y modo de mapeo (relevante para elegir el tipo de memoria
// "host visible" en Vulkan). Se prueba con el blitter, en dos sentidos y dos rondas:
//  (a) la GPU lee lo que la CPU escribió por el mapeo (sin flush explícito);
//  (b) la CPU lee lo que la GPU escribió, tras haber cacheado antes datos viejos.
// El buffer "testigo" es PAT por defecto (UC en GPU) + mmap WC: coherente por construcción.
void coherence(Probe& P) {
    const int bcs = P.engine_index(1);
    std::printf("\n== Coherencia CPU↔GPU por PAT/mmap (blit de 4 MiB, 2 rondas, sin clflush)\n");
    if (bcs < 0) return;
    const u64 bytes = 4ull << 20;
    const u32 n = u32(bytes / 4);
    Bo ref;
    if (ref.create(P.dev, bytes) || ref.mmap(MapMode::WC)) { std::printf("  no se pudo crear el testigo\n"); return; }
    ref.gpu_va = P.alloc_va(bytes);
    struct { int pat; MapMode m; const char* d; } vs[] = {
        {-1, MapMode::WC, "PAT defecto + mmap WC"}, {-1, MapMode::WB, "PAT defecto + mmap WB"},
        {0, MapMode::WB, "PAT0 WB + mmap WB"},      {0, MapMode::WC, "PAT0 WB + mmap WC"},
        {3, MapMode::WB, "PAT3 coh1 + mmap WB"},    {4, MapMode::WB, "PAT4 coh2 + mmap WB"},
    };
    auto blit = [&](const Bo& dst, const Bo& src) {
        Batch b = P.begin();
        b.xy_fast_copy(dst.gpu_va, src.gpu_va, 4096, 1024, u32(bytes / 4096));
        b.mi_flush_dw();
        b.end();
        const Bo* extra[2] = {&src, &dst};
        return P.run(u32(bcs), b, extra, 2);
    };
    for (auto& v : vs) {
        if (!P.gpu_ok) break;
        Bo t;
        int rc = t.create(P.dev, bytes, v.pat);
        if (!rc) rc = t.mmap(v.m);
        if (rc) { std::printf("  %-24s: %s\n", v.d, errstr(rc).c_str()); continue; }
        t.gpu_va = P.alloc_va(bytes);
        u64 bad_a = 0, bad_b = 0;
        for (u32 round = 1; round <= 2 && P.gpu_ok; ++round) {
            const u32 key = 0x1234567u * round;
            // (a) CPU escribe t (almacenes normales: en WB quedan líneas sucias en caché) → GPU copia a ref.
            volatile u32* tp = t.u32p();
            for (u32 i = 0; i < n; ++i) tp[i] = pattern(i) ^ key;
            if (blit(ref, t)) break;
            const u32* rp = ref.u32p();
            for (u32 i = 0; i < n; i += 61) bad_a += rp[i] != (pattern(i) ^ key);
            // (b) CPU lee t (cachea lo viejo) → GPU copia ref' → t → CPU relee t.
            u32 sink = 0;
            for (u32 i = 0; i < n; ++i) sink += tp[i];
            fill_wc(ref.u32p(), n);   // ref = pattern(i)
            if (blit(t, ref)) break;
            for (u32 i = 0; i < n; ++i) bad_b += tp[i] != pattern(i);
            (void)sink;
        }
        std::printf("  %-24s: GPU ve escrituras CPU: %s | CPU ve escrituras GPU: %s\n", v.d,
                    bad_a ? ("NO (" + std::to_string(bad_a) + " muestras viejas)").c_str() : "sí",
                    bad_b ? ("NO (" + std::to_string(bad_b) + " dwords viejos)").c_str() : "sí");
    }
}

// MI_COPY_MEM_MEM: copia de dwords hecha por el propio command streamer (sin EUs/blitter).
void cs_copy(Probe& P) {
    std::printf("\n== MI_COPY_MEM_MEM (el command streamer copia dword a dword)\n");
    const u32 K = 8192;   // 8192 comandos × 20 B = 160 KiB de lote, copian 32 KiB
    Bo buf;
    if (int rc = buf.create(P.dev, 2 * K * 4); rc) { std::printf("  %s\n", errstr(rc).c_str()); return; }
    buf.gpu_va = P.alloc_va(buf.size);
    for (size_t e = 0; e < P.map.size() && P.gpu_ok; ++e) {
        const u32 base = engine_mmio_base(P.map[e].cls, 0);
        std::vector<double> ns;
        for (int rep = 0; rep < 6; ++rep) {
            Batch b = P.begin();
            b.mi_store_reg(base + kRingTimestamp, P.res.gpu_va + 256);
            for (u32 i = 0; i < K; ++i) b.mi_copy_mem_mem(buf.gpu_va + K * 4 + i * 4, buf.gpu_va + i * 4);
            b.mi_store_reg(base + kRingTimestamp, P.res.gpu_va + 260);
            b.end();
            if (b.overflow()) { std::printf("  lote demasiado grande\n"); return; }
            const Bo* extra[1] = {&buf};
            if (int rc = P.run(u32(e), b, extra, 1); rc) { std::printf("  %s0: %s\n", engine_class_name(P.map[e].cls), errstr(rc).c_str()); break; }
            if (rep) ns.push_back(ticks_to_ns(i64(P.res.u32p()[65] - P.res.u32p()[64]), P.ts_freq));
        }
        if (ns.empty()) continue;
        const auto s = stats(ns);
        std::printf("  %s0: %u copias en %.1f µs → %.1f ns/comando, %.3f GB/s\n", engine_class_name(P.map[e].cls), K,
                    s.med * 1e-3, s.med / K, double(K) * 4 / s.med);
    }
}

// Ancho de banda CPU a través de los mapeos del BO (relevante para leer campos de la GPU).
void cpu_maps(Probe& P) {
    std::printf("\n== CPU ↔ BO: ancho de banda de lectura/escritura por el mapeo (64 MiB)\n");
    const u64 bytes = 64ull << 20;
    std::vector<u8> host(bytes);
    struct { int pat; MapMode m; const char* d; } vs[] = {
        {-1, MapMode::WC, "PAT defecto, mmap WC"},
        {-1, MapMode::WB, "PAT defecto, mmap WB"},
        {3, MapMode::WB, "PAT3 coherente, mmap WB"},
    };
    for (auto& v : vs) {
        Bo b;
        int rc = b.create(P.dev, bytes, v.pat);
        if (!rc) rc = b.mmap(v.m);
        if (rc) { std::printf("  %-26s: %s\n", v.d, errstr(rc).c_str()); continue; }
        std::memset(b.map, 1, bytes);   // poblar páginas
        std::vector<double> wr, rd, rd_nt;
        for (int rep = 0; rep < (P.rapido ? 2 : 5); ++rep) {
            u64 t0 = now_ns();
            std::memcpy(b.map, host.data(), bytes);
            wr.push_back(double(now_ns() - t0));
            t0 = now_ns();
            std::memcpy(host.data(), b.map, bytes);
            rd.push_back(double(now_ns() - t0));
            // Lectura con MOVNTDQA (cargas "streaming", pensadas para memoria WC).
            t0 = now_ns();
            __m256i acc = _mm256_setzero_si256();
            const __m256i* s = static_cast<const __m256i*>(b.map);
            for (u64 i = 0; i < bytes / 32; i += 4) {
                acc = _mm256_xor_si256(acc, _mm256_stream_load_si256(s + i));
                acc = _mm256_xor_si256(acc, _mm256_stream_load_si256(s + i + 1));
                acc = _mm256_xor_si256(acc, _mm256_stream_load_si256(s + i + 2));
                acc = _mm256_xor_si256(acc, _mm256_stream_load_si256(s + i + 3));
            }
            rd_nt.push_back(double(now_ns() - t0));
            volatile int sink = _mm256_extract_epi32(acc, 0);
            (void)sink;
        }
        auto gb = [&](const std::vector<double>& t) { return double(bytes) / stats(t).med; };
        std::printf("  %-26s: escritura memcpy %.1f GB/s | lectura memcpy %.2f GB/s | lectura MOVNTDQA %.2f GB/s (1 hilo)\n",
                    v.d, gb(wr), gb(rd), gb(rd_nt));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = "/dev/dri/renderD128";
    Probe P;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rapido")) P.rapido = true;
        else path = argv[i];
    }
    std::printf("drm_probe — iGPU Intel por ioctl DRM crudos (i915), sin bibliotecas\n");
    if (!P.dev.open(path)) { std::printf("ERROR: %s\n", P.dev.err.c_str()); return 1; }
    std::printf("\n== DRM_IOCTL_VERSION (%s)\n  driver %s %d.%d.%d (%s) — %s\n", path, P.dev.drv_name.c_str(),
                P.dev.ver_major, P.dev.ver_minor, P.dev.ver_patch, P.dev.drv_date.c_str(), P.dev.drv_desc.c_str());
    report_params(P);
    report_queries(P);
    if (!P.ts_freq) { std::printf("\nsin frecuencia de timestamp: no se pueden medir tiempos GPU\n"); return 1; }
    if (!setup(P)) { std::printf("\nno se pudo preparar el envío de lotes: fin\n"); return 1; }
    functional(P);
    if (P.gpu_ok) latency(P);
    if (P.gpu_ok) cs_copy(P);
    if (P.gpu_ok) bandwidth(P);
    if (P.gpu_ok) coherence(P);
    cpu_maps(P);
    if (P.ctx) P.dev.context_destroy(P.ctx);
    if (P.vm) P.dev.vm_destroy(P.vm);
    std::printf("\nfin (%s)\n", P.gpu_ok ? "todo el trabajo GPU terminó" : "hubo errores GPU, ver arriba");
    return P.gpu_ok ? 0 : 2;
}
