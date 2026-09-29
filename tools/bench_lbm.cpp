// ============================================================================
//  tools/bench_lbm.cpp — benchmark del solver LBM + ancho de banda de memoria.
//
//  1) "STREAM" AVX2 propio (lectura, escritura NT, copia, triada y
//     lectura-modificación-escritura in-place = patrón de Esoteric-Pull) con
//     el pool de hilos, para el techo (roofline) de ancho de banda.
//  2) MLUPS del solver: rejillas × precisión × colisión × hilos × afinidad.
//  Todas las cifras son MEDIANAS de ≥5 repeticiones (la máquina tiene carga
//  de fondo: qemu, rustc...).
//
//  Uso:
//    bench_lbm stream  [--threads 6,12,20] [--mb 512] [--runs 5]
//    bench_lbm lbm     [--grids 256x128x96,...] [--prec fp16,fp32] [--coll reg,bgk]
//                      [--threads 20] [--pin 0,1] [--steps 20] [--runs 5] [--geom car|empty]
//                      [--grain G] [--pf P] [--nt 0|1]
//    bench_lbm quick   (resumen < 1 min)
//  Salida: tabla de texto; con --md añade filas Markdown listas para pegar.
// ============================================================================
#include "lbm/solver.hpp"
#include "core/mem.hpp"
#include "core/simd.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"

#include <algorithm>
#include <sched.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <vector>

using namespace cfd;
using namespace cfd::lbm;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const usize n = v.size();
    return n == 0 ? 0 : (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

static std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    while (*s) {
        char* e;
        v.push_back(static_cast<int>(std::strtol(s, &e, 10)));
        s = (*e == ',') ? e + 1 : e;
        if (e == s && *s != ',') break;
    }
    return v;
}
static std::vector<std::string> parse_list(const char* s) {
    std::vector<std::string> v;
    std::string cur;
    for (; *s; ++s) {
        if (*s == ',') { v.push_back(cur); cur.clear(); } else cur += *s;
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

static void restart_pool(int n, bool pin) {
    pool().stop();
    if (pin) {
        pool().set_affinity(true, detect_topology().order);
    } else {
        // Sin afinidad: devolver el hilo principal a TODAS las CPUs (si una ronda anterior lo fijó,
        // los hilos nuevos heredarían su máscara y todo correría en un solo núcleo).
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c = 0; c < CPU_SETSIZE && c < 1024; ++c) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof(set), &set);
        pool().set_affinity(false, {});
    }
    pool().start(n);
}

// ---------------------------------------------------------------------------------------------
//  STREAM AVX2
// ---------------------------------------------------------------------------------------------
struct StreamRes { double read, write_nt, copy, triad, rmw; };

static StreamRes stream_bench(usize mb, int runs) {
    const usize n = mb * (1u << 20) / sizeof(float);
    Buffer<float> a(n), b(n), c(n);
    parallel_for(0, static_cast<i64>(n / 8), 1 << 12, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i)
            for (int l = 0; l < 8; ++l) { a[i * 8 + l] = 1.0f; b[i * 8 + l] = 2.0f; c[i * 8 + l] = 0.0f; }
    });
    const i64 nb = static_cast<i64>(n / 8);
    const i64 grain = 1 << 13;   // 256 KiB por trozo
    std::vector<double> tr, tw, tc, tt, tm;
    volatile float sink = 0;
    for (int r = 0; r < runs; ++r) {
        // Lectura: suma con 4 acumuladores.
        {
            Padded<float> part[128];
            const double t0 = now_sec();
            parallel_for(0, nb, grain, [&](i64 lo, i64 hi) {
                __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
                const float* p = a.data();
                i64 i = lo;
                for (; i + 4 <= hi; i += 4) {
                    s0 = _mm256_add_ps(s0, _mm256_load_ps(p + i * 8));
                    s1 = _mm256_add_ps(s1, _mm256_load_ps(p + i * 8 + 8));
                    s2 = _mm256_add_ps(s2, _mm256_load_ps(p + i * 8 + 16));
                    s3 = _mm256_add_ps(s3, _mm256_load_ps(p + i * 8 + 24));
                }
                for (; i < hi; ++i) s0 = _mm256_add_ps(s0, _mm256_load_ps(p + i * 8));
                const int w = std::max(0, ThreadPool::worker_index()) & 127;
                part[w].value += simd::hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            });
            tr.push_back(now_sec() - t0);
            float s = 0;
            for (auto& p : part) s += p.value;
            sink = s;
        }
        // Escritura no temporal.
        {
            const double t0 = now_sec();
            parallel_for(0, nb, grain, [&](i64 lo, i64 hi) {
                const __m256 v = _mm256_set1_ps(3.0f);
                float* p = c.data();
                for (i64 i = lo; i < hi; ++i) _mm256_stream_ps(p + i * 8, v);
                _mm_sfence();
            });
            tw.push_back(now_sec() - t0);
        }
        // Copia (stores normales: incluye RFO).
        {
            const double t0 = now_sec();
            parallel_for(0, nb, grain, [&](i64 lo, i64 hi) {
                const float* pa = a.data();
                float* pc = c.data();
                for (i64 i = lo; i < hi; ++i) _mm256_store_ps(pc + i * 8, _mm256_load_ps(pa + i * 8));
            });
            tc.push_back(now_sec() - t0);
        }
        // Triada a = b + s·c.
        {
            const double t0 = now_sec();
            parallel_for(0, nb, grain, [&](i64 lo, i64 hi) {
                const __m256 s = _mm256_set1_ps(0.5f);
                float* pa = a.data();
                const float* pb = b.data();
                const float* pc = c.data();
                for (i64 i = lo; i < hi; ++i)
                    _mm256_store_ps(pa + i * 8, _mm256_fmadd_ps(s, _mm256_load_ps(pc + i * 8), _mm256_load_ps(pb + i * 8)));
            });
            tt.push_back(now_sec() - t0);
        }
        // Lectura-modificación-escritura in-place (lo que hace Esoteric-Pull con cada posición).
        {
            const double t0 = now_sec();
            parallel_for(0, nb, grain, [&](i64 lo, i64 hi) {
                const __m256 s = _mm256_set1_ps(0.999f);
                float* pb = b.data();
                for (i64 i = lo; i < hi; ++i) _mm256_store_ps(pb + i * 8, _mm256_mul_ps(s, _mm256_load_ps(pb + i * 8)));
            });
            tm.push_back(now_sec() - t0);
        }
    }
    (void)sink;
    const double bytes = static_cast<double>(n) * 4;
    StreamRes r;
    r.read = bytes / median(tr) * 1e-9;
    r.write_nt = bytes / median(tw) * 1e-9;
    r.copy = 2 * bytes / median(tc) * 1e-9;       // bytes "útiles" (lectura + escritura), sin contar el RFO
    r.triad = 3 * bytes / median(tt) * 1e-9;
    r.rmw = 2 * bytes / median(tm) * 1e-9;
    return r;
}

// ---------------------------------------------------------------------------------------------
//  Geometría del benchmark: "coche" romo con 4 ruedas girando + suelo móvil (caso realista:
//  bloques con máscara alrededor del cuerpo y ruta escalar en la capa del suelo y en las ruedas).
// ---------------------------------------------------------------------------------------------
static void make_car(std::vector<u8>& g, int nx, int ny, int nz, Solver& s) {
    g.assign(static_cast<usize>(nx) * ny * nz, 0);
    const float L = nx * 0.30f, W = ny * 0.30f, H = nz * 0.18f;
    const float x0 = nx * 0.2f, yc = (ny - 1) * 0.5f, z0 = nz * 0.04f + 1;
    const float R = H * 0.45f;
    for (int z = 1; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const usize n = x + static_cast<usize>(nx) * (y + static_cast<usize>(ny) * z);
                // Cuerpo: caja redondeada (elipsoide superior)
                const float dx = (x - (x0 + L * 0.5f)) / (L * 0.5f), dy = (y - yc) / (W * 0.5f), dz = (z - (z0 + H * 0.5f)) / (H * 0.5f);
                if (dx * dx * dx * dx + dy * dy * dy * dy + dz * dz * dz * dz < 1.0f) g[n] = 1;
                // Ruedas (cilindros en y)
                for (int wi = 0; wi < 4; ++wi) {
                    const float wx = x0 + (wi & 1 ? L * 0.85f : L * 0.15f);
                    const float wy = (wi & 2) ? yc + W * 0.5f + R * 0.3f : yc - W * 0.5f - R * 0.3f;
                    const float ex = x - wx, ez = z - (R + 0.5f);
                    if (ex * ex + ez * ez < R * R && std::fabs(y - wy) < R * 0.6f) g[n] = static_cast<u8>(2 + wi);
                }
            }
    s.set_geometry(g.data());
    if (std::getenv("BENCH_STATIC_WHEELS")) return;   // diagnóstico: ruedas sin girar (sin ruta escalar)
    const float u = s.config().u_inf;
    for (int wi = 0; wi < 4; ++wi) {
        WallMotion m;
        const float wx = x0 + (wi & 1 ? L * 0.85f : L * 0.15f);
        const float wy = (wi & 2) ? yc + W * 0.5f + R * 0.3f : yc - W * 0.5f - R * 0.3f;
        m.omega = Vec3(0, u / R, 0);
        m.center = Vec3(wx, wy, R + 0.5f);
        s.set_wall_motion(static_cast<u8>(2 + wi), m);
    }
}

struct LbmRes { double mlups, mlups_kernel, force_ms, gbs; };

// Un caso vivo (solver ya inicializado) y sus mediciones acumuladas.
struct Case {
    std::string grid, prec, coll;
    int nx, ny, nz;
    Solver* s = nullptr;
    std::vector<double> m[64], mk[64], fm[64];   // por configuración de hilos (índice)
};

static float g_nu = -1.0f;   // --nu: fuerza la viscosidad (todas las colisiones)
static float g_cs = -1.0f;   // --cs: fuerza la constante de Smagorinsky
static int g_cpu = -1;       // --cpu: CPU lógica para el modo single (por defecto el primer P-core)
static Solver* make_solver(int nx, int ny, int nz, Precision prec, Collision coll, const std::string& geom, const Solver::Tuning& tun) {
    Config c;
    c.nx = nx; c.ny = ny; c.nz = nz;
    c.precision = prec; c.collision = coll;
    // Configuración realista de la app (Regularizado + LES, ν=2e-4). BGK con τ≈0.5006 diverge en
    // este caso (medido): para BGK se usa ν=0.01 (τ=0.53). El coste por celda no depende de ν.
    c.u_inf = 0.08f; c.nu = coll == Collision::BGK ? 0.01f : 2e-4f; c.cs_smag = 0.16f;
    if (g_nu > 0.0f) c.nu = g_nu;
    if (g_cs >= 0.0f) c.cs_smag = g_cs;
    c.ground = geom == "empty" ? GroundMode::None : GroundMode::Moving;
    c.ramp_steps = 0;
    Solver* s = new Solver;
    s->init(c);
    s->set_tuning(tun);
    std::vector<u8> g;
    if (geom != "empty") make_car(g, nx, ny, nz, *s);
    return s;
}

static double loadavg() {
    double l = 0;
    if (FILE* f = std::fopen("/proc/loadavg", "r")) { if (std::fscanf(f, "%lf", &l) != 1) l = 0; std::fclose(f); }
    return l;
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "quick";
    std::vector<int> threads;
    std::vector<int> pins = {0};
    std::vector<int> grains = {0}, pfs = {-1}, pairs = {-1}, maxts = {0};
    std::vector<std::string> grids, precs = {"fp16"}, colls = {"reg"};
    std::string geom = "car";
    int steps = 20, runs = 5;
    usize mb = 512;
    bool md = false;
    Solver::Tuning tun;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--threads") threads = parse_ints(next());
        else if (a == "--pin") pins = parse_ints(next());
        else if (a == "--grids") grids = parse_list(next());
        else if (a == "--prec") precs = parse_list(next());
        else if (a == "--coll") colls = parse_list(next());
        else if (a == "--geom") geom = next();
        else if (a == "--steps") steps = std::atoi(next());
        else if (a == "--runs") runs = std::atoi(next());
        else if (a == "--mb") mb = static_cast<usize>(std::atoi(next()));
        else if (a == "--grain") grains = parse_ints(next());
        else if (a == "--pf") pfs = parse_ints(next());
        else if (a == "--pair") pairs = parse_ints(next());
        else if (a == "--maxt") maxts = parse_ints(next());
        else if (a == "--nu") g_nu = static_cast<float>(std::atof(next()));
        else if (a == "--cs") g_cs = static_cast<float>(std::atof(next()));
        else if (a == "--cpu") g_cpu = std::atoi(next());
        else if (a == "--nt") tun.nt_macro = std::atoi(next());
        else if (a == "--ftz") tun.ftz = std::atoi(next());
        else if (a == "--md") md = true;
    }
    const CpuTopology topo = detect_topology();
    std::printf("CPU: %d P + %d E + %d LP-E, recomendado %d hilos\n", topo.n_pcores, topo.n_ecores, topo.n_lpe, topo.recommended);

    if (mode == "stream") {
        if (threads.empty()) threads = {6, 12, 14, 16, 20, 22};
        std::printf("STREAM AVX2 (%zu MiB por arreglo, mediana de %d)  GB/s\n", mb, runs);
        std::printf("%-7s %-4s %8s %8s %8s %8s %8s\n", "hilos", "pin", "lectura", "escr.NT", "copia", "triada", "RMW");
        for (int pin : pins)
            for (int t : threads) {
                restart_pool(t, pin != 0);
                const StreamRes r = stream_bench(mb, runs);
                std::printf("%-7d %-4d %8.1f %8.1f %8.1f %8.1f %8.1f\n", t, pin, r.read, r.write_nt, r.copy, r.triad, r.rmw);
                if (md) std::printf("| %d | %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n", t, pin ? "sí" : "no", r.read, r.write_nt, r.copy, r.triad, r.rmw);
                std::fflush(stdout);
            }
        pool().stop();
        return 0;
    }
    if (mode == "single") {
        // Eficiencia del kernel por núcleo con tiempo de CPU del hilo (robusto frente a la carga de
        // fondo: el tiempo en que el SO nos expulsa no cuenta). 1 hilo fijado a un P-core.
        if (grids.empty()) grids = {"256x128x96"};
        pool().stop();
        if (g_cpu >= 0) pool().set_affinity(true, {g_cpu});
        else pool().set_affinity(true, detect_topology().order);
        pool().start(1);
        std::printf("%-12s %-5s %-4s %10s %10s %10s\n", "rejilla", "prec", "col", "MLUPS-cpu", "MLUPS-wall", "F ms/paso");
        for (const std::string& gs : grids) {
            int nx = 0, ny = 0, nz = 0;
            std::sscanf(gs.c_str(), "%dx%dx%d", &nx, &ny, &nz);
            for (const std::string& ps : precs)
                for (const std::string& cs : colls)
                for (int pr : pairs) {
                    Solver::Tuning tt = tun;
                    tt.pair_blocks = pr;
                    Solver* sv = make_solver(nx, ny, nz, ps == "fp32" ? Precision::FP32 : Precision::FP16S,
                                             cs == "bgk" ? Collision::BGK : (cs == "rr" ? Collision::Recursive : Collision::Regularized), geom, tt);
                    sv->step(2, true);
                    std::vector<double> mc, mw, fm;
                    for (int r = 0; r < runs; ++r) {
                        timespec a, b;
                        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &a);
                        const double w0 = now_sec();
                        sv->step(steps, true);
                        const double w = now_sec() - w0;
                        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
                        const double cpu = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
                        const double cells = static_cast<double>(nx) * ny * nz * steps;
                        mc.push_back(cells / cpu * 1e-6);
                        mw.push_back(cells / w * 1e-6);
                        fm.push_back(sv->last_force_seconds() * 1e3 / steps);
                    }
                    std::printf("%-12s %-5s %-4s par=%-2d %10.2f %10.2f %10.3f\n", gs.c_str(), ps.c_str(), cs.c_str(), pr, median(mc), median(mw), median(fm));
                    std::fflush(stdout);
                    delete sv;
                }
        }
        pool().stop();
        return 0;
    }
    if (mode == "quick") {
        if (threads.empty()) threads = {topo.recommended};
        if (grids.empty()) grids = {"256x128x96"};
        precs = {"fp16", "fp32"};
        colls = {"reg", "bgk"};
    }
    if (threads.empty()) threads = {topo.recommended};
    if (grids.empty()) grids = {"192x96x64", "256x128x96", "384x160x128"};
    // Configuraciones de hilos (hilos × afinidad). Las mediciones se INTERCALAN por rondas:
    // ronda r → para cada configuración de hilos → para cada caso: un step(steps). Así la carga
    // de fondo (variable) afecta por igual a todas las configuraciones.
    struct TC { int t, pin, grain, pf, pair, maxt; };
    std::vector<TC> tcfg;
    for (int pin : pins) for (int t : threads) for (int g : grains) for (int pf : pfs) for (int pr : pairs) for (int mt : maxts)
        tcfg.push_back({t, pin, g, pf, pr, mt});
    // Case::m/mk/fm tienen 64 entradas (una por configuración): más combinaciones desbordarían (revisión).
    CFD_CHECK(tcfg.size() <= 64, "bench_lbm: demasiadas combinaciones de hilos/afinidad/ajustes (máx. 64)");
    std::printf("LBM D3Q19 Esoteric-Pull, geometría '%s', %d pasos por medición, mediana de %d rondas intercaladas\n", geom.c_str(), steps, runs);
    std::printf("carga media al empezar: %.1f\n", loadavg());
    std::printf("%-12s %-5s %-4s %-5s %-4s %-5s %-3s %-4s %-4s %9s %9s %8s %8s\n", "rejilla", "prec", "col", "hilos", "pin", "grano", "pf",
                "par", "maxt", "MLUPS", "MLUPS-k", "GB/s-k", "F ms/p");
    for (const std::string& gs : grids) {
        int nx = 0, ny = 0, nz = 0;
        std::sscanf(gs.c_str(), "%dx%dx%d", &nx, &ny, &nz);
        std::vector<Case> cases;
        restart_pool(topo.recommended, false);
        for (const std::string& ps : precs)
            for (const std::string& cs : colls) {
                Case c;
                c.grid = gs; c.prec = ps; c.coll = cs; c.nx = nx; c.ny = ny; c.nz = nz;
                c.s = make_solver(nx, ny, nz, ps == "fp32" ? Precision::FP32 : Precision::FP16S,
                                  cs == "bgk" ? Collision::BGK : (cs == "rr" ? Collision::Recursive : Collision::Regularized), geom, tun);
                c.s->step(steps, true);   // calentamiento
                cases.push_back(std::move(c));
            }
        const double N = static_cast<double>(nx) * ny * nz;
        for (int r = 0; r < runs; ++r)
            for (usize ti = 0; ti < tcfg.size(); ++ti) {
                restart_pool(tcfg[ti].t, tcfg[ti].pin != 0);
                // Calentamiento tras recrear el pool: los hilos nuevos nacen junto al padre y el planificador
                // (CFS) tarda decenas de ms en repartirlos. Sin esto, el PRIMER caso medido tras cada reinicio
                // salía penalizado un 20-60 % (artefacto detectado: siempre era FP16-Reg). Además se rota el
                // orden de los casos en cada ronda.
                {
                    const double t0 = now_sec();
                    while (now_sec() - t0 < 0.15) cases[0].s->step(1, false);
                }
                for (usize ci = 0; ci < cases.size(); ++ci) {
                    Case& c = cases[(ci + static_cast<usize>(r)) % cases.size()];
                    Solver::Tuning tt = tun;
                    tt.row_grain = tcfg[ti].grain;
                    tt.prefetch = tcfg[ti].pf;
                    tt.pair_blocks = tcfg[ti].pair;
                    tt.max_threads = tcfg[ti].maxt;
                    c.s->set_tuning(tt);
                    c.s->step(steps, true);
                    c.m[ti].push_back(c.s->last_mlups());
                    c.mk[ti].push_back(N * steps / c.s->last_kernel_seconds() * 1e-6);
                    c.fm[ti].push_back(c.s->last_force_seconds() * 1e3 / steps);
                }
            }
        for (usize ti = 0; ti < tcfg.size(); ++ti)
            for (Case& c : cases) {
                const double mk = median(c.mk[ti]);
                const double bpc = (c.prec == "fp32" ? 4.0 : 2.0) * 19 * 2;
                const double gbs = mk * 1e6 * bpc * 1e-9;
                std::printf("%-12s %-5s %-4s %-5d %-4d %-5d %-3d %-4d %-4d %9.1f %9.1f %8.1f %8.3f\n", gs.c_str(), c.prec.c_str(),
                            c.coll.c_str(), tcfg[ti].t, tcfg[ti].pin, tcfg[ti].grain, tcfg[ti].pf, tcfg[ti].pair, tcfg[ti].maxt,
                            median(c.m[ti]), mk, gbs, median(c.fm[ti]));
                if (md)
                    std::printf("| %s | %s | %s | %d | %s | %.0f | %.0f | %.1f | %.3f |\n", gs.c_str(), c.prec.c_str(), c.coll.c_str(),
                                tcfg[ti].t, tcfg[ti].pin ? "sí" : "no", median(c.m[ti]), mk, gbs, median(c.fm[ti]));
                if (c.s->diverged()) std::printf("  [aviso] divergencia\n");
            }
        std::fflush(stdout);
        for (Case& c : cases) delete c.s;
    }
    std::printf("carga media al terminar: %.1f\n", loadavg());
    pool().stop();
    return 0;
}
