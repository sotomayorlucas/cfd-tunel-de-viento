// ============================================================================
//  tests/test_flowvis.cpp — pruebas del módulo flowvis con campos analíticos
//  sintéticos (flujo uniforme, tubos de vórtice de Lamb-Oseen, rotación de
//  sólido rígido, deformación pura, flujo potencial alrededor de una esfera,
//  estela gaussiana). PASA/FALLA con código de salida ≠ 0 si algo falla.
//
//  Compilar (sin el rasterizador → se omite la imagen compuesta):
//    g++ -std=c++23 -O3 -march=native -Isrc -pthread -DFLOWVIS_NO_RASTER tests/test_flowvis.cpp  (sigue)
//        src/render/flowvis.cpp src/render/flowvis_lines.cpp src/render/flowvis_volume.cpp  (sigue)
//        src/core/threadpool.cpp src/core/png.cpp -o build/flowvis/test_flowvis
//  Con el rasterizador: quitar -DFLOWVIS_NO_RASTER y añadir src/render/flowvis_draw.cpp src/render/raster_*.cpp.
//  Imágenes: build/flowvis/*.png
// ============================================================================
#include "core/png.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "render/flowvis.hpp"
#ifndef FLOWVIS_NO_RASTER
#include "render/raster.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace cfd;
using namespace cfd::flowvis;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                                                      \
    do {                                                                                      \
        if (cond) ++g_pass;                                                                   \
        else {                                                                                \
            ++g_fail;                                                                         \
            std::printf("  FALLO %s:%d: %s — ", __FILE__, __LINE__, #cond);                   \
            std::printf(__VA_ARGS__);                                                         \
            std::printf("\n");                                                                \
        }                                                                                     \
    } while (0)

static const char* k_out = "build/flowvis";

// Carga media del sistema (1 min). Con carga alta (otros procesos compitiendo por los
// núcleos) los objetivos de tiempo se informan como AVISO y no cuentan como fallo;
// FLOWVIS_STRICT_PERF=1 los hace obligatorios siempre.
static double load_avg() {
    double l = 0;
    if (FILE* fp = std::fopen("/proc/loadavg", "r")) { if (std::fscanf(fp, "%lf", &l) != 1) l = 0; std::fclose(fp); }
    return l;
}
static bool g_strict_perf = false;
// Compilaciones instrumentadas (ASan/UBSan/TSan) o sin optimizar: los tiempos no significan nada
// → objetivos sólo informativos salvo FLOWVIS_STRICT_PERF=1.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || !defined(__OPTIMIZE__)
static constexpr bool k_instrumented = true;
#else
static constexpr bool k_instrumented = false;
#endif
#define CHECK_PERF(cond, ...)                                                                 \
    do {                                                                                      \
        if (g_strict_perf) CHECK(cond, __VA_ARGS__);                                          \
        else if (!(cond)) { std::printf("  AVISO (carga alta, no cuenta): "); std::printf(__VA_ARGS__); std::printf("\n"); } \
        else ++g_pass;                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
//  Campo sintético
// ---------------------------------------------------------------------------
struct TestField {
    int nx, ny, nz;
    float u_inf;
    Buffer<float> rho, ux, uy, uz;
    Buffer<u8> flags, sid;
    TestField(int x, int y, int z, float u) : nx(x), ny(y), nz(z), u_inf(u) {
        const usize n = static_cast<usize>(x) * y * z;
        rho.resize(n); ux.resize(n); uy.resize(n); uz.resize(n); flags.resize(n, true); sid.resize(n, true);
        rho.fill(1.0f); ux.fill(u); uy.fill(0.0f); uz.fill(0.0f);
    }
    usize idx(int x, int y, int z) const { return static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * z); }
    lbm::FieldView view() const {
        lbm::FieldView v;
        v.nx = nx; v.ny = ny; v.nz = nz;
        v.rho = rho.data(); v.ux = ux.data(); v.uy = uy.data(); v.uz = uz.data();
        v.flags = flags.data(); v.solid_id = sid.data(); v.u_inf = u_inf;
        return v;
    }
    // fn(Vec3 p, Vec3& u, float& rho, bool& solid)
    template <class F>
    void fill(F&& fn) {
        parallel_for(0, static_cast<i64>(ny) * nz, 4, [&](i64 lo, i64 hi) {
            for (i64 r = lo; r < hi; ++r) {
                const int y = static_cast<int>(r % ny), z = static_cast<int>(r / ny);
                for (int x = 0; x < nx; ++x) {
                    Vec3 u{u_inf, 0, 0};
                    float rr = 1.0f;
                    bool s = false;
                    fn(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)), u, rr, s);
                    const usize n = idx(x, y, z);
                    ux[n] = s ? 0.0f : u.x; uy[n] = s ? 0.0f : u.y; uz[n] = s ? 0.0f : u.z;
                    rho[n] = rr;
                    flags[n] = s ? lbm::kSolid : 0;
                    sid[n] = s ? 1 : 0;
                }
            }
        });
    }
};

// Tubo de Lamb-Oseen con eje paralelo a X por (yc, zc): velocidad inducida en el plano (y,z).
static Vec3 lamb_x(Vec3 p, float yc, float zc, float rc, float G) {
    const float dy = p.y - yc, dz = p.z - zc, r2 = dy * dy + dz * dz;
    if (r2 < 1e-12f) return {0, 0, 0};
    const float k = G / (2.0f * k_pi * r2) * (1.0f - std::exp(-r2 / (rc * rc)));
    return {0, -k * dz, k * dy};
}
// Ídem con eje paralelo a Z por (xc, yc).
static Vec3 lamb_z(Vec3 p, float xc, float yc, float rc, float G) {
    const float dx = p.x - xc, dy = p.y - yc, r2 = dx * dx + dy * dy;
    if (r2 < 1e-12f) return {0, 0, 0};
    const float k = G / (2.0f * k_pi * r2) * (1.0f - std::exp(-r2 / (rc * rc)));
    return {-k * dy, k * dx, 0};
}
// Flujo potencial alrededor de una esfera (radio R, centro c) con U en +X.
static Vec3 sphere_flow(Vec3 p, Vec3 c, float R, float U) {
    const Vec3 d = p - c;
    const float r2 = length2(d), r = std::sqrt(r2);
    if (r < 1e-6f) return {0, 0, 0};
    const float R3 = R * R * R, r3 = r2 * r, r5 = r3 * r2;
    const float k = 0.5f * U * R3;
    return Vec3(U, 0, 0) + (Vec3(1, 0, 0) * (1.0f / r3) - d * (3.0f * d.x / r5)) * k;
}
static float bernoulli_rho(Vec3 u, float U) { return 1.0f + 1.5f * (U * U - length2(u)); }

template <class F>
static double median_ms(int reps, F&& f) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_sec();
        f();
        t.push_back((now_sec() - t0) * 1e3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Escena de referencia de rendimiento: 256×128×96, esfera + par de vórtices de estela.
static void fill_bench_scene(TestField& F) {
    const float U = F.u_inf;
    const Vec3 c{70, 64, 40};
    const float R = 14.0f, G = U * 2.0f * k_pi * 6.0f;
    F.fill([&](Vec3 p, Vec3& u, float& rho, bool& s) {
        if (length(p - c) < R) { s = true; u = {0, 0, 0}; rho = 1.0f; return; }
        u = sphere_flow(p, c, R, U);
        const float ramp = smoothstep(85.0f, 115.0f, p.x);
        u += (lamb_x(p, 52.0f, 40.0f, 4.0f, G) + lamb_x(p, 76.0f, 40.0f, 4.0f, -G)) * ramp;
        const float def = 0.3f * ramp * (std::exp(-(sq(p.y - 52.0f) + sq(p.z - 40.0f)) / 16.0f) + std::exp(-(sq(p.y - 76.0f) + sq(p.z - 40.0f)) / 16.0f));
        u.x *= 1.0f - def;
        rho = bernoulli_rho(u, U);
    });
}

// ============================================================================
//  1. Líneas de corriente en flujo uniforme
// ============================================================================
static void test_uniform_streamlines() {
    std::printf("[1] líneas de corriente en flujo uniforme\n");
    TestField F(64, 32, 32, 0.08f);
    const auto f = F.view();
    FlowSampler S;
    S.update(f);
    Streamlines L;
    // (a) paso fijo: 20 puntos, longitud exacta 19·0.5
    L.params.max_steps = 20;
    L.params.step = L.params.min_step = L.params.max_step = 0.5f;
    const Vec3 seed{5.3f, 16.2f, 15.7f};
    L.set_seeds(std::span<const Vec3>(&seed, 1));
    L.compute(S);
    CHECK(L.line_count() == 1, "líneas=%zu", L.line_count());
    if (L.line_count() == 1) {
        const auto pts = L.points();
        const u32 s0 = L.starts()[0], cnt = L.counts()[0];
        CHECK(cnt == 20, "puntos=%u", cnt);
        const Vec3 a = pts[s0], b = pts[s0 + cnt - 1];
        CHECK(std::fabs((b.x - a.x) - 9.5f) < 0.01f, "longitud=%f (esperado 9.5)", b.x - a.x);
        float dev = 0;
        for (u32 i = 0; i < cnt; ++i) dev = max_(dev, max_(std::fabs(pts[s0 + i].y - seed.y), std::fabs(pts[s0 + i].z - seed.z)));
        CHECK(dev < 1e-4f, "desviación lateral=%g", dev);
    }
    // (b) adaptativo hasta salir del dominio: recta, monótona, termina en el borde
    L.params = StreamlineParams{};
    L.compute(S);
    if (L.line_count() == 1) {
        const auto pts = L.points();
        const u32 s0 = L.starts()[0], cnt = L.counts()[0];
        const Vec3 b = pts[s0 + cnt - 1];
        bool mono = true;
        for (u32 i = 1; i < cnt; ++i) mono &= pts[s0 + i].x > pts[s0 + i - 1].x;
        CHECK(mono, "x no monótona");
        CHECK(b.x > 61.9f && b.x <= 63.0f, "fin en x=%f (esperado ~63)", b.x);
        CHECK(cnt < 400, "no debió agotar pasos (%u)", cnt);
        std::printf("    adaptativo: %u puntos hasta x=%.2f (paso crece hasta max_step)\n", cnt, b.x);
    }
    // (c) en ambos sentidos: de x≈0 a x≈63, ordenada aguas arriba → aguas abajo
    L.params.both_directions = true;
    const Vec3 seed2{30.0f, 10.0f, 20.0f};
    L.set_seeds(std::span<const Vec3>(&seed2, 1));
    L.compute(S);
    CHECK(L.line_count() == 1, "líneas=%zu", L.line_count());
    if (L.line_count() == 1) {
        const auto pts = L.points();
        const u32 s0 = L.starts()[0], cnt = L.counts()[0];
        bool mono = true;
        for (u32 i = 1; i < cnt; ++i) mono &= pts[s0 + i].x > pts[s0 + i - 1].x;
        CHECK(mono, "bidireccional no monótona");
        CHECK(pts[s0].x < 1.0f && pts[s0 + cnt - 1].x > 62.0f, "extremos %f..%f", pts[s0].x, pts[s0 + cnt - 1].x);
    }
    // (d) semillas inválidas (fuera / sólido) no generan líneas
    const Vec3 bad[2] = {{-5, 1, 1}, {100, 3, 3}};
    L.set_seeds(bad);
    L.compute(S);
    CHECK(L.line_count() == 0, "líneas inválidas=%zu", L.line_count());
}

// ============================================================================
//  2. Línea de corriente alrededor de un vórtice (radio acotado)
// ============================================================================
static void test_vortex_streamlines() {
    std::printf("[2] líneas de corriente en un vórtice de Lamb-Oseen\n");
    const float U = 0.08f, rc = 4.0f, G = U * 2.0f * k_pi * 8.0f;
    TestField F(64, 64, 16, U);
    F.fill([&](Vec3 p, Vec3& u, float& rho, bool&) { u = lamb_z(p, 32, 32, rc, G); rho = 1.0f; });
    FlowSampler S;
    S.update(F.view());
    const float radii[3] = {3.0f, 8.0f, 14.0f};
    std::vector<Vec3> seeds;
    for (float r : radii) seeds.push_back({32.0f + r, 32.0f, 8.0f});
    for (int pass = 0; pass < 2; ++pass) {
        Streamlines L;
        L.params.max_steps = 400;
        L.params.min_speed = 0.001f;
        L.set_seeds(seeds);
        if (pass == 0) L.compute(S); else L.compute_reference(F.view());
        CHECK(L.line_count() == 3, "líneas=%zu", L.line_count());
        for (usize l = 0; l < L.line_count(); ++l) {
            const auto pts = L.points();
            const u32 s0 = L.starts()[l], cnt = L.counts()[l];
            const float r0 = radii[l];
            float drift = 0, dz = 0, ang = 0;
            for (u32 i = 0; i < cnt; ++i) {
                const Vec3 p = pts[s0 + i];
                drift = max_(drift, std::fabs(std::sqrt(sq(p.x - 32.0f) + sq(p.y - 32.0f)) - r0));
                dz = max_(dz, std::fabs(p.z - 8.0f));
                if (i) {
                    const Vec3 a = pts[s0 + i - 1] - Vec3(32, 32, 0), b = p - Vec3(32, 32, 0);
                    ang += std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y);
                }
            }
            CHECK(cnt == 400, "r0=%g: puntos=%u", r0, cnt);
            CHECK(drift / r0 < 0.02f, "r0=%g: deriva radial %.4f celdas (%.2f%%)", r0, drift, 100 * drift / r0);
            CHECK(dz < 1e-3f, "r0=%g: deriva en z %g", r0, dz);
            CHECK(ang > 2.0f * k_pi, "r0=%g: no completó una vuelta (%.2f rad)", r0, ang);
            std::printf("    %s r0=%4.1f: %.2f vueltas, deriva radial máx %.4f celdas (%.3f%%)\n", pass ? "FP32 ref " : "FP16 emp.", r0,
                        ang / (2 * k_pi), drift, 100 * drift / r0);
        }
    }
}

// ============================================================================
//  3. Q-criterio y vorticidad
// ============================================================================
static void test_q_vorticity() {
    std::printf("[3] criterio Q y vorticidad\n");
    const float U = 0.08f;
    // (a) rotación de sólido rígido: |ω| = 2Ω, Q = Ω²
    {
        const float Om = 0.004f;
        TestField F(48, 40, 24, U);
        F.fill([&](Vec3 p, Vec3& u, float&, bool&) { u = {-Om * (p.y - 20.0f), Om * (p.x - 24.0f), 0.01f}; });
        const auto f = F.view();
        std::vector<float> row(48);
        float eq = 0, ew = 0;
        for (int z = 1; z < 23; ++z)
            for (int y = 1; y < 39; ++y) {
                detail::quantity_row(f, Quantity::QCriterion, y, z, row.data());
                for (int x = 1; x < 47; ++x) eq = max_(eq, std::fabs(row[x] - Om * Om / (U * U)));
                detail::quantity_row(f, Quantity::Vorticity, y, z, row.data());
                for (int x = 1; x < 47; ++x) ew = max_(ew, std::fabs(row[x] - 2 * Om / U));
            }
        CHECK(eq < 1e-4f * Om * Om / (U * U) + 1e-6f, "Q sólido rígido: error %g (esperado %g)", eq, Om * Om / (U * U));
        CHECK(ew < 1e-3f, "|ω| sólido rígido: error %g (esperado %g)", ew, 2 * Om / U);
        const float qf = f.q_criterion(20, 20, 12) / (U * U);
        CHECK(std::fabs(qf - cell_quantity(f, Quantity::QCriterion, 20, 20, 12)) < 1e-5f, "Q vs FieldView::q_criterion: %g vs %g", qf,
              cell_quantity(f, Quantity::QCriterion, 20, 20, 12));
    }
    // (b) deformación pura: Q = -a² < 0, ω = 0
    {
        const float a = 0.002f;
        TestField F(40, 40, 16, U);
        F.fill([&](Vec3 p, Vec3& u, float&, bool&) { u = {U + a * (p.x - 20), -a * (p.y - 20), 0}; });
        const auto f = F.view();
        std::vector<float> row(40);
        float qmax = -1e30f, err = 0, wmax = 0;
        for (int z = 1; z < 15; ++z)
            for (int y = 1; y < 39; ++y) {
                detail::quantity_row(f, Quantity::QCriterion, y, z, row.data());
                for (int x = 1; x < 39; ++x) { qmax = max_(qmax, row[x]); err = max_(err, std::fabs(row[x] + a * a / (U * U))); }
                detail::quantity_row(f, Quantity::Vorticity, y, z, row.data());
                for (int x = 1; x < 39; ++x) wmax = max_(wmax, row[x]);
            }
        CHECK(qmax < 0.0f, "Q en deformación pura debe ser < 0 (máx %g)", qmax);
        CHECK(err < 1e-5f, "Q deformación: error %g (esperado %g)", err, -a * a / (U * U));
        CHECK(wmax < 1e-4f, "|ω| en deformación pura %g", wmax);
    }
    // (c) Lamb-Oseen: Q > 0 en el núcleo, Q < 0 fuera (r = 2.5 rc)
    {
        const float rc = 4.0f, G = U * 2.0f * k_pi * 8.0f;
        TestField F(40, 64, 64, U);
        F.fill([&](Vec3 p, Vec3& u, float&, bool&) { u = lamb_x(p, 32, 32, rc, G) + Vec3(U, 0, 0); });
        const auto f = F.view();
        const float qc = cell_quantity(f, Quantity::QCriterion, 20, 32, 32);
        const float qo = cell_quantity(f, Quantity::QCriterion, 20, 32, 42);
        const float qn = sample_quantity(f, Quantity::QCriterion, {20.3f, 32.4f, 31.6f});
        CHECK(qc > 0.0f, "Q centro = %g", qc);
        CHECK(qn > 0.0f, "Q interpolado en núcleo = %g", qn);
        CHECK(qo < 0.0f, "Q a 2.5 rc = %g", qo);
        std::printf("    Lamb-Oseen: Q(núcleo)=%.4g  Q(2.5rc)=%.4g\n", qc, qo);
    }
    // (d) coherencia AVX2 vs escalar para todas las magnitudes, con sólidos
    {
        TestField F(64, 24, 20, U);
        F.fill([&](Vec3 p, Vec3& u, float& rho, bool& s) {
            s = (p.x > 20 && p.x < 30 && p.y > 8 && p.y < 14 && p.z > 5 && p.z < 12) || (static_cast<int>(p.x * 7 + p.y * 3 + p.z) % 23 == 0);
            u = {U * (1.0f + 0.3f * std::sin(p.y * 0.3f + p.z * 0.2f)), 0.01f * std::cos(p.x * 0.2f), 0.02f * std::sin(p.x * 0.1f + p.y * 0.4f)};
            rho = 1.0f + 0.01f * std::sin(p.x * 0.3f);
        });
        const auto f = F.view();
        std::vector<float> row(64);
        float err = 0;
        int nan_mismatch = 0;
        for (int q = 0; q < static_cast<int>(Quantity::Count); ++q)
            for (int z = 0; z < 20; ++z)
                for (int y = 0; y < 24; ++y) {
                    detail::quantity_row(f, static_cast<Quantity>(q), y, z, row.data());
                    for (int x = 0; x < 64; ++x) {
                        const float s = cell_quantity(f, static_cast<Quantity>(q), x, y, z);
                        if ((s != s) != (row[x] != row[x])) { ++nan_mismatch; continue; }
                        if (s == s) err = max_(err, std::fabs(s - row[x]) / (1.0f + std::fabs(s)));
                    }
                }
        CHECK(nan_mismatch == 0, "sólidos NaN distintos entre AVX2 y escalar: %d", nan_mismatch);
        CHECK(err < 1e-4f, "error AVX2 vs escalar %g", err);
    }
}

// ============================================================================
//  4. Plano de corte: orientación, valores, colores, sondas
// ============================================================================
static void test_slices() {
    std::printf("[4] planos de corte (orientación / colores / sonda)\n");
    const float U = 0.08f;
    TestField F(40, 24, 16, U);
    // u_x/U = 0.2 + 0.01x + 0.02y + 0.03z (lineal → la interpolación es exacta)
    auto ana = [](float x, float y, float z) { return 0.2f + 0.01f * x + 0.02f * y + 0.03f * z; };
    F.fill([&](Vec3 p, Vec3& u, float&, bool& s) {
        u = {U * ana(p.x, p.y, p.z), 0, 0};
        s = (p.x == 30 && p.y == 7 && p.z == 3);
    });
    const auto f = F.view();
    SliceView sv;
    sv.set_quantity(Quantity::Ux);
    sv.params.scale = {Colormap::Viridis, 0.0f, 2.0f, false};
    struct Case { Axis a; float pos; int tw, th; };
    const Case cases[3] = {{Axis::X, 7.25f, 24, 16}, {Axis::Y, 5.0f, 40, 16}, {Axis::Z, 2.6f, 40, 24}};
    for (const Case& c : cases) {
        sv.params.axis = c.a;
        sv.params.pos = c.pos;
        sv.update(f);
        CHECK(sv.tex_w() == c.tw && sv.tex_h() == c.th, "eje %s: textura %dx%d (esperado %dx%d)", axis_name(c.a), sv.tex_w(), sv.tex_h(), c.tw, c.th);
        float err = 0;
        int bad_col = 0;
        for (int v = 0; v < sv.val_h(); ++v)
            for (int u = 0; u < sv.val_w(); ++u) {
                float x, y, z;
                if (c.a == Axis::X) { x = c.pos; y = static_cast<float>(u); z = static_cast<float>(v); }
                else if (c.a == Axis::Y) { x = static_cast<float>(u); y = c.pos; z = static_cast<float>(v); }
                else { x = static_cast<float>(u); y = static_cast<float>(v); z = c.pos; }
                const float val = sv.values()[static_cast<usize>(v) * sv.val_w() + u];
                if (val != val) continue;
                err = max_(err, std::fabs(val - ana(x, y, z)));
                const u32 expect = map_color(sv.params.scale, val) & 0x00FFFFFFu;
                bad_col += (sv.texture()[static_cast<usize>(v) * sv.val_w() + u] & 0x00FFFFFFu) != expect;
            }
        CHECK(err < 1e-4f, "eje %s: error de valor %g", axis_name(c.a), err);
        CHECK(bad_col == 0, "eje %s: %d colores incorrectos", axis_name(c.a), bad_col);
        // texel (0,0) en la esquina 0 del quad; esquina 1 = u máximo
        Vec3 cr[4];
        sv.corners(cr);
        const int ax = static_cast<int>(c.a);
        CHECK(std::fabs(cr[0][ax] - c.pos) < 1e-6f && cr[0][(ax + 1) % 3] == -0.5f, "esquinas eje %s", axis_name(c.a));
        // value_at en coordenadas fraccionarias
        const float uq = 3.4f, vq = 2.7f;
        float xq, yq, zq;
        if (c.a == Axis::X) { xq = c.pos; yq = uq; zq = vq; }
        else if (c.a == Axis::Y) { xq = uq; yq = c.pos; zq = vq; }
        else { xq = uq; yq = vq; zq = c.pos; }
        CHECK(std::fabs(sv.value_at(uq, vq) - ana(xq, yq, zq)) < 1e-4f, "value_at eje %s: %f vs %f", axis_name(c.a), sv.value_at(uq, vq), ana(xq, yq, zq));
    }
    // Sólido → gris oscuro (eje Y en y=7, celda (30,7,3))
    sv.params.axis = Axis::Y; sv.params.pos = 7.0f;
    sv.update(f);
    CHECK(sv.texture()[3 * 40 + 30] == sv.params.solid_color, "texel sólido = %08x", sv.texture()[3 * 40 + 30]);
    CHECK(sv.texture()[3 * 40 + 29] != sv.params.solid_color, "texel vecino no sólido");
    // Sonda con rayo de cámara: mirar el plano Y desde -Y
    Camera cam;
    cam.target = {20, 7, 8}; cam.yaw = 0.0f; cam.pitch = 0.0f; cam.distance = 60.0f;
    cam.update({0, 0, 400, 300});
    float sx = 0, sy = 0, dep = 0;
    const Vec3 target{12.3f, 7.0f, 9.6f};
    cam.project(target, sx, sy, dep);
    Vec3 hit;
    float val = 0;
    const bool ok = sv.pick(cam, sx, sy, hit, val);
    CHECK(ok && length(hit - target) < 1e-3f, "pick: hit (%f,%f,%f)", hit.x, hit.y, hit.z);
    CHECK(ok && std::fabs(val - ana(target.x, target.y, target.z)) < 1e-4f, "pick: valor %f vs %f", val, ana(target.x, target.y, target.z));
    // Rango automático robusto ≈ extremos
    sv.params.auto_range = true;
    sv.update(f);
    const ColorScale e = sv.effective_scale();
    {
        std::vector<float> vs;
        for (usize i = 0; i < static_cast<usize>(sv.val_w()) * sv.val_h(); ++i) if (sv.values()[i] == sv.values()[i]) vs.push_back(sv.values()[i]);
        std::sort(vs.begin(), vs.end());
        // Percentil discreto: se admite ± un valor vecino de la lista ordenada y ± 2 cubetas del histograma.
        const usize il = static_cast<usize>(0.01 * vs.size()), ih = static_cast<usize>(0.99 * vs.size());
        const float bin = (vs.back() - vs.front()) / 1024.0f * 2.0f;
        const bool ok_lo = e.lo >= vs[il - 1] - bin && e.lo <= vs[il + 1] + bin;
        const bool ok_hi = e.hi >= vs[ih - 1] - bin && e.hi <= vs[ih + 1] + bin;
        CHECK(ok_lo && ok_hi, "auto: [%f,%f] vs percentiles exactos [%f,%f]", e.lo, e.hi, vs[il], vs[ih]);
        CHECK(sv.value_min() == vs.front() && std::fabs(sv.value_max() - vs.back()) < 1e-6f, "extremos %f..%f vs %f..%f", sv.value_min(), sv.value_max(), vs.front(), vs.back());
    }
    // LIC en flujo uniforme a lo largo de u (corte Y de u_x): vetas horizontales →
    // diferencias entre subtexeles vecinos en u mucho menores que en v.
    {
        TestField G(64, 16, 48, U);
        SliceView lic;
        lic.params.axis = Axis::Y; lic.params.pos = 8.0f; lic.params.lic = 3;
        lic.update(G.view());
        CHECK(lic.tex_w() == 64 * 3 && lic.tex_h() == 48 * 3 && lic.val_w() == 64, "LIC: textura %dx%d", lic.tex_w(), lic.tex_h());
        double du = 0, dv = 0;
        const int W = lic.tex_w(), Hh = lic.tex_h();
        for (int v = 20; v < Hh - 20; ++v)
            for (int u = 40; u < W - 40; ++u) {
                const u32 c = lic.texture()[static_cast<usize>(v) * W + u], cr = lic.texture()[static_cast<usize>(v) * W + u + 1], cu = lic.texture()[static_cast<usize>(v + 1) * W + u];
                du += std::abs(static_cast<int>(c & 255) - static_cast<int>(cr & 255)) + std::abs(static_cast<int>((c >> 8) & 255) - static_cast<int>((cr >> 8) & 255));
                dv += std::abs(static_cast<int>(c & 255) - static_cast<int>(cu & 255)) + std::abs(static_cast<int>((c >> 8) & 255) - static_cast<int>((cu >> 8) & 255));
            }
        CHECK(dv > 4.0 * du && dv > 0, "LIC: vetas no alineadas con el flujo (Δu=%g, Δv=%g)", du, dv);
        std::printf("    LIC flujo uniforme: variación a lo largo %.3g vs transversal %.3g (razón %.1f)\n", du, dv, dv / max_(du, 1.0));
    }
    // Divergente: 0 en el centro del mapa
    ColorScale d{Colormap::CoolWarm, -2.5f, 1.0f, true};
    CHECK(map_color(d, 0.0f) == render::colormap_lut(Colormap::CoolWarm)[128], "pivote del mapa divergente");
    CHECK(map_color(d, -2.5f) == render::colormap_lut(Colormap::CoolWarm)[0] && map_color(d, 1.0f) == render::colormap_lut(Colormap::CoolWarm)[255], "extremos divergentes");
    alignas(32) float vals[8] = {-2.5f, -1.25f, 0.0f, 0.5f, 1.0f, 3.0f, -9.0f, std::numeric_limits<float>::quiet_NaN()};
    alignas(32) u32 c8[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(c8), map_color8(d, simd::f8::load(vals)));
    int mism = 0;
    for (int i = 0; i < 8; ++i) mism += c8[i] != map_color(d, vals[i]);
    CHECK(mism == 0, "map_color8 vs map_color: %d distintos", mism);
}

// ============================================================================
//  5. Partículas: conservación, emisión, muertes, velocidad
// ============================================================================
static void test_particles() {
    std::printf("[5] partículas de humo\n");
    const float U = 0.08f;
    TestField F(64, 32, 32, U);
    // bloque sólido aguas abajo (con u uniforme alrededor: prueba de la lógica de muerte)
    F.fill([&](Vec3 p, Vec3& u, float&, bool& s) { s = p.x >= 44 && p.x <= 50 && p.y >= 12 && p.y <= 20 && p.z >= 12 && p.z <= 20; u = {U, 0, 0}; });
    FlowSampler S;
    S.update(F.view());
    Particles P;
    P.params.capacity = 30000;
    P.params.rate = 50.0f;
    P.params.max_age = 400.0f;
    P.params.jitter = 0.3f;
    const Rake rk = Rake::line({2, 16, 6}, {2, 16, 26}, 8);
    P.set_emitters(std::span<const Rake>(&rk, 1));
    bool conserved = true, inside = true;
    for (int fr = 0; fr < 200; ++fr) {
        P.step(S, 10.0f);
        const auto& st = P.stats();
        conserved &= st.emitted == st.alive + st.died_outside + st.died_solid + st.died_age + st.overwritten;
        for (const Vec3& p : P.points()) inside &= p.x >= 0 && p.x <= 63 && p.y >= 0 && p.y <= 31 && p.z >= 0 && p.z <= 31;
    }
    const auto& st = P.stats();
    CHECK(conserved, "emitidas != vivas + muertas + sobrescritas");
    CHECK(inside, "partícula fuera del dominio en la salida");
    CHECK(std::llabs(static_cast<long long>(st.emitted) - 50LL * 10 * 200) <= 1, "emitidas=%llu (esperado 100000)", static_cast<unsigned long long>(st.emitted));
    const double expect_alive = 50.0 * 400.0;   // limitado por la edad (cruce = 760 pasos > 400)
    CHECK(std::fabs(static_cast<double>(st.alive) - expect_alive) < 0.03 * expect_alive, "vivas=%zu (esperado ≈ %.0f)", st.alive, expect_alive);
    CHECK(st.died_age > 0 && st.overwritten == 0, "muertes por edad=%llu sobrescritas=%llu", static_cast<unsigned long long>(st.died_age),
          static_cast<unsigned long long>(st.overwritten));
    std::printf("    emitidas %llu, vivas %zu, fuera %llu, sólido %llu, edad %llu, sobrescritas %llu\n", static_cast<unsigned long long>(st.emitted), st.alive,
                static_cast<unsigned long long>(st.died_outside), static_cast<unsigned long long>(st.died_solid),
                static_cast<unsigned long long>(st.died_age), static_cast<unsigned long long>(st.overwritten));
    // Vida larga → llegan al bloque o al final del dominio
    P.params.max_age = 5000.0f;
    P.params.rate = 10.0f;   // el anillo de 30000 no se llena antes de 3000 pasos (240 celdas)
    P.reset();
    for (int fr = 0; fr < 150; ++fr) P.step(S, 10.0f);
    CHECK(P.stats().died_solid > 0, "ninguna murió en el sólido");
    CHECK(P.stats().died_outside > 0, "ninguna salió del dominio");
    // Capacidad pequeña → reciclaje en anillo
    P.params.capacity = 1000;
    P.params.max_age = 5000.0f;
    P.step(S, 1.0f);   // redimensiona
    for (int fr = 0; fr < 50; ++fr) P.step(S, 10.0f);
    CHECK(P.stats().overwritten > 0 && P.stats().alive <= P.capacity(), "anillo: sobrescritas=%llu vivas=%zu cap=%zu",
          static_cast<unsigned long long>(P.stats().overwritten), P.stats().alive, P.capacity());
    // Velocidad: en flujo uniforme la nube avanza U·dt por cuadro (media de x)
    {
        TestField G(64, 32, 32, U);
        FlowSampler S2;
        S2.update(G.view());
        Particles Q;
        Q.params.capacity = 5000; Q.params.rate = 20.0f; Q.params.max_age = 1e9f; Q.params.jitter = 0.0f;
        const Rake r2 = Rake::line({4, 16, 16}, {4, 16, 16}, 1);
        Q.set_emitters(std::span<const Rake>(&r2, 1));
        Q.step(S2, 20.0f);
        const usize n0 = Q.stats().alive;
        double mx0 = 0;
        for (const Vec3& p : Q.points()) mx0 += p.x;
        mx0 /= static_cast<double>(n0);
        Q.params.rate = 1e-6f;   // sin nuevas emisiones
        Q.step(S2, 50.0f);
        double mx1 = 0;
        for (const Vec3& p : Q.points()) mx1 += p.x;
        mx1 /= static_cast<double>(Q.stats().alive);
        CHECK(Q.stats().alive == n0, "la nube perdió partículas");
        // Tolerancia relativa 4e-4: U = 0.08 se guarda en FP16 como 0.0800171 (error relativo 2.1e-4 ≤ 2^-11)
        CHECK(std::fabs((mx1 - mx0) - 50.0 * U) < 4e-4 * 50.0 * U + 1e-4, "avance medio %f (esperado %f)", mx1 - mx0, 50.0 * U);
        // Avance grande en un cuadro (150 pasos → subpasos topados en max_substeps): sigue exacto en flujo uniforme
        Q.step(S2, 150.0f);
        double mx2 = 0;
        for (const Vec3& p : Q.points()) mx2 += p.x;
        mx2 /= static_cast<double>(Q.stats().alive);
        CHECK(Q.stats().alive == n0 && std::fabs((mx2 - mx1) - 150.0 * U) < 4e-4 * 150.0 * U + 1e-4, "avance grande %f (esperado %f)", mx2 - mx1, 150.0 * U);
        // posiciones de emisión: semilla + U·dt·frac → en [4, 4 + U·20]
        CHECK(mx0 > 4.0 && mx0 < 4.0 + 20.0 * U, "centro de emisión %f", mx0);
    }
}

// ============================================================================
//  6. Malla coloreada (Cp en superficie) y sondas — flujo potencial en esfera
// ============================================================================
static void build_sphere_mesh(Mesh& m, Vec3 c, float R, int nu, int nv) {
    m.clear();
    for (int j = 0; j <= nv; ++j)
        for (int i = 0; i <= nu; ++i) {
            const float th = k_pi * static_cast<float>(j) / nv, ph = 2 * k_pi * static_cast<float>(i) / nu;
            const Vec3 n{std::cos(th), std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph)};   // polo en ±X
            m.pos.push_back(c + n * R);
            m.nrm.push_back(n);
            m.group.push_back(1);
        }
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i) {
            const u32 a = static_cast<u32>(j * (nu + 1) + i), b = a + 1, d = a + static_cast<u32>(nu + 1), e = d + 1;
            m.tri.insert(m.tri.end(), {a, d, b, b, d, e});
        }
    m.recompute_bounds();
}

static void test_mesh_probe() {
    std::printf("[6] colores de malla (Cp) y sondas en flujo potencial\n");
    const float U = 0.08f, R = 10.0f;
    const Vec3 c{40, 32, 32};
    TestField F(96, 64, 64, U);
    F.fill([&](Vec3 p, Vec3& u, float& rho, bool& s) {
        s = length(p - c) < R;
        u = sphere_flow(p, c, R, U);
        rho = bernoulli_rho(u, U);
    });
    const auto f = F.view();
    Mesh m;
    build_sphere_mesh(m, c, R, 64, 32);
    SurfaceParams sp;
    sp.mode = SurfaceMode::Cp;
    sp.offset = 1.0f;
    color_mesh(m, f, sp);
    CHECK(m.color.size() == m.pos.size(), "tamaño de colores");
    // Error de Cp muestreado vs analítico en el punto de muestreo (r = R+1)
    float err = 0;
    for (usize i = 0; i < m.pos.size(); ++i) {
        const Vec3 q = m.pos[i] + m.nrm[i] * 1.0f;
        const float cs = sample_quantity(f, Quantity::Cp, q);
        const float ca = 1.0f - length2(sphere_flow(q, c, R, U)) / (U * U);
        err = max_(err, std::fabs(cs - ca));
    }
    CHECK(err < 0.12f, "Cp superficie: error máx %f", err);
    // Punto de remanso (polo -X) rojo, ecuador azul
    const u32 front = m.color[0];   // j = 0 → θ = 0 → n = +X... (polo +X = aguas abajo)
    const Vec3 fr = render::unpack_rgb(front);
    const u32 eq = m.color[static_cast<usize>(16 * 65)];
    const Vec3 ec = render::unpack_rgb(eq);
    CHECK(fr.x > fr.z, "remanso posterior debería ser rojizo (r=%f b=%f)", fr.x, fr.z);
    CHECK(ec.z > ec.x, "ecuador debería ser azulado (r=%f b=%f)", ec.x, ec.z);
    // Sonda
    const Probe pr = probe(f, {c.x - R - 2.0f, c.y, c.z});
    const Vec3 ua = sphere_flow({c.x - R - 2.0f, c.y, c.z}, c, R, U);
    CHECK(pr.valid && !pr.solid, "sonda válida");
    CHECK(std::fabs(pr.speed - length(ua) / U) < 0.02f, "sonda |u| %f vs %f", pr.speed, length(ua) / U);
    CHECK(std::fabs(pr.cp0 - 1.0f) < 0.02f, "sonda Cp0 %f (potencial → ≈1)", pr.cp0);
    const Probe ps = probe(f, c);
    CHECK(ps.solid, "sonda dentro de la esfera debería ser sólida");
    // FlowSampler (FP16) vs FieldView::velocity (FP32) en puntos aleatorios fluidos
    FlowSampler S;
    S.update(f);
    WyRand rng(7);
    float e16 = 0;
    for (int i = 0; i < 20000; ++i) {
        const Vec3 p{rng.uniform(0, 95), rng.uniform(0, 63), rng.uniform(0, 63)};
        if (length(p - c) < R + 2.0f) continue;
        e16 = max_(e16, length(S.velocity(p) - f.velocity(p)) / U);
    }
    CHECK(e16 < 2e-3f, "FP16 vs FP32: error relativo máx %g", e16);
    std::printf("    Cp superficie err máx %.4f; muestreador FP16 err rel. máx %.2e\n", err, e16);
}

// ============================================================================
//  7. Estela (wake survey) y huella en el suelo
// ============================================================================
static void test_wake_ground() {
    std::printf("[7] estela y huella en el suelo\n");
    const float U = 0.08f, A = 0.4f, sg = 5.0f;
    TestField F(64, 48, 48, U);
    F.fill([&](Vec3 p, Vec3& u, float& rho, bool& s) {
        const float g = std::exp(-(sq(p.y - 24) + sq(p.z - 20)) / (sg * sg));
        u = {U * (1.0f - A * g), 0, 0};
        rho = 1.0f;
        // suelo en z = 0 + succión (ρ<1) en una mancha bajo el "coche"
        if (p.z == 0) s = true;
        if (p.z <= 2 && std::fabs(p.x - 30) < 8 && std::fabs(p.y - 24) < 6) rho = 1.0f - 1.5f * U * U * 2.0f;   // Cp = -2
    });
    for (usize n = 0; n < F.sid.size(); ++n) if (F.flags[n]) F.sid[n] = lbm::k_ground_id;
    const auto f = F.view();
    const WakeStats w = wake_survey(f, 40.0f);
    const double expect = 2 * A * k_pi * sg * sg - A * A * k_pi * sg * sg / 2;
    CHECK(std::fabs(w.loss_area - expect) < 0.03 * expect, "∫(1-Cp0)dA = %f (esperado %f)", w.loss_area, expect);
    CHECK(std::fabs(w.centroid.x - 24) < 0.1f && std::fabs(w.centroid.y - 20) < 0.1f, "centroide (%f,%f)", w.centroid.x, w.centroid.y);
    CHECK(std::fabs(w.min_cp0 - sq(1 - A)) < 0.01f, "Cp0 mín %f (esperado %f)", w.min_cp0, sq(1 - A));
    std::printf("    pérdida ∫(1-Cp0)dA = %.2f celdas² (analítico %.2f), Cp0 mín %.3f\n", w.loss_area, expect, w.min_cp0);
    GroundFootprint gf;
    gf.update(f);
    CHECK(gf.slice().plane_pos() == 1.0f, "capa automática %f (esperado 1)", gf.slice().plane_pos());
    CHECK(gf.tex_w() == 64 && gf.tex_h() == 48, "textura %dx%d", gf.tex_w(), gf.tex_h());
    const Aabb e = gf.extent();
    CHECK(e.lo.z == 0.5f && e.lo.x == -0.5f && e.hi.y == 47.5f, "extensión");
    const u32 under = gf.texture()[24 * 64 + 30], outside = gf.texture()[5 * 64 + 5];
    const float vu = gf.slice().values()[24 * 64 + 30];
    CHECK(std::fabs(vu + 2.0f) < 1e-3f, "Cp bajo el coche %f (esperado -2)", vu);
    CHECK(under == map_color(gf.params.scale, vu), "huella bajo el coche %08x vs %08x", under, map_color(gf.params.scale, vu));
    CHECK(outside == map_color(gf.params.scale, 0.0f), "huella fuera %08x", outside);
    const auto ws = wake_slice_params(40.0f);
    CHECK(ws.axis == Axis::X && ws.quantity == Quantity::Cp0, "parámetros de estela");
}

// ============================================================================
//  8. Volumen de vórtices (imagen) + oclusión por profundidad
// ============================================================================
static void test_volume_image() {
    std::printf("[8] volumen de vórtices (PNG)\n");
    const float U = 0.08f, G = U * 2.0f * k_pi * 6.0f;
    TestField F(160, 80, 64, U);
    F.fill([&](Vec3 p, Vec3& u, float& rho, bool&) {
        const float ramp = smoothstep(10.0f, 30.0f, p.x);
        const float wob = 1.5f * std::sin(p.x * 0.06f);
        u = Vec3(U, 0, 0) + (lamb_x(p, 28.0f + wob, 32.0f, 3.5f, G) + lamb_x(p, 52.0f - wob, 32.0f, 3.5f, -G)) * ramp;
        const float def = 0.5f * ramp * (std::exp(-(sq(p.y - 28 - wob) + sq(p.z - 32)) / 12.25f) + std::exp(-(sq(p.y - 52 + wob) + sq(p.z - 32)) / 12.25f));
        u.x *= 1.0f - def;
        rho = bernoulli_rho(u, U);
    });
    const auto f = F.view();
    VortexVolume V;
    V.update(f);
    const auto& st = V.stats();
    CHECK(st.bricks_nonempty > 0 && st.bricks_nonempty < st.bricks / 2, "ladrillos no vacíos %zu de %zu", st.bricks_nonempty, st.bricks);
    CHECK(st.max_value > V.params.threshold, "Q máx %f", st.max_value);
    std::printf("    rejilla %dx%dx%d, ladrillos %zu/%zu no vacíos, Q máx %.4f\n", st.vx, st.vy, st.vz, st.bricks_nonempty, st.bricks, st.max_value);
    const int W = 960, H = 600;
    Framebuffer fb;
    fb.resize(W, H);
    Camera cam;
    cam.target = {80, 40, 32}; cam.yaw = -0.75f; cam.pitch = 0.45f; cam.distance = 190.0f;
    cam.update({0, 0, W, H});
    const u32 bg_top = 0xFF20242Cu, bg_bot = 0xFF0C0E12u;
    // "Escena": un bloque opaco (profundidad sintética) delante de parte de los vórtices
    float sx = 0, sy = 0, dep = 0;
    cam.project({60, 40, 32}, sx, sy, dep);
    const Rect occ{static_cast<int>(sx) - 60, static_cast<int>(sy) - 50, 120, 100};
    auto paint = [&] {
        fb.clear_depth();
        fb.gradient({0, 0, W, H}, bg_top, bg_bot);
        for (int y = occ.y; y < occ.y + occ.h; ++y)
            for (int x = occ.x; x < occ.x + occ.w; ++x) { fb.row(y)[x] = 0xFF7A7F88u; fb.drow(y)[x] = dep - 30.0f; }
    };
    for (int mode = 0; mode < 2; ++mode) {
        V.params.half_res = mode == 0;
        paint();
        std::vector<u32> before(fb.color.data(), fb.color.data() + fb.color.size());
        V.render(fb, cam);
        int changed = 0, occ_changed = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const bool ch = fb.row(y)[x] != before[static_cast<usize>(y) * fb.stride + x];
                changed += ch;
                if (ch && occ.contains(x, y) && x > occ.x + 2 && y > occ.y + 2 && x < occ.x + occ.w - 3 && y < occ.y + occ.h - 3) ++occ_changed;
            }
        CHECK(changed > W * H / 100, "%s: sólo %d píxeles cambiados", mode ? "completa" : "media", changed);
        CHECK(occ_changed == 0, "%s: %d píxeles del oclusor modificados", mode ? "completa" : "media", occ_changed);
        const std::string path = std::string(k_out) + (mode ? "/volume_full.png" : "/volume_half.png");
        CHECK(png::write_argb(path, fb.color.data(), W, H, fb.stride), "escribir %s", path.c_str());
        std::printf("    %s: %d píxeles con vórtice → %s\n", mode ? "resolución completa" : "media resolución", changed, path.c_str());
    }
    // Estilo nube, y color por magnitud (Inferno)
    V.params.half_res = true;
    V.params.style = VolumeStyle::Cloud;
    paint();
    V.render(fb, cam);
    png::write_argb(std::string(k_out) + "/volume_cloud.png", fb.color.data(), W, H, fb.stride);
    V.params.style = VolumeStyle::Surface;
    V.params.color_by = VolumeColor::Magnitude;
    V.update(f);
    paint();
    V.render(fb, cam);
    png::write_argb(std::string(k_out) + "/volume_magnitude.png", fb.color.data(), W, H, fb.stride);
    // El umbral se aplica en render(): subirlo por encima del máximo vacía el volumen sin update()
    V.params.full = st.max_value * 2.0f;
    V.update(f);
    V.params.threshold = st.max_value * 1.2f;
    paint();
    std::vector<u32> before2(fb.color.data(), fb.color.data() + fb.color.size());
    V.render(fb, cam);
    CHECK(std::equal(before2.begin(), before2.end(), fb.color.data()) && V.stats().bricks_nonempty == 0, "umbral > máximo: el volumen debería quedar vacío");

    // Planos translúcidos: un plano opaco (opacity 1) entre la cámara y los vórtices los oculta
    // por completo; con opacity 0.5 la aportación se reduce ~a la mitad.
    V.params = VolumeParams{};
    V.update(f);
    auto energy = [&](std::span<const TranslucentPlane> pl) {
        fb.clear_depth();
        fb.clear_color(0xFF000000u);
        V.render(fb, cam, pl);
        double e = 0;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { const u32 c = fb.row(y)[x]; e += ((c >> 16) & 255) + ((c >> 8) & 255) + (c & 255); }
        return e;
    };
    TranslucentPlane pl;   // rectángulo ilimitado: todas las visuales hacia los tubos lo cruzan
    pl.axis = Axis::Y; pl.pos = 4.0f;                                                   // entre la cámara (-Y) y los tubos
    const double e0 = energy({});
    pl.opacity = 1.0f;
    const double e1 = energy(std::span<const TranslucentPlane>(&pl, 1));
    pl.opacity = 0.5f;
    const double e5 = energy(std::span<const TranslucentPlane>(&pl, 1));
    CHECK(e0 > 0 && e1 == 0, "plano opaco: energía %g (sin plano %g)", e1, e0);
    CHECK(e5 > 0.4 * e0 && e5 < 0.6 * e0, "plano al 50%%: energía relativa %.3f", e5 / e0);
    pl.pos = 70.0f;   // detrás de los tubos: no cambia nada
    const double eb = energy(std::span<const TranslucentPlane>(&pl, 1));
    CHECK(std::fabs(eb - e0) < 1e-6 * e0, "plano detrás de los vórtices no debe influir (%g vs %g)", eb, e0);
    // plane_transmission: puntos a ambos lados del plano
    CHECK(plane_transmission(cam, {80, 40, 32}, std::span<const TranslucentPlane>(&pl, 1)) == 1.0f, "punto delante del plano");
    CHECK(std::fabs(plane_transmission(cam, {80, 75, 32}, std::span<const TranslucentPlane>(&pl, 1)) - 0.5f) < 1e-6f, "punto detrás del plano");
    TranslucentPlane small = pl; small.lo = {0, 0}; small.hi = {1, 1};
    CHECK(plane_transmission(cam, {80, 75, 32}, std::span<const TranslucentPlane>(&small, 1)) == 1.0f, "fuera del rectángulo del plano");
}

// ============================================================================
//  9. Rendimiento (medianas) en la escena de referencia 256×128×96
// ============================================================================
static void test_performance() {
    const double la = load_avg();
    const char* env = std::getenv("FLOWVIS_STRICT_PERF");
    g_strict_perf = (env && env[0] == '1') || (!k_instrumented && la < 6.0);
    std::printf("[9] rendimiento (mediana; carga media 1 min = %.1f%s → objetivos %s)\n", la, k_instrumented ? ", compilación instrumentada" : "",
                g_strict_perf ? "obligatorios" : "sólo informativos");
    const float U = 0.08f;
    TestField F(256, 128, 96, U);
    fill_bench_scene(F);
    const auto f = F.view();
    FlowSampler S;
    const double t_pack = median_ms(15, [&] { S.update(f); });
    std::printf("    FlowSampler::update (3.1 M celdas → %.1f MB FP16): %.2f ms\n", S.memory_bytes() / 1048576.0, t_pack);

    // Microbenchmark de muestreo: FP16 empaquetado vs FieldView::velocity (FP32 SoA), 1 hilo, puntos aleatorios
    {
        const int NP = 1 << 20;
        std::vector<Vec3> pts(NP);
        WyRand rng(3);
        for (auto& p : pts) p = {rng.uniform(0, 255), rng.uniform(0, 127), rng.uniform(0, 95)};
        volatile float sink = 0;
        const double t16 = median_ms(5, [&] {
            __m128 acc = _mm_setzero_ps();
            for (const Vec3& p : pts) acc = _mm_add_ps(acc, S.sample(_mm_setr_ps(p.x, p.y, p.z, 0)));
            sink = sink + _mm_cvtss_f32(acc);
        });
        const double t32 = median_ms(5, [&] {
            float acc = 0;
            for (const Vec3& p : pts) { const Vec3 u = f.velocity(p); acc += u.x + u.y + u.z + f.sample(f.rho, p); }
            sink = sink + acc;
        });
        std::printf("    muestreo trilineal aleatorio (u + ρ, 1 hilo): FP16 empaquetado %.1f ns/muestra, FieldView FP32 %.1f ns/muestra\n",
                    t16 * 1e6 / NP, t32 * 1e6 / NP);
    }

    // Líneas de corriente: 2000 semillas × 400 pasos
    Streamlines L;
    L.params.max_steps = 400;
    L.params.max_step = 0.5f;   // 400 pasos ≈ 200 celdas: casi todas completan los 400
    const Rake rk = Rake::grid({20, 20, 8}, {0, 88, 0}, {0, 0, 72}, 50, 40);
    L.set_rakes(std::span<const Rake>(&rk, 1));
    const double t_sl = median_ms(9, [&] { L.compute(S); });
    const usize npts = L.total_points();
    const double t_ref = median_ms(3, [&] { L.compute_reference(f); });
    std::printf("    líneas de corriente 2000×400: FP16 %.2f ms (%zu puntos, %.1f ns/paso·hilo-eq), referencia FP32 %.2f ms  [objetivo < 10 ms]\n",
                t_sl, npts, t_sl * 1e6 / static_cast<double>(npts) * pool().size(), t_ref);
    CHECK(npts > 2000u * 380u, "puntos totales %zu", npts);
    CHECK_PERF(t_sl < 10.0, "líneas de corriente %.2f ms > 10 ms", t_sl);

    // Partículas: 300 k vivas
    Particles P;
    P.params.capacity = 300000;
    P.params.max_age = 3000.0f;
    P.params.rate = 300000.0f / 3000.0f * 1.02f;   // estado estacionario lleno (reciclaje en anillo)
    const Rake emit = Rake::grid({4, 30, 10}, {0, 68, 0}, {0, 0, 60}, 24, 16);
    P.set_emitters(std::span<const Rake>(&emit, 1));
    for (int i = 0; i < 320; ++i) P.step(S, 10.0f);
    const double t_p = median_ms(21, [&] { P.step(S, 10.0f); });
    std::printf("    partículas: %zu vivas, paso %.2f ms  [objetivo 300k < 3 ms]\n", P.stats().alive, t_p);
    CHECK(P.stats().alive > 280000, "vivas %zu", P.stats().alive);
    CHECK_PERF(t_p < 3.0, "partículas %.2f ms > 3 ms", t_p);

    // Planos de corte
    SliceView sv;
    sv.params.axis = Axis::Y; sv.params.pos = 64.0f; sv.set_quantity(Quantity::Speed);
    const double t_sy = median_ms(21, [&] { sv.update(f); });
    sv.set_quantity(Quantity::Vorticity); sv.params.pos = 63.5f;
    const double t_syv = median_ms(21, [&] { sv.update(f); });
    sv.params.axis = Axis::Z; sv.params.pos = 40.0f; sv.set_quantity(Quantity::Cp);
    const double t_sz = median_ms(21, [&] { sv.update(f); });
    sv.params.axis = Axis::X; sv.params.pos = 150.0f; sv.set_quantity(Quantity::QCriterion);
    const double t_sx = median_ms(21, [&] { sv.update(f); });
    std::printf("    corte Y |u| %.3f ms, corte Y |ω| interpolado %.3f ms, corte Z Cp %.3f ms, corte X Q %.3f ms\n", t_sy, t_syv, t_sz, t_sx);

    // Volumen
    VortexVolume V;
    const double t_vu = median_ms(9, [&] { V.update(f); });
    Framebuffer fb;
    fb.resize(1500, 1100);
    Camera cam;
    cam.target = {128, 64, 48}; cam.yaw = -0.7f; cam.pitch = 0.4f; cam.distance = 330.0f;
    cam.update({0, 0, 1500, 1100});
    fb.clear_depth();
    V.params.half_res = true;
    const double t_vh = median_ms(15, [&] { fb.clear_color(0xFF101418u); V.render(fb, cam); });
    png::write_argb(std::string(k_out) + "/bench_volume_half.png", fb.color.data(), fb.w, fb.h, fb.stride);
    V.params.half_res = false;
    const double t_vf = median_ms(9, [&] { fb.clear_color(0xFF101418u); V.render(fb, cam); });
    V.params.half_res = true;
    V.params.shading = false;
    const double t_vns = median_ms(15, [&] { fb.clear_color(0xFF101418u); V.render(fb, cam); });
    V.params.shading = true;
    std::printf("    volumen: update %.2f ms (%zu/%zu ladrillos), render 1500×1100 media res. %.2f ms (sin sombreado %.2f), completa %.2f ms  [objetivo < 15 ms]\n",
                t_vu, V.stats().bricks_nonempty, V.stats().bricks, t_vh, t_vns, t_vf);
    CHECK_PERF(t_vh < 15.0, "volumen %.2f ms > 15 ms", t_vh);

    // Malla coloreada: esfera de ~200k vértices
    Mesh m;
    build_sphere_mesh(m, {70, 64, 40}, 14.0f, 630, 315);
    SurfaceParams sp;
    const double t_cm = median_ms(15, [&] { color_mesh(m, f, sp); });
    std::printf("    color_mesh %zu vértices: %.2f ms\n", m.pos.size(), t_cm);
}

#ifndef FLOWVIS_NO_RASTER
// ============================================================================
//  10. Imágenes compuestas (rasterizador): corte translúcido + líneas + humo +
//      vórtices; y efecto suelo (huella de Cp) con una esfera cerca del suelo.
// ============================================================================
static void test_composed_image() {
    std::printf("[10] imágenes compuestas (rasterizador)\n");
    const float U = 0.08f;
    {
        TestField F(256, 128, 96, U);
        fill_bench_scene(F);
        const auto f = F.view();
        FlowSampler S;
        S.update(f);
        const int W = 1600, H = 1000;
        Framebuffer fb;
        fb.resize(W, H);
        Camera cam;
        cam.target = {120, 64, 40}; cam.yaw = -0.62f; cam.pitch = 0.38f; cam.distance = 300.0f;
        cam.update({0, 0, W, H});
        fb.clear_depth();
        fb.gradient({0, 0, W, H}, 0xFF2A3038u, 0xFF0E1014u);
        Mesh m;
        build_sphere_mesh(m, {70, 64, 40}, 14.0f, 96, 48);
        SurfaceParams sp;
        color_mesh(m, f, sp);
        render::draw_mesh(fb, cam, m, render::Light{}, render::MeshStyle{});
        SliceView sv;
        sv.params.axis = Axis::Y; sv.params.pos = 64.0f;
        sv.set_quantity(Quantity::Cp);
        sv.params.opacity = 0.7f;
        sv.update(f);
        sv.draw(fb, cam);
        const TranslucentPlane tp = sv.translucent_plane();
        const std::span<const TranslucentPlane> behind(&tp, 1);
        Streamlines L;
        L.params.max_steps = 500;
        const Rake rk = Rake::grid({10, 44, 26}, {0, 40, 0}, {0, 0, 28}, 14, 10);
        L.set_rakes(std::span<const Rake>(&rk, 1));
        L.compute(S);
        L.draw(fb, cam, true, behind);
        Particles P;
        P.params.capacity = 150000;
        const Rake emit = Rake::line({4, 58, 20}, {4, 58, 60}, 24);
        P.set_emitters(std::span<const Rake>(&emit, 1));
        for (int i = 0; i < 300; ++i) P.step(S, 10.0f);
        P.draw(fb, cam, behind);
        VortexVolume V;
        V.update(f);
        V.render(fb, cam, behind);
        const std::string path = std::string(k_out) + "/composed.png";
        CHECK(png::write_argb(path, fb.color.data(), W, H, fb.stride), "escribir %s", path.c_str());
        std::printf("    → %s (%zu líneas, %zu partículas)\n", path.c_str(), L.line_count(), P.stats().alive);
        // Sólo humo (varita vertical + horizontal) sobre la esfera
        fb.clear_depth();
        fb.gradient({0, 0, W, H}, 0xFF1A1E24u, 0xFF08090Cu);
        render::draw_mesh(fb, cam, m, render::Light{}, render::MeshStyle{});
        Particles Q;
        Q.params.capacity = 200000;
        const Rake wands[2] = {Rake::line({4, 64, 22}, {4, 64, 58}, 16), Rake::line({4, 40, 40}, {4, 88, 40}, 16)};
        Q.set_emitters(wands);
        for (int i = 0; i < 400; ++i) Q.step(S, 10.0f);
        Q.draw(fb, cam);
        png::write_argb(std::string(k_out) + "/smoke.png", fb.color.data(), W, H, fb.stride);
    }
    {
        // Esfera cerca del suelo: flujo potencial + imagen especular (z → -z) → pared z = 0.5 impermeable.
        const Vec3 c{60, 48, 12.5f};
        const float R = 9.0f;
        const Vec3 ci{c.x, c.y, 1.0f - c.z};   // imagen respecto al plano z = 0.5
        TestField F(192, 96, 48, U);
        F.fill([&](Vec3 p, Vec3& u, float& rho, bool& s) {
            s = p.z < 0.5f || length(p - c) < R;
            u = sphere_flow(p, c, R, U) + sphere_flow(p, ci, R, U) - Vec3(U, 0, 0);
            rho = bernoulli_rho(u, U);
        });
        for (usize n = 0; n < F.sid.size(); ++n) if (F.flags[n] && n < static_cast<usize>(192 * 96)) F.sid[n] = lbm::k_ground_id;
        const auto f = F.view();
        FlowSampler S;
        S.update(f);
        GroundFootprint gf;
        gf.update(f);
        // Succión máxima bajo la esfera (Cp < 0 en el suelo, entre la esfera y su imagen)
        const float cp_under = gf.slice().values()[48 * 192 + 60];
        const float cp_far = gf.slice().values()[10 * 192 + 5];
        CHECK(cp_under < -0.3f && std::fabs(cp_far) < 0.05f, "huella: Cp bajo la esfera %f, lejos %f", cp_under, cp_far);
        std::printf("    efecto suelo: Cp en el suelo bajo la esfera %.3f (lejos %.3f)\n", cp_under, cp_far);
        const int W = 1400, H = 900;
        Framebuffer fb;
        fb.resize(W, H);
        Camera cam;
        cam.target = {80, 48, 8}; cam.yaw = -0.35f; cam.pitch = 0.55f; cam.distance = 190.0f;
        cam.update({0, 0, W, H});
        fb.clear_depth();
        fb.gradient({0, 0, W, H}, 0xFF2A3038u, 0xFF0E1014u);
        gf.draw(fb, cam, 8.0f, 0.0f, 0xFF3A3F48u, 0xFF5A606Cu);
        Mesh m;
        build_sphere_mesh(m, c, R, 96, 48);
        SurfaceParams sp;
        color_mesh(m, f, sp);
        render::draw_mesh(fb, cam, m, render::Light{}, render::MeshStyle{});
        Streamlines L;
        L.params.max_steps = 600;
        L.params.scale = default_scale(Quantity::Speed);
        const Rake r1 = rake_floor({{c.x - R, c.y - R, 0}, {c.x + R, c.y + R, c.z + R}}, 25.0f, 2.0f, 28, 0.3f);
        const Rake r2 = rake_vertical({{c.x - R, c.y - R, 0}, {c.x + R, c.y + R, c.z + R}}, 25.0f, c.y + 1.0f, 12, 0.1f);
        const Rake rakes[2] = {r1, r2};
        L.set_rakes(rakes);
        L.compute(S);
        L.draw(fb, cam);
        const std::string path = std::string(k_out) + "/ground_effect.png";
        CHECK(png::write_argb(path, fb.color.data(), W, H, fb.stride), "escribir %s", path.c_str());
        std::printf("    → %s (%zu líneas)\n", path.c_str(), L.line_count());
        // Estela del corte X aguas abajo (Cp0) como textura suelta (referencia visual)
        SliceView wk;
        wk.params = wake_slice_params(90.0f);
        wk.update(f);
        std::vector<u32> img(static_cast<usize>(wk.tex_w()) * wk.tex_h());
        for (int v = 0; v < wk.tex_h(); ++v)   // PNG con z hacia arriba
            for (int u = 0; u < wk.tex_w(); ++u) img[static_cast<usize>(wk.tex_h() - 1 - v) * wk.tex_w() + u] = wk.texture()[static_cast<usize>(v) * wk.tex_w() + u];
        png::write_argb(std::string(k_out) + "/wake_slice.png", img.data(), wk.tex_w(), wk.tex_h(), wk.tex_w());
    }
    {
        // Cortes con LIC de la escena de referencia (PNG con z hacia arriba)
        TestField F(256, 128, 96, U);
        fill_bench_scene(F);
        auto save = [&](const SliceView& sv, const char* name) {
            std::vector<u32> img(static_cast<usize>(sv.tex_w()) * sv.tex_h());
            for (int v = 0; v < sv.tex_h(); ++v)
                for (int u = 0; u < sv.tex_w(); ++u) img[static_cast<usize>(sv.tex_h() - 1 - v) * sv.tex_w() + u] = sv.texture()[static_cast<usize>(v) * sv.tex_w() + u];
            png::write_argb(std::string(k_out) + name, img.data(), sv.tex_w(), sv.tex_h(), sv.tex_w());
        };
        SliceView sx;
        sx.params.axis = Axis::X; sx.params.pos = 150.0f; sx.set_quantity(Quantity::Ux); sx.params.lic = 4; sx.params.lic_length = 5.0f;
        double t = median_ms(5, [&] { sx.update(F.view()); });
        save(sx, "/lic_x150.png");
        SliceView sy;
        sy.params.axis = Axis::Y; sy.params.pos = 64.0f; sy.set_quantity(Quantity::Speed); sy.params.lic = 3;
        const double t2 = median_ms(5, [&] { sy.update(F.view()); });
        save(sy, "/lic_y64.png");
        std::printf("    LIC: corte X %dx%d en %.2f ms, corte Y %dx%d en %.2f ms\n", sx.tex_w(), sx.tex_h(), t, sy.tex_w(), sy.tex_h(), t2);
    }
}
#endif

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    ::mkdir("build", 0755);
    ::mkdir(k_out, 0755);
    pool().start();
    std::printf("flowvis: %d hilos\n", pool().size());
    const double t0 = now_sec();
    test_uniform_streamlines();
    test_vortex_streamlines();
    test_q_vorticity();
    test_slices();
    test_particles();
    test_mesh_probe();
    test_wake_ground();
    test_volume_image();
    test_performance();
#ifndef FLOWVIS_NO_RASTER
    test_composed_image();
#endif
    std::printf("\n%s: %d comprobaciones correctas, %d fallos (%.1f s)\n", g_fail ? "FALLA" : "PASA", g_pass, g_fail, now_sec() - t0);
    pool().stop();
    return g_fail ? 1 : 0;
}
