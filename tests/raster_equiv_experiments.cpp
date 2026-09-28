// ============================================================================
//  tests/test_raster_equiv.cpp — equivalencia de las rutas rápidas del
//  rasterizador con sus rutas de referencia (interruptores RZ_EXPERIMENTS).
//
//  Cada optimización "que no debe cambiar la imagen" tiene un interruptor A/B en
//  rz::g_rz_exp (ver raster_internal.hpp). Aquí se renderizan varias escenas con
//  la ruta rápida y con la de referencia y se exige igualdad BIT A BIT de color
//  y profundidad:
//    bit 0  sellos 4×2        vs filas de 8           (triángulos pequeños)
//    bit 2  Hi-Z de traseras  vs sin Hi-Z
//    bit 3  binning SIMD      vs binning escalar      (carriles especiales: recorte,
//                                                      índices inválidos, triángulos grandes)
//    bit 7  sellos en puntos  vs filas
//    bit 11 orden LPT         vs orden natural de tiles
//    bit 10 vértices AVX2     vs escalar: NO es bit a bit por diseño. La Z de vista sale con otro
//                             orden de FMA (±1 ulp) y algún vértice redondea distinto a 1/16 px;
//                             en triángulos casi de canto (siluetas) eso mueve la profundidad
//                             interpolada hasta ~4e-4 relativo. Se exige color casi idéntico
//                             (≤ 200 ppm de píxeles) y profundidad con error relativo < 1e-3.
//
//  Compilar TODO el módulo con -DRZ_EXPERIMENTS (el build normal no lleva los interruptores):
//    g++ -std=c++23 -O3 -march=native -ffp-contract=fast -DRZ_EXPERIMENTS -DNDEBUG -pthread -Isrc
//        tests/test_raster_equiv.cpp src/render/raster*.cpp src/core/threadpool.cpp
//        -o build/raster/test_raster_equiv
// ============================================================================
#ifndef RZ_EXPERIMENTS
#error "test_raster_equiv requiere -DRZ_EXPERIMENTS (también en src/render/raster*.cpp)"
#endif
#include "render/raster_internal.hpp"
#include "render/colormap.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cfd;
using namespace cfd::render;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; std::printf("  PASS  "); } else { ++g_fail; std::printf("  FAIL  "); } \
                              std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); } while (0)

static void add_vertex(Mesh& m, Vec3 p, Vec3 n, u32 c) { m.pos.push_back(p); m.nrm.push_back(n); m.color.push_back(c); m.group.push_back(1); }
static void make_sphere(Mesh& m, Vec3 c, float r, int nu, int nv) {
    const u32 base = static_cast<u32>(m.pos.size());
    for (int j = 0; j <= nv; ++j)
        for (int i = 0; i <= nu; ++i) {
            const float th = k_pi * j / nv, ph = 2.0f * k_pi * i / nu;
            const Vec3 n{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
            add_vertex(m, c + n * r, n, colormap(Colormap::Turbo, 0.1f + 0.8f * j / nv));
        }
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i) {
            const u32 a = base + j * (nu + 1) + i, b = a + 1, d = a + (nu + 1), e = d + 1;
            if (j != 0) m.tri.insert(m.tri.end(), {a, d, b});
            if (j != nv - 1) m.tri.insert(m.tri.end(), {b, d, e});
        }
}
static void make_torus(Mesh& m, Vec3 c, float R, float r, int nu, int nv) {
    const u32 base = static_cast<u32>(m.pos.size());
    for (int j = 0; j <= nv; ++j)
        for (int i = 0; i <= nu; ++i) {
            const float v = 2.0f * k_pi * j / nv, u = 2.0f * k_pi * i / nu;
            const Vec3 ring{std::cos(u), std::sin(u), 0}, n = ring * std::cos(v) + Vec3(0, 0, std::sin(v));
            add_vertex(m, c + ring * R + n * r, n, colormap(Colormap::CoolWarm, 0.5f + 0.5f * std::sin(u * 4.0f)));
        }
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i) {
            const u32 a = base + j * (nu + 1) + i, b = a + 1, d = a + (nu + 1), e = d + 1;
            m.tri.insert(m.tri.end(), {a, b, d, b, e, d});
        }
}
static Camera make_cam(Rect vp, Vec3 target, float yaw, float pitch, float dist, bool ortho = false) {
    Camera c; c.target = target; c.yaw = yaw; c.pitch = pitch; c.distance = dist; c.ortho = ortho; c.update(vp); return c;
}

struct Snap { std::vector<u32> c; std::vector<float> z; };
static Snap snap(const Framebuffer& fb) {
    Snap s;
    s.c.assign(fb.color.begin(), fb.color.end());
    s.z.assign(fb.depth.begin(), fb.depth.end());
    return s;
}
struct Diff { usize color = 0, depth_bits = 0; float depth_rel = 0; };
static Diff diff(const Snap& a, const Snap& b) {
    Diff d;
    for (usize i = 0; i < a.c.size(); ++i) {
        d.color += a.c[i] != b.c[i];
        if (std::memcmp(&a.z[i], &b.z[i], 4) != 0) {
            ++d.depth_bits;
            const float za = a.z[i], zb = b.z[i];
            const float rel = (za < 1e30f && zb < 1e30f) ? std::fabs(za - zb) / std::fabs(za) : 1.0f;
            d.depth_rel = rel > d.depth_rel ? rel : d.depth_rel;
        }
    }
    return d;
}

int main() {
    pool().start();
    Framebuffer fb;
    fb.resize(1280, 800);
    const Rect vp{13, 7, 1200, 760};   // viewport no alineado
    // Malla: esfera + toros densos + triángulos grandes (> 256 px: carril escalar del binning SIMD) +
    // suelo gigante que cruza el plano cercano y la banda de guarda + índices inválidos.
    Mesh m;
    make_sphere(m, {0, 0, 40}, 50.0f, 200, 100);
    make_torus(m, {0, 0, 40}, 100.0f, 25.0f, 220, 70);
    make_torus(m, {30, 20, 90}, 30.0f, 8.0f, 24, 8);                  // triángulos grandes
    {
        const u32 b = static_cast<u32>(m.pos.size());
        const float S = 6000.0f;
        add_vertex(m, {-S, -S, -20}, {0, 0, 1}, 0xFF808080u); add_vertex(m, {S, -S, -20}, {0, 0, 1}, 0xFF808080u);
        add_vertex(m, {S, S, -20}, {0, 0, 1}, 0xFF909090u);   add_vertex(m, {-S, S, -20}, {0, 0, 1}, 0xFF909090u);
        m.tri.insert(m.tri.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
        m.tri.insert(m.tri.end(), {b, 0xFFFFFFF0u, b + 1});                // índice inválido
    }
    std::vector<Vec3> pts;
    std::vector<u32> cols;
    WyRand rng(17);
    for (int i = 0; i < 40000; ++i) {
        pts.push_back({rng.uniform(-250, 250), rng.uniform(-150, 150), rng.uniform(-20, 150)});
        cols.push_back(colormap(Colormap::Inferno, rng.uniform()) & 0x90FFFFFFu);
    }
    struct Scene { const char* name; Camera cam; MeshStyle st; };
    MeshStyle opaque, one, blend, wire;
    one.two_sided = false; blend.alpha = 0.45f; wire.wireframe_overlay = true;
    const Scene scenes[] = {
        {"persp opaca", make_cam(vp, {0, 0, 40}, -0.5f, 0.4f, 380.0f), opaque},
        {"persp una cara", make_cam(vp, {0, 0, 40}, 0.9f, 0.2f, 300.0f), one},
        {"rasante (recorte cercano)", make_cam(vp, {0, 0, -5}, 0.3f, 0.03f, 40.0f), opaque},
        {"orto alambre", make_cam(vp, {0, 0, 40}, 0.2f, 0.7f, 400.0f, true), wire},
        {"translúcida", make_cam(vp, {0, 0, 40}, -0.5f, 0.4f, 380.0f), blend},
    };
    struct Bit { int bit; const char* name; usize tol_ppm; };
    const Bit bits[] = {{0, "filas en vez de sellos", 0}, {2, "sin Hi-Z", 0}, {3, "binning escalar", 0},
                        {7, "puntos sin sellos", 0}, {11, "sin orden LPT", 0}, {10, "vértices escalares", 200}};
    for (const Scene& sc : scenes) {
        auto render = [&] {
            fb.clear_color(0xFF202428u); fb.clear_depth();
            draw_mesh(fb, sc.cam, m, Light{}, sc.st);
            draw_points(fb, sc.cam, pts, cols, 3.0f, true);
            draw_points(fb, sc.cam, std::span<const Vec3>(pts.data(), 2000), cols, 5.0f, false);
            return snap(fb);
        };
        rz::g_rz_exp = 0;
        const Snap ref = render();
        const RasterStats rs = last_mesh_stats();
        std::printf("[%s] tris repartidos %llu, recortados %llu\n", sc.name, (unsigned long long)rs.tris_binned, (unsigned long long)rs.tris_clipped);
        for (const Bit& b : bits) {
            rz::g_rz_exp = 1 << b.bit;
            const Snap s = render();
            rz::g_rz_exp = 0;
            const Diff d = diff(ref, s);
            const usize lim = ref.c.size() * b.tol_ppm / 1000000;
            if (b.tol_ppm == 0)
                CHECK(d.color == 0 && d.depth_bits == 0, "%-26s %-24s color distinto: %zu, depth distinto: %zu (exacto)", sc.name, b.name, d.color, d.depth_bits);
            else
                CHECK(d.color <= lim && d.depth_rel < 1e-3f, "%-26s %-24s color distinto: %zu (tol. %zu), depth: %zu bits distintos, error rel. máx %.2e",
                      sc.name, b.name, d.color, lim, d.depth_bits, static_cast<double>(d.depth_rel));
        }
    }
    std::printf("\nRESUMEN: %d PASS, %d FAIL\n", g_pass, g_fail);
    pool().stop();
    return g_fail ? 1 : 0;
}
