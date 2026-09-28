// ============================================================================
//  tests/test_raster.cpp — pruebas del rasterizador por software (render core).
//
//  Correctitud: cobertura exacta, estanqueidad (sin grietas ni dobles impactos en
//  aristas compartidas), orden de profundidad, recorte en el plano cercano, UV
//  perspectiva-correctas, integral de cobertura de líneas AA, centrado de splats.
//  Robustez: coordenadas enormes, NaN/inf, tamaños absurdos, índices fuera de
//  rango, viewport fuera del framebuffer → sin cuelgues ni escrituras fuera.
//  Rendimiento: medianas de varias repeticiones (la máquina tiene carga de fondo).
//  Guarda PNGs en build/raster/ para inspección visual.
//  Añadido en la revisión: orden de la mezcla translúcida, normales nulas/ausentes, FXAA y SSAO
//  funcionales, determinismo paralelo == serie de TODAS las primitivas (carreras), estanqueidad con
//  cientos de triángulos recortados por el plano cercano. Las rutas rápidas frente a las de
//  referencia (sellos, Hi-Z, binning SIMD, LPT…) se comprueban en tests/test_raster_equiv.cpp.
//
//  Uso: test_raster [--quick]   (--quick omite los benchmarks)
// ============================================================================
#include "render/raster.hpp"
#include "render/colormap.hpp"
#include "core/png.hpp"
#include "core/util.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace cfd;
using namespace cfd::render;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; std::printf("  PASS  "); } else { ++g_fail; std::printf("  FAIL  "); } \
                              std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); } while (0)

static const char* k_out = "build/raster/";
static const float k_nan = std::numeric_limits<float>::quiet_NaN();
static const float k_inf = std::numeric_limits<float>::infinity();

// ---- Mallas procedimentales -----------------------------------------------------------------
static void add_vertex(Mesh& m, Vec3 p, Vec3 n, u32 c) { m.pos.push_back(p); m.nrm.push_back(n); m.color.push_back(c); m.group.push_back(1); }

static void make_sphere(Mesh& m, Vec3 c, float r, int nu, int nv, u32 (*col)(float, float)) {
    const u32 base = static_cast<u32>(m.pos.size());
    for (int j = 0; j <= nv; ++j) {
        const float th = k_pi * static_cast<float>(j) / static_cast<float>(nv);
        for (int i = 0; i <= nu; ++i) {
            const float ph = 2.0f * k_pi * static_cast<float>(i) / static_cast<float>(nu);
            const Vec3 n{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
            add_vertex(m, c + n * r, n, col(static_cast<float>(i) / nu, static_cast<float>(j) / nv));
        }
    }
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i) {
            const u32 a = base + static_cast<u32>(j * (nu + 1) + i), b = a + 1, d = a + static_cast<u32>(nu + 1), e = d + 1;
            // antihorario visto desde fuera (normal hacia fuera)
            if (j != 0) { m.tri.push_back(a); m.tri.push_back(d); m.tri.push_back(b); }
            if (j != nv - 1) { m.tri.push_back(b); m.tri.push_back(d); m.tri.push_back(e); }
        }
    m.recompute_bounds();
}
static void make_torus(Mesh& m, Vec3 c, float R, float r, int nu, int nv, u32 (*col)(float, float)) {
    const u32 base = static_cast<u32>(m.pos.size());
    for (int j = 0; j <= nv; ++j) {
        const float v = 2.0f * k_pi * static_cast<float>(j) / static_cast<float>(nv);
        for (int i = 0; i <= nu; ++i) {
            const float u = 2.0f * k_pi * static_cast<float>(i) / static_cast<float>(nu);
            const Vec3 ring{std::cos(u), std::sin(u), 0};
            const Vec3 n = ring * std::cos(v) + Vec3(0, 0, std::sin(v));
            add_vertex(m, c + ring * R + n * r, n, col(static_cast<float>(i) / nu, static_cast<float>(j) / nv));
        }
    }
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i) {
            const u32 a = base + static_cast<u32>(j * (nu + 1) + i), b = a + 1, d = a + static_cast<u32>(nu + 1), e = d + 1;
            m.tri.push_back(a); m.tri.push_back(b); m.tri.push_back(d);
            m.tri.push_back(b); m.tri.push_back(e); m.tri.push_back(d);
        }
    m.recompute_bounds();
}
// Rejilla plana z = z0 de nx×ny celdas (2 triángulos por celda); vértices interiores perturbados.
static void make_grid(Mesh& m, float x0, float y0, float size, int nx, int ny, float z0, u32 color, bool jitter) {
    const u32 base = static_cast<u32>(m.pos.size());
    WyRand rng(1234);
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            float px = x0 + size * static_cast<float>(i) / nx, py = y0 + size * static_cast<float>(j) / ny;
            if (jitter && i > 0 && j > 0 && i < nx && j < ny) {
                // ±0.2 celdas: sin pliegues (el peor caso perpendicular a una diagonal es 0.57 < 0.707)
                px += rng.uniform(-0.2f, 0.2f) * size / nx;
                py += rng.uniform(-0.2f, 0.2f) * size / ny;
            }
            add_vertex(m, {px, py, z0}, {0, 0, 1}, color);
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const u32 a = base + static_cast<u32>(j * (nx + 1) + i), b = a + 1, d = a + static_cast<u32>(nx + 1), e = d + 1;
            const bool flip = ((i * 7 + j * 13) % 3) == 0;   // diagonales mezcladas
            if (!flip) { m.tri.insert(m.tri.end(), {a, b, e}); m.tri.insert(m.tri.end(), {a, e, d}); }
            else { m.tri.insert(m.tri.end(), {a, b, d}); m.tri.insert(m.tri.end(), {b, e, d}); }
        }
    m.recompute_bounds();
}
// Cuadrilátero plano (2 triángulos) con normal n.
static void make_quad(Mesh& m, Vec3 a, Vec3 b, Vec3 c, Vec3 d, u32 col) {
    const u32 base = static_cast<u32>(m.pos.size());
    const Vec3 n = normalize(cross(b - a, d - a));
    add_vertex(m, a, n, col); add_vertex(m, b, n, col); add_vertex(m, c, n, col); add_vertex(m, d, n, col);
    m.tri.insert(m.tri.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    m.recompute_bounds();
}

static u32 col_sphere(float u, float v) { (void)u; return colormap(Colormap::Turbo, 0.12f + 0.76f * v); }
static u32 col_torus(float u, float v) { (void)v; return colormap(Colormap::CoolWarm, 0.5f + 0.5f * std::sin(u * 6.2831853f * 2.0f)); }

static void save(const Framebuffer& fb, const std::string& name) {
    const std::string p = std::string(k_out) + name;
    if (!png::write_argb(p, fb.color.data(), fb.w, fb.h, fb.stride)) std::printf("  (no se pudo escribir %s)\n", p.c_str());
    else std::printf("  → %s\n", p.c_str());
}
static void clear(Framebuffer& fb, u32 c = 0xFF000000u) { fb.clear_color(c); fb.clear_depth(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static void background(Framebuffer& fb, Rect r) { fb.gradient(r, 0xFF30353Fu, 0xFF121418u); }

static Camera make_cam(Rect vp, Vec3 target, float yaw, float pitch, float dist, bool ortho = false) {
    Camera c;
    c.target = target; c.yaw = yaw; c.pitch = pitch; c.distance = dist; c.ortho = ortho;
    c.update(vp);
    return c;
}
// Área (px²) del polígono proyectado (shoelace).
static double projected_area(const Camera& cam, const Vec3* p, int n) {
    double a = 0;
    float sx[8] = {}, sy[8] = {}, d = 0;
    for (int i = 0; i < n; ++i) cam.project(p[i], sx[i], sy[i], d);
    for (int i = 0; i < n; ++i) { const int j = (i + 1) % n; a += static_cast<double>(sx[i]) * sy[j] - static_cast<double>(sx[j]) * sy[i]; }
    return std::fabs(a) * 0.5;
}

// ============================================================================================
//  Correctitud de mallas
// ============================================================================================
static void test_coverage(Framebuffer& fb) {
    std::printf("[malla] cobertura de un cuadrado plano vs área proyectada\n");
    for (int ortho = 0; ortho < 2; ++ortho) {
        Mesh m;
        const Vec3 q[4] = {{-60, -45, 0}, {60, -45, 0}, {60, 45, 0}, {-60, 45, 0}};
        make_quad(m, q[0], q[1], q[2], q[3], 0xFFFFFFFFu);
        const Camera cam = make_cam({0, 0, fb.w, fb.h}, {3, 2, 0}, 0.3f, 0.9f, 400.0f, ortho != 0);
        clear(fb, 0xFF000000u);
        MeshStyle st; st.debug_overdraw = true; st.two_sided = true;
        draw_mesh(fb, cam, m, Light{}, st);
        usize n = 0, dbl = 0;
        for (int y = 0; y < fb.h; ++y)
            for (int x = 0; x < fb.w; ++x) { const u32 v = fb.row(y)[x] & 0xFFFFFFu; n += v != 0; dbl += v > 1; }
        const double area = projected_area(cam, q, 4);
        const double err = (static_cast<double>(n) - area) / area;
        CHECK(std::fabs(err) < 0.01 && dbl == 0, "%s: %zu píxeles vs área %.1f (error %.3f %%), dobles=%zu",
              ortho ? "ortográfica" : "perspectiva", n, area, err * 100.0, dbl);
    }
    {   // Cara frontal/trasera: esfera con two_sided=false sigue viéndose (orientación correcta)
        Mesh m;
        make_sphere(m, {0, 0, 0}, 50.0f, 48, 24, col_sphere);
        const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.7f, 0.3f, 300.0f);
        clear(fb, 0xFF000000u);
        MeshStyle st; st.debug_overdraw = true; st.two_sided = true;
        draw_mesh(fb, cam, m, Light{}, st);
        usize two = 0;
        for (int y = 0; y < fb.h; ++y) for (int x = 0; x < fb.w; ++x) two += (fb.row(y)[x] & 0xFFFFFFu) >= 2;
        clear(fb, 0xFF000000u);
        MeshStyle s1; s1.two_sided = false;
        draw_mesh(fb, cam, m, Light{}, s1);
        const u64 binned = last_mesh_stats().tris_binned;
        usize one = 0;
        for (int y = 0; y < fb.h; ++y) for (int x = 0; x < fb.w; ++x) one += (fb.row(y)[x] & 0xFFFFFFu) != 0;
        CHECK(one > 10000 && binned < m.tri_count() * 6 / 10 && two > one * 9 / 10,
              "culling de caras traseras: %zu px visibles, %llu/%zu tris repartidos (doble cara: %zu px con 2 capas)",
              one, (unsigned long long)binned, m.tri_count(), two);
    }
}

// Estanqueidad: rejilla densa de triángulos con vértices perturbados → cada píxel interior
// recibe EXACTAMENTE un fragmento (0 = grieta, ≥2 = doble impacto en arista compartida).
static void test_watertight(Framebuffer& fb) {
    std::printf("[malla] estanqueidad (regla top-left, punto fijo 28.4)\n");
    struct Case { float yaw, pitch, dist; bool ortho; int n; };
    const Case cases[] = {{0.2f, 0.8f, 300.0f, false, 120}, {-0.9f, 0.35f, 260.0f, false, 300}, {0.0f, 1.5f, 400.0f, true, 200}, {2.1f, 0.25f, 180.0f, false, 400}};
    std::vector<u32> ref;
    for (const Case& cs : cases) {
        const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, cs.yaw, cs.pitch, cs.dist, cs.ortho);
        MeshStyle st; st.debug_overdraw = true;
        Mesh big;   // referencia: el mismo cuadrado con 2 triángulos
        make_grid(big, -100, -100, 200, 1, 1, 0, 0xFFFFFFFFu, false);
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, big, Light{}, st);
        ref.assign(static_cast<usize>(fb.w) * fb.h, 0);
        for (int y = 0; y < fb.h; ++y)
            for (int x = 0; x < fb.w; ++x) ref[static_cast<usize>(y) * fb.w + x] = fb.row(y)[x] & 0xFFFFFFu;
        Mesh m;
        make_grid(m, -100, -100, 200, cs.n, cs.n, 0, 0xFFFFFFFFu, true);
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, m, Light{}, st);
        usize interior = 0, holes = 0, doubles = 0;
        for (int y = 1; y < fb.h - 1; ++y)
            for (int x = 1; x < fb.w - 1; ++x) {
                const u32 v = fb.row(y)[x] & 0xFFFFFFu;
                doubles += v > 1;
                bool in = true;
                for (int dy = -1; dy <= 1 && in; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) in &= ref[static_cast<usize>(y + dy) * fb.w + x + dx] == 1;
                if (!in) continue;
                ++interior;
                holes += v == 0;
            }
        CHECK(interior > 10000 && holes == 0 && doubles == 0, "%zu tris (%s yaw=%.1f pitch=%.2f): %zu px interiores, grietas=%zu, dobles=%zu",
              m.tri_count(), cs.ortho ? "orto" : "persp", cs.yaw, cs.pitch, interior, holes, doubles);
    }
    {   // (revisión) Estanqueidad con MUCHOS triángulos recortados: znear grande → el plano cercano corta
        // cientos de aristas compartidas de la rejilla; los puntos de corte deben coincidir exactamente.
        Camera cam;
        cam.target = {0, 0, 0}; cam.yaw = 0.4f; cam.pitch = 0.6f; cam.distance = 300.0f; cam.znear = 230.0f;
        cam.update({0, 0, fb.w, fb.h});
        MeshStyle st; st.debug_overdraw = true;
        Mesh big, m;
        make_grid(big, -150, -150, 300, 1, 1, 0, 0xFFFFFFFFu, false);
        make_grid(m, -150, -150, 300, 150, 150, 0, 0xFFFFFFFFu, true);
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, big, Light{}, st);
        ref.assign(static_cast<usize>(fb.w) * fb.h, 0);
        for (int y = 0; y < fb.h; ++y)
            for (int x = 0; x < fb.w; ++x) ref[static_cast<usize>(y) * fb.w + x] = fb.row(y)[x] & 0xFFFFFFu;
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, m, Light{}, st);
        const u64 nclip = last_mesh_stats().tris_clipped;
        usize interior = 0, holes = 0, doubles = 0;
        for (int y = 1; y < fb.h - 1; ++y)
            for (int x = 1; x < fb.w - 1; ++x) {
                const u32 v = fb.row(y)[x] & 0xFFFFFFu;
                doubles += v > 1;
                bool in = true;
                for (int dy = -1; dy <= 1 && in; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) in &= ref[static_cast<usize>(y + dy) * fb.w + x + dx] == 1;
                if (!in) continue;
                ++interior;
                holes += v == 0;
            }
        CHECK(nclip > 100 && interior > 10000 && holes == 0 && doubles == 0,
              "znear=230 cortando la rejilla: %llu tris de recorte, %zu px interiores, grietas=%zu, dobles=%zu",
              (unsigned long long)nclip, interior, holes, doubles);
    }
}

static void test_depth_order(Framebuffer& fb) {
    std::printf("[malla] orden de profundidad (planos que se cortan, ambos órdenes de envío)\n");
    // Plano A: z = 0.5x (rojo), plano B: z = -0.5x (verde): se cortan en x = 0. Visto desde arriba,
    // para x > 0 el más alto (más cercano) es A; para x < 0, B.
    for (int order = 0; order < 2; ++order) {
        Mesh m;
        auto addA = [&] { make_quad(m, {-80, -60, -40}, {80, -60, 40}, {80, 60, 40}, {-80, 60, -40}, 0xFFFF0000u); };
        auto addB = [&] { make_quad(m, {-80, -60, 40}, {80, -60, -40}, {80, 60, -40}, {-80, 60, 40}, 0xFF00FF00u); };
        if (order == 0) { addA(); addB(); } else { addB(); addA(); }
        const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.0f, 1.45f, 500.0f);
        clear(fb, 0xFF000000u);
        MeshStyle st; st.two_sided = true;
        draw_mesh(fb, cam, m, Light{}, st);
        int ok = 0, total = 0;
        for (float x : {-60.0f, -30.0f, -10.0f, 10.0f, 30.0f, 60.0f})
            for (float y : {-40.0f, 0.0f, 40.0f}) {
                float sx = 0, sy = 0, d = 0;
                cam.project({x, y, 0}, sx, sy, d);
                const u32 c = fb.row(static_cast<int>(sy))[static_cast<int>(sx)];
                const int r = (c >> 16) & 255, g = (c >> 8) & 255;
                ++total;
                ok += x > 0 ? (r > 2 * g + 10) : (g > 2 * r + 10);
            }
        CHECK(ok == total, "orden de envío %d: %d/%d muestras con el plano correcto delante", order, ok, total);
    }
}

static void test_near_clip(Framebuffer& fb) {
    std::printf("[malla] recorte en el plano cercano\n");
    {   // Cámara DENTRO de una esfera grande: todo el viewport debe cubrirse (caras interiores).
        Mesh m;
        make_sphere(m, {0, 0, 0}, 100.0f, 96, 48, col_sphere);
        const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.4f, 0.3f, 3.0f);
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, m, Light{}, MeshStyle{});
        usize n = 0;
        for (int y = 0; y < fb.h; ++y)
            for (int x = 0; x < fb.w; ++x) n += (fb.row(y)[x] & 0xFFFFFFu) != 0;
        const double frac = static_cast<double>(n) / (static_cast<double>(fb.w) * fb.h);
        CHECK(frac > 0.999, "cámara dentro de una esfera: %.3f %% del viewport cubierto", frac * 100.0);
    }
    {   // Suelo enorme que cruza el plano cercano con cámara rasante: se recorta, no se descarta.
        Mesh m;
        make_grid(m, -5000, -5000, 10000, 60, 60, 0, 0xFFB0B0B0u, false);
        Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 2}, 0.3f, 0.05f, 6.0f);
        clear(fb, 0xFF000000u);
        MeshStyle st; st.use_vertex_color = false;
        draw_mesh(fb, cam, m, Light{}, st);
        const RasterStats rs = last_mesh_stats();
        usize bottom = 0;
        for (int x = 0; x < fb.w; ++x) bottom += (fb.row(fb.h - 1)[x] & 0xFFFFFFu) != 0;
        CHECK(rs.tris_clipped > 0 && bottom == static_cast<usize>(fb.w), "suelo rasante: %llu triángulos de recorte, fila inferior cubierta %zu/%d",
              (unsigned long long)rs.tris_clipped, bottom, fb.w);
        // Estanqueidad también en triángulos recortados (cercano + banda de guarda)
        MeshStyle od; od.debug_overdraw = true;
        Mesh g;
        make_grid(g, -3000, -3000, 6000, 150, 150, 0, 0xFFFFFFFFu, true);
        clear(fb, 0xFF000000u);
        draw_mesh(fb, cam, g, Light{}, od);
        usize dbl = 0, holes = 0;
        for (int y = fb.h / 2 + 40; y < fb.h; ++y)
            for (int x = 0; x < fb.w; ++x) { const u32 v = fb.row(y)[x] & 0xFFFFFFu; dbl += v > 1; holes += v == 0; }
        CHECK(dbl == 0 && holes == 0, "rejilla recortada: grietas=%zu dobles=%zu (tris de recorte %llu)", holes, dbl,
              (unsigned long long)last_mesh_stats().tris_clipped);
    }
}

static void test_mesh_visual(Framebuffer& fb) {
    std::printf("[malla] escena esfera + toro + suelo (PNG)\n");
    Mesh m;
    make_sphere(m, {0, 0, 60}, 55.0f, 160, 80, col_sphere);
    make_torus(m, {0, 0, 60}, 105.0f, 26.0f, 180, 60, col_torus);
    const Rect vp{0, 0, fb.w, fb.h};
    const Camera cam = make_cam(vp, {0, 0, 45}, -0.55f, 0.38f, 470.0f);
    clear(fb);
    background(fb, vp);
    Aabb ext; ext.lo = {-400, -300, 0}; ext.hi = {400, 300, 0};
    draw_ground(fb, cam, -5.0f, ext, 20.0f, 0.0f, 0xFF3A3F48u, 0xFF8A93A6u);
    draw_mesh(fb, cam, m, Light{}, MeshStyle{});
    const RasterStats st = last_mesh_stats();
    std::printf("  tris=%llu repartidos=%llu recortados=%llu refs=%llu tiles=%d/%d\n", (unsigned long long)st.tris_in,
                (unsigned long long)st.tris_binned, (unsigned long long)st.tris_clipped, (unsigned long long)st.bin_entries, st.tiles_busy, st.tiles);
    save(fb, "mesh_sphere_torus.png");
    screen_space_edges(fb, vp, 0.6f);
    save(fb, "mesh_sphere_torus_ao.png");
    fxaa(fb, vp);
    save(fb, "mesh_sphere_torus_ao_fxaa.png");
    // variantes: alambre, translúcido, ortográfica
    Framebuffer f2; f2.resize(960, 600);
    const Rect vp2{0, 0, f2.w, f2.h};
    const Camera c2 = make_cam(vp2, {0, 0, 45}, -0.55f, 0.38f, 470.0f);
    Mesh s;
    make_sphere(s, {0, 0, 60}, 55.0f, 40, 20, col_sphere);
    make_torus(s, {0, 0, 60}, 105.0f, 26.0f, 48, 16, col_torus);
    clear(f2); background(f2, vp2);
    MeshStyle w; w.wireframe_overlay = true;
    draw_mesh(f2, c2, s, Light{}, w);
    save(f2, "mesh_wire.png");
    clear(f2); background(f2, vp2);
    draw_ground(f2, c2, -5.0f, ext, 20.0f, 0.0f, 0xFF3A3F48u, 0xFF8A93A6u);
    MeshStyle a; a.alpha = 0.45f;
    draw_mesh(f2, c2, m, Light{}, a);
    save(f2, "mesh_alpha.png");
    clear(f2); background(f2, vp2);
    const Camera c3 = make_cam(vp2, {0, 0, 45}, 0.0f, 0.0f, 470.0f, true);
    draw_mesh(f2, c3, m, Light{}, MeshStyle{});
    save(f2, "mesh_ortho_side.png");
    CHECK(st.tris_binned > 0, "la escena produce triángulos");
}

// ============================================================================================
//  Pruebas añadidas por la revisión (defectos encontrados y cobertura que faltaba)
// ============================================================================================
static void make_cp_texture(std::vector<u32>& tex, int tw, int th);
static Vec3 g_view_dir;   // hacia el ojo (para colorear hemisferios)
static u32 col_near_far(float u, float v) {
    const float th = k_pi * v, ph = 2.0f * k_pi * u;
    const Vec3 n{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
    return dot(n, g_view_dir) > 0.0f ? 0xFF00FF00u : 0xFFFF0000u;   // cercano verde, lejano rojo
}
// Malla translúcida: en una malla cerrada la cara CERCANA debe quedar encima (composición "over").
// Antes de la revisión se mezclaban las traseras después de las frontales y dominaba la cara lejana.
static void test_translucent_order(Framebuffer& fb) {
    std::printf("[malla] translúcida: la cara cercana domina (orden trasera → frontal)\n");
    const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.3f, 0.2f, 300.0f);
    g_view_dir = normalize(cam.eye - cam.target);
    Mesh m;
    make_sphere(m, {0, 0, 0}, 60.0f, 96, 48, col_near_far);
    Light flat; flat.ambient = 1.0f; flat.diffuse = 0.0f; flat.specular = 0.0f; flat.rim = 0.0f;
    for (int two = 0; two < 2; ++two) {
        clear(fb, 0xFF000000u);
        MeshStyle st; st.alpha = 0.5f; st.two_sided = two != 0;
        draw_mesh(fb, cam, m, flat, st);
        float sx = 0, sy = 0, d = 0;
        cam.project({0, 0, 0}, sx, sy, d);
        const u32 c = fb.row(static_cast<int>(sy))[static_cast<int>(sx)];
        const int r = (c >> 16) & 255, g = (c >> 8) & 255;
        // doble cara: ≈ 0.5·verde + 0.25·rojo (orden correcto) frente a 0.5·rojo + 0.25·verde (invertido)
        CHECK(two ? (g > r + 30) : (g > 80 && r < 10), "%s: centro R=%d G=%d", two ? "doble cara" : "una cara", r, g);
    }
}
// Normales nulas (degeneradas) no deben dar NaN → píxeles negros; sin normales, una cara trasera en
// doble cara se sombrea igual que la frontal (la normal por defecto ya mira a la cámara).
static void test_degenerate_normals(Framebuffer& fb) {
    std::printf("[malla] normales nulas / ausentes\n");
    const Camera cam = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.3f, 0.4f, 300.0f);
    Mesh m;
    make_sphere(m, {0, 0, 0}, 60.0f, 48, 24, col_sphere);
    for (Vec3& n : m.nrm) n = {0, 0, 0};
    clear(fb, 0xFF000000u);
    draw_mesh(fb, cam, m, Light{}, MeshStyle{});
    usize covered = 0, black = 0;
    for (int y = 0; y < fb.h; ++y)
        for (int x = 0; x < fb.w; ++x) {
            const bool cov = fb.drow(y)[x] < k_inf;
            covered += cov;
            black += cov && (fb.row(y)[x] & 0xFFFFFFu) == 0;
        }
    CHECK(covered > 10000 && black == 0, "normales nulas: %zu píxeles cubiertos, %zu negros (NaN)", covered, black);
    const Camera c2 = make_cam({0, 0, fb.w, fb.h}, {0, 0, 0}, 0.0f, 0.6f, 300.0f);
    u32 px[2] = {};
    for (int flip = 0; flip < 2; ++flip) {
        Mesh q;
        q.pos = {{-50, -50, 0}, {50, -50, 0}, {50, 50, 0}, {-50, 50, 0}};
        q.color.assign(4, 0xFFC0C0C0u);
        q.tri = flip ? std::vector<u32>{0, 2, 1, 0, 3, 2} : std::vector<u32>{0, 1, 2, 0, 2, 3};
        clear(fb, 0xFF000000u);
        draw_mesh(fb, c2, q, Light{}, MeshStyle{});
        float sx = 0, sy = 0, d = 0;
        c2.project({0, 0, 0}, sx, sy, d);
        px[flip] = fb.row(static_cast<int>(sy))[static_cast<int>(sx)];
    }
    CHECK(px[0] == px[1], "sin normales: cara frontal 0x%08X == cara trasera 0x%08X", px[0], px[1]);
}

// Escena con TODAS las primitivas (malla opaca + translúcida + alambre, suelo con textura, cuadrilátero,
// polilíneas, líneas, puntos aditivos/alfa, flechas, SSAO, FXAA). Devuelve el hash de color + depth.
static u64 render_all(Framebuffer& fb, const Mesh& m, const Mesh& small, const std::vector<Vec3>& pts,
                      const std::vector<u32>& cols, const std::vector<u32>& starts, const std::vector<u32>& counts,
                      const std::vector<u32>& tex, int tw, int th) {
    const Rect vp{37, 21, fb.w - 60, fb.h - 40};
    const Camera cam = make_cam(vp, {0, 0, 30}, -0.6f, 0.35f, 420.0f);
    clear(fb, 0xFF000000u);
    background(fb, vp);
    Aabb ext; ext.lo = {-400, -250, 0}; ext.hi = {500, 250, 0};
    draw_ground(fb, cam, -40.0f, ext, 16.0f, 3.0f, 0xFF3A3F48u, 0xFF9AA3B6u, tex.data(), tw, th);
    draw_mesh(fb, cam, m, Light{}, MeshStyle{});
    MeshStyle wire; wire.wireframe_overlay = true; wire.two_sided = false;
    draw_mesh(fb, cam, small, Light{}, wire);
    const Vec3 sl[4] = {{-300, 0, -40}, {300, 0, -40}, {300, 0, 120}, {-300, 0, 120}};
    draw_textured_quad(fb, cam, sl, tex.data(), tw, th, 0.7f, true, false);
    draw_polylines(fb, cam, pts, cols, starts, counts, 1.7f, true);
    draw_lines(fb, cam, std::span<const Vec3>(pts.data(), 400), cols, true, 3.0f, true);
    draw_points(fb, cam, pts, cols, 3.0f, true);
    draw_points(fb, cam, std::span<const Vec3>(pts.data(), 3000), cols, 6.0f, false);
    draw_arrow(fb, cam, {0, 0, 90}, {0, 0, 170}, 0xFFE04040u, 3.0f);
    MeshStyle tr; tr.alpha = 0.4f;
    draw_mesh(fb, cam, small, Light{}, tr);
    screen_space_edges(fb, vp, 0.6f);
    fxaa(fb, vp);
    u64 h = 1469598103934665603ull;
    for (int y = 0; y < fb.h; ++y)
        for (int x = 0; x < fb.w; ++x) {
            float z = fb.drow(y)[x];
            u32 zb;
            std::memcpy(&zb, &z, 4);
            h = (h ^ fb.row(y)[x]) * 1099511628211ull;
            h = (h ^ zb) * 1099511628211ull;
        }
    return h;
}
// Determinismo: el mismo cuadro renderizado en paralelo (pool completo) y en serie (llamado desde
// dentro de un trabajo del pool → todos los parallel_for internos se ejecutan en serie) debe ser
// idéntico bit a bit. Detecta carreras entre tiles/filas y dependencias del orden de ejecución.
static void test_determinism(Framebuffer& fb) {
    std::printf("[determinismo] paralelo vs serie, todas las primitivas\n");
    Mesh m, small;
    make_sphere(m, {0, 0, 50}, 55.0f, 160, 80, col_sphere);
    make_torus(m, {0, 0, 50}, 105.0f, 26.0f, 180, 60, col_torus);
    make_grid(m, -2000, -2000, 4000, 40, 40, -38.0f, 0xFF707070u, false);   // cruza el plano cercano / banda de guarda
    make_torus(small, {60, -40, 120}, 40.0f, 10.0f, 48, 16, col_torus);
    std::vector<Vec3> pts;
    std::vector<u32> cols, starts, counts;
    WyRand rng(99);
    for (int k = 0; k < 400; ++k) {
        starts.push_back(static_cast<u32>(pts.size()));
        const float y0 = rng.uniform(-200, 200), z0 = rng.uniform(-30, 150);
        const int n = 20 + static_cast<int>(rng.below(60));
        for (int i = 0; i < n; ++i) {
            pts.push_back({-300.0f + 600.0f * i / n, y0 + 30.0f * std::sin(i * 0.2f + k), z0 + 20.0f * std::cos(i * 0.13f)});
            cols.push_back(colormap(Colormap::Turbo, static_cast<float>(i) / n) & 0xD0FFFFFFu);
        }
        counts.push_back(static_cast<u32>(n));
    }
    std::vector<u32> tex;
    make_cp_texture(tex, 128, 64);
    const u64 hp = render_all(fb, m, small, pts, cols, starts, counts, tex, 128, 64);
    const u64 hp2 = render_all(fb, m, small, pts, cols, starts, counts, tex, 128, 64);
    u64 hs = 0;
    pool().run_slots([&](int s, int) { if (s == 0) hs = render_all(fb, m, small, pts, cols, starts, counts, tex, 128, 64); }, 2);
    CHECK(hp == hs && hp == hp2, "hash paralelo %016llx, repetido %016llx, serie %016llx", (unsigned long long)hp, (unsigned long long)hp2, (unsigned long long)hs);
    save(fb, "determinism_scene.png");
}

// FXAA: imagen plana → sin cambios; borde duro diagonal → sólo cambian píxeles del borde y a valores
// intermedios. SSAO: plano frontal (profundidad constante) y fondo (inf) → sin cambios.
static void test_post_functional(Framebuffer& fb) {
    std::printf("[postproceso] FXAA y SSAO funcionales\n");
    const Rect vp{0, 0, fb.w, fb.h};
    fb.clear_color(0xFF406080u);
    fb.clear_depth();
    std::vector<u32> a(fb.color.begin(), fb.color.end());
    fxaa(fb, vp);
    screen_space_edges(fb, vp, 1.0f);
    CHECK(std::equal(a.begin(), a.end(), fb.color.begin()), "imagen plana y fondo infinito: FXAA + SSAO no cambian nada");
    // Borde en escalera (x < y·0.37 → blanco) sobre negro, profundidad constante
    for (int y = 0; y < fb.h; ++y)
        for (int x = 0; x < fb.w; ++x) {
            fb.row(y)[x] = static_cast<float>(x) < static_cast<float>(y) * 0.37f + 300.0f ? 0xFFFFFFFFu : 0xFF000000u;
            fb.drow(y)[x] = 250.0f;
        }
    a.assign(fb.color.begin(), fb.color.end());
    screen_space_edges(fb, vp, 1.0f);
    const bool ssao_flat = std::equal(a.begin(), a.end(), fb.color.begin());
    fxaa(fb, vp);
    usize changed = 0, far_changed = 0, mid = 0;
    for (int y = 1; y < fb.h - 1; ++y)
        for (int x = 1; x < fb.w - 1; ++x) {
            const usize i = static_cast<usize>(y) * fb.stride + x;
            if (fb.color.data()[i] == a[i]) continue;
            ++changed;
            const float edge = static_cast<float>(y) * 0.37f + 300.0f;
            far_changed += std::fabs(static_cast<float>(x) - edge) > 3.0f;
            const u32 g = (fb.color.data()[i] >> 8) & 255;
            mid += g > 10 && g < 245;
        }
    CHECK(ssao_flat, "SSAO: superficie de profundidad constante sin oscurecer");
    CHECK(changed > static_cast<usize>(fb.h) / 2 && far_changed == 0 && mid * 10 >= changed * 9,
          "FXAA: %zu píxeles de borde suavizados (%zu intermedios), %zu lejos del borde", changed, mid, far_changed);
}

// ============================================================================================
//  Líneas, puntos, cuadriláteros, suelo
// ============================================================================================
static void test_lines(Framebuffer& fb) {
    std::printf("[líneas] integral de cobertura AA y aspecto\n");
    const Rect vp{0, 0, fb.w, fb.h};
    // Cámara ortográfica cenital; línea horizontal: la suma de cobertura en una columna = ancho.
    const Camera cam = make_cam(vp, {0, 0, 0}, 0.0f, 1.5f, 500.0f, true);
    for (float width : {1.0f, 1.5f, 3.0f, 7.0f}) {
        clear(fb, 0xFF000000u);
        const Vec3 p[2] = {{-100, 0.37f, 0}, {100, 0.37f, 0}};
        const u32 c[1] = {0xFFFFFFFFu};
        draw_lines(fb, cam, p, c, false, width, false);
        float sx = 0, sy = 0, d = 0;
        cam.project({0, 0.37f, 0}, sx, sy, d);
        double sum = 0;
        const int x = static_cast<int>(sx);
        for (int y = 0; y < fb.h; ++y) sum += static_cast<double>(fb.row(y)[x] & 255) / 255.0;
        CHECK(std::fabs(sum - width) < 0.08 * width + 0.1, "ancho %.1f px: integral de cobertura %.3f", width, sum);
    }
    // Aspecto: polilíneas (líneas de corriente) con profundidad contra una esfera + caja + flechas
    const Camera c2 = make_cam(vp, {0, 0, 40}, -0.5f, 0.4f, 420.0f);
    clear(fb); background(fb, vp);
    Aabb ext; ext.lo = {-300, -200, 0}; ext.hi = {300, 200, 0};
    draw_ground(fb, c2, -10.0f, ext, 20.0f, 0.0f, 0xFF3A3F48u, 0xFF8A93A6u);
    Mesh m;
    make_sphere(m, {0, 0, 40}, 45.0f, 64, 32, col_sphere);
    draw_mesh(fb, c2, m, Light{}, MeshStyle{});
    std::vector<Vec3> pts;
    std::vector<u32> cols;
    std::vector<u32> starts, counts;
    for (int k = 0; k < 60; ++k) {
        starts.push_back(static_cast<u32>(pts.size()));
        const float y0 = -120.0f + 4.0f * k, z0 = 40.0f + 25.0f * std::sin(k * 0.3f);
        for (int i = 0; i <= 200; ++i) {
            const float x = -250.0f + 2.5f * i;
            const float bump = 50.0f * std::exp(-sq(x / 60.0f)) * std::exp(-sq(y0 / 70.0f));
            pts.push_back({x, y0 + 8.0f * std::sin(x * 0.03f + k), z0 + bump});
            cols.push_back(colormap(Colormap::Turbo, static_cast<float>(i) / 200.0f));
        }
        counts.push_back(201);
    }
    draw_polylines(fb, c2, pts, cols, starts, counts, 1.6f, true);
    Aabb box; box.lo = {-150, -130, -10}; box.hi = {150, 130, 110};
    draw_box_wire(fb, c2, box, 0xC0A0B0C8u, 1.2f);
    const u64 before = last_mesh_stats().tris_in;
    draw_arrow(fb, c2, {0, 0, 100}, {0, 0, 170}, 0xFFE04040u, 3.0f);
    draw_arrow(fb, c2, {0, 0, 100}, {80, 0, 100}, 0xFF40A0E0u, 3.0f);
    CHECK(last_mesh_stats().tris_in == before && before == m.tri_count(), "draw_arrow no pisa last_mesh_stats() (%llu tris)", (unsigned long long)before);
    save(fb, "lines_streamlines.png");
}

static void test_points(Framebuffer& fb) {
    std::printf("[puntos] centrado del splat y aspecto\n");
    const Rect vp{0, 0, fb.w, fb.h};
    const Camera cam = make_cam(vp, {0, 0, 0}, 0.3f, 0.6f, 300.0f);
    clear(fb, 0xFF000000u);
    const Vec3 p[1] = {{3.3f, -2.7f, 1.1f}};
    const u32 c[1] = {0xFFFFFFFFu};
    draw_points(fb, cam, p, c, 7.0f, true);
    float sx = 0, sy = 0, d = 0;
    cam.project(p[0], sx, sy, d);
    double w = 0, mx = 0, my = 0;
    for (int y = static_cast<int>(sy) - 10; y <= static_cast<int>(sy) + 10; ++y)
        for (int x = static_cast<int>(sx) - 10; x <= static_cast<int>(sx) + 10; ++x) {
            const double v = fb.row(y)[x] & 255;
            w += v; mx += v * (x + 0.5); my += v * (y + 0.5);
        }
    mx /= w; my /= w;
    CHECK(std::fabs(mx - sx) < 0.08 && std::fabs(my - sy) < 0.08, "centroide (%.3f, %.3f) vs proyección (%.3f, %.3f)", mx, my, sx, sy);
    // Aditivo: dos veces el mismo punto = el doble (saturado)
    clear(fb, 0xFF000000u);
    const u32 dim[1] = {0xFF404040u};
    draw_points(fb, cam, p, dim, 7.0f, true);
    const u32 once = fb.row(static_cast<int>(sy))[static_cast<int>(sx)] & 255;
    draw_points(fb, cam, p, dim, 7.0f, true);
    const u32 twice = fb.row(static_cast<int>(sy))[static_cast<int>(sx)] & 255;
    CHECK(once > 40 && twice >= 2 * once - 1 && twice <= 2 * once + 1, "aditivo: %u → %u", once, twice);
    // Aspecto: humo aditivo + partículas alfa alrededor de una esfera
    clear(fb); background(fb, vp);
    Mesh m;
    make_sphere(m, {0, 0, 0}, 50.0f, 64, 32, col_sphere);
    const Camera c2 = make_cam(vp, {0, 0, 0}, -0.5f, 0.4f, 380.0f);
    draw_mesh(fb, c2, m, Light{}, MeshStyle{});
    std::vector<Vec3> pts;
    std::vector<u32> cols;
    WyRand rng(7);
    for (int i = 0; i < 60000; ++i) {
        const float t = rng.uniform(), a = rng.uniform(0, 6.2831f), r = 60.0f + 25.0f * rng.uniform();
        pts.push_back({-200.0f + 400.0f * t, r * std::cos(a) * (0.3f + t), r * std::sin(a) * (0.3f + t)});
        cols.push_back(colormap(Colormap::Inferno, 0.25f + 0.7f * t) & 0x60FFFFFFu);
    }
    draw_points(fb, c2, pts, cols, 3.0f, true);
    std::vector<Vec3> p2;
    std::vector<u32> c2v;
    for (int i = 0; i < 400; ++i) { p2.push_back({rng.uniform(-150, 150), rng.uniform(-150, 150), rng.uniform(-60, 60)}); c2v.push_back(0xE0F0F4FFu); }
    draw_points(fb, c2, p2, c2v, 6.0f, false);
    save(fb, "points_smoke.png");
}

static void test_textured_quad(Framebuffer& fb) {
    std::printf("[cuadrilátero texturizado] UV perspectiva-correctas\n");
    const Rect vp{0, 0, fb.w, fb.h};
    // Textura 2×1: mitad izquierda roja, derecha verde. En perspectiva oblicua la frontera debe
    // caer en la proyección del punto medio 3D (no en el punto medio de pantalla).
    const u32 tex2[2] = {0xFFFF0000u, 0xFF00FF00u};
    const Camera cam = make_cam(vp, {0, 0, 0}, 1.2f, 0.2f, 160.0f);
    const Vec3 q[4] = {{-150, 0, -30}, {150, 0, -30}, {150, 0, 30}, {-150, 0, 30}};
    clear(fb, 0xFF000000u);
    draw_textured_quad(fb, cam, q, tex2, 2, 1, 1.0f, true, true);
    float sx = 0, sy = 0, d = 0;
    cam.project({0, 0, 0}, sx, sy, d);
    const int y = static_cast<int>(sy);
    int trans = -1;
    for (int x = 1; x < fb.w; ++x) {
        const u32 a = fb.row(y)[x - 1], b = fb.row(y)[x];
        const int ga = (a >> 8) & 255, ra = (a >> 16) & 255, gb = (b >> 8) & 255, rb = (b >> 16) & 255;
        if ((ra > ga) != (rb > gb) && (a & 0xFFFFFF) && (b & 0xFFFFFF)) { trans = x; break; }
    }
    float ax = 0, ay = 0, bx = 0, by = 0;
    cam.project(q[0] * 0.5f + q[3] * 0.5f, ax, ay, d);
    cam.project(q[1] * 0.5f + q[2] * 0.5f, bx, by, d);
    const float screen_mid = 0.5f * (ax + bx);
    CHECK(trans >= 0 && std::fabs(static_cast<float>(trans) - sx) < 2.0f && std::fabs(screen_mid - sx) > 20.0f,
          "frontera en x=%d, punto medio 3D proyectado %.1f (punto medio de pantalla %.1f)", trans, sx, screen_mid);
    // Aspecto: plano de corte con mapa de colores sobre esfera, translúcido y con agujero (alfa 0)
    clear(fb); background(fb, vp);
    const int tw = 256, th = 128;
    std::vector<u32> tex(static_cast<usize>(tw) * th);
    for (int j = 0; j < th; ++j)
        for (int i = 0; i < tw; ++i) {
            const float u = (i + 0.5f) / tw, v = (j + 0.5f) / th;
            const float r = std::sqrt(sq((u - 0.5f) * 2.0f) + sq((v - 0.5f) * 1.0f));
            const float val = 0.5f + 0.45f * std::sin(u * 12.0f) * std::exp(-r * 1.5f);
            u32 c = colormap(Colormap::Viridis, val);
            if (r < 0.15f) c &= 0x00FFFFFFu;
            tex[static_cast<usize>(j) * tw + i] = c;
        }
    Mesh m;
    make_sphere(m, {0, 0, 0}, 40.0f, 64, 32, col_sphere);
    const Camera c2 = make_cam(vp, {0, 0, 0}, -0.6f, 0.45f, 380.0f);
    draw_mesh(fb, c2, m, Light{}, MeshStyle{});
    const Vec3 sl[4] = {{-200, 0, -80}, {200, 0, -80}, {200, 0, 80}, {-200, 0, 80}};
    draw_textured_quad(fb, c2, sl, tex.data(), tw, th, 0.85f, true, false);
    save(fb, "textured_quad.png");
}

static void make_cp_texture(std::vector<u32>& tex, int tw, int th) {
    tex.resize(static_cast<usize>(tw) * th);
    for (int j = 0; j < th; ++j)
        for (int i = 0; i < tw; ++i) {
            const float u = (i + 0.5f) / tw - 0.42f, v = (j + 0.5f) / th - 0.5f;
            const float r = std::sqrt(sq(u * 5.0f) + sq(v * 4.0f));
            const float a = std::exp(-r * r * 1.2f);
            u32 c = colormap(Colormap::CoolWarm, 0.5f - 0.45f * std::exp(-r * r * 3.0f) + 0.3f * std::exp(-sq(r - 1.0f) * 8.0f));
            c = (c & 0x00FFFFFFu) | (static_cast<u32>(a * 230.0f) << 24);
            tex[static_cast<usize>(j) * tw + i] = c;
        }
}

static void test_ground(Framebuffer& fb) {
    std::printf("[suelo] rejilla AA, cinta, textura de Cp, profundidad\n");
    const Rect vp{0, 0, fb.w, fb.h};
    const Camera cam = make_cam(vp, {0, 0, 20}, -0.6f, 0.32f, 520.0f);
    clear(fb); background(fb, vp);
    Aabb ext; ext.lo = {-500, -260, 0}; ext.hi = {700, 260, 0};
    const int tw = 128, th = 64;
    std::vector<u32> tex;
    make_cp_texture(tex, tw, th);
    draw_ground(fb, cam, 0.0f, ext, 16.0f, 5.0f, 0xFF3A3F48u, 0xFF9AA3B6u, tex.data(), tw, th);
    usize written = 0, top_written = 0;
    for (int x = 0; x < fb.w; ++x) {
        written += fb.drow(fb.h - 1)[x] < k_inf;
        top_written += fb.drow(0)[x] < k_inf;
    }
    CHECK(written > static_cast<usize>(fb.w) / 2 && top_written == 0, "profundidad escrita en fila inferior %zu/%d, fila superior %zu", written, fb.w, top_written);
    float sx = 0, sy = 0, d = 0;
    cam.project({100, 50, 0}, sx, sy, d);
    const float zd = fb.drow(static_cast<int>(sy))[static_cast<int>(sx)];
    CHECK(std::fabs(zd - d) < 0.02f * d, "profundidad del suelo %.2f vs project() %.2f", zd, d);
    save(fb, "ground.png");
    // Cinta: desplazar 10 celdas (periodo común de damero ×2 y líneas mayores ×5) reproduce la imagen.
    std::vector<u32> a0(fb.color.begin(), fb.color.end());
    clear(fb); background(fb, vp);
    draw_ground(fb, cam, 0.0f, ext, 16.0f, 5.0f + 16.0f * 10.0f, 0xFF3A3F48u, 0xFF9AA3B6u, tex.data(), tw, th);
    usize diff = 0, maxd = 0;
    for (usize i = 0; i < a0.size(); ++i) {
        const u32 x = a0[i], y = fb.color.data()[i];
        if (x != y) {
            ++diff;
            for (int s = 0; s < 24; s += 8) maxd = max_(maxd, static_cast<usize>(std::abs(static_cast<int>((x >> s) & 255) - static_cast<int>((y >> s) & 255))));
        }
    }
    CHECK(maxd <= 3, "cinta: desplazar un periodo reproduce la imagen (%zu píxeles con diferencia ≤ %zu niveles)", diff, maxd);
    clear(fb); background(fb, vp);
    draw_ground(fb, cam, 0.0f, ext, 16.0f, 11.0f, 0xFF3A3F48u, 0xFF9AA3B6u, tex.data(), tw, th);
    diff = 0;
    for (usize i = 0; i < a0.size(); ++i) diff += a0[i] != fb.color.data()[i];
    CHECK(diff > a0.size() / 50, "cinta: desplazar media celda cambia la imagen (%zu píxeles)", diff);
}

// ============================================================================================
//  Robustez
// ============================================================================================
static void test_robustness(Framebuffer& fb) {
    std::printf("[robustez] entradas degeneradas, sin escrituras fuera del viewport\n");
    const u32 sentinel = 0xFF123456u;
    const Rect vp{203, 151, 1301, 899};      // viewport interior no alineado
    fb.clear_color(sentinel); fb.clear_depth();
    const Camera cams[3] = {make_cam(vp, {0, 0, 0}, 0.4f, 0.5f, 300.0f), make_cam(vp, {0, 0, 0}, 0.4f, 0.5f, 2.0f),
                            make_cam(vp, {0, 0, 0}, 1.1f, -0.2f, 150.0f, true)};
    std::vector<Vec3> pts = {{0, 0, 0}, {1e30f, 0, 0}, {-1e30f, 5, 5}, {k_nan, 0, 0}, {0, k_inf, 0}, {1e7f, -1e7f, 3},
                             {-50, -50, -50}, {50, 50, 50}, {0, 0, 1e30f}, {0, 0, -1e30f}, {3, 3, 3}, {k_nan, k_nan, k_nan},
                             {-k_inf, 0, 0}, {1e5f, 1e5f, 1e5f}, {-2, 1, 0.5f}, {2, -1, -0.5f}};
    std::vector<u32> cols = {0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu, 0xFFFFFFFFu};
    std::vector<u32> starts = {0, 3, 100, 0xFFFFFFF0u, 5, 14}, counts = {16, 5, 3, 20, 0xFFFFFFFFu, 2};
    Mesh bad;
    for (const Vec3& p : pts) { bad.pos.push_back(p); bad.nrm.push_back({0, 0, 1}); }
    WyRand rng(3);
    for (int i = 0; i < 3000; ++i) bad.tri.push_back(rng.below(20));   // índices fuera de rango incluidos
    bad.tri.push_back(1);                                              // no múltiplo de 3
    Mesh big;
    make_sphere(big, {0, 0, 0}, 1e6f, 16, 8, col_sphere);
    const Vec3 qn[4] = {{k_nan, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    const Vec3 qh[4] = {{-1e9f, -1e9f, 0}, {1e9f, -1e9f, 0}, {1e9f, 1e9f, 0}, {-1e9f, 1e9f, 0}};
    const Vec3 qb[4] = {{-50, -50, 0}, {50, -50, 0}, {50, 50, 0}, {-50, 50, 0}};
    const u32 tex[4] = {0xFFFFFFFFu, 0x80FF0000u, 0xFF00FF00u, 0x000000FFu};
    for (const Camera& cam : cams) {
        for (float w : {2.0f, 1e9f, k_nan, -5.0f, 0.0f, 0.3f, 300.0f}) {
            draw_lines(fb, cam, pts, cols, true, w, true);
            draw_lines(fb, cam, pts, cols, false, w, false);
            draw_polylines(fb, cam, pts, cols, starts, counts, w, true);
            draw_polylines(fb, cam, pts, {}, starts, counts, w, false);
            draw_box_wire(fb, cam, big.bounds, 0xFFFFFFFFu, w);
            draw_arrow(fb, cam, {0, 0, 0}, {k_nan, 0, 0}, 0xFFFF0000u, w);
            draw_arrow(fb, cam, {0, 0, 0}, {0, 0, 0}, 0xFFFF0000u, w);
            draw_arrow(fb, cam, {-1e8f, 0, 0}, {1e8f, 3, 0}, 0xFFFF0000u, w);
            draw_points(fb, cam, pts, cols, w, true);
            draw_points(fb, cam, pts, {}, w, false);
        }
        draw_lines(fb, cam, {}, {}, true, 1.0f, true);
        draw_points(fb, cam, {}, {}, 1.0f, true);
        draw_polylines(fb, cam, pts, cols, {}, {}, 1.0f, true);
        MeshStyle st;
        draw_mesh(fb, cam, bad, Light{}, st);
        st.two_sided = false; st.alpha = 0.5f; st.wireframe_overlay = true;
        draw_mesh(fb, cam, bad, Light{}, st);
        draw_mesh(fb, cam, big, Light{}, MeshStyle{});
        Mesh nonrm = big; nonrm.nrm.clear(); nonrm.color.clear();
        draw_mesh(fb, cam, nonrm, Light{}, MeshStyle{});
        draw_mesh(fb, cam, Mesh{}, Light{}, MeshStyle{});
        draw_textured_quad(fb, cam, qn, tex, 2, 2, 1.0f);
        draw_textured_quad(fb, cam, qh, tex, 2, 2, 1.0f);
        draw_textured_quad(fb, cam, qb, tex, 2, 2, k_nan);
        draw_textured_quad(fb, cam, qb, nullptr, 2, 2, 1.0f);
        draw_textured_quad(fb, cam, qb, tex, 0, 2, 1.0f);
        draw_textured_quad(fb, cam, qb, tex, 2, 2, 0.5f, false, true);
        Aabb e; e.lo = {-1e9f, -1e9f, 0}; e.hi = {1e9f, 1e9f, 0};
        Aabb en; en.lo = {k_nan, 0, 0}; en.hi = {1, 1, 0};
        draw_ground(fb, cam, 0.0f, e, 10.0f, 0.0f, 0xFF404040u, 0xFFFFFFFFu, tex, 2, 2);
        draw_ground(fb, cam, k_nan, e, 10.0f, 0.0f, 0xFF404040u, 0xFFFFFFFFu);
        draw_ground(fb, cam, 0.0f, en, 0.0f, k_nan, 0xFF404040u, 0xFFFFFFFFu, tex, 2, 2);
        draw_ground(fb, cam, 1e20f, Aabb{}, 1e-20f, 1e20f, 0xFF404040u, 0xFFFFFFFFu);
        screen_space_edges(fb, vp, 1.0f);
        screen_space_edges(fb, vp, k_nan);
        fxaa(fb, vp);
    }
    // Postprocesos con rectángulos fuera del framebuffer
    screen_space_edges(fb, {-50, -50, 100, 100}, 1.0f);
    fxaa(fb, {fb.w - 5, fb.h - 5, 100, 100});
    fxaa(fb, {5000, 5000, 10, 10});
    usize bad_px = 0, bad_z = 0;
    for (int y = 0; y < fb.h; ++y)
        for (int x = 0; x < fb.w; ++x) {
            if (vp.contains(x, y)) continue;
            if (x >= fb.w - 5 && y >= fb.h - 5) continue;   // tocado a propósito por el FXAA de la esquina
            if (x < 50 && y < 50) continue;                 // SSAO en (-50,-50,100,100)
            bad_px += fb.row(y)[x] != sentinel;
            bad_z += fb.drow(y)[x] != k_inf;
        }
    usize inside = 0;
    for (int y = vp.y; y < vp.y + vp.h; ++y)
        for (int x = vp.x; x < vp.x + vp.w; ++x) inside += fb.row(y)[x] != sentinel;
    CHECK(bad_px == 0 && bad_z == 0 && inside > 0, "sin cuelgues; píxeles fuera del viewport modificados: color=%zu depth=%zu (dentro: %zu)", bad_px, bad_z, inside);
    // Viewport que se sale del framebuffer: se recorta
    const Rect vbig{-300, -200, fb.w + 700, fb.h + 500};
    const Camera cb = make_cam(vbig, {0, 0, 0}, 0.4f, 0.5f, 50.0f);
    Mesh s;
    make_sphere(s, {0, 0, 0}, 40.0f, 64, 32, col_sphere);
    draw_mesh(fb, cb, s, Light{}, MeshStyle{});
    draw_ground(fb, cb, -40.0f, Aabb{}, 10.0f, 0.0f, 0xFF404040u, 0xFFFFFFFFu);
    draw_polylines(fb, cb, pts, cols, starts, counts, 3.0f, true);
    draw_points(fb, cb, pts, cols, 30.0f, true);
    screen_space_edges(fb, vbig, 0.8f);
    fxaa(fb, vbig);
    CHECK(true, "viewport mayor que el framebuffer: sin cuelgues");
}

// ============================================================================================
//  Rendimiento
// ============================================================================================
// Devuelve la mediana (ms); g_last_min guarda el mínimo. Los objetivos se juzgan con el mínimo
// (capacidad de la máquina) porque la mediana sufre con la carga de fondo de otros procesos.
static double g_last_min = 0;
template <class Prep, class Draw>
static double bench(int reps, Prep&& prep, Draw&& draw) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        prep();
        const double t0 = now_sec();
        draw();
        t.push_back(now_sec() - t0);
    }
    g_last_min = *std::min_element(t.begin(), t.end()) * 1e3;
    return median(t) * 1e3;
}
static double load_avg() {
    double l = 0;
    if (FILE* f = std::fopen("/proc/loadavg", "r")) { if (std::fscanf(f, "%lf", &l) != 1) l = 0; std::fclose(f); }
    return l;
}
// Objetivo de rendimiento: con carga de fondo alta (loadavg > nº de CPUs / 2) un fallo se informa
// como AVISO y no cuenta (la prueba sería no determinista); si no, es un FAIL normal.
static bool g_heavy_load = false;
#define PERF_CHECK(ok, ...) do { if ((ok) || !g_heavy_load) { CHECK((ok), __VA_ARGS__); } \
                                 else { std::printf("  AVISO "); std::printf(__VA_ARGS__); std::printf("  [carga alta: no cuenta]\n"); } } while (0)

// Malla "tipo coche": esfera + 2 toros densos que llenan buena parte del viewport 3D.
static void make_bench_mesh(Mesh& m, int target_tris) {
    m.clear();
    const int k = static_cast<int>(std::sqrt(static_cast<double>(target_tris) / 10.0));
    make_sphere(m, {0, 0, 0}, 70.0f, 2 * k, k, col_sphere);
    make_torus(m, {0, 0, 0}, 150.0f, 40.0f, 2 * k, k, col_torus);
    make_torus(m, {0, 0, 40.0f}, 100.0f, 18.0f, 2 * k, k / 2, col_torus);
}

static void run_benchmarks(Framebuffer& fb) {
    const double la = load_avg();
    const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    g_heavy_load = la > 0.5 * ncpu;
    std::printf("[rendimiento] framebuffer %dx%d, viewport 3D 1500x1150, %d hilos; mediana (mín) de 31; carga media %.1f%s\n",
                fb.w, fb.h, pool().size(), la, g_heavy_load ? " (ALTA: los objetivos no alcanzados sólo avisan)" : "");
    const Rect vp{0, 0, 1500, 1150};
    const Camera cam = make_cam(vp, {0, 0, 10}, -0.5f, 0.5f, 520.0f);
    for (int target : {100000, 300000, 600000}) {
        Mesh m;
        make_bench_mesh(m, target);
        std::vector<double> tv, tb, tr;
        const double ms = bench(31, [&] { fb.clear_color(0xFF181A20u); fb.clear_depth(); }, [&] {
            draw_mesh(fb, cam, m, Light{}, MeshStyle{});
            const RasterStats s = last_mesh_stats();
            tv.push_back(s.t_vertex); tb.push_back(s.t_bin); tr.push_back(s.t_raster);
        });
        const RasterStats s = last_mesh_stats();
        const double mn = g_last_min;
        const bool ok = target != 300000 || mn < 8.0;
        PERF_CHECK(ok, "draw_mesh %zu tris: %.2f ms (mín %.2f)  [vértices %.2f | binning %.2f | raster %.2f]  (refs %llu, tiles %d)%s", m.tri_count(), ms, mn,
              median(tv) * 1e3, median(tb) * 1e3, median(tr) * 1e3, (unsigned long long)s.bin_entries, s.tiles_busy,
              target == 300000 ? "  objetivo < 8 ms" : "");
        if (target == 300000) {
            MeshStyle one; one.two_sided = false;
            const double ms1 = bench(31, [&] { fb.clear_color(0xFF181A20u); fb.clear_depth(); }, [&] { draw_mesh(fb, cam, m, Light{}, one); });
            std::printf("        (una cara: %.2f ms, mín %.2f)\n", ms1, g_last_min);
            save(fb, "bench_mesh.png");
        }
    }
    // Líneas: 500 polilíneas × 101 puntos = 50 000 segmentos, ancho 1.5, test de profundidad
    Mesh sm;
    make_bench_mesh(sm, 100000);
    std::vector<Vec3> pts;
    std::vector<u32> cols, starts, counts;
    for (int k = 0; k < 500; ++k) {
        starts.push_back(static_cast<u32>(pts.size()));
        const float y0 = -250.0f + static_cast<float>(k), z0 = -60.0f + 120.0f * ((k * 37) % 100) / 100.0f;
        for (int i = 0; i <= 100; ++i) {
            const float x = -400.0f + 8.0f * i;
            pts.push_back({x, y0 + 20.0f * std::sin(x * 0.02f + k), z0 + 30.0f * std::cos(x * 0.015f + k * 0.1f)});
            cols.push_back(colormap(Colormap::Turbo, i / 100.0f));
        }
        counts.push_back(101);
    }
    auto prep_scene = [&] { fb.clear_color(0xFF181A20u); fb.clear_depth(); draw_mesh(fb, cam, sm, Light{}, MeshStyle{}); };
    double ms = bench(31, prep_scene, [&] { draw_polylines(fb, cam, pts, cols, starts, counts, 1.5f, true); });
    PERF_CHECK(g_last_min < 3.0, "draw_polylines 50 000 segmentos (ancho 1.5, depth): %.2f ms (mín %.2f)  objetivo < 3 ms", ms, g_last_min);
    save(fb, "bench_lines.png");
    // Puntos: 200 000 splats de 3 px
    std::vector<Vec3> pp;
    std::vector<u32> pc;
    WyRand rng(11);
    for (int i = 0; i < 200000; ++i) {
        const float t = rng.uniform(), a = rng.uniform(0, 6.2831f), r = 60.0f + 120.0f * rng.uniform();
        pp.push_back({-350.0f + 700.0f * t, r * std::cos(a), r * std::sin(a) * 0.6f});
        pc.push_back(colormap(Colormap::Inferno, 0.2f + 0.8f * t) & 0x50FFFFFFu);
    }
    ms = bench(31, prep_scene, [&] { draw_points(fb, cam, pp, pc, 3.0f, true); });
    PERF_CHECK(g_last_min < 3.0, "draw_points 200 000 splats 3 px aditivos: %.2f ms (mín %.2f)  objetivo < 3 ms", ms, g_last_min);
    save(fb, "bench_points.png");
    ms = bench(31, prep_scene, [&] { draw_points(fb, cam, pp, pc, 3.0f, false); });
    std::printf("        (mezcla alfa: %.2f ms, mín %.2f)\n", ms, g_last_min);
    // Suelo con textura de Cp
    std::vector<u32> tex;
    make_cp_texture(tex, 128, 64);
    Aabb ext; ext.lo = {-600, -300, 0}; ext.hi = {800, 300, 0};
    const Camera gcam = make_cam(vp, {0, 0, 10}, -0.5f, 0.35f, 520.0f);
    ms = bench(31, [&] { fb.clear_color(0xFF181A20u); fb.clear_depth(); }, [&] {
        draw_ground(fb, gcam, -60.0f, ext, 16.0f, 3.0f, 0xFF3A3F48u, 0xFF9AA3B6u, tex.data(), 128, 64);
    });
    PERF_CHECK(g_last_min < 2.0, "draw_ground con textura (viewport 1500x1150): %.2f ms (mín %.2f)  objetivo < 2 ms", ms, g_last_min);
    ms = bench(31, [&] { fb.clear_color(0xFF181A20u); fb.clear_depth(); }, [&] {
        draw_ground(fb, gcam, -60.0f, ext, 16.0f, 3.0f, 0xFF3A3F48u, 0xFF9AA3B6u);
    });
    std::printf("        (sin textura: %.2f ms, mín %.2f)\n", ms, g_last_min);
    // Plano de corte grande texturizado
    std::vector<u32> st(512 * 256);
    for (int i = 0; i < 512 * 256; ++i) st[static_cast<usize>(i)] = colormap(Colormap::Viridis, static_cast<float>((i * 7) % 256) / 255.0f) & 0xD0FFFFFFu;
    const Vec3 sl[4] = {{-500, 0, -150}, {500, 0, -150}, {500, 0, 150}, {-500, 0, 150}};
    ms = bench(31, prep_scene, [&] { draw_textured_quad(fb, cam, sl, st.data(), 512, 256, 0.8f, true, false); });
    std::printf("  INFO  draw_textured_quad (plano de corte 512x256, bilineal): %.2f ms (mín %.2f)\n", ms, g_last_min);
    // Postprocesos sobre una escena completa
    prep_scene();
    draw_ground(fb, cam, -60.0f, ext, 16.0f, 3.0f, 0xFF3A3F48u, 0xFF9AA3B6u);
    std::vector<u32> saved(fb.color.begin(), fb.color.end());
    auto restore = [&] { std::memcpy(fb.color.data(), saved.data(), saved.size() * 4); };
    ms = bench(31, restore, [&] { screen_space_edges(fb, vp, 0.6f); });
    std::printf("  INFO  screen_space_edges (1500x1150, 12 muestras): %.2f ms (mín %.2f)\n", ms, g_last_min);
    ms = bench(31, restore, [&] { fxaa(fb, vp); });
    std::printf("  INFO  fxaa (1500x1150): %.2f ms (mín %.2f)\n", ms, g_last_min);
    ms = bench(31, restore, [&] { fb.gradient(vp, 0xFF30353Fu, 0xFF121418u); });
    std::printf("  INFO  referencia Framebuffer::gradient (1500x1150): %.2f ms (mín %.2f)\n", ms, g_last_min);
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    pool().start();
    std::printf("hilos del pool: %d\n", pool().size());
    Framebuffer fb;
    fb.resize(1920, 1200);
    test_coverage(fb);
    test_watertight(fb);
    test_depth_order(fb);
    test_near_clip(fb);
    test_mesh_visual(fb);
    test_lines(fb);
    test_points(fb);
    test_textured_quad(fb);
    test_ground(fb);
    test_translucent_order(fb);
    test_degenerate_normals(fb);
    test_post_functional(fb);
    test_determinism(fb);
    test_robustness(fb);
    if (!quick) run_benchmarks(fb);
    std::printf("\nRESUMEN: %d PASS, %d FAIL\n", g_pass, g_fail);
    pool().stop();
    return g_fail ? 1 : 0;
}
