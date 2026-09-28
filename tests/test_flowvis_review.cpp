// ============================================================================
//  tests/test_flowvis_review.cpp — pruebas de la revisión adversarial del módulo
//  flowvis: casos límite que test_flowvis.cpp no cubría.
//
//   R1  valores no finitos en el campo (±inf, NaN, rangos que desbordan): nada
//       debe fallar ni escribir fuera de los búferes (regresión de un fallo real:
//       robust_range indexaba el histograma con (int)NaN → escritura fuera de pila).
//   R2  dominios con nx NO múltiplo de 8 y tamaños impares: núcleos AVX2 = escalar,
//       FlowSampler (ruta de stores no alineados, ρ-1 de los sólidos), cortes.
//   R3  detail::mask_near_wall (SWAR) = fuerza bruta con flags aleatorios.
//   R4  volumen en un viewport desplazado: no toca píxeles fuera de cam.vp y la media
//       resolución es invariante a la traslación del viewport.
//   R5  SliceView: cambiar params.axis sin update() no desalinea textura/esquinas/sonda.
//   R6  color_mesh: el color de cada vértice = map_color(escala, Cp muestreado).
//   R7  líneas de corriente en ambos sentidos con semillas inválidas intercaladas.
//   R8  partículas: cambio de capacidad, pausa (dt = 0) y conservación.
//   R9  robust_range: casos directos (todo NaN, constante, percentiles conocidos).
//   R10 volumen: parámetros de update() (color_by, full) frente a los de render() (threshold).
//   R11 dominio largo (nx-1 ≥ 2048): la sujeción trilineal no lee fuera del búfer.
//
//  Compilar (no necesita el rasterizador):
//    g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_flowvis_review.cpp  (sigue)
//        src/render/flowvis.cpp src/render/flowvis_lines.cpp src/render/flowvis_volume.cpp (sigue)
//        src/core/threadpool.cpp -o build/flowvis/test_flowvis_review
//  Con ASan/UBSan: -O1 -g -fsanitize=address,undefined,float-cast-overflow → build/flowvis-asan/.
// ============================================================================
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "render/flowvis.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
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

static constexpr float k_inf = std::numeric_limits<float>::infinity();
static constexpr float k_nan = std::numeric_limits<float>::quiet_NaN();

struct Field {
    int nx, ny, nz;
    float u_inf = 0.08f;
    std::vector<float> rho, ux, uy, uz;
    std::vector<u8> flags, sid;
    Field(int x, int y, int z) : nx(x), ny(y), nz(z) {
        const usize n = static_cast<usize>(x) * y * z;
        rho.assign(n, 1.0f); ux.assign(n, u_inf); uy.assign(n, 0.0f); uz.assign(n, 0.0f);
        flags.assign(n, 0); sid.assign(n, 0);
    }
    usize idx(int x, int y, int z) const { return static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * z); }
    lbm::FieldView view() const {
        lbm::FieldView v;
        v.nx = nx; v.ny = ny; v.nz = nz;
        v.rho = rho.data(); v.ux = ux.data(); v.uy = uy.data(); v.uz = uz.data();
        v.flags = flags.data(); v.solid_id = sid.data(); v.u_inf = u_inf;
        return v;
    }
    // Campo pseudoaleatorio suave + sólidos dispersos (con velocidad de pared 0 y ρ = 1, como el solver).
    void random_fill(u64 seed, float solid_frac) {
        WyRand r(seed);
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const usize n = idx(x, y, z);
                    const bool s = r.uniform() < solid_frac;
                    flags[n] = s ? lbm::kSolid : 0;
                    sid[n] = s ? 1 : 0;
                    ux[n] = s ? 0.0f : u_inf * (1.0f + 0.4f * std::sin(0.37f * x + 0.21f * y) + 0.2f * r.uniform(-1, 1));
                    uy[n] = s ? 0.0f : u_inf * (0.3f * std::cos(0.29f * z + 0.11f * x) + 0.1f * r.uniform(-1, 1));
                    uz[n] = s ? 0.0f : u_inf * (0.3f * std::sin(0.23f * y - 0.17f * z) + 0.1f * r.uniform(-1, 1));
                    rho[n] = s ? 1.0f : 1.0f + 0.004f * r.uniform(-1, 1);
                }
    }
};

static bool finite(float v) { return std::fabs(v) <= std::numeric_limits<float>::max(); }

// ============================================================================
//  R1. Valores no finitos
// ============================================================================
static void test_nonfinite() {
    std::printf("[R1] valores no finitos en el campo\n");
    struct Case { const char* name; float a, b; };
    const Case cases[] = {{"+inf", k_inf, 1.0f}, {"-inf", -k_inf, 1.0f}, {"desbordamiento ±3e38", 3e38f, -3e38f}, {"NaN", k_nan, 1.0f}};
    for (const Case& c : cases) {
        Field F(40, 24, 20);
        F.random_fill(11, 0.0f);
        const usize n = F.idx(9, 12, 10);
        F.rho[n] = c.a; F.rho[n + 1] = c.b;
        F.ux[n + 2] = c.a;                      // también en la velocidad (|u|, Q, muestreador)
        const auto f = F.view();
        bool ok = true;
        for (int ax = 0; ax < 3; ++ax)
            for (int q = 0; q < static_cast<int>(Quantity::Count); ++q)
                for (int ar = 0; ar < 2; ++ar) {
                    SliceView s;
                    s.params.axis = static_cast<Axis>(ax);
                    s.params.pos = ax == 0 ? 9.0f : (ax == 1 ? 12.0f : 10.0f);
                    s.set_quantity(static_cast<Quantity>(q));
                    s.params.auto_range = ar != 0;
                    s.params.fade_below = 0.1f;
                    s.update(f);
                    const ColorScale e = s.effective_scale();
                    ok &= finite(s.value_min()) && finite(s.value_max()) && finite(e.lo) && finite(e.hi) && e.hi > e.lo;
                }
        CHECK(ok, "%s: rango de corte no finito", c.name);
        for (int ax = 0; ax < 3; ++ax) {   // LIC con el valor no finito en el plano
            SliceView s;
            s.params.axis = static_cast<Axis>(ax);
            s.params.pos = ax == 0 ? 9.0f : (ax == 1 ? 12.0f : 10.0f);
            s.set_quantity(Quantity::Speed);
            s.params.auto_range = true;
            s.params.lic = 2;
            s.update(f);
        }
        // Volumen, muestreador, líneas y partículas: sólo no deben fallar (ASan/UBSan lo vigila).
        VortexVolume V;
        V.params.downsample = 1;
        V.update(f);
        CHECK(finite(V.stats().max_value) || V.stats().max_value == k_inf, "%s: máximo del volumen", c.name);
        render::Framebuffer fb;
        fb.resize(160, 120);
        fb.clear_color(0xFF000000u);
        render::Camera cam;
        cam.target = {20, 12, 10}; cam.distance = 80.0f;
        cam.update({0, 0, 160, 120});
        V.render(fb, cam);
        FlowSampler S;
        S.update(f);
        Streamlines L;
        const Rake rk = Rake::grid({1, 2, 2}, {0, 20, 0}, {0, 0, 16}, 6, 6);
        L.set_rakes(std::span<const Rake>(&rk, 1));
        L.compute(S);
        Particles P;
        P.params.capacity = 2000;
        P.set_emitters(std::span<const Rake>(&rk, 1));
        for (int i = 0; i < 20; ++i) P.step(S, 10.0f);
        const auto& st = P.stats();
        CHECK(st.emitted == st.alive + st.died_outside + st.died_solid + st.died_age + st.overwritten, "%s: conservación", c.name);
        bool pts_ok = true;
        for (const Vec3& p : P.points()) pts_ok &= finite(p.x) && finite(p.y) && finite(p.z);
        CHECK(pts_ok, "%s: partícula con posición no finita en la salida", c.name);
        WakeStats w = wake_survey(f, 9.0f);
        (void)w;
    }
    // Posición del corte NaN / fuera de rango: se sujeta (sin UB).
    Field F(24, 16, 16);
    SliceView s;
    s.params.pos = k_nan;
    s.update(F.view());
    CHECK(s.plane_pos() >= 0.0f && s.plane_pos() <= 15.0f, "pos NaN → %f", s.plane_pos());
    s.params.pos = 1e9f;
    s.update(F.view());
    CHECK(s.plane_pos() == 15.0f, "pos enorme → %f", s.plane_pos());
}

// ============================================================================
//  R2. nx no múltiplo de 8 y tamaños impares
// ============================================================================
static void test_odd_sizes() {
    std::printf("[R2] dominios con nx no múltiplo de 8\n");
    const int sizes[][3] = {{37, 13, 11}, {9, 5, 7}, {2, 2, 2}, {17, 3, 4}};
    for (const auto& sz : sizes) {
        Field F(sz[0], sz[1], sz[2]);
        F.random_fill(5 + sz[0], 0.12f);
        const auto f = F.view();
        // Núcleos por filas = escalar
        std::vector<float> row(static_cast<usize>(F.nx));
        int nan_mis = 0;
        float err = 0;
        for (int q = 0; q < static_cast<int>(Quantity::Count); ++q)
            for (int z = 0; z < F.nz; ++z)
                for (int y = 0; y < F.ny; ++y) {
                    detail::quantity_row(f, static_cast<Quantity>(q), y, z, row.data());
                    for (int x = 0; x < F.nx; ++x) {
                        const float s = cell_quantity(f, static_cast<Quantity>(q), x, y, z);
                        if ((s != s) != (row[x] != row[x])) { ++nan_mis; continue; }
                        if (s == s) err = max_(err, std::fabs(s - row[x]) / (1.0f + std::fabs(s)));
                    }
                }
        CHECK(nan_mis == 0 && err < 1e-4f, "%dx%dx%d: filas AVX2 vs escalar (NaN %d, err %g)", sz[0], sz[1], sz[2], nan_mis, err);
        // FlowSampler: valores por celda (u de la celda; ρ-1 del sólido = media de vecinas fluidas)
        FlowSampler S;
        S.update(f);
        float eu = 0, er = 0;
        for (int z = 0; z < F.nz; ++z)
            for (int y = 0; y < F.ny; ++y)
                for (int x = 0; x < F.nx; ++x) {
                    const usize n = F.idx(x, y, z);
                    alignas(16) float r[4];
                    _mm_store_ps(r, S.sample(_mm_setr_ps(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 0.0f)));
                    // en el borde superior la sujeción deja t = 1-1e-4: tolerancia acorde
                    eu = max_(eu, std::fabs(r[0] - F.ux[n]) + std::fabs(r[1] - F.uy[n]) + std::fabs(r[2] - F.uz[n]));
                    float expect = F.rho[n] - 1.0f;
                    if (F.flags[n] & lbm::kSolid) {
                        float acc = 0; int cnt = 0;
                        const int nb[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
                        for (const auto& d : nb) {
                            const int a = x + d[0], b = y + d[1], c = z + d[2];
                            if (a < 0 || b < 0 || c < 0 || a >= F.nx || b >= F.ny || c >= F.nz) continue;
                            const usize m = F.idx(a, b, c);
                            if (!(F.flags[m] & lbm::kSolid)) { acc += F.rho[m] - 1.0f; ++cnt; }
                        }
                        expect = cnt ? acc / static_cast<float>(cnt) : 0.0f;
                    }
                    const bool edge = x == F.nx - 1 || y == F.ny - 1 || z == F.nz - 1;
                    if (!edge) er = max_(er, std::fabs(r[3] - expect));
                }
        CHECK(eu < 2e-3f * F.u_inf * 3.0f + 1e-4f * 0.2f, "%dx%dx%d: FlowSampler u err %g", sz[0], sz[1], sz[2], eu);
        CHECK(er < 1e-5f, "%dx%dx%d: FlowSampler ρ-1 (sólidos = media de vecinas fluidas) err %g", sz[0], sz[1], sz[2], er);
        // Cortes en los 3 ejes: valor por celda = cell_quantity en capas enteras
        int bad = 0;
        for (int ax = 0; ax < 3; ++ax) {
            SliceView s;
            s.params.axis = static_cast<Axis>(ax);
            const int nax = ax == 0 ? F.nx : (ax == 1 ? F.ny : F.nz);
            s.params.pos = static_cast<float>(nax / 2);
            s.set_quantity(Quantity::Vorticity);
            s.update(f);
            for (int v = 0; v < s.val_h(); ++v)
                for (int u = 0; u < s.val_w(); ++u) {
                    int x, y, z;
                    if (ax == 0) { x = nax / 2; y = u; z = v; } else if (ax == 1) { x = u; y = nax / 2; z = v; } else { x = u; y = v; z = nax / 2; }
                    const float a = s.values()[static_cast<usize>(v) * s.val_w() + u], b = cell_quantity(f, Quantity::Vorticity, x, y, z);
                    bad += (a != a) != (b != b) || (a == a && std::fabs(a - b) > 1e-4f * (1.0f + std::fabs(b)));
                }
        }
        CHECK(bad == 0, "%dx%dx%d: %d valores de corte distintos", sz[0], sz[1], sz[2], bad);
        // Volumen (ds 1 y 2) + render en media/completa: sin fallos
        for (int ds = 1; ds <= 2; ++ds) {
            VortexVolume V;
            V.params.downsample = ds;
            V.params.threshold = 1e-6f;
            V.update(f);
            render::Framebuffer fb;
            fb.resize(97, 61);
            render::Camera cam;
            cam.target = {F.nx * 0.5f, F.ny * 0.5f, F.nz * 0.5f}; cam.distance = 3.0f * static_cast<float>(F.nx + F.ny + F.nz);
            cam.update({0, 0, 97, 61});
            for (int h = 0; h < 2; ++h) { V.params.half_res = h == 0; fb.clear_color(0xFF000000u); fb.clear_depth(); V.render(fb, cam); }
        }
    }
}

// ============================================================================
//  R3. mask_near_wall (SWAR) = fuerza bruta
// ============================================================================
static void test_mask_near_wall() {
    std::printf("[R3] máscara junto a paredes (SWAR vs fuerza bruta)\n");
    for (int nx : {8, 9, 10, 11, 17, 37, 64}) {
        Field F(nx, 9, 7);
        F.random_fill(100 + nx, 0.05f);
        const auto f = F.view();
        std::vector<float> row(static_cast<usize>(nx));
        int bad = 0;
        for (int z = 0; z < F.nz; ++z)
            for (int y = 0; y < F.ny; ++y) {
                std::fill(row.begin(), row.end(), 1.0f);
                detail::mask_near_wall(f, y, z, row.data());
                for (int x = 0; x < nx; ++x) {
                    bool near = false;
                    const int nb[7][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
                    for (const auto& d : nb) {
                        const int a = x + d[0], b = y + d[1], c = z + d[2];
                        if (a < 0 || b < 0 || c < 0 || a >= nx || b >= F.ny || c >= F.nz) continue;
                        near |= (F.flags[F.idx(a, b, c)] & lbm::kSolid) != 0;
                    }
                    bad += near != (row[x] != row[x]);
                }
            }
        CHECK(bad == 0, "nx=%d: %d celdas mal enmascaradas", nx, bad);
    }
}

// ============================================================================
//  R4. Volumen en un viewport desplazado
// ============================================================================
static void test_volume_viewport() {
    std::printf("[R4] volumen en viewport desplazado\n");
    const float U = 0.08f, G = U * 2.0f * k_pi * 6.0f;
    Field F(96, 48, 40);
    for (int z = 0; z < F.nz; ++z)
        for (int y = 0; y < F.ny; ++y)
            for (int x = 0; x < F.nx; ++x) {
                const float dy = static_cast<float>(y) - 24.0f, dz = static_cast<float>(z) - 20.0f, r2 = dy * dy + dz * dz;
                const float k = r2 < 1e-12f ? 0.0f : G / (2.0f * k_pi * r2) * (1.0f - std::exp(-r2 / 9.0f));
                const usize n = F.idx(x, y, z);
                F.ux[n] = U; F.uy[n] = -k * dz; F.uz[n] = k * dy;
            }
    const auto f = F.view();
    VortexVolume V;
    V.update(f);
    const int W = 301, H = 203, OX = 37, OY = 21;
    render::Camera cam;
    cam.target = {48, 24, 20}; cam.yaw = -0.5f; cam.pitch = 0.3f; cam.distance = 130.0f;
    constexpr u32 k_sent = 0xFF123456u;
    for (int h = 0; h < 2; ++h) {
        V.params.half_res = h == 0;
        render::Framebuffer big, small;
        big.resize(W + 90, H + 70);
        small.resize(W, H);
        big.clear_color(k_sent); big.clear_depth();
        small.clear_color(k_sent); small.clear_depth();
        cam.update({OX, OY, W, H});
        V.render(big, cam);
        cam.update({0, 0, W, H});
        V.render(small, cam);
        int outside = 0, diff = 0, inside_changed = 0;
        for (int y = 0; y < big.h; ++y)
            for (int x = 0; x < big.w; ++x) {
                const bool in = x >= OX && y >= OY && x < OX + W && y < OY + H;
                const u32 c = big.row(y)[x];
                if (!in) { outside += c != k_sent; continue; }
                inside_changed += c != k_sent;
                if (h == 0) diff += c != small.row(y - OY)[x - OX];
            }
        CHECK(outside == 0, "%s: %d píxeles fuera del viewport modificados", h ? "completa" : "media", outside);
        CHECK(inside_changed > W * H / 50, "%s: sólo %d píxeles con vórtice", h ? "completa" : "media", inside_changed);
        if (h == 0) CHECK(diff == 0, "media resolución: %d píxeles distintos al desplazar el viewport", diff);
        // viewport que se sale del framebuffer por la izquierda/arriba (primer plano: el vórtice
        // llena la vista, así que se trazan filas/columnas negativas): se recorta sin escribir fuera
        render::Camera close = cam;
        close.distance = 14.0f;
        close.update({-40, -30, W, H});
        render::Framebuffer fb2;
        fb2.resize(W - 60, H - 50);
        fb2.clear_color(k_sent); fb2.clear_depth();
        V.render(fb2, close);
        int touched = 0;
        for (int y = 0; y < fb2.h; ++y) for (int x = 0; x < fb2.w; ++x) touched += fb2.row(y)[x] != k_sent;
        CHECK(touched > fb2.w * fb2.h / 4, "%s: primer plano recortado: sólo %d píxeles", h ? "completa" : "media", touched);
    }
}

// ============================================================================
//  R10. Volumen: parámetros horneados en update() (color_by, full) no cambian el render
//       hasta el siguiente update(); los de render (threshold) sí.
// ============================================================================
static void test_volume_baked_params() {
    std::printf("[R10] volumen: parámetros de update() frente a los de render()\n");
    const float U = 0.08f, G = U * 2.0f * k_pi * 6.0f;
    Field F(64, 40, 40);
    for (int z = 0; z < F.nz; ++z)
        for (int y = 0; y < F.ny; ++y)
            for (int x = 0; x < F.nx; ++x) {
                const float dy = static_cast<float>(y) - 20.0f, dz = static_cast<float>(z) - 20.0f, r2 = dy * dy + dz * dz;
                const float k = r2 < 1e-12f ? 0.0f : G / (2.0f * k_pi * r2) * (1.0f - std::exp(-r2 / 9.0f));
                const usize n = F.idx(x, y, z);
                F.ux[n] = U * (1.0f - 0.5f * std::exp(-r2 / 9.0f)); F.uy[n] = -k * dz; F.uz[n] = k * dy;
            }
    VortexVolume V;
    V.update(F.view());
    render::Camera cam;
    cam.target = {32, 20, 20}; cam.yaw = -0.6f; cam.pitch = 0.3f; cam.distance = 110.0f;
    cam.update({0, 0, 240, 160});
    auto shot = [&] {
        render::Framebuffer fb;
        fb.resize(240, 160);
        fb.clear_color(0xFF000000u); fb.clear_depth();
        V.render(fb, cam);
        return std::vector<u32>(fb.color.data(), fb.color.data() + fb.color.size());
    };
    const auto a = shot();
    V.params.color_by = VolumeColor::Magnitude;
    V.params.full *= 3.0f;
    const auto b = shot();
    CHECK(a == b, "color_by/full sin update() alteraron el render");
    V.update(F.view());
    const auto c = shot();
    CHECK(a != c, "tras update() el color debería cambiar");
    V.params.threshold *= 4.0f;   // de render: efecto inmediato
    const auto d = shot();
    CHECK(c != d, "threshold es de render: debe cambiar sin update()");
}

// ============================================================================
//  R11. Dominio largo (n-1 ≥ 2048): la sujeción trilineal n-1-1e-4 redondeaba a n-1 y la
//       celda x0+1 se leía fuera del búfer (ASan: heap-buffer-overflow). Ahora x0 ≤ n-2.
// ============================================================================
static void test_long_domain() {
    std::printf("[R11] dominio largo (nx = 2056): muestreo en la última celda\n");
    Field F(2056, 2, 3);
    for (int x = 0; x < F.nx; ++x)
        for (int z = 0; z < F.nz; ++z)
            for (int y = 0; y < F.ny; ++y) F.ux[F.idx(x, y, z)] = F.u_inf * (1.0f + 1e-4f * static_cast<float>(x));
    const auto f = F.view();
    FlowSampler S;
    S.update(f);
    const Vec3 u = S.velocity({2055.0f, 1.0f, 2.0f});
    CHECK(std::fabs(u.x - F.ux[F.idx(2055, 1, 2)]) < 1e-3f * F.u_inf, "FlowSampler en la última celda: %f vs %f", u.x, F.ux[F.idx(2055, 1, 2)]);
    const float sp = sample_quantity(f, Quantity::Speed, {2055.0f, 1.0f, 2.0f});
    CHECK(std::fabs(sp - F.ux[F.idx(2055, 1, 2)] / F.u_inf) < 1e-4f, "sample_quantity en la última celda: %f", sp);
    const Probe pr = probe(f, {2055.0f, 1.0f, 2.0f});
    CHECK(pr.valid && std::fabs(pr.speed - sp) < 1e-4f, "sonda en la última celda");
    Mesh m;
    m.pos.push_back({2055.0f, 1.0f, 2.0f});
    m.nrm.push_back({1.0f, 0.0f, 0.0f});
    color_mesh(m, f, SurfaceParams{});
    CHECK(m.color.size() == 1, "color_mesh en la esquina");
}

// ============================================================================
//  R5. SliceView: cambiar el eje sin update()
// ============================================================================
static void test_slice_axis_consistency() {
    std::printf("[R5] corte: cambio de eje sin update()\n");
    Field F(40, 24, 16);
    for (int z = 0; z < F.nz; ++z)
        for (int y = 0; y < F.ny; ++y)
            for (int x = 0; x < F.nx; ++x) F.ux[F.idx(x, y, z)] = F.u_inf * (0.1f + 0.01f * x + 0.02f * y + 0.03f * z);
    SliceView s;
    s.set_quantity(Quantity::Ux);
    s.params.axis = Axis::Y; s.params.pos = 5.0f;
    s.update(F.view());
    Vec3 c0[4];
    s.corners(c0);
    const float v0 = s.value_at_world({12.0f, 5.0f, 7.0f});
    s.params.axis = Axis::X;                 // la UI cambia el eje; aún no hay update()
    Vec3 c1[4];
    s.corners(c1);
    bool same = true;
    for (int i = 0; i < 4; ++i) same &= length(c0[i] - c1[i]) == 0.0f;
    CHECK(same, "las esquinas deben seguir siendo las del corte calculado (eje Y) hasta el próximo update()");
    CHECK(s.value_at_world({12.0f, 5.0f, 7.0f}) == v0, "value_at_world debe usar el eje del corte calculado");
    CHECK(s.translucent_plane().axis == Axis::Y, "translucent_plane debe usar el eje calculado");
    s.update(F.view());
    s.corners(c1);
    CHECK(c1[0].x == 5.0f && c1[1].y == 23.5f, "tras update() el corte pasa a X (esquinas %f %f)", c1[0].x, c1[1].y);
}

// ============================================================================
//  R6. color_mesh = map_color(Cp muestreado)
// ============================================================================
static void test_color_mesh_exact() {
    std::printf("[R6] color_mesh: colores exactos\n");
    Field F(48, 32, 32);
    F.random_fill(77, 0.0f);
    const Vec3 c{24, 16, 16};
    for (int z = 0; z < F.nz; ++z)
        for (int y = 0; y < F.ny; ++y)
            for (int x = 0; x < F.nx; ++x)
                if (length(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)) - c) < 7.0f) {
                    const usize n = F.idx(x, y, z);
                    F.flags[n] = lbm::kSolid; F.ux[n] = F.uy[n] = F.uz[n] = 0.0f; F.rho[n] = 1.0f;
                }
    const auto f = F.view();
    Mesh m;
    WyRand r(3);
    for (int i = 0; i < 4000; ++i) {
        Vec3 d{r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)};
        d = normalize(d);
        m.pos.push_back(c + d * 7.0f);
        m.nrm.push_back(d);
        m.group.push_back(static_cast<u8>(1 + (i % 3)));
    }
    for (int mode = 0; mode < 2; ++mode) {
        SurfaceParams sp;
        sp.mode = mode ? SurfaceMode::Speed : SurfaceMode::Cp;
        if (mode) sp.scale = default_scale(Quantity::Speed);
        color_mesh(m, f, sp);
        int bad = 0;
        for (usize i = 0; i < m.pos.size(); ++i) {
            const float v = sample_quantity(f, mode ? Quantity::Speed : Quantity::Cp, m.pos[i] + m.nrm[i] * sp.offset);
            bad += v == v && m.color[i] != map_color(sp.scale, v);
        }
        CHECK(bad == 0, "%s: %d colores de vértice distintos de map_color(muestra)", mode ? "Speed" : "Cp", bad);
    }
    // Componente con paleta propia
    std::vector<u32> pal(256, 0xFF000000u);
    pal[1] = 0xFFFF0000u; pal[2] = 0xFF00FF00u; pal[3] = 0xFF0000FFu;
    SurfaceParams sp;
    sp.mode = SurfaceMode::Component;
    sp.group_colors = pal;
    color_mesh(m, f, sp);
    int bad = 0;
    for (usize i = 0; i < m.pos.size(); ++i) bad += m.color[i] != pal[m.group[i]];
    CHECK(bad == 0, "componente: %d colores incorrectos", bad);
}

// ============================================================================
//  R7. Líneas en ambos sentidos con semillas inválidas intercaladas
// ============================================================================
static void test_streamlines_mixed() {
    std::printf("[R7] líneas de corriente: semillas mixtas en ambos sentidos\n");
    Field F(64, 24, 24);
    for (int z = 10; z < 14; ++z)
        for (int y = 10; y < 14; ++y)
            for (int x = 30; x < 34; ++x) { const usize n = F.idx(x, y, z); F.flags[n] = lbm::kSolid; F.ux[n] = 0.0f; }
    FlowSampler S;
    S.update(F.view());
    const Vec3 seeds[6] = {{20, 5, 5}, {-3, 5, 5}, {31, 11, 11}, {40, 18, 6}, {10, 30, 5}, {50, 3, 20}};
    Streamlines L;
    L.params.both_directions = true;
    L.params.max_steps = 150;
    L.set_seeds(seeds);
    L.compute(S);
    CHECK(L.line_count() == 3, "líneas = %zu (esperado 3 válidas de 6)", L.line_count());
    bool mono = true, disjoint = true, ends = true;
    for (usize l = 0; l < L.line_count(); ++l) {
        const u32 s0 = L.starts()[l], n = L.counts()[l];
        for (u32 i = 1; i < n; ++i) mono &= L.points()[s0 + i].x > L.points()[s0 + i - 1].x;
        ends &= L.points()[s0].x < 1.0f;                       // hacia atrás llega a la entrada
        for (usize k = l + 1; k < L.line_count(); ++k) {
            const u32 t0 = L.starts()[k], m = L.counts()[k];
            disjoint &= s0 + n <= t0 || t0 + m <= s0;
        }
    }
    CHECK(mono && ends, "líneas no monótonas o no llegan a la entrada");
    CHECK(disjoint, "rangos de puntos solapados");
    // Referencia FP32: mismas líneas (mismo número de puntos ±1 por el redondeo FP16 del paso adaptativo)
    Streamlines R;
    R.params = L.params;
    R.set_seeds(seeds);
    R.compute_reference(F.view());
    bool same = R.line_count() == L.line_count();
    for (usize l = 0; same && l < L.line_count(); ++l) same &= std::abs(static_cast<int>(R.counts()[l]) - static_cast<int>(L.counts()[l])) <= 2;
    CHECK(same, "FP16 vs FP32: nº de líneas/puntos distinto");
}

// ============================================================================
//  R8. Partículas: capacidad, pausa, conservación
// ============================================================================
static void test_particles_state() {
    std::printf("[R8] partículas: capacidad, pausa y conservación\n");
    Field F(48, 24, 24);
    FlowSampler S;
    S.update(F.view());
    Particles P;
    P.params.capacity = 5000;
    P.params.rate = 30.0f;
    const Rake rk = Rake::line({2, 4, 4}, {2, 20, 20}, 9);
    P.set_emitters(std::span<const Rake>(&rk, 1));
    for (int i = 0; i < 30; ++i) P.step(S, 5.0f);
    const usize alive = P.stats().alive;
    std::vector<Vec3> before(P.points().begin(), P.points().end());
    P.step(S, 0.0f);   // pausa
    bool same = P.stats().alive == alive;
    for (usize i = 0; same && i < alive; ++i) same = length(before[i] - P.points()[i]) == 0.0f;
    CHECK(same, "la pausa cambió la salida");
    P.params.capacity = 777;   // cambio de capacidad → reinicio limpio
    P.step(S, 5.0f);
    const auto& st = P.stats();
    CHECK(P.capacity() == 784 && st.alive <= P.capacity(), "capacidad %zu vivas %zu", P.capacity(), st.alive);
    bool cons = true;
    for (int i = 0; i < 60; ++i) {
        P.step(S, 7.0f);
        const auto& t = P.stats();
        cons &= t.emitted == t.alive + t.died_outside + t.died_solid + t.died_age + t.overwritten && t.alive <= P.capacity();
    }
    CHECK(cons, "conservación tras cambiar la capacidad");
    P.set_emitters({});
    for (int i = 0; i < 200; ++i) P.step(S, 20.0f);
    CHECK(P.stats().alive == 0, "sin emisores deberían morir todas (vivas %zu)", P.stats().alive);
}

// ============================================================================
//  R9. robust_range directo
// ============================================================================
static void test_robust_range() {
    std::printf("[R9] robust_range\n");
    std::vector<float> v(1000, k_nan);
    float lo = 0, hi = 0;
    CHECK(!detail::robust_range(v.data(), v.size(), 0.01f, 0.99f, lo, hi), "todo NaN → false");
    v.assign(37, 2.5f);
    CHECK(detail::robust_range(v.data(), v.size(), 0.01f, 0.99f, lo, hi) && lo < 2.5f && hi > 2.5f && hi - lo < 0.01f, "constante [%f,%f]", lo, hi);
    v.resize(10001);
    for (int i = 0; i <= 10000; ++i) v[static_cast<usize>(i)] = static_cast<float>(i);
    v.push_back(k_inf); v.push_back(-k_inf); v.push_back(k_nan);
    CHECK(detail::robust_range(v.data(), v.size(), 0.0f, 1.0f, lo, hi) && lo == 0.0f && hi == 10000.0f, "mín/máx exactos [%f,%f]", lo, hi);
    CHECK(detail::robust_range(v.data(), v.size(), 0.01f, 0.99f, lo, hi) && std::fabs(lo - 100.0f) < 20.0f && std::fabs(hi - 9900.0f) < 20.0f,
          "percentiles 1-99 %% [%f,%f] (esperado ≈[100,9900])", lo, hi);
    // n no múltiplo de 8, con NaN en la cola escalar
    const float t[11] = {k_nan, 3, 1, 4, 1, 5, 9, 2, 6, k_nan, -7};
    CHECK(detail::robust_range(t, 11, 0.0f, 1.0f, lo, hi) && lo == -7.0f && hi == 9.0f, "cola escalar [%f,%f]", lo, hi);
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    pool().start();
    const double t0 = now_sec();
    test_robust_range();
    test_odd_sizes();
    test_mask_near_wall();
    test_volume_viewport();
    test_slice_axis_consistency();
    test_color_mesh_exact();
    test_streamlines_mixed();
    test_particles_state();
    test_volume_baked_params();
    test_long_domain();
    test_nonfinite();
    std::printf("\n%s: %d comprobaciones correctas, %d fallos (%.1f s)\n", g_fail ? "FALLA" : "PASA", g_pass, g_fail, now_sec() - t0);
    pool().stop();
    return g_fail ? 1 : 0;
}
