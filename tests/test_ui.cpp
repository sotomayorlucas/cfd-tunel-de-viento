// ============================================================================
//  tests/test_ui.cpp — pruebas del módulo ui + plataforma (headless).
//
//  * draw2d: glifos con acentos (píxel a píxel contra la fuente), UTF-8,
//    mezcla AVX2 == SWAR, recorte (nada fuera del clip ni del framebuffer, con
//    coordenadas enormes/negativas/NaN), render paralelo == serie.
//  * UI: clic de botón (true exactamente una vez), slider (monótono y sujeto),
//    checkbox/radio/toggle, combo (abre, elige, cierra; clic fuera), rueda,
//    wants_mouse / wants_keyboard, Tab/Espacio/flechas, valor tecleado.
//  * Plataforma: escalado entero AVX2 == referencia escalar; ventana headless.
//  * Rendimiento: panel completo (≈45 widgets + 2 gráficas) < 1.5 ms.
//  * Capturas: build/ui/ui_demo*.png (revisadas a ojo).
//  Salida: PASS/FAIL, código ≠ 0 si algo falla.
// ============================================================================
#include "../src/core/png.hpp"
#include "../src/core/util.hpp"
#include "../src/platform/platform.hpp"
#include "../src/render/draw2d.hpp"
#include "../src/render/font_data.hpp"
#include "../src/ui/demo.hpp"
#include "../src/ui/ui.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace cfd;
using render::Framebuffer;
using render::Painter;
using render::Rect;

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                          \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(c)) { ++g_fail; std::printf("    FALLO %s:%d: %s\n", __FILE__, __LINE__, #c); } \
    } while (0)
#define CHECKF(c, ...)                                                                    \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(c)) {                                                                       \
            ++g_fail;                                                                     \
            std::printf("    FALLO %s:%d: %s — ", __FILE__, __LINE__, #c);                \
            std::printf(__VA_ARGS__);                                                     \
            std::printf("\n");                                                            \
        }                                                                                 \
    } while (0)

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------------------------------------
//  Simulador de entrada
// ---------------------------------------------------------------------------------------------
struct Sim {
    ui::Context ui;
    platform::Input in;
    Framebuffer fb;
    double t = 0;
    explicit Sim(int w = 800, int h = 600) {
        fb.resize(w, h);
        ui.style().smooth_scroll = false;
    }
    template <class F>
    void frame(F&& build) {
        ui.begin_frame(in, fb.w, fb.h, t);
        build();
        ui.end_frame();
        t += 1.0 / 60.0;
        in.begin_frame();
    }
    void move(int x, int y) { in.mouse_dx += x - in.mouse_x; in.mouse_dy += y - in.mouse_y; in.mouse_x = x; in.mouse_y = y; }
    void press(int b = 0) { in.mouse_down[b] = true; in.mouse_pressed[b] = true; }
    void release(int b = 0) { in.mouse_down[b] = false; in.mouse_released[b] = true; }
    void key(int k) { in.key_pressed[k] = true; }
    void type(const char* s) { std::snprintf(in.text, sizeof in.text, "%s", s); in.text_len = static_cast<int>(std::strlen(in.text)); }
    template <class F>
    void click(int x, int y, F&& build) { move(x, y); frame(build); press(); frame(build); release(); frame(build); }
};
static const Rect k_panel{500, 0, 300, 600};
static int cx(Rect r) { return r.x + r.w / 2; }
static int cy(Rect r) { return r.y + r.h / 2; }

// ---------------------------------------------------------------------------------------------
static void test_utf8() {
    std::printf("[utf8] decodificación y conteo\n");
    const char* s = "¿Qué ñandú?";
    CHECK(render::utf8_count(s) == 11);
    CHECK(render::text_width(s) == 88);
    CHECK(render::text_width(s, render::Font::Large) == 176);
    u8 g[64];
    int n = render::utf8_to_glyphs("áéíóúñÑ¿¡º°²", -1, g, 64);
    const u8 exp[] = {0xE1, 0xE9, 0xED, 0xF3, 0xFA, 0xF1, 0xD1, 0xBF, 0xA1, 0xBA, 0xB0, 0xB2};
    CHECK(n == 12);
    for (int i = 0; i < n && i < 12; ++i) CHECKF(g[i] == exp[i] - 0x20, "glifo %d = %d", i, g[i]);
    // Inválidos → '?', sin leer fuera de rango
    const char bad[] = {'a', static_cast<char>(0xC3), 'b', static_cast<char>(0xFF), static_cast<char>(0xE2), static_cast<char>(0x82), 0};
    n = render::utf8_to_glyphs(bad, -1, g, 64);
    CHECK(n == 5);
    CHECK(g[0] == 'a' - 0x20 && g[1] == '?' - 0x20 && g[2] == 'b' - 0x20 && g[3] == '?' - 0x20 && g[4] == '?' - 0x20);
    // Extras: Δ ρ → ≈ ✓ existen (índices ≥ 224) y μ griega → µ Latin-1
    n = render::utf8_to_glyphs("Δρ→≈✓μ", -1, g, 64);
    CHECK(n == 6);
    for (int i = 0; i < 5; ++i) CHECK(g[i] >= 224);
    CHECK(g[5] == 0xB5 - 0x20);
    // Ruta SWAR (8 bytes ASCII) == ruta escalar, incluidos controles
    std::string a;
    for (int i = 0; i < 300; ++i) a.push_back(static_cast<char>(i % 7 == 3 ? '\t' : 0x20 + (i * 37) % 95));
    std::vector<u8> out(400);
    n = render::utf8_to_glyphs(a.c_str(), -1, out.data(), 400);
    CHECK(n == 300);
    bool ok = true;
    for (int i = 0; i < 300; ++i) {
        const u8 c = static_cast<u8>(a[static_cast<usize>(i)]);
        ok &= out[static_cast<usize>(i)] == (c < 0x20 ? 0 : c - 0x20);
    }
    CHECK(ok);
    CHECK(render::utf8_prefix_bytes("añb", -1, 2) == 3);
    // Conteo SWAR en cadenas largas mixtas
    std::string m;
    for (int i = 0; i < 100; ++i) m += "aé€";
    CHECK(render::utf8_count(m.c_str()) == 300);
}

static void test_glyphs() {
    std::printf("[texto] glifos con acentos píxel a píxel (8×16 y 16×32)\n");
    Framebuffer fb;
    fb.resize(400, 80);
    fb.clear_color(0xFF000000u);
    Painter p(fb);
    const char* s = "áéíóúñÑ¿¡º";
    const u8 codes[] = {0xE1, 0xE9, 0xED, 0xF3, 0xFA, 0xF1, 0xD1, 0xBF, 0xA1, 0xBA};
    const int end = p.text(8, 4, s, 0xFFFFFFFFu);
    CHECK(end == 8 + 10 * 8);
    int bad = 0;
    for (int k = 0; k < 10; ++k)
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 8; ++c) {
                const bool on = (font::k_small_glyphs[codes[k] - 0x20][r] >> (7 - c)) & 1;
                const u32 px = fb.row(4 + r)[8 + 8 * k + c];
                bad += (px == 0xFFFFFFFFu) != on;
            }
    CHECKF(bad == 0, "%d píxeles distintos (fuente pequeña)", bad);
    fb.clear_color(0xFF000000u);
    p.text(3, 40, s, 0xFFFFFFFFu, render::Font::Large);
    bad = 0;
    for (int k = 0; k < 10; ++k)
        for (int r = 0; r < 32 && 40 + r < fb.h; ++r)
            for (int c = 0; c < 16; ++c) {
                const bool on = (font::k_large_glyphs[codes[k] - 0x20][r] >> (15 - c)) & 1;
                const u32 px = fb.row(40 + r)[3 + 16 * k + c];
                bad += (px == 0xFFFFFFFFu) != on;
            }
    CHECKF(bad == 0, "%d píxeles distintos (fuente grande)", bad);
    // Texto translúcido: mezcla correcta en los píxeles del glifo, fondo intacto fuera.
    fb.clear_color(0xFF102030u);
    p.text(0, 0, "H", 0x80FFFFFFu);
    const u32 exp = render::blend_px(0xFF102030u, 0xFFFFFFFFu, render::alpha256(0x80));
    int on = 0, off_ok = 1;
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c) {
            const bool bit = (font::k_small_glyphs['H' - 0x20][r] >> (7 - c)) & 1;
            const u32 px = fb.row(r)[c];
            if (bit) on += px == exp; else off_ok &= px == 0xFF102030u;
        }
    CHECK(on > 10 && off_ok);
}

static void test_blend_exact() {
    std::printf("[mezcla] AVX2 == SWAR escalar (bit-exacto)\n");
    Framebuffer fb, ref;
    fb.resize(64, 16);
    ref.resize(64, 16);
    WyRand rng(7);
    for (usize i = 0; i < fb.color.size(); ++i) fb.color[i] = ref.color[i] = static_cast<u32>(rng.next()) | 0xFF000000u;
    const u32 cols[] = {0x80FF8040u, 0x01FFFFFFu, 0xFE102030u, 0x40000000u};
    Painter p(fb);
    int y = 0;
    for (u32 c : cols) {
        p.fill_rect({3, y, 45, 3}, c);
        for (int yy = y; yy < y + 3; ++yy)
            for (int x = 3; x < 48; ++x) ref.row(yy)[x] = render::blend_px(ref.row(yy)[x], c, render::alpha256(c >> 24));
        y += 4;
    }
    CHECK(std::memcmp(fb.color.data(), ref.color.data(), fb.color.size() * 4) == 0);
}

// Dibuja primitivas aleatorias (con coordenadas absurdas) en un Painter o DrawList.
template <class P>
static void random_ops(P& p, WyRand& rng, int nops, const u32* img) {
    auto coord = [&]() -> float {
        switch (rng.below(10)) {
            case 0: return 1e9f;
            case 1: return -1e9f;
            case 2: return static_cast<float>(rng.below(2000)) - 1000.0f;
            case 3: return std::numeric_limits<float>::quiet_NaN();
            default: return rng.uniform(-60.0f, 400.0f);
        }
    };
    auto icoord = [&]() -> int {
        switch (rng.below(10)) {
            case 0: return std::numeric_limits<int>::max() - 3;
            case 1: return std::numeric_limits<int>::min() + 3;
            case 2: return static_cast<int>(rng.below(2000)) - 1000;
            default: return static_cast<int>(rng.below(460)) - 60;
        }
    };
    auto col = [&]() -> u32 { const u32 a = rng.below(3) == 0 ? 255u : rng.below(256); return (a << 24) | static_cast<u32>(rng.next() & 0xFFFFFF); };
    for (int i = 0; i < nops; ++i) {
        const Rect r{icoord(), icoord(), icoord(), icoord()};
        // Grosor: normalmente 1-4 px; a veces INT_MAX (antes: desborde con signo en t*2, UBSan).
        auto thick = [&](int k) { return rng.below(8) == 0 ? std::numeric_limits<int>::max() : 1 + static_cast<int>(rng.below(static_cast<u32>(k))); };
        switch (rng.below(17)) {
            case 0: p.fill_rect(r, col()); break;
            case 1: p.rect(r, col(), thick(4)); break;
            case 2: p.fill_round_rect(r, coord(), col(), rng.below(16)); break;
            case 3: p.round_rect(r, coord(), col(), thick(3), rng.below(16)); break;
            case 4: p.shadow(r, rng.uniform(0, 20), rng.uniform(0, 30), col(), rng.below(2)); break;
            case 5: p.gradient_v(r, col(), col()); break;
            case 6: p.gradient_h(r, col(), col()); break;
            case 7: p.line(coord(), coord(), coord(), coord(), col(), rng.uniform(0.2f, 8.0f)); break;
            case 8: {
                Vec2 pts[12];
                for (Vec2& q : pts) q = {coord(), coord()};
                p.polyline(pts, 1 + static_cast<int>(rng.below(12)), col(), rng.uniform(0.5f, 5.0f));
                break;
            }
            case 9: p.fill_circle(coord(), coord(), rng.below(4) ? rng.uniform(0, 80) : 1e9f, col()); break;
            case 10: p.circle(coord(), coord(), rng.uniform(0, 80), col(), rng.uniform(0.5f, 6.0f)); break;
            case 11: p.fill_triangle({coord(), coord()}, {coord(), coord()}, {coord(), coord()}, col()); break;
            case 12: p.blit(img, 32, 32, 32, icoord(), icoord(), rng.below(2), rng.below(256)); break;
            case 13: p.blit_scaled(img, 32, 32, 32, r, rng.below(2)); break;
            case 14: p.text(icoord(), icoord(), "¿Clip? ñáéíóú Δρ→ texto largo que se sale del recorte por la derecha", col(),
                            rng.below(2) ? render::Font::Large : render::Font::Small); break;
            case 15: p.text_shadow(icoord(), icoord(), "sombra ¡sí!", col()); break;
            default: {
                Vec2 pts[8];
                float x = coord();
                for (Vec2& q : pts) { q = {x, coord()}; x += rng.uniform(0, 60); }
                p.fill_area(pts, 8, coord(), col(), col());
                break;
            }
        }
    }
}

static int count_outside(const Framebuffer& fb, Rect clip, u32 canary) {
    int bad = 0;
    for (int y = 0; y < fb.h; ++y)
        for (int x = 0; x < fb.stride; ++x) {                       // incluye el relleno del stride
            const bool inside = x < fb.w && clip.contains(x, y);
            if (!inside && fb.row(y)[x] != canary) ++bad;
        }
    return bad;
}

static void test_clipping() {
    std::printf("[recorte] sin escrituras fuera del clip ni del framebuffer (coords enormes/negativas/NaN)\n");
    std::vector<u32> img(32 * 32);
    for (int i = 0; i < 32 * 32; ++i) img[static_cast<usize>(i)] = (static_cast<u32>(i * 2654435761u) & 0x00FFFFFFu) | (static_cast<u32>(i * 7) << 24);
    const u32 canary = 0xFF123456u;
    const Rect clips[] = {{40, 30, 150, 100}, {0, 0, 333, 211}, {-50, -50, 100, 100}, {300, 180, 1000, 1000}, {10, 10, 0, 50}};
    for (int pass = 0; pass < 2; ++pass) {
        for (const Rect& clip : clips) {
            Framebuffer fb;
            fb.resize(333, 211);                                    // stride 336 → 3 columnas de relleno
            for (usize i = 0; i < fb.color.size(); ++i) fb.color[i] = canary;
            WyRand rng(1234 + static_cast<u64>(clip.x));
            Rect eff = clip;                                        // recorte efectivo ∩ framebuffer
            {
                const int x0 = std::max(0, clip.x), y0 = std::max(0, clip.y);
                const int x1 = std::min(fb.w, clip.x + clip.w), y1 = std::min(fb.h, clip.y + clip.h);
                eff = {x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
            }
            if (pass == 0) {
                Painter p(fb);
                p.push_clip(clip);
                random_ops(p, rng, 1500, img.data());
                p.pop_clip();
            } else {
                render::DrawList dl;
                dl.push_clip(clip);
                random_ops(dl, rng, 1500, img.data());
                dl.pop_clip();
                dl.render(fb);
            }
            const int bad = count_outside(fb, eff, canary);
            CHECKF(bad == 0, "%s clip(%d,%d,%d,%d): %d píxeles fuera modificados", pass ? "DrawList" : "Painter", clip.x, clip.y, clip.w, clip.h, bad);
            if (eff.w > 20 && eff.h > 20) {
                int changed = 0;
                for (int y = eff.y; y < eff.y + eff.h; ++y)
                    for (int x = eff.x; x < eff.x + eff.w; ++x) changed += fb.row(y)[x] != canary;
                CHECK(changed > 0);
            }
        }
    }
}

static void test_parallel_equals_serial() {
    std::printf("[drawlist] render paralelo por franjas == ejecución en serie\n");
    std::vector<u32> img(32 * 32, 0x80FF00FFu);
    render::DrawList dl;
    WyRand rng(99);
    dl.push_clip({5, 3, 700, 590});
    random_ops(dl, rng, 3000, img.data());
    dl.pop_clip();
    Framebuffer a, b;
    a.resize(640, 480);
    b.resize(640, 480);
    a.clear_color(0xFF202020u);
    b.clear_color(0xFF202020u);
    dl.render(a);                         // paralelo, franjas automáticas
    Painter p(b);
    dl.execute(p);                        // serie
    bool same = true;
    for (int y = 0; y < a.h; ++y) same &= std::memcmp(a.row(y), b.row(y), static_cast<usize>(a.w) * 4) == 0;
    CHECK(same);
    Framebuffer c;
    c.resize(640, 480);
    c.clear_color(0xFF202020u);
    dl.render(c, 7);                      // otro número de franjas
    same = true;
    for (int y = 0; y < a.h; ++y) same &= std::memcmp(a.row(y), c.row(y), static_cast<usize>(a.w) * 4) == 0;
    CHECK(same);
}

// Painter inmediato == DrawList grabada y reproducida (misma secuencia de llamadas): detecta
// diferencias de GRABACIÓN (culling, redondeos, recuento de glifos), que la prueba anterior
// (DrawList paralela vs DrawList en serie) no puede ver porque ambas usan la misma grabación.
static void test_painter_equals_drawlist() {
    std::printf("[drawlist] Painter inmediato == DrawList grabada (mismas llamadas)\n");
    std::vector<u32> img(32 * 32);
    for (int i = 0; i < 32 * 32; ++i) img[static_cast<usize>(i)] = (static_cast<u32>(i * 2654435761u) & 0x00FFFFFFu) | (static_cast<u32>(i * 5) << 24);
    for (int seed = 0; seed < 3; ++seed) {
        Framebuffer a, b;
        a.resize(500, 400);
        b.resize(500, 400);
        a.clear_color(0xFF202020u);
        b.clear_color(0xFF202020u);
        {
            WyRand rng(4242 + static_cast<u64>(seed));
            Painter p(a);
            p.push_clip({7, 5, 480, 380});
            random_ops(p, rng, 1500, img.data());
        }
        {
            WyRand rng(4242 + static_cast<u64>(seed));
            render::DrawList dl;
            dl.push_clip({7, 5, 480, 380});
            random_ops(dl, rng, 1500, img.data());
            dl.render(b);
        }
        int rows = 0;
        for (int y = 0; y < a.h; ++y) rows += std::memcmp(a.row(y), b.row(y), static_cast<usize>(a.w) * 4) != 0;
        CHECKF(rows == 0, "semilla %d: %d filas distintas", seed, rows);
    }
    // Texto con bytes de continuación huérfanos: antes utf8_count (DrawList) ≠ glifos decodificados
    // (Painter) → la DrawList recortaba la cola y text_width no medía lo dibujado.
    const char* odd = "\x80" "ab\xBF" "c é\x80\x80 d";
    Framebuffer a, b;
    a.resize(200, 20);
    b.resize(200, 20);
    a.clear_color(0xFF000000u);
    b.clear_color(0xFF000000u);
    Painter p(a);
    const int xe_p = p.text(2, 2, odd, 0xFFFFFFFFu);
    render::DrawList dl;
    const int xe_d = dl.text(2, 2, odd, 0xFFFFFFFFu);
    dl.render(b);
    CHECKF(xe_p == xe_d && xe_p == 2 + render::text_width(odd), "x final Painter %d DrawList %d medido %d", xe_p, xe_d, 2 + render::text_width(odd));
    bool same = true;
    for (int y = 0; y < a.h; ++y) same &= std::memcmp(a.row(y), b.row(y), static_cast<usize>(a.w) * 4) == 0;
    CHECK(same);
}

// Propiedad: para CUALQUIER secuencia de bytes, nº de glifos decodificados == utf8_count ==
// nº de cortes de utf8_prefix_bytes (layout coherente con lo que se dibuja).
static void test_utf8_fuzz() {
    std::printf("[utf8] fuzz: glifos decodificados == utf8_count para bytes aleatorios\n");
    WyRand rng(31337);
    const u8 pool_bytes[] = {'a', ' ', 0x09, 0x7F, 0x80, 0xBF, 0xC0, 0xC2, 0xC3, 0xA9, 0xDF, 0xE0, 0xE2, 0x82, 0xAC, 0xED, 0xA0, 0xEF, 0xF0, 0x90, 0xF4, 0xF5, 0xFF};
    int bad = 0;
    std::vector<u8> out(300);
    for (int it = 0; it < 20000; ++it) {
        char buf[64];
        const int n = 1 + static_cast<int>(rng.below(40));
        for (int i = 0; i < n; ++i)
            buf[i] = static_cast<char>(rng.below(3) == 0 ? static_cast<u8>(0x20 + rng.below(95)) : pool_bytes[rng.below(sizeof pool_bytes)]);
        const int ng = render::utf8_to_glyphs(buf, n, out.data(), 300);
        const int nc = render::utf8_count(buf, n);
        bad += ng != nc;
        if (ng != nc && bad < 3) {
            std::printf("    bytes:");
            for (int i = 0; i < n; ++i) std::printf(" %02X", static_cast<u8>(buf[i]));
            std::printf("  → glifos %d, count %d\n", ng, nc);
        }
    }
    CHECKF(bad == 0, "%d cadenas con recuento incoherente", bad);
}

// Degradado vertical: el tramado Bayer debe tener periodo 4 en x en TODA la fila (también en
// la unión entre la cabeza pelada a 32 B y el cuerpo alineado), para cualquier x inicial.
static void test_gradient_dither_phase() {
    std::printf("[degradado] tramado Bayer con periodo 4 en x para cualquier alineación\n");
    Framebuffer f;
    f.resize(160, 16);
    Painter p(f);
    int bad = 0;
    for (int x0 = 0; x0 < 8; ++x0) {
        f.clear_color(0xFF000000u);
        p.gradient_v({x0, 0, 120, 16}, 0xFF101418u, 0xFF1A2026u);
        for (int y = 0; y < 16; ++y)
            for (int x = x0; x + 4 < x0 + 120; ++x) bad += f.row(y)[x] != f.row(y)[x + 4];
    }
    CHECKF(bad == 0, "%d pares (x, x+4) distintos", bad);
}

// Regresiones del revisor (cada una fallaba antes de su corrección).
static void test_review_regressions() {
    std::printf("[ui] regresiones: Mayús en slider_int, Esc/foco y teclado, hover del combo, series con offset raro\n");
    // 1) slider_int + Mayús: 200 px lentos (1 px/cuadro) deben mover ≈ 10 % de 200 px de recorrido.
    {
        Sim s;
        int iv = 10;
        Rect ir{};
        auto b = [&] { s.ui.begin_panel("p", k_panel); s.ui.slider_int("Pasos", &iv, 0, 100); ir = s.ui.last_rect(); s.ui.end_panel(); };
        s.frame(b);
        s.move(ir.x + 4 + (ir.w - 8) / 10, cy(ir)); s.frame(b);
        s.press(); s.frame(b);
        const int start = iv;
        for (int i = 0; i < 200; ++i) { s.in.shift = true; s.in.mouse_down[0] = true; s.move(s.in.mouse_x + 1, s.in.mouse_y); s.frame(b); }
        s.in.shift = false; s.release(); s.frame(b);
        const float expect = 200.0f * 100.0f / static_cast<float>(ir.w - 8) * 0.1f;
        CHECKF(std::fabs(static_cast<float>(iv - start) - expect) <= 1.0f, "inicio %d fin %d (esperado +%.1f)", start, iv, expect);
    }
    // 2) Esc que cierra un popup: wants_keyboard() ese cuadro (la app no debe usar ese Esc).
    {
        Sim s;
        int sel = 0;
        const char* items[] = {"a", "b", "c"};
        Rect cb{};
        bool hov = false;
        auto b = [&] { s.ui.begin_panel("p", k_panel); s.ui.combo("C", &sel, items, 3); cb = s.ui.last_rect(); hov = s.ui.item_hovered(); s.ui.end_panel(); };
        s.frame(b);
        s.click(cx(cb), cy(cb), b);
        CHECK(s.ui.popup_open());
        s.frame(b);
        CHECK(hov);                                     // 3) item_hovered() del combo con popup abierto
        s.key(platform::KeyEscape); s.frame(b);
        CHECK(!s.ui.popup_open());
        CHECK(s.ui.wants_keyboard());
        s.frame(b);
        CHECK(!s.ui.wants_keyboard());
        // Esc sin nada que cerrar: la UI no lo consume.
        s.key(platform::KeyEscape); s.frame(b);
        CHECK(!s.ui.wants_keyboard());
    }
    // 4) Tab (foco visible) y luego clic en el visor: la UI suelta el teclado.
    {
        Sim s;
        bool a = false;
        auto b = [&] { s.ui.begin_panel("p", k_panel); s.ui.checkbox("X", &a); s.ui.end_panel(); };
        s.frame(b);
        s.key(platform::KeyTab); s.frame(b);
        CHECK(s.ui.wants_keyboard());
        s.click(100, 100, b);
        CHECK(!s.ui.wants_keyboard());
        CHECK(!a);
        s.key(platform::KeyTab); s.frame(b);            // Tab vuelve a entrar
        CHECK(s.ui.wants_keyboard());
    }
    // 5) PlotSeries con offset negativo / ≥ count: mismo dibujo que el offset normalizado
    //    (antes: lectura fuera del búfer, ASan).
    {
        float d[50];
        for (int i = 0; i < 50; ++i) d[i] = std::sin(0.3f * static_cast<float>(i));
        u64 h[3] = {};
        const int offs[3] = {7, 7 - 50, 7 + 3 * 50};
        for (int k = 0; k < 3; ++k) {
            Sim s;
            s.fb.clear_color(0xFF000000u);
            const ui::PlotSeries ps{d, 50, offs[k], 0, "x"};
            auto b = [&] { s.ui.begin_panel("p", k_panel); s.ui.plot_lines("g", &ps, 1, 140); s.ui.end_panel(); };
            s.frame(b);
            s.ui.begin_frame(s.in, s.fb.w, s.fb.h, s.t); b(); s.ui.end_frame();
            s.ui.render(s.fb);
            h[k] = 0xcbf29ce484222325ull;
            for (int y = 0; y < s.fb.h; ++y)
                for (int x = 0; x < s.fb.w; ++x) { h[k] ^= s.fb.row(y)[x]; h[k] *= 0x100000001b3ull; }
        }
        CHECK(h[0] == h[1] && h[0] == h[2]);
        // Serie nula / count negativo: se ignora sin fallar.
        Sim s;
        const ui::PlotSeries bad[2] = {{nullptr, 10, 0, 0, "nula"}, {d, -5, 0, 0, "neg"}};
        s.frame([&] { s.ui.begin_panel("p", k_panel); s.ui.plot_lines("g", bad, 2, 120); s.ui.end_panel(); });
        CHECK(true);
    }
    // 7) round_rect/rect nunca pintan fuera de SU rectángulo (antes, con grosor ≥ medio lado,
    //    las celdas de esquina ri = max(r, t) se salían; el test de recorte no lo veía porque
    //    seguía dentro del clip).
    {
        Framebuffer fb;
        fb.resize(64, 64);
        WyRand rng(77);
        int outside = 0;
        Painter p(fb);
        for (int it = 0; it < 3000; ++it) {
            fb.clear_color(0xFF000000u);
            const Rect r{20 + static_cast<int>(rng.below(8)), 20 + static_cast<int>(rng.below(8)), 1 + static_cast<int>(rng.below(14)), 1 + static_cast<int>(rng.below(14))};
            const int t = 1 + static_cast<int>(rng.below(8));
            const u32 c = rng.below(2) ? 0xFFFFFFFFu : 0x80FFFFFFu;
            if (it % 3 == 2) p.rect(r, c, it % 2 ? t : (1 << 30));
            else p.round_rect(r, rng.uniform(0.0f, 10.0f), c, it % 7 == 0 ? std::numeric_limits<int>::max() : t, rng.below(16));
            for (int y = 0; y < fb.h; ++y)
                for (int x = 0; x < fb.w; ++x) outside += !r.contains(x, y) && fb.row(y)[x] != 0xFF000000u;
        }
        CHECKF(outside == 0, "%d píxeles pintados fuera del rectángulo", outside);
    }
    // 6) Mezcla AVX2 == SWAR también en el canal alfa con destino NO opaco (alfa 0 al crear).
    {
        Framebuffer fb;
        fb.resize(64, 4);
        WyRand rng(3);
        std::vector<u32> ref(fb.color.size());
        for (usize i = 0; i < fb.color.size(); ++i) fb.color[i] = ref[i] = static_cast<u32>(rng.next());   // alfa aleatorio
        Painter p(fb);
        p.fill_rect({1, 0, 50, 4}, 0x80FF8040u);
        for (int y = 0; y < 4; ++y)
            for (int x = 1; x < 51; ++x) ref[static_cast<usize>(y * fb.stride + x)] = render::blend_px(ref[static_cast<usize>(y * fb.stride + x)], 0x80FF8040u, render::alpha256(0x80));
        CHECK(std::memcmp(fb.color.data(), ref.data(), ref.size() * 4) == 0);
    }
}

// ---------------------------------------------------------------------------------------------
//  Widgets
// ---------------------------------------------------------------------------------------------
static void test_button() {
    std::printf("[ui] botón: true exactamente una vez por clic\n");
    Sim s;
    Rect br{};
    int trues = 0;
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        if (s.ui.button("Pulsar")) ++trues;
        br = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.move(cx(br), cy(br));
    s.frame(build);
    CHECK(trues == 0);
    CHECK(s.ui.hot_id() != 0);
    s.press();
    s.frame(build);
    CHECK(trues == 0);
    CHECK(s.ui.active_id() != 0);
    s.frame(build);
    s.release();
    s.frame(build);
    CHECK(trues == 1);
    s.frame(build);
    s.frame(build);
    CHECK(trues == 1);
    CHECK(s.ui.active_id() == 0);
    // Pulsar fuera (visor) y soltar dentro → nada.
    s.move(10, 10); s.press(); s.frame(build);
    s.move(cx(br), cy(br)); s.frame(build);
    s.release(); s.frame(build);
    CHECK(trues == 1);
    // Pulsar dentro, salir y soltar fuera → nada.
    s.press(); s.frame(build);
    s.move(cx(br), br.y + br.h + 40); s.frame(build);
    s.release(); s.frame(build);
    CHECK(trues == 1);
    // Pulsar y soltar en el mismo cuadro → clic.
    s.move(cx(br), cy(br)); s.frame(build);
    s.press(); s.release(); s.frame(build);
    CHECK(trues == 2);
}

static void test_slider() {
    std::printf("[ui] slider: arrastre monótono, sujeción, captura del ratón, ajuste fino\n");
    Sim s;
    float v = 100.0f;
    Rect tr{};
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        s.ui.slider_float("Velocidad", &v, 50.0f, 350.0f, "%.0f", "km/h");
        tr = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.move(tr.x + 6, cy(tr));
    s.frame(build);
    s.press();
    s.frame(build);
    CHECK(v < 60.0f);
    float prev = v;
    bool mono = true;
    for (int x = tr.x + 6; x < tr.x + tr.w + 150; x += 9) {
        s.move(x, cy(tr) + (x % 3));                     // un poco de temblor vertical
        s.frame(build);
        mono &= v >= prev;
        prev = v;
        CHECK(s.ui.wants_mouse());
    }
    CHECK(mono);
    CHECK(v == 350.0f);
    s.move(20, 300);                                     // muy a la izquierda, dentro del visor
    s.frame(build);
    CHECK(v == 50.0f);
    CHECK(s.ui.wants_mouse());                           // sigue capturado aunque esté fuera del panel
    s.release();
    s.frame(build);
    CHECK(s.ui.active_id() == 0);
    // Ajuste fino con Mayús: 100 px → 10 % del recorrido absoluto.
    s.move(cx(tr), cy(tr)); s.frame(build);
    s.press(); s.frame(build);
    const float v0 = v;
    s.in.shift = true;
    s.move(cx(tr) + 50, cy(tr)); s.frame(build);
    const float fine = v - v0;
    s.in.shift = false;
    s.release(); s.frame(build);
    const float full = 50.0f * 300.0f / static_cast<float>(tr.w - 8);
    CHECKF(fine > 0.0f && std::fabs(fine - 0.1f * full) < 0.02f * full, "fino=%g completo=%g", fine, full);
    // slider_int redondea y sujeta
    int iv = 5;
    Rect ir{};
    auto bi = [&] { s.ui.begin_panel("p", k_panel); s.ui.slider_int("Pasos", &iv, 1, 32); ir = s.ui.last_rect(); s.ui.end_panel(); };
    s.frame(bi);
    s.move(ir.x + ir.w + 30, cy(ir)); s.frame(bi);
    s.move(ir.x + 30, cy(ir)); s.frame(bi);
    s.press(); s.frame(bi);
    CHECK(iv >= 1 && iv <= 32);
    s.move(ir.x + ir.w + 100, cy(ir)); s.frame(bi);
    CHECK(iv == 32);
    s.release(); s.frame(bi);
}

static void test_checkbox_radio_toggle() {
    std::printf("[ui] checkbox / radio / interruptor / botón conmutable\n");
    Sim s;
    bool a = false, sw = false, tb = false;
    int r = 0;
    Rect ra{}, rr1{}, rr2{}, rsw{}, rtb{};
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        s.ui.checkbox("Humo", &a); ra = s.ui.last_rect();
        s.ui.row(2);
        s.ui.radio("Fijo", &r, 1); rr1 = s.ui.last_rect();
        s.ui.radio("Cinta", &r, 2); rr2 = s.ui.last_rect();
        s.ui.toggle("Ruedas", &sw); rsw = s.ui.last_rect();
        s.ui.toggle_button("DRS", &tb); rtb = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.click(cx(ra), cy(ra), build);
    CHECK(a);
    s.click(cx(ra), cy(ra), build);
    CHECK(!a);
    s.click(cx(rr2), cy(rr2), build);
    CHECK(r == 2);
    s.click(cx(rr1), cy(rr1), build);
    CHECK(r == 1);
    CHECK(rr1.y == rr2.y && rr2.x > rr1.x);               // misma fila
    s.click(cx(rsw), cy(rsw), build);
    CHECK(sw);
    s.click(cx(rtb), cy(rtb), build);
    CHECK(tb);
}

static void test_combo() {
    std::printf("[ui] combo: abre, elige, cierra; clic fuera cierra sin cambiar ni activar\n");
    Sim s;
    int sel = 0, top_clicks = 0;
    const char* items[] = {"Uno", "Dos", "Tres", "Cuatro"};
    Rect cb{}, tb{};
    bool changed = false;
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        if (s.ui.button("Arriba")) ++top_clicks;
        tb = s.ui.last_rect();
        changed |= s.ui.combo("Modelo", &sel, items, 4);
        cb = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.click(cx(cb), cy(cb), build);
    CHECK(s.ui.popup_open());
    CHECK(s.ui.wants_mouse());
    const int rh = s.ui.style().row_h;
    const int item_y = cb.y + cb.h + 3 + 4 + 2 * rh + rh / 2;
    s.click(cx(cb), item_y, build);
    CHECK(sel == 2);
    CHECK(changed);
    CHECK(!s.ui.popup_open());
    // Abrir y pulsar fuera (en otro botón): se cierra y el botón NO se activa.
    s.click(cx(cb), cy(cb), build);
    CHECK(s.ui.popup_open());
    s.click(cx(tb), cy(tb), build);
    CHECK(!s.ui.popup_open());
    CHECK(top_clicks == 0);
    CHECK(sel == 2);
    // Clic fuera en el visor.
    s.click(cx(cb), cy(cb), build);
    CHECK(s.ui.popup_open());
    s.click(50, 50, build);
    CHECK(!s.ui.popup_open());
    CHECK(sel == 2);
    // Segundo clic en la cabecera cierra.
    s.click(cx(cb), cy(cb), build);
    CHECK(s.ui.popup_open());
    s.click(cx(cb), cy(cb), build);
    CHECK(!s.ui.popup_open());
    // Escape cierra.
    s.click(cx(cb), cy(cb), build);
    s.key(platform::KeyEscape);
    s.frame(build);
    CHECK(!s.ui.popup_open());
}

static void test_scroll() {
    std::printf("[ui] panel: la rueda desplaza y se sujeta al contenido\n");
    Sim s;
    const Rect pr{500, 0, 300, 300};
    Rect first{}, last{};
    auto build = [&] {
        s.ui.begin_panel("lista", pr);
        for (int i = 0; i < 40; ++i) {
            s.ui.push_id(i);
            s.ui.button("Elemento");
            if (i == 0) first = s.ui.last_rect();
            if (i == 39) last = s.ui.last_rect();
            s.ui.pop_id();
        }
        s.ui.end_panel();
    };
    s.frame(build);
    s.frame(build);
    const int y0 = first.y;
    s.move(pr.x + 100, 150);
    s.in.wheel = -1.0f;
    s.frame(build);
    s.frame(build);
    CHECKF(first.y == y0 - 3 * s.ui.style().row_h, "y0=%d y=%d", y0, first.y);
    for (int i = 0; i < 100; ++i) { s.in.wheel = -1.0f; s.frame(build); }
    s.frame(build);
    const int pad = s.ui.style().pad;
    CHECKF(last.y + last.h == pr.y + pr.h - pad, "último elemento acaba en %d (esperado %d)", last.y + last.h, pr.y + pr.h - pad);
    // Rueda fuera del panel: sin cambio.
    const int yb = first.y;
    s.move(100, 150);
    s.in.wheel = 5.0f;
    s.frame(build);
    s.frame(build);
    CHECK(first.y == yb);
    // Rueda hacia arriba: vuelve al inicio y se sujeta.
    s.move(pr.x + 100, 150);
    for (int i = 0; i < 100; ++i) { s.in.wheel = 1.0f; s.frame(build); }
    s.frame(build);
    CHECK(first.y == y0);
}

static void test_wants_mouse() {
    std::printf("[ui] wants_mouse: panel sí, visor no; arrastres capturados por quien pulsó\n");
    Sim s;
    float v = 0.5f;
    Rect tr{};
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        s.ui.slider_float("Opacidad", &v, 0.0f, 1.0f);
        tr = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.move(600, 400); s.frame(build);
    CHECK(s.ui.wants_mouse());
    s.move(200, 300); s.frame(build);
    CHECK(!s.ui.wants_mouse());
    // Pulsar en el visor y arrastrar sobre el slider: la app conserva el ratón.
    s.press(); s.frame(build);
    CHECK(!s.ui.wants_mouse());
    s.move(tr.x + 10, cy(tr)); s.frame(build);
    CHECK(!s.ui.wants_mouse());
    s.move(tr.x + tr.w - 10, cy(tr)); s.frame(build);
    CHECK(v == 0.5f);
    s.release(); s.frame(build);
    CHECK(v == 0.5f);
    // Tras soltar, sobre el panel la UI vuelve a querer el ratón.
    s.frame(build);
    CHECK(s.ui.wants_mouse());
    CHECK(!s.ui.wants_keyboard());
}

static void test_keyboard() {
    std::printf("[ui] teclado: Tab, Espacio, flechas, valor tecleado (coma decimal), Esc\n");
    Sim s;
    bool a = false;
    float v = 10.0f;
    int clicks = 0;
    Rect tr{};
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        s.ui.checkbox("Casilla", &a);
        s.ui.slider_float("Valor", &v, 0.0f, 100.0f, "%.1f");
        tr = s.ui.last_rect();
        if (s.ui.button("Aceptar")) ++clicks;
        s.ui.end_panel();
    };
    s.frame(build);
    s.key(platform::KeyTab); s.frame(build);           // foco → casilla
    CHECK(s.ui.wants_keyboard());
    s.key(' '); s.frame(build);
    CHECK(a);
    s.key(platform::KeyTab); s.frame(build);           // foco → slider
    s.key(platform::KeyRight); s.frame(build);
    CHECKF(std::fabs(v - 11.0f) < 1e-4f, "v=%g", v);
    s.in.shift = true;
    s.key(platform::KeyLeft); s.frame(build);          // paso fino 0.1
    s.in.shift = false;
    CHECKF(std::fabs(v - 10.9f) < 1e-4f, "v=%g", v);
    s.key(platform::KeyEnter); s.frame(build);         // escribir valor
    CHECK(s.ui.editing());
    s.type("42,5"); s.frame(build);
    s.key(platform::KeyEnter); s.frame(build);
    CHECK(!s.ui.editing());
    CHECKF(std::fabs(v - 42.5f) < 1e-4f, "v=%g", v);
    s.key(platform::KeyTab); s.frame(build);           // foco → botón
    s.key(platform::KeyEnter); s.frame(build);
    CHECK(clicks == 1);
    s.key(platform::KeyEscape); s.frame(build);
    CHECK(s.ui.wants_keyboard());                       // este Esc lo consumió la UI (quitar el foco)
    s.frame(build);
    CHECK(!s.ui.wants_keyboard());
    // Ctrl+clic → edición; valor fuera de rango se sujeta; Esc cancela.
    s.move(cx(tr), cy(tr)); s.frame(build);
    s.in.ctrl = true;
    s.press(); s.frame(build);
    s.in.ctrl = false;
    s.release(); s.frame(build);
    CHECK(s.ui.editing());
    CHECK(s.ui.wants_keyboard());
    s.type("1e6"); s.frame(build);
    s.key(platform::KeyEnter); s.frame(build);
    CHECK(v == 100.0f);
    // Doble clic: el primer clic mueve el valor, el doble lo restaura y abre la edición.
    v = 30.0f;
    s.frame(build);
    s.move(tr.x + tr.w - 20, cy(tr)); s.frame(build);
    s.press(); s.frame(build);
    CHECK(v > 80.0f);
    s.release(); s.frame(build);
    s.press(); s.in.double_click = true; s.frame(build);
    s.release(); s.frame(build);
    CHECK(s.ui.editing());
    CHECKF(v == 30.0f, "v=%g", v);
    s.key(platform::KeyEscape); s.frame(build);
    CHECK(!s.ui.editing());
    CHECK(v == 30.0f);
}

static void test_tooltip_and_toast() {
    std::printf("[ui] tooltip tras 0.5 s de hover; avisos que expiran\n");
    Sim s;
    Rect br{};
    auto build = [&] {
        s.ui.begin_panel("p", k_panel);
        s.ui.button("Info");
        s.ui.tooltip("Texto de ayuda con acentos: ¿qué tal?");
        br = s.ui.last_rect();
        s.ui.end_panel();
    };
    s.frame(build);
    s.move(cx(br), cy(br));
    for (int i = 0; i < 40; ++i) s.frame(build);
    // El tooltip se dibuja en la capa superior: render y buscar píxeles del color de fondo del tooltip.
    s.fb.clear_color(0xFF000000u);
    s.ui.begin_frame(s.in, s.fb.w, s.fb.h, s.t);
    build();
    s.ui.end_frame();
    s.ui.render(s.fb);
    int tip_px = 0;
    for (int y = br.y + br.h; y < std::min(s.fb.h, br.y + br.h + 80); ++y)
        for (int x = 0; x < s.fb.w; ++x) tip_px += s.fb.row(y)[x] != 0xFF000000u && x < k_panel.x;
    CHECK(tip_px > 200);                                 // el tooltip sobresale del panel hacia el visor
    s.ui.toast(0, "Aviso de prueba");
    for (int i = 0; i < 60 * 5; ++i) s.frame(build);     // 5 s > 3.5 s
    s.ui.begin_frame(s.in, s.fb.w, s.fb.h, s.t);
    s.ui.end_frame();
    CHECK(s.ui.stats().cmds == 0);                       // sin panel ni avisos → nada que dibujar
}

// ---------------------------------------------------------------------------------------------
//  Plataforma
// ---------------------------------------------------------------------------------------------
static void test_upscale() {
    std::printf("[plataforma] escalado entero AVX2 == referencia (escala 1-3, márgenes, NT)\n");
    WyRand rng(5);
    const int sw = 37, sh = 19, ss = 40;
    std::vector<u32> src(static_cast<usize>(ss * sh));
    for (u32& p : src) p = static_cast<u32>(rng.next());
    for (int scale = 1; scale <= 3; ++scale)
        for (int extra = -3; extra <= 9; extra += 6)
            for (int nt = 0; nt < 2; ++nt)
                for (int par = 0; par < 2; ++par) {
                    const int dw = sw * scale + extra, dh = sh * scale + extra / 2;
                    const int ds = (dw + 15) & ~15;
                    Buffer<u32> dst(static_cast<usize>(ds * dh));
                    dst.fill(0xDEADBEEFu);
                    platform::upscale_argb(src.data(), sw, sh, ss, dst.data(), dw, dh, ds, scale, par, nt);
                    int bad = 0;
                    for (int y = 0; y < dh; ++y)
                        for (int x = 0; x < ds; ++x) {
                            u32 e;
                            if (x >= dw) e = 0xDEADBEEFu;                       // relleno intacto
                            else if (x < sw * scale && y < sh * scale) e = src[static_cast<usize>((y / scale) * ss + x / scale)];
                            else e = 0;
                            bad += dst[static_cast<usize>(y * ds + x)] != e;
                        }
                    CHECKF(bad == 0, "escala %d extra %d nt %d par %d: %d píxeles", scale, extra, nt, par, bad);
                }
    auto w = platform::create_headless_window();
    CHECK(w && w->open("prueba", 320, 200, 2));
    platform::Input in;
    in.begin_frame();
    w->poll(in);
    CHECK(in.fb_w == 320 && in.fb_h == 200 && in.resized && w->is_headless() && w->pixel_scale() == 2);
    Framebuffer fb;
    fb.resize(320, 200);
    w->present(fb);
    CHECK(w->present_stats().frames == 1);
}

// ---------------------------------------------------------------------------------------------
//  Rendimiento + capturas
// ---------------------------------------------------------------------------------------------
static void save_crop(const Framebuffer& fb, Rect r, int zoom, const char* path) {
    std::vector<u32> img(static_cast<usize>(r.w * zoom) * static_cast<usize>(r.h * zoom));
    for (int y = 0; y < r.h * zoom; ++y)
        for (int x = 0; x < r.w * zoom; ++x) img[static_cast<usize>(y * r.w * zoom + x)] = fb.row(r.y + y / zoom)[r.x + x / zoom];
    png::write_argb(path, img.data(), r.w * zoom, r.h * zoom, r.w * zoom);
}

static void test_perf_and_screenshots() {
    std::printf("[rendimiento] panel completo de demostración (1920×1200)\n");
    Framebuffer fb;
    fb.resize(1920, 1200);
    ui::Context ui;
    ui::DemoState st;
    ui::demo_init(st);
    platform::Input in;
    const int pw = 420;
    const Rect panel{fb.w - pw, 0, pw, fb.h};
    const Rect vp{0, 0, fb.w - pw, fb.h};
    ui.set_toast_area(vp);
    double t = 0;
    auto run = [&](bool render) {
        ui.begin_frame(in, fb.w, fb.h, t);
        ui::demo_panel(ui, st, panel, 60.0);
        ui.end_frame();
        if (render) ui.render(fb);
        in.begin_frame();
        t += 1.0 / 60.0;
        ui::demo_tick(st, 1.0 / 60.0);
    };
    // Calentamiento (reservas de vectores, cachés).
    in.mouse_x = panel.x + 200; in.mouse_y = 700;
    for (int i = 0; i < 30; ++i) run(true);
    std::vector<double> tb, tr, tt;
    int widgets = 0;
    usize cmds = 0;
    for (int i = 0; i < 300; ++i) {
        in.mouse_x = panel.x + 40 + (i * 7) % 360;          // ratón en movimiento (hover variable)
        in.mouse_y = 100 + (i * 13) % 1000;
        const double t0 = now_sec();
        ui.begin_frame(in, fb.w, fb.h, t);
        ui::demo_panel(ui, st, panel, 60.0);
        ui.end_frame();
        const double t1 = now_sec();
        ui.render(fb);
        const double t2 = now_sec();
        tb.push_back((t1 - t0) * 1e3);
        tr.push_back((t2 - t1) * 1e3);
        tt.push_back((t2 - t0) * 1e3);
        widgets = ui.stats().widgets;
        cmds = ui.stats().cmds;
        in.begin_frame();
        t += 1.0 / 60.0;
        ui::demo_tick(st, 1.0 / 60.0);
    }
    const double mb = median(tb), mr = median(tr), mt = median(tt);
    std::printf("    widgets=%d comandos=%zu  construcción %.3f ms  render %.3f ms  total %.3f ms (medianas de 300)\n",
                widgets, cmds, mb, mr, mt);
    CHECKF(mt < 1.5, "total %.3f ms ≥ 1.5 ms", mt);
    CHECK(widgets >= 40);
    // Render en serie (1 franja) como referencia.
    {
        std::vector<double> ts;
        ui.begin_frame(in, fb.w, fb.h, t);
        ui::demo_panel(ui, st, panel, 60.0);
        ui.end_frame();
        for (int i = 0; i < 100; ++i) {
            render::Painter p(fb);
            const double t0 = now_sec();
            // reproducir las 3 capas en serie
            ui.render(fb);   // calentar
            (void)p;
            ts.push_back((now_sec() - t0) * 1e3);
        }
        std::printf("    render (paralelo, repetido sin reconstruir): %.3f ms\n", median(ts));
    }

    // --- Capturas ---
    auto frame_full = [&](int mx, int my, int wheel, bool press, bool release) {
        in.mouse_x = mx; in.mouse_y = my;
        in.wheel = static_cast<float>(wheel);
        if (press) { in.mouse_down[0] = true; in.mouse_pressed[0] = true; }
        if (release) { in.mouse_down[0] = false; in.mouse_released[0] = true; }
        ui.begin_frame(in, fb.w, fb.h, t);
        ui::demo_panel(ui, st, panel, 60.0);
        ui.end_frame();
        ui::demo_viewport(fb, vp, 60.0, ui.stats().build_ms + ui.stats().render_ms);
        ui.render(fb);
        in.begin_frame();
        t += 1.0 / 60.0;
    };
    ui.style().smooth_scroll = false;
    for (int i = 0; i < 40; ++i) frame_full(panel.x + 300, 300, 1, false, false);   // arriba del todo
    // Hover sobre el slider de velocidad (2º control de "Túnel de viento") para ver su tooltip.
    for (int i = 0; i < 50; ++i) frame_full(panel.x + 300, 284, 0, false, false);
    ui.toast(ui.style().success, "Geometría voxelizada en 38 ms (1.2 M celdas sólidas)");
    for (int i = 0; i < 20; ++i) frame_full(panel.x + 300, 284, 0, false, false);
    png::write_argb("build/ui/ui_demo.png", fb.color.data(), fb.w, fb.h, fb.stride);
    save_crop(fb, {panel.x, 0, pw, 600}, 2, "build/ui/ui_panel_top_x2.png");
    save_crop(fb, {panel.x, 600, pw, 600}, 2, "build/ui/ui_panel_mid_x2.png");
    // Desplazar al final del panel.
    for (int i = 0; i < 60; ++i) frame_full(panel.x + 200, 600, -1, false, false);
    for (int i = 0; i < 5; ++i) frame_full(panel.x + 200, 600, 0, false, false);
    png::write_argb("build/ui/ui_demo_bottom.png", fb.color.data(), fb.w, fb.h, fb.stride);
    save_crop(fb, {panel.x, 0, pw, 600}, 2, "build/ui/ui_panel_bot1_x2.png");
    save_crop(fb, {panel.x, 600, pw, 600}, 2, "build/ui/ui_panel_bot2_x2.png");
    // Combo abierto (volver arriba y abrir "Modelo", 1er control tras la cabecera).
    for (int i = 0; i < 60; ++i) frame_full(panel.x + 300, 300, 1, false, false);
    frame_full(panel.x + 330, 119, 0, false, false);
    frame_full(panel.x + 330, 119, 0, true, false);
    frame_full(panel.x + 330, 119, 0, false, true);
    CHECK(ui.popup_open());
    for (int i = 0; i < 4; ++i) frame_full(panel.x + 250, 119 + 12 + 3 * 24 + 12, 0, false, false);
    png::write_argb("build/ui/ui_demo_popup.png", fb.color.data(), fb.w, fb.h, fb.stride);
    save_crop(fb, {panel.x - 20, 0, pw + 20, 520}, 2, "build/ui/ui_popup_x2.png");
    std::printf("    capturas: build/ui/ui_demo.png, ui_demo_bottom.png, ui_demo_popup.png (+ recortes ×2)\n");
}

// Escala ×2: framebuffer nativo 3840×2400 (sin escalado al presentar) y UI con métricas ×2
// y fuente 16×32. Comprueba que el diseño escala (misma estructura, sin solapes) y lo guarda.
static void test_scale2() {
    std::printf("[ui] escala 2 (framebuffer 3840×2400 nativo)\n");
    Framebuffer fb;
    fb.resize(3840, 2400);
    ui::Context ui;
    ui.set_scale(2.0f);
    ui.style().smooth_scroll = false;
    CHECK(ui.style().font == render::Font::Large && ui.style().row_h == 48);
    ui::DemoState st;
    ui::demo_init(st);
    platform::Input in;
    in.mouse_x = 3000; in.mouse_y = 560;
    const Rect panel{fb.w - 840, 0, 840, fb.h};
    std::vector<double> tt;
    for (int i = 0; i < 40; ++i) {
        const double t0 = now_sec();
        ui.begin_frame(in, fb.w, fb.h, i / 60.0);
        ui::demo_panel(ui, st, panel, 60.0);
        ui.end_frame();
        ui::demo_viewport(fb, {0, 0, fb.w - 840, fb.h}, 60.0, 0.0);
        ui.render(fb);
        tt.push_back((now_sec() - t0) * 1e3);
        in.begin_frame();
    }
    std::printf("    UI+visor a 3840×2400: %.3f ms (mediana)\n", median(tt));
    CHECK(ui.stats().widgets >= 40);
    save_crop(fb, {panel.x, 0, 840, 1200}, 1, "build/ui/ui_escala2_top.png");
}

int main(int argc, char** argv) {
    // --rapido: omite rendimiento y capturas (para compilaciones con sanitizers).
    const bool quick = argc > 1 && std::strcmp(argv[1], "--rapido") == 0;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    pool().start();
    const double t0 = now_sec();
    test_utf8();
    test_glyphs();
    test_blend_exact();
    test_clipping();
    test_parallel_equals_serial();
    test_painter_equals_drawlist();
    test_utf8_fuzz();
    test_gradient_dither_phase();
    test_review_regressions();
    test_button();
    test_slider();
    test_checkbox_radio_toggle();
    test_combo();
    test_scroll();
    test_wants_mouse();
    test_keyboard();
    test_tooltip_and_toast();
    test_upscale();
    if (!quick) test_perf_and_screenshots();
    if (!quick) test_scale2();
    std::printf("\n%s: %d comprobaciones, %d fallos (%.1f s)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
