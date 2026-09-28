// ============================================================================
//  tests/test_ui_bench.cpp — micro-benchmarks del módulo ui (antes/después).
//
//  Cada truco se compara con una versión escalar "ingenua" escrita aquí mismo
//  (misma salida), con medianas de N repeticiones (la máquina tiene carga de
//  fondo). Comprueba además que ambas versiones den el mismo resultado.
//  Uso: build/ui/test_ui_bench [--rapido]
// ============================================================================
#include "../src/core/util.hpp"
#include "../src/platform/platform.hpp"
#include "../src/render/draw2d.hpp"
#include "../src/render/font_data.hpp"
#include "../src/ui/demo.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cfd;
using render::Framebuffer;
using render::Painter;
using render::Rect;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("    FALLO %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

template <class F>
static double bench_ms(int reps, F&& f) {
    std::vector<double> v;
    v.reserve(static_cast<usize>(reps));
    f();                                       // calentamiento
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_sec();
        f();
        v.push_back((now_sec() - t0) * 1e3);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static void row(const char* name, double before, double after, const char* unit = "ms") {
    std::printf("  %-52s antes %9.4f %s  después %9.4f %s  (×%.2f)\n", name, before, unit, after, unit, before / after);
}

// ---- Referencias escalares ---------------------------------------------------------------------------
static CFD_NOINLINE void ref_fill(Framebuffer& fb, Rect r, u32 c) {
    for (int y = r.y; y < r.y + r.h; ++y) {
        u32* d = fb.row(y) + r.x;
        for (int x = 0; x < r.w; ++x) { d[x] = c; __asm__ volatile("" ::: "memory"); }   // sin autovectorizar
    }
}
static CFD_NOINLINE void ref_fill_memset_like(Framebuffer& fb, Rect r, u32 c) {
    for (int y = r.y; y < r.y + r.h; ++y) std::fill_n(fb.row(y) + r.x, r.w, c);         // lo que haría GCC -O3
}
static CFD_NOINLINE void ref_blend(Framebuffer& fb, Rect r, u32 c) {
    const u32 a = render::alpha256(c >> 24);
    for (int y = r.y; y < r.y + r.h; ++y) {
        u32* d = fb.row(y) + r.x;
        for (int x = 0; x < r.w; ++x) d[x] = render::blend_px(d[x], c, a);
    }
}
// Mezcla "de libro" con floats por canal (lo que se suele escribir primero).
static CFD_NOINLINE void ref_blend_float(Framebuffer& fb, Rect r, u32 c) {
    const float a = static_cast<float>(c >> 24) / 255.0f;
    for (int y = r.y; y < r.y + r.h; ++y) {
        u32* d = fb.row(y) + r.x;
        for (int x = 0; x < r.w; ++x) {
            u32 o = 0xFF000000u;
            for (int s = 0; s < 24; s += 8) {
                const float cs = static_cast<float>((c >> s) & 255), cd = static_cast<float>((d[x] >> s) & 255);
                o |= static_cast<u32>(cd + (cs - cd) * a + 0.5f) << s;
            }
            d[x] = o;
        }
    }
}
// Texto: bucle por bit (sin LUT ni VPMASKMOVD).
static CFD_NOINLINE void ref_text(Framebuffer& fb, int x, int y, const u8* g, int n, u32 c) {
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 16; ++r) {
            const u8 bits = font::k_small_glyphs[g[i] < 224 ? g[i] : 31][r];
            u32* d = fb.row(y + r) + x + i * 8;
            for (int b = 0; b < 8; ++b) if (bits & (0x80 >> b)) d[b] = c;
        }
}
static CFD_NOINLINE int ref_utf8_count(const char* s) {
    int n = 0;
    for (; *s; ++s) n += (static_cast<u8>(*s) & 0xC0) != 0x80;
    return n;
}
// Escalado 2× escalar (fila a fila, la segunda fila por memcpy de la primera).
static CFD_NOINLINE void ref_upscale2(const u32* src, int sw, int sh, int ss, u32* dst, int ds) {
    for (int y = 0; y < sh; ++y) {
        u32* d0 = dst + static_cast<usize>(2 * y) * static_cast<usize>(ds);
        const u32* s = src + static_cast<usize>(y) * static_cast<usize>(ss);
        for (int x = 0; x < sw; ++x) { d0[2 * x] = s[x]; d0[2 * x + 1] = s[x]; }
        std::memcpy(d0 + ds, d0, static_cast<usize>(2 * sw) * 4);
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const bool quick = argc > 1 && !std::strcmp(argv[1], "--rapido");
    const int R = quick ? 21 : 101;
    pool().start();
    std::printf("[bench] %d hilos, medianas de %d repeticiones\n", pool().size(), R);
    Framebuffer fb, fb2;
    fb.resize(1920, 1200);
    fb2.resize(1920, 1200);
    fb.clear_color(0xFF202020u);
    fb2.clear_color(0xFF202020u);
    const Rect panel{1500, 0, 420, 1200};
    Painter p(fb);

    // 1) Relleno opaco del panel (420×1200).
    {
        const double a = bench_ms(R, [&] { ref_fill(fb2, panel, 0xFF15171Cu); });
        const double b = bench_ms(R, [&] { ref_fill_memset_like(fb2, panel, 0xFF15171Cu); });
        const double c = bench_ms(R, [&] { p.fill_rect(panel, 0xFF15171Cu); });
        row("fill_rect opaco 420×1200 (escalar vs AVX2+maskstore)", a, c);
        row("fill_rect opaco (std::fill_n autovect. vs AVX2)", b, c);
    }
    // 2) Relleno translúcido (mezcla).
    {
        const u32 col = 0x80336699u;
        for (int y = 0; y < fb.h; ++y) for (int x = 0; x < fb.w; ++x) fb.row(y)[x] = fb2.row(y)[x] = 0xFF000000u | static_cast<u32>(x * 2654435761u + static_cast<u32>(y));
        p.fill_rect(panel, col);
        ref_blend(fb2, panel, col);
        bool same = true;
        for (int y = 0; y < fb.h; ++y) same &= !std::memcmp(fb.row(y) + panel.x, fb2.row(y) + panel.x, 420 * 4);
        CHECK(same);
        const double a = bench_ms(R, [&] { ref_blend_float(fb2, panel, col); });
        const double b = bench_ms(R, [&] { ref_blend(fb2, panel, col); });
        const double c = bench_ms(R, [&] { p.fill_rect(panel, col); });
        row("mezcla 420×1200: float por canal vs AVX2 16 bits", a, c);
        row("mezcla 420×1200: SWAR escalar vs AVX2 16 bits", b, c);
    }
    // 3) Texto: 200 líneas × 50 glifos.
    {
        const char* line = "Coeficiente de presión Cp: -1.234 · ñandú ¿qué? 12";
        u8 g[64];
        const int n = render::utf8_to_glyphs(line, -1, g, 64);
        fb.clear_color(0);
        fb2.clear_color(0);
        for (int k = 0; k < 60; ++k) { p.text_glyphs(20, 10 + k * 18, g, n, 0xFFFFFFFFu); ref_text(fb2, 20, 10 + k * 18, g, n, 0xFFFFFFFFu); }
        bool same = true;
        for (int y = 0; y < fb.h; ++y) same &= !std::memcmp(fb.row(y), fb2.row(y), static_cast<usize>(fb.w) * 4);
        CHECK(same);
        const double a = bench_ms(R, [&] { for (int k = 0; k < 60; ++k) ref_text(fb2, 20, 10 + k * 18, g, n, 0xFFFFFFFFu); });
        const double b = bench_ms(R, [&] { for (int k = 0; k < 60; ++k) p.text_glyphs(20, 10 + k * 18, g, n, 0xFFFFFFFFu); });
        std::printf("    (%d glifos por repetición)\n", 60 * n);
        row("texto 8×16: bucle por bit vs LUT+VPMASKMOVD", a, b);
    }
    // 4) UTF-8: contar glifos de un texto largo.
    {
        std::string s;
        for (int i = 0; i < 2000; ++i) s += "Túnel de viento — ñ ";
        const int n1 = ref_utf8_count(s.c_str()), n2 = render::utf8_count(s.c_str());
        CHECK(n1 == n2);
        const double a = bench_ms(R, [&] { volatile int k = ref_utf8_count(s.c_str()); (void)k; });
        const double b = bench_ms(R, [&] { volatile int k = render::utf8_count(s.c_str()); (void)k; });
        row("utf8_count 45 KB: byte a byte vs SWAR+popcount", a * 1e3, b * 1e3, "µs");
    }
    // 5) Primitivas del visor ficticio (tiempos absolutos, 1 hilo).
    {
        fb.clear_color(0);
        const Rect vp{0, 0, 1500, 1200};
        const double tg = bench_ms(R, [&] { p.gradient_v(vp, 0xFF2A2F38u, 0xFF0D0F13u); });
        Vec2 pts[160];
        for (int i = 0; i < 160; ++i) pts[i] = {40.0f + i * 8.8f, 500.0f + 60.0f * std::sin(i * 0.05f)};
        const double tp = bench_ms(R, [&] { p.polyline(pts, 160, 0xFF4AA8FFu, 1.6f); });
        const double tl = bench_ms(R, [&] { p.line(700, 540, 20, 1199, 0x28FFFFFFu, 1.0f); });
        const double th = bench_ms(R, [&] { p.line(0, 700, 1500, 700, 0x28FFFFFFu, 1.0f); });
        const double ts = bench_ms(R, [&] { p.shadow({580, 520, 340, 40}, 20.0f, 18.0f, 0x80000000u, false); });
        std::printf("  gradiente 1500×1200 con Bayer %.3f ms | polilínea 160 pts %.3f ms | línea diagonal %.3f ms | línea horizontal 1500 px %.3f ms | sombra %.3f ms\n",
                    tg, tp, tl, th, ts);
    }
    // 6) Escalado 1920×1200 → 3840×2400.
    {
        Buffer<u32> dst(static_cast<usize>(3840) * 2400);
        dst.zero();
        const double a = bench_ms(R, [&] { ref_upscale2(fb.color.data(), 1920, 1200, fb.stride, dst.data(), 3840); });
        const double b = bench_ms(R, [&] { platform::upscale_argb(fb.color.data(), 1920, 1200, fb.stride, dst.data(), 3840, 2400, 3840, 2, false, false); });
        const double c = bench_ms(R, [&] { platform::upscale_argb(fb.color.data(), 1920, 1200, fb.stride, dst.data(), 3840, 2400, 3840, 2, false, true); });
        const double d = bench_ms(R, [&] { platform::upscale_argb(fb.color.data(), 1920, 1200, fb.stride, dst.data(), 3840, 2400, 3840, 2, true, false); });
        const double e = bench_ms(R, [&] { platform::upscale_argb(fb.color.data(), 1920, 1200, fb.stride, dst.data(), 3840, 2400, 3840, 2, true, true); });
        row("escalado 2× 1 hilo: escalar+memcpy vs AVX2 (stores normales)", a, b);
        row("escalado 2× 1 hilo: AVX2 normal vs AVX2 NT", b, c);
        row("escalado 2× AVX2 NT: 1 hilo vs pool", c, e);
        row("escalado 2× pool: stores normales vs NT", d, e);
        std::printf("    ancho de banda efectivo (pool+NT): %.1f GB/s escritos\n", 3840.0 * 2400 * 4 / (e * 1e-3) / 1e9);
    }
    // 7) Panel de demostración completo: construcción + render paralelo vs serie.
    {
        ui::Context ui;
        ui::DemoState st;
        ui::demo_init(st);
        platform::Input in;
        in.mouse_x = panel.x + 200; in.mouse_y = 400;
        double t = 0;
        auto build = [&] { ui.begin_frame(in, fb.w, fb.h, t); ui::demo_panel(ui, st, panel, 60.0); ui.end_frame(); t += 1.0 / 60; };
        const double tb = bench_ms(R, build);
        build();
        const double tpar = bench_ms(R, [&] { const render::DrawList* l0 = &ui.draw(); render::DrawList::render(fb, &l0, 1, 0); });
        // Serie: reproducir con un Painter (sin franjas). Se usa la capa principal vía render con 1 franja.
        const double tser = bench_ms(R, [&] {
            // 1 franja = un solo hilo recorre todos los comandos
            const render::DrawList* l0 = &ui.draw();
            render::DrawList::render(fb, &l0, 1, 1);
        });
        std::printf("  panel demo: construcción %.3f ms (%d widgets, %zu comandos)\n", tb, ui.stats().widgets, ui.stats().cmds);
        row("render panel: 1 franja (serie) vs franjas en paralelo", tser, tpar);
        // Visor ficticio (DrawList paralela).
        const double tv = bench_ms(R, [&] { ui::demo_viewport(fb, {0, 0, 1500, 1200}, 60, 0.1); });
        std::printf("  visor ficticio completo (DrawList paralela): %.3f ms\n", tv);
    }
    std::printf("\n%s (%d fallos)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
