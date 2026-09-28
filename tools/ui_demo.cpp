// ============================================================================
//  tools/ui_demo.cpp — demo interactiva X11 del panel de UI + visor ficticio.
//
//    build/ui/ui_demo              ventana (escala automática: 2× en 3840×2400)
//    build/ui/ui_demo --frames 60  sale sola tras N cuadros e imprime tiempos
//    build/ui/ui_demo --headless --frames 300   sin ventana (sólo UI)
//    build/ui/ui_demo --scale 1 --size 1280x800
//  Teclas: F11 pantalla completa, Esc (sin foco en la UI) salir.
// ============================================================================
#include "../src/core/util.hpp"
#include "../src/platform/platform.hpp"
#include "../src/ui/demo.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace cfd;

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[static_cast<usize>(p * static_cast<double>(v.size() - 1))];
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    long frames = -1;
    bool headless = false;
    int scale = 0, fw = 1920, fh = 1200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--headless")) headless = true;
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) scale = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--size") && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &fw, &fh);
        else { std::printf("uso: %s [--frames N] [--headless] [--scale S] [--size WxH]\n", argv[0]); return 1; }
    }
    pool().start();
    std::unique_ptr<platform::Window> win = headless ? nullptr : platform::create_x11_window();
    if (!win) {
        if (!headless) std::printf("[ui_demo] sin servidor X: modo headless\n");
        win = platform::create_headless_window();
    }
    if (scale <= 0) scale = win->is_headless() ? 1 : platform::detect_pixel_scale();
    if (!win->open("Túnel de viento CFD — demo de UI", fw, fh, scale)) {
        std::printf("[ui_demo] no se pudo abrir la ventana\n");
        return 1;
    }
    render::Framebuffer fb;
    fb.resize(fw, fh);
    ui::Context ui;
    if (fb.h >= 2000 && scale == 1) ui.set_scale(2.0f);   // framebuffer nativo 4K sin escalado → UI ×2
    ui::DemoState st;
    ui::demo_init(st);
    platform::Input in;
    double t_prev = now_sec(), fps = 60.0;
    std::vector<double> t_ui, t_vp, t_present, t_up, t_put, t_wait, t_frame;
    long n = 0;
    bool fullscreen = false;
    while (frames < 0 || n < frames) {
        const double t0 = now_sec();
        in.begin_frame();
        win->poll(in);
        if (in.quit) break;
        if (in.resized && in.fb_w > 0 && in.fb_h > 0) fb.resize(in.fb_w, in.fb_h);
        const int pw = std::min(static_cast<int>(420 * ui.style().scale), fb.w / 2);
        const render::Rect panel{fb.w - pw, 0, pw, fb.h};
        const render::Rect vp{0, 0, fb.w - pw, fb.h};
        ui.set_toast_area(vp);
        const double t1 = now_sec();
        ui.begin_frame(in, fb.w, fb.h, t1);
        ui::demo_panel(ui, st, panel, fps);
        ui.end_frame();
        if (!ui.wants_keyboard()) {
            if (in.key_pressed[platform::KeyEscape]) break;
            if (in.key_pressed[platform::KeyF11]) { fullscreen = !fullscreen; win->set_fullscreen(fullscreen); }
        }
        const double t2 = now_sec();
        ui::demo_viewport(fb, vp, fps, ui.stats().build_ms + ui.stats().render_ms);
        const double t3 = now_sec();
        ui.render(fb);
        const double t4 = now_sec();
        win->present(fb);
        const double t5 = now_sec();
        ui::demo_tick(st, t5 - t_prev);
        const double dt = t5 - t_prev;
        t_prev = t5;
        if (dt > 0) fps += (1.0 / dt - fps) * 0.1;
        if (n >= 5) {                                   // descarta el arranque
            t_ui.push_back((t2 - t1 + t4 - t3) * 1e3);
            t_vp.push_back((t3 - t2) * 1e3);
            t_present.push_back((t5 - t4) * 1e3);
            const platform::PresentStats ps = win->present_stats();
            t_up.push_back(ps.upscale_ms);
            t_put.push_back(ps.put_ms);
            t_wait.push_back(ps.wait_ms);
            t_frame.push_back((t5 - t0) * 1e3);
        }
        ++n;
    }
    const platform::PresentStats ps = win->present_stats();
    std::printf("[ui_demo] %ld cuadros, fb %dx%d → ventana %dx%d (escala %d, %s)\n", n, fb.w, fb.h, ps.win_w, ps.win_h,
                win->pixel_scale(), win->is_headless() ? "headless" : (ps.shm ? "MIT-SHM" : "XPutImage"));
    std::printf("[ui_demo] medianas: UI %.3f ms | visor %.3f ms | present %.3f ms (escalado %.3f, put %.3f, espera %.3f) | cuadro %.3f ms\n",
                median(t_ui), median(t_vp), median(t_present), median(t_up), median(t_put), median(t_wait), median(t_frame));
    std::printf("[ui_demo] p10/p50/p90: present %.2f/%.2f/%.2f  escalado %.2f/%.2f/%.2f  put %.2f/%.2f/%.2f  espera %.2f/%.2f/%.2f ms\n",
                pct(t_present, .1), pct(t_present, .5), pct(t_present, .9), pct(t_up, .1), pct(t_up, .5), pct(t_up, .9),
                pct(t_put, .1), pct(t_put, .5), pct(t_put, .9), pct(t_wait, .1), pct(t_wait, .5), pct(t_wait, .9));
    return 0;
}
