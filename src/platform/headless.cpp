// ============================================================================
//  platform/headless.cpp — ventana ficticia (tests, benchmarks, capturas PNG).
//
//  No abre nada: poll() sólo informa del tamaño del framebuffer y present()
//  cuenta cuadros. Los tests de la UI construyen `Input` a mano para simular
//  secuencias de ratón/teclado.
// ============================================================================
#include "platform.hpp"
#include "../core/util.hpp"

#include <cstring>

namespace cfd::platform {
namespace {

class HeadlessWindow final : public Window {
public:
    bool open(const char* title, int fb_w, int fb_h, int pixel_scale) override {
        set_title(title);
        fb_w_ = fb_w > 0 ? fb_w : 1;
        fb_h_ = fb_h > 0 ? fb_h : 1;
        scale_ = pixel_scale > 0 ? pixel_scale : 1;
        first_ = true;
        return true;
    }
    void poll(Input& in) override {
        in.fb_w = fb_w_;
        in.fb_h = fb_h_;
        if (first_) { in.resized = true; first_ = false; }
    }
    void present(const render::Framebuffer& fb) override {
        const double t0 = now_sec();
        (void)fb;
        ++stats_.frames;
        stats_.total_ms = (now_sec() - t0) * 1e3;
        stats_.win_w = fb_w_ * scale_;
        stats_.win_h = fb_h_ * scale_;
    }
    void set_title(const char* title) override {
        std::snprintf(title_, sizeof title_, "%s", title ? title : "");
    }
    int pixel_scale() const override { return scale_; }
    bool is_headless() const override { return true; }
    PresentStats present_stats() const override { return stats_; }

private:
    int fb_w_ = 1, fb_h_ = 1, scale_ = 1;
    bool first_ = true;
    char title_[256] = {};
    PresentStats stats_;
};

} // namespace

std::unique_ptr<Window> create_headless_window() { return std::make_unique<HeadlessWindow>(); }

} // namespace cfd::platform
