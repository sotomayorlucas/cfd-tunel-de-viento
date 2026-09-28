// ============================================================================
//  ui/ui.cpp — implementación de la GUI inmediata.
//
//  Ruta caliente = construir ~40 widgets + 2 gráficas por cuadro: sólo aritmética
//  entera, hashing FNV-1a, snprintf y grabación de comandos POD en DrawList
//  (vectores reutilizados: cero asignaciones en régimen estacionario). El coste
//  de píxeles se paga en render() en paralelo por franjas.
// ============================================================================
#include "ui.hpp"
#include "../core/util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cfd::ui {

using render::DrawList;
using render::Font;
using render::HAlign;
using render::VAlign;
using render::font_center;
using render::font_h;
using render::font_w;
using render::mul_alpha;
using render::with_alpha;

struct Context::PanelState {
    Id id = 0;
    Rect rect{}, view{};
    float scroll = 0, target = 0;
    int content_h = 0, max_scroll = 0;
    u32 frame = 0;
};
static constexpr int k_max_panels = 8;
static constexpr float k_tooltip_delay = 0.5f;

// ---------------------------------------------------------------------------------------
Style Style::dark(float s) {
    Style st;
    st.scale = s;
    auto S = [s](int v) { return max_(1, static_cast<int>(std::lround(static_cast<float>(v) * s))); };
    st.pad = S(12); st.spacing = S(5); st.row_h = S(24); st.text_h = S(18); st.header_h = S(30);
    st.radius = S(4); st.check = S(16); st.scrollbar_w = S(6); st.indent = S(8);
    st.font = s >= 1.5f ? Font::Large : Font::Small;
    st.font_big = Font::Large;
    return st;
}

Context::Context() {
    st_ = Style::dark(1.0f);
    panels_ = new PanelState[k_max_panels];
    slots_ = static_cast<Slot*>(std::calloc(k_slots, sizeof(Slot)));
    id_stack_[0] = 0xcbf29ce484222325ull;
    id_depth_ = 1;
    scratch_.reserve(8192);
#ifndef NDEBUG
    seen_ = static_cast<Id*>(std::calloc(1024, sizeof(Id)));
#endif
}
Context::~Context() {
    delete[] panels_;
    std::free(slots_);
    std::free(seen_);
}

void Context::set_scale(float s) {
    const Style d = Style::dark(s);
    st_.scale = d.scale; st_.pad = d.pad; st_.spacing = d.spacing; st_.row_h = d.row_h; st_.text_h = d.text_h;
    st_.header_h = d.header_h; st_.radius = d.radius; st_.check = d.check; st_.scrollbar_w = d.scrollbar_w;
    st_.indent = d.indent; st_.font = d.font; st_.font_big = d.font_big;
}

// ---- Ids / almacenamiento ---------------------------------------------------------------
Id Context::get_id(const char* label) const {
    const char* p = label ? label : "";
    if (const char* t = std::strstr(p, "###")) p = t;
    u64 h = id_stack_[id_depth_ - 1];
    for (; *p; ++p) { h ^= static_cast<u8>(*p); h *= 0x100000001b3ull; }
    return h ? h : 1;
}
void Context::push_id(const char* s) {
    CFD_CHECK(id_depth_ < 32, "ui: pila de ids llena");
    const Id h = get_id(s);
    id_stack_[id_depth_++] = h;
}
void Context::push_id(int i) {
    CFD_CHECK(id_depth_ < 32, "ui: pila de ids llena");
    u64 h = id_stack_[id_depth_ - 1];
    for (int k = 0; k < 4; ++k) { h ^= static_cast<u8>(static_cast<u32>(i) >> (8 * k)); h *= 0x100000001b3ull; }
    id_stack_[id_depth_++] = h ? h : 1;
}
void Context::pop_id() { if (id_depth_ > 1) --id_depth_; }

int Context::display_len(const char* label) const {
    if (!label) return 0;
    const char* h = std::strstr(label, "##");
    return h ? static_cast<int>(h - label) : static_cast<int>(std::strlen(label));
}

Context::Slot* Context::slot(Id id) {
    const u32 m = k_slots - 1;
    u32 i = static_cast<u32>(id ^ (id >> 31)) & m;
    for (int probe = 0; probe < k_slots; ++probe, i = (i + 1) & m) {
        if (slots_[i].key == id) return &slots_[i];
        if (slots_[i].key == 0) return &slots_[i];
    }
    return &slots_[0];
}
float* Context::state_f(Id id, float init) {
    Slot* s = slot(id);
    if (s->key != id) { s->key = id; s->f = init; s->i = 0; }
    return &s->f;
}
int* Context::state_i(Id id, int init) {
    Slot* s = slot(id);
    if (s->key != id) { s->key = id; s->i = init; s->f = 0; }
    return &s->i;
}
Context::PanelState* Context::panel_state(Id id) {
    for (int i = 0; i < npanels_; ++i) if (panels_[i].id == id) return &panels_[i];
    int k = npanels_ < k_max_panels ? npanels_++ : 0;
    if (npanels_ == k_max_panels) {              // reutiliza el más antiguo
        u32 oldest = ~0u;
        for (int i = 0; i < npanels_; ++i) if (panels_[i].frame < oldest) { oldest = panels_[i].frame; k = i; }
    }
    panels_[k] = PanelState{};
    panels_[k].id = id;
    return &panels_[k];
}

const char* Context::fmtv(const char* f, va_list ap) {
    std::vsnprintf(fmtbuf_, sizeof fmtbuf_, f, ap);
    return fmtbuf_;
}
const char* Context::fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* r = fmtv(f, ap);
    va_end(ap);
    return r;
}

// ---- Cuadro -------------------------------------------------------------------------------
static CFD_INLINE bool in_rects(const Rect* rs, int n, int x, int y) {
    for (int i = 0; i < n; ++i) if (rs[i].contains(x, y)) return true;
    return false;
}

void Context::begin_frame(const platform::Input& in, int fb_w, int fb_h, double t) {
    t_begin_ = now_sec();
    in_ = in;
    dt_ = frame_ ? clamp_(t - time_, 0.0, 0.25) : 1.0 / 60.0;
    time_ = t;
    ++frame_;
    fb_w_ = fb_w; fb_h_ = fb_h;
    for (DrawList& l : layers_) { l.clear(); l.set_base_clip({0, 0, fb_w, fb_h}); }
    std::memcpy(ui_rects_prev_, ui_rects_, sizeof(Rect) * static_cast<usize>(ui_nrects_));
    ui_nrects_prev_ = ui_nrects_;
    ui_nrects_ = 0;
    hot_id_ = 0;
    active_alive_ = false;
    popup_alive_ = false;
    edit_alive_ = false;
    focus_n_ = 0;
    last_id_ = 0;
    last_hovered_ = false;
    tip_pending_ = false;
    click_consumed_ = false;
    popup_opened_now_ = false;
    cur_layer_ = 0;
    in_popup_ = false;
    row_cols_ = 0;
    stats_.widgets = 0;
    if (seen_) std::memset(seen_, 0, 1024 * sizeof(Id));

    const int mx = in_.mouse_x, my = in_.mouse_y;
    const bool any_pressed = in_.mouse_pressed[0] || in_.mouse_pressed[1] || in_.mouse_pressed[2];
    const bool over_ui = in_rects(ui_rects_prev_, ui_nrects_prev_, mx, my);
    // Propiedad del arrastre: quien recibe la pulsación se queda el ratón hasta soltar.
    if (any_pressed && !ui_owns_ && !app_owns_) {
        if (over_ui || popup_id_) ui_owns_ = true;
        else {
            app_owns_ = true;
            // Clic en el visor: se abandona la navegación por teclado (si no, tras un Tab
            // wants_keyboard() quedaba a true para siempre y la app no recibía teclas).
            focus_visible_ = false;
        }
    }
    // Clic fuera del popup: lo cierra y se consume (no activa lo de debajo).
    if (popup_id_ && any_pressed && !popup_rect_.contains(mx, my)) {
        popup_id_ = 0;
        click_consumed_ = true;
    }
    // Esc que usa la UI (cancelar edición, cerrar popup, quitar el foco visible) queda
    // CONSUMIDO: wants_keyboard() es true este cuadro para que la app no lo use también
    // (p.ej. salir de la aplicación al cerrar un desplegable con Esc).
    esc_consumed_ = false;
    if (in_.key_pressed[platform::KeyEscape]) {
        if (edit_id_) { edit_cancel_ = true; esc_consumed_ = true; }
        else if (popup_id_) { popup_id_ = 0; esc_consumed_ = true; }
        else {
            esc_consumed_ = focus_visible_ && focus_id_ != 0;
            focus_id_ = 0; focus_visible_ = false;
        }
    }
}

void Context::end_frame() {
    flush_row();
    if (active_id_ && !active_alive_) clear_active();
    if (popup_id_ && !popup_alive_) popup_id_ = 0;
    if (edit_id_ && !edit_alive_) edit_id_ = 0;
    // Navegación con Tab por los widgets de este cuadro.
    if (in_.key_pressed[platform::KeyTab] && !edit_id_ && focus_n_ > 0) {
        int idx = -1;
        for (int i = 0; i < focus_n_; ++i) if (focus_list_[i] == focus_id_) { idx = i; break; }
        if (idx < 0) idx = in_.shift ? 0 : focus_n_ - 1;
        idx = (idx + (in_.shift ? focus_n_ - 1 : 1)) % focus_n_;
        focus_id_ = focus_list_[idx];
        focus_visible_ = true;
        focus_scroll_ = true;
        popup_id_ = 0;
    }
    const int mx = in_.mouse_x, my = in_.mouse_y;
    const bool over_now = in_rects(ui_rects_, ui_nrects_, mx, my);
    wants_mouse_ = ui_owns_ || active_id_ != 0 || popup_id_ != 0 || (!app_owns_ && over_now);
    wants_keyboard_ = esc_consumed_ || edit_id_ != 0 || (focus_visible_ && focus_id_ != 0);
    if (!in_.mouse_down[0] && !in_.mouse_down[1] && !in_.mouse_down[2]) ui_owns_ = app_owns_ = false;
    last_hot_ = hot_id_;
    draw_tooltip();
    draw_toasts();
    stats_.cmds = layers_[0].size() + layers_[1].size() + layers_[2].size();
    stats_.build_ms = (now_sec() - t_begin_) * 1e3;
}

void Context::render(render::Framebuffer& fb) {
    const double t0 = now_sec();
    const DrawList* ls[3] = {&layers_[0], &layers_[1], &layers_[2]};
    DrawList::render(fb, ls, 3);
    stats_.render_ms = (now_sec() - t0) * 1e3;
}

// ---- Interacción ---------------------------------------------------------------------------
void Context::set_active(Id id) { active_id_ = id; active_alive_ = true; }
void Context::clear_active() { active_id_ = 0; }

bool Context::hoverable(Rect r) const {
    if (app_owns_) return false;
    const int mx = in_.mouse_x, my = in_.mouse_y;
    if (!r.contains(mx, my)) return false;
    if (!layers_[cur_layer_].clip().contains(mx, my)) return false;
    if (popup_id_ && !in_popup_ && popup_rect_.contains(mx, my)) return false;
    return true;
}

void Context::ensure_visible(Rect r) {
    PanelState* ps = cur_panel_;
    if (!ps) return;
    const int top = ps->rect.y + st_.pad, bot = ps->rect.y + ps->rect.h - st_.pad;
    if (r.y < top) ps->target -= static_cast<float>(top - r.y);
    else if (r.y + r.h > bot) ps->target += static_cast<float>(r.y + r.h - bot);
}

bool Context::item_add(Id id, Rect r, bool focusable) {
    ++stats_.widgets;
    if (seen_ && id) {                              // depuración: mismo id dos veces en un cuadro
        u32 i = static_cast<u32>(id) & 1023;
        for (int k = 0; k < 1024 && seen_[i]; ++k, i = (i + 1) & 1023)
            if (seen_[i] == id) {
                if (!dup_warned_) std::fprintf(stderr, "[ui] aviso: id duplicado %016llx (usar \"##sufijo\" o push_id)\n", static_cast<unsigned long long>(id));
                dup_warned_ = true;
                break;
            }
        if (!seen_[i]) seen_[i] = id;
    }
    last_id_ = id;
    last_rect_ = r;
    if (focusable && focus_n_ < 512) focus_list_[focus_n_++] = id;
    if (focus_scroll_ && focus_visible_ && focus_id_ == id) { ensure_visible(r); focus_scroll_ = false; }
    const bool hov = hoverable(r) && (active_id_ == 0 || active_id_ == id);
    if (hov) {
        hot_id_ = id;
        if (id != last_hot_) hover_t0_ = time_;
    }
    last_hovered_ = hov;
    return hov;
}

bool Context::key_activate(Id id) const {
    return focus_visible_ && focus_id_ == id && !edit_id_ &&
           (in_.key_pressed[platform::KeyEnter] || in_.key_pressed[' ']);
}

bool Context::press_behavior(Id id, Rect r, bool* out_hov, bool* out_held, bool focusable) {
    const bool hov = item_add(id, r, focusable);
    bool pressed = false;
    if (hov && !click_consumed_ && in_.mouse_pressed[0]) {
        set_active(id);
        if (focus_id_ != id) focus_visible_ = false;
        focus_id_ = id;
    }
    if (active_id_ == id) {
        active_alive_ = true;
        if (!in_.mouse_down[0]) {                    // soltado (o pulsado y soltado en el mismo cuadro)
            if (hov && (in_.mouse_released[0] || in_.mouse_pressed[0])) pressed = true;
            clear_active();
        }
    }
    if (key_activate(id)) pressed = true;
    if (out_hov) *out_hov = hov;
    if (out_held) *out_held = active_id_ == id;
    return pressed;
}

void Context::focus_ring(Id id, Rect r) {
    if (!(focus_visible_ && focus_id_ == id)) return;
    layers_[cur_layer_].round_rect({r.x - 2, r.y - 2, r.w + 4, r.h + 4}, static_cast<float>(st_.radius + 2), st_.accent, 1);
}

// ---- Distribución ----------------------------------------------------------------------------
void Context::flush_row() {
    if (row_cols_ > 0 && row_idx_ > 0) cur_y_ += row_maxh_ + st_.spacing;
    row_cols_ = 0; row_idx_ = 0; row_maxh_ = 0;
}
void Context::row(int ncols) {
    flush_row();
    row_cols_ = max_(ncols, 0);
}
Rect Context::next_rect(int h) {
    if (row_cols_ > 0) {
        const int gap = st_.spacing + 1;
        const int w = (cur_w_ - gap * (row_cols_ - 1)) / row_cols_;
        const int x = cur_x_ + row_idx_ * (w + gap);
        const int ww = row_idx_ == row_cols_ - 1 ? cur_x_ + cur_w_ - x : w;
        const Rect r{x, cur_y_, ww, h};
        row_maxh_ = max_(row_maxh_, h);
        if (++row_idx_ >= row_cols_) flush_row();
        return r;
    }
    const Rect r{cur_x_, cur_y_, cur_w_, h};
    cur_y_ += h + st_.spacing;
    return r;
}
void Context::spacing(int px) { flush_row(); cur_y_ += px < 0 ? st_.spacing * 2 : px; }
void Context::indent(int px) { const int d = px < 0 ? st_.indent : px; cur_x_ += d; cur_w_ -= d; }
void Context::unindent(int px) { const int d = px < 0 ? st_.indent : px; cur_x_ -= d; cur_w_ += d; }
void Context::separator() {
    flush_row();
    const Rect r = next_rect(st_.spacing + 1);
    layers_[0].fill_rect({r.x, r.y + st_.spacing / 2, r.w, 1}, st_.separator);
}

// ---- Paneles ------------------------------------------------------------------------------------
bool Context::begin_panel(const char* name, Rect r) {
    const Id id = get_id(name);
    PanelState* ps = panel_state(id);
    ps->rect = r;
    ps->frame = frame_;
    DrawList& dl = layers_[0];
    dl.fill_rect(r, st_.panel);
    dl.fill_rect({r.x, r.y, 1, r.h}, st_.panel_edge);
    if (ui_nrects_ < 16) ui_rects_[ui_nrects_++] = r;
    const int mx = in_.mouse_x, my = in_.mouse_y;
    const bool over = r.contains(mx, my) && !(popup_id_ && popup_rect_.contains(mx, my)) && !app_owns_;
    const int visible = r.h - 2 * st_.pad;
    ps->max_scroll = max_(0, ps->content_h - visible);
    if (over && in_.wheel != 0.0f && !edit_id_) ps->target -= in_.wheel * static_cast<float>(st_.row_h * 3);
    ps->target = clamp_(ps->target, 0.0f, static_cast<float>(ps->max_scroll));
    if (st_.smooth_scroll) {
        const float k = 1.0f - std::exp(-static_cast<float>(dt_) * 18.0f);
        ps->scroll += (ps->target - ps->scroll) * k;
        if (std::fabs(ps->target - ps->scroll) < 0.5f) ps->scroll = ps->target;
    } else {
        ps->scroll = ps->target;
    }
    ps->view = {r.x + 1, r.y, r.w - 1, r.h};
    dl.push_clip(ps->view);
    cur_panel_ = ps;
    const bool bar = ps->max_scroll > 0;
    cur_x_ = r.x + st_.pad;
    cur_w_ = r.w - 2 * st_.pad - (bar ? st_.scrollbar_w + 2 : 0);
    cur_y_ = r.y + st_.pad - static_cast<int>(std::lround(ps->scroll));
    row_cols_ = 0;
    push_id(name);
    return true;
}

void Context::end_panel() {
    PanelState* ps = cur_panel_;
    if (!ps) return;
    flush_row();
    pop_id();
    const Rect r = ps->rect;
    const int top = r.y + st_.pad - static_cast<int>(std::lround(ps->scroll));
    ps->content_h = max_(0, cur_y_ - st_.spacing - top);
    const int visible = r.h - 2 * st_.pad;
    ps->max_scroll = max_(0, ps->content_h - visible);
    ps->target = clamp_(ps->target, 0.0f, static_cast<float>(ps->max_scroll));
    ps->scroll = clamp_(ps->scroll, 0.0f, static_cast<float>(ps->max_scroll));
    DrawList& dl = layers_[0];
    if (ps->max_scroll > 0) {
        const Rect track{r.x + r.w - st_.scrollbar_w - 3, r.y + 4, st_.scrollbar_w, r.h - 8};
        const float frac = static_cast<float>(visible) / static_cast<float>(ps->content_h);
        const int th = max_(st_.row_h, static_cast<int>(static_cast<float>(track.h) * frac));
        const int ty = track.y + static_cast<int>(static_cast<float>(track.h - th) * ps->scroll / static_cast<float>(ps->max_scroll));
        const Rect thumb{track.x, ty, track.w, th};
        const Id sid = get_id("##barra_desplazamiento") ^ ps->id;
        const Rect hit{track.x - 4, track.y, track.w + 7, track.h};
        const bool hov = item_add(sid, hit, false);
        if (hov && !click_consumed_ && in_.mouse_pressed[0]) {
            set_active(sid);
            if (thumb.contains(in_.mouse_x, in_.mouse_y) || (in_.mouse_x >= thumb.x - 4 && in_.mouse_y >= thumb.y && in_.mouse_y < thumb.y + thumb.h))
                drag_grab_ = static_cast<float>(in_.mouse_y - thumb.y);
            else
                drag_grab_ = static_cast<float>(th) * 0.5f;          // clic en la pista: salta ahí
        }
        if (active_id_ == sid) {
            active_alive_ = true;
            if (in_.mouse_down[0]) {
                const float t = (static_cast<float>(in_.mouse_y - track.y) - drag_grab_) / static_cast<float>(max_(1, track.h - th));
                ps->target = ps->scroll = clamp_(t, 0.0f, 1.0f) * static_cast<float>(ps->max_scroll);
            } else {
                clear_active();
            }
        }
        const u32 tc = active_id_ == sid ? st_.accent : (hov ? st_.text_dim : st_.border);
        dl.fill_round_rect(track, track.w * 0.5f, with_alpha(st_.frame, 0x80));
        dl.fill_round_rect(thumb, track.w * 0.5f, tc);
    }
    dl.pop_clip();
    cur_panel_ = nullptr;
}

// ---- Texto ----------------------------------------------------------------------------------------
// Etiqueta a la izquierda de r, recortada con "…" si no cabe.
void Context::label_left(Rect r, const char* label, int nbytes, u32 color) {
    if (!label || nbytes <= 0) return;
    const int fw = st_.fw();
    const int maxg = r.w / fw;
    const int ng = render::utf8_count(label, nbytes);
    const int y = r.y + r.h / 2 - font_center(st_.font);
    DrawList& dl = layers_[cur_layer_];
    if (ng <= maxg) { dl.text(r.x, y, label, color, st_.font, nbytes); return; }
    if (maxg <= 1) return;
    const int cut = render::utf8_prefix_bytes(label, nbytes, maxg - 1);
    const int x = dl.text(r.x, y, label, color, st_.font, cut);
    dl.text(x, y, "…", color, st_.font);
}

void Context::title(const char* text, const char* subtitle) {
    flush_row();
    const int hb = font_h(st_.font_big);
    const Rect r = next_rect(hb - 6 + (subtitle ? st_.text_h + 2 : 0));
    DrawList& dl = layers_[0];
    if (st_.font_big == st_.font) dl.text(r.x + 1, r.y - 5, text, st_.text_strong, st_.font_big);   // negrita falsa
    dl.text(r.x, r.y - 5, text, st_.text_strong, st_.font_big);
    if (subtitle) dl.text(r.x + 1, r.y + hb - 4, subtitle, st_.text_dim, st_.font);
    item_add(get_id(text), r, false);
}

bool Context::header(const char* label, bool default_open) {
    flush_row();
    const Id id = get_id(label) * 0x9E3779B97F4A7C15ull + 0x5EC7u;   // id salado de sección
    int* open = state_i(id, default_open ? 1 : 0);
    cur_y_ += st_.spacing;
    const Rect pr = cur_panel_ ? cur_panel_->rect : Rect{cur_x_ - st_.pad, cur_y_, cur_w_ + 2 * st_.pad, st_.header_h};
    const Rect r{pr.x + 1, cur_y_, pr.w - 1 - (cur_panel_ && cur_panel_->max_scroll > 0 ? st_.scrollbar_w + 5 : 0), st_.header_h};
    cur_y_ += st_.header_h + st_.spacing + 2;
    bool hov = false;
    if (press_behavior(id, r, &hov, nullptr)) *open = !*open;
    DrawList& dl = layers_[0];
    dl.fill_rect(r, hov ? st_.header_hover : st_.header);
    dl.fill_rect({r.x, r.y, r.w, 1}, st_.panel_edge);
    if (*open) dl.fill_rect({r.x, r.y + 1, 3, r.h - 1}, st_.accent);
    // Chevrón: ▾ abierto / ▸ cerrado (triángulo antialias).
    const float cx = static_cast<float>(cur_x_ + 5), cy = static_cast<float>(r.y) + static_cast<float>(r.h) * 0.5f;
    const float s = 4.0f * st_.scale;
    const u32 cc = *open ? st_.accent : st_.text_dim;
    if (*open) dl.fill_triangle({cx - s, cy - s * 0.55f}, {cx + s, cy - s * 0.55f}, {cx, cy + s * 0.65f}, cc);
    else dl.fill_triangle({cx - s * 0.55f, cy - s}, {cx + s * 0.65f, cy}, {cx - s * 0.55f, cy + s}, cc);
    const int n = display_len(label);
    const int ty = r.y + r.h / 2 - font_center(st_.font);
    dl.text(cur_x_ + static_cast<int>(16 * st_.scale), ty, label, *open ? st_.text_strong : st_.text, st_.font, n);
    focus_ring(id, {r.x + 2, r.y + 2, r.w - 4, r.h - 4});
    return *open != 0;
}

void Context::label(const char* t, u32 color) {
    const Rect r = next_rect(st_.text_h);
    label_left(r, t, display_len(t), color ? color : st_.text);
}

void Context::text(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    const Rect r = next_rect(st_.text_h);
    label_left(r, s, static_cast<int>(std::strlen(s)), st_.text);
}
void Context::text_colored(u32 color, const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    const Rect r = next_rect(st_.text_h);
    label_left(r, s, static_cast<int>(std::strlen(s)), color);
}

// Ajuste de línea por palabras con fuente de ancho fijo (conteo de glifos UTF-8).
template <class F>
static void wrap_lines(const char* s, int max_glyphs, F&& emit) {
    if (max_glyphs < 4) max_glyphs = 4;
    const int n = static_cast<int>(std::strlen(s));
    int start = 0;
    while (start < n) {
        while (start < n && s[start] == ' ') ++start;
        if (start >= n) break;
        int g = 0, i = start, last_space = -1, end = n;
        bool newline = false;
        for (; i < n; ++i) {
            if (s[i] == '\n') { end = i; newline = true; break; }
            if ((static_cast<u8>(s[i]) & 0xC0) == 0x80) continue;
            if (s[i] == ' ') last_space = i;
            if (g == max_glyphs) { end = last_space > start ? last_space : i; break; }
            ++g;
        }
        if (i >= n && !newline) end = n;
        emit(s + start, end - start);
        start = end + (newline ? 1 : 0);
    }
}

void Context::wrapped_core(u32 color, const char* s) {
    flush_row();
    const int maxg = cur_w_ / st_.fw();
    wrap_lines(s, maxg, [&](const char* p, int nb) {
        const Rect r = next_rect(st_.text_h - 2);
        layers_[0].text(r.x, r.y + (r.h - st_.fh()) / 2 + 1, p, color, st_.font, nb);
    });
}
void Context::text_wrapped(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    wrapped_core(st_.text_dim, s);
}
void Context::text_wrapped_colored(u32 color, const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    wrapped_core(color, s);
}

void Context::value(const char* label, const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    const Rect r = next_rect(st_.text_h + 1);
    const int vw = render::text_width(s, st_.font);
    layers_[0].text(r.x + r.w - vw, r.y + r.h / 2 - font_center(st_.font), s, st_.text_strong, st_.font);
    label_left({r.x, r.y, r.w - vw - st_.fw(), r.h}, label, display_len(label), st_.text_dim);
}
void Context::value_colored(const char* label, u32 color, const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    const char* s = fmtv(f, ap);
    va_end(ap);
    const Rect r = next_rect(st_.text_h + 1);
    const int vw = render::text_width(s, st_.font);
    layers_[0].text(r.x + r.w - vw, r.y + r.h / 2 - font_center(st_.font), s, color, st_.font);
    label_left({r.x, r.y, r.w - vw - st_.fw(), r.h}, label, display_len(label), st_.text_dim);
}

void Context::metric(const char* label, const char* value, const char* unit, u32 color) {
    const int hb = font_h(st_.font_big);
    const int ip = static_cast<int>(8 * st_.scale);
    const Rect r = next_rect(ip + st_.fh() + hb - 8 + ip / 2);
    DrawList& dl = layers_[0];
    dl.fill_round_rect(r, static_cast<float>(st_.radius + 1), 0xFF1A1D24u);
    dl.round_rect(r, static_cast<float>(st_.radius + 1), st_.panel_edge, 1);
    const u32 vc = color ? color : st_.text_strong;
    dl.fill_round_rect({r.x, r.y + ip, 3, r.h - 2 * ip}, 1.5f, vc);
    label_left({r.x + ip + 2, r.y + ip - 3, r.w - ip - 4, st_.fh()}, label, display_len(label), st_.text_dim);
    const int vy = r.y + ip + st_.fh() - 6;
    const int x = dl.text(r.x + ip + 2, vy, value, vc, st_.font_big);
    if (unit) dl.text(x + 4, vy + render::font_baseline(st_.font_big) - render::font_baseline(st_.font), unit, st_.text_dim, st_.font);
    item_add(get_id(label), r, false);
}

// ---- Botones y casillas -----------------------------------------------------------------------------
bool Context::button(const char* label, ButtonKind kind) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    bool hov = false, held = false;
    const bool pressed = press_behavior(id, r, &hov, &held);
    u32 bg, fg = st_.text, bd = st_.border;
    switch (kind) {
        case ButtonKind::Primary:
            bg = held ? st_.accent_dim : (hov ? st_.accent_hover : st_.accent); fg = st_.accent_text; bd = 0; break;
        case ButtonKind::Danger:
            bg = held ? 0xFF6B2525u : (hov ? 0xFF8A2E2Eu : 0xFF5A2226u); fg = 0xFFFFD9D9u; bd = 0xFF7A3036u; break;
        case ButtonKind::Subtle:
            bg = held ? st_.frame_active : (hov ? st_.frame_hover : 0); bd = 0; break;
        default:
            bg = held ? st_.frame_active : (hov ? st_.frame_hover : st_.frame);
            if (hov) bd = 0xFF3E4452u;
            break;
    }
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    if (bg) dl.fill_round_rect(r, rad, bg);
    if (bd) dl.round_rect(r, rad, bd, 1);
    const int n = display_len(label);
    const int tw = render::text_width(label, n, st_.font);
    dl.text(r.x + (r.w - tw) / 2, r.y + r.h / 2 - font_center(st_.font) + (held ? 1 : 0), label, fg, st_.font, n);
    focus_ring(id, r);
    return pressed;
}

bool Context::toggle_button(const char* label, bool* v) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    bool hov = false, held = false;
    const bool pressed = press_behavior(id, r, &hov, &held);
    if (pressed) *v = !*v;
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    if (*v) {
        dl.fill_round_rect(r, rad, held ? st_.accent_dim : (hov ? st_.accent_hover : st_.accent));
    } else {
        dl.fill_round_rect(r, rad, held ? st_.frame_active : (hov ? st_.frame_hover : st_.frame));
        dl.round_rect(r, rad, hov ? 0xFF3E4452u : st_.border, 1);
    }
    const int n = display_len(label);
    const int tw = render::text_width(label, n, st_.font);
    dl.text(r.x + (r.w - tw) / 2, r.y + r.h / 2 - font_center(st_.font), label, *v ? st_.accent_text : st_.text, st_.font, n);
    focus_ring(id, r);
    return pressed;
}

bool Context::toggle(const char* label, bool* v) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    bool hov = false;
    const bool pressed = press_behavior(id, r, &hov, nullptr);
    if (pressed) *v = !*v;
    float* anim = state_f(id, *v ? 1.0f : 0.0f);
    const float target = *v ? 1.0f : 0.0f;
    *anim += (target - *anim) * (1.0f - std::exp(-static_cast<float>(dt_) * 20.0f));
    if (std::fabs(*anim - target) < 0.01f) *anim = target;
    const int sh = st_.check + 2, sw = sh * 2 - 2;
    const Rect sr{r.x + r.w - sw, r.y + (r.h - sh) / 2, sw, sh};
    DrawList& dl = layers_[0];
    const u32 track = render::lerp_color(hov ? st_.frame_active : st_.frame_hover, hov ? st_.accent_hover : st_.accent, *anim);
    dl.fill_round_rect(sr, sh * 0.5f, track);
    if (*anim < 0.99f) dl.round_rect(sr, sh * 0.5f, mul_alpha(st_.border, 1.0f - *anim), 1);
    const float kr = static_cast<float>(sh) * 0.5f - 3.0f;
    const float kx = static_cast<float>(sr.x) + sh * 0.5f + (*anim) * static_cast<float>(sw - sh);
    dl.fill_circle(kx, static_cast<float>(sr.y) + sh * 0.5f, kr, *v ? 0xFFFFFFFFu : 0xFFB9C0CCu);
    label_left({r.x, r.y, r.w - sw - 8, r.h}, label, display_len(label), st_.text);
    focus_ring(id, r);
    return pressed;
}

bool Context::checkbox(const char* label, bool* v) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    const int n = display_len(label);
    const Rect hit{r.x, r.y, min_(r.w, st_.check + gap_px() + render::text_width(label, n, st_.font) + 4), r.h};
    bool hov = false;
    const bool pressed = press_behavior(id, hit, &hov, nullptr);
    if (pressed) *v = !*v;
    const int c = st_.check;
    const Rect b{r.x, r.y + (r.h - c) / 2, c, c};
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius) - 1.0f;
    if (*v) {
        dl.fill_round_rect(b, rad, hov ? st_.accent_hover : st_.accent);
        const float s = static_cast<float>(c) / 16.0f;
        const Vec2 pts[3] = {{b.x + 3.8f * s, b.y + 8.2f * s}, {b.x + 6.8f * s, b.y + 11.2f * s}, {b.x + 12.4f * s, b.y + 4.9f * s}};
        dl.polyline(pts, 3, 0xFFFFFFFFu, 2.0f * s);
    } else {
        dl.fill_round_rect(b, rad, hov ? st_.frame_hover : st_.frame);
        dl.round_rect(b, rad, hov ? st_.accent : 0xFF434957u, 1);
    }
    dl.text(r.x + c + gap_px(), r.y + r.h / 2 - font_center(st_.font), label, st_.text, st_.font, n);
    focus_ring(id, hit);
    return pressed;
}

bool Context::radio(const char* label, int* v, int value) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    const int n = display_len(label);
    const Rect hit{r.x, r.y, min_(r.w, st_.check + gap_px() + render::text_width(label, n, st_.font) + 4), r.h};
    bool hov = false;
    const bool pressed = press_behavior(id, hit, &hov, nullptr);
    if (pressed) *v = value;
    const bool on = *v == value;
    const float rr = static_cast<float>(st_.check) * 0.5f;
    const float cx = static_cast<float>(r.x) + rr, cy = static_cast<float>(r.y) + static_cast<float>(r.h) * 0.5f;
    DrawList& dl = layers_[0];
    if (on) {
        dl.fill_circle(cx, cy, rr, hov ? st_.accent_hover : st_.accent);
        dl.fill_circle(cx, cy, rr * 0.4f, 0xFFFFFFFFu);
    } else {
        dl.fill_circle(cx, cy, rr, hov ? st_.frame_hover : st_.frame);
        dl.circle(cx, cy, rr - 0.5f, hov ? st_.accent : 0xFF434957u, 1.0f);
    }
    dl.text(r.x + st_.check + gap_px(), r.y + r.h / 2 - font_center(st_.font), label, on ? st_.text_strong : st_.text, st_.font, n);
    focus_ring(id, hit);
    return pressed;
}

bool Context::segmented(const char* idlabel, const char* const* items, int n, int* sel) {
    if (n <= 0) return false;
    const Id id = get_id(idlabel);
    const Rect r = next_rect(st_.row_h + 2);
    item_add(id, r);                                  // parada de Tab del control completo
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    dl.fill_round_rect(r, rad, st_.frame);
    dl.round_rect(r, rad, st_.border, 1);
    bool changed = false;
    // Teclado: ←/→ con el foco en el control.
    if (focus_visible_ && focus_id_ == id && !edit_id_) {
        if (in_.key_pressed[platform::KeyLeft] && *sel > 0) { --*sel; changed = true; }
        if (in_.key_pressed[platform::KeyRight] && *sel < n - 1) { ++*sel; changed = true; }
    }
    push_id(idlabel);
    const int w = r.w / n;
    for (int i = 0; i < n; ++i) {
        const Rect s{r.x + i * w, r.y, i == n - 1 ? r.x + r.w - (r.x + i * w) : w, r.h};
        push_id(i);
        const Id sid = get_id("##seg");
        pop_id();
        bool hov = false;
        // Los segmentos no son parada de Tab (focusable = false; antes se hacía --focus_n_,
        // que con la lista llena de 512 borraba la parada de otro widget).
        if (press_behavior(sid, s, &hov, nullptr, false) && *sel != i) { *sel = i; changed = true; focus_id_ = id; }
        --stats_.widgets;
        const bool on = *sel == i;
        if (on) dl.fill_round_rect({s.x + 2, s.y + 2, s.w - 4, s.h - 4}, rad - 1.0f, st_.accent);
        else if (hov) dl.fill_round_rect({s.x + 2, s.y + 2, s.w - 4, s.h - 4}, rad - 1.0f, st_.frame_hover);
        else if (i > 0 && *sel != i - 1) dl.fill_rect({s.x, s.y + 6, 1, s.h - 12}, st_.border);
        const int tw = render::text_width(items[i], st_.font);
        const int maxw = s.w - 6;
        const Rect tr{s.x + max_(3, (s.w - min_(tw, maxw)) / 2), s.y, maxw, s.h};
        label_left(tr, items[i], static_cast<int>(std::strlen(items[i])), on ? st_.accent_text : (hov ? st_.text_strong : st_.text));
    }
    pop_id();
    last_id_ = id;
    last_rect_ = r;
    last_hovered_ = r.contains(in_.mouse_x, in_.mouse_y) && hoverable(r);
    focus_ring(id, r);
    return changed;
}

// ---- Edición de texto (valor tecleado) -------------------------------------------------------------
bool Context::text_edit(Id id, Rect r, char* buf, int cap) {
    edit_alive_ = true;
    edit_rect_ = r;
    bool commit = edit_commit_;
    edit_commit_ = false;
    // Teclas
    const auto& kp = in_.key_pressed;
    if (edit_cancel_) { edit_cancel_ = false; edit_id_ = 0; return false; }
    if (kp[platform::KeyEnter] || kp[platform::KeyTab]) commit = true;
    if (in_.mouse_pressed[0] && !r.contains(in_.mouse_x, in_.mouse_y)) commit = true;
    for (int i = 0; i < in_.text_len; ++i) {
        char ch = in_.text[i];
        if (ch == ',') ch = '.';                     // coma decimal (teclado español)
        const bool ok = (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E';
        if (!ok) continue;
        if (edit_select_all_) { edit_len_ = edit_cur_ = 0; edit_select_all_ = false; }
        if (edit_len_ < cap - 1) {
            std::memmove(buf + edit_cur_ + 1, buf + edit_cur_, static_cast<usize>(edit_len_ - edit_cur_));
            buf[edit_cur_++] = ch;
            ++edit_len_;
        }
    }
    if (kp[platform::KeyBackspace]) {
        if (edit_select_all_) { edit_len_ = edit_cur_ = 0; edit_select_all_ = false; }
        else if (edit_cur_ > 0) {
            std::memmove(buf + edit_cur_ - 1, buf + edit_cur_, static_cast<usize>(edit_len_ - edit_cur_));
            --edit_cur_; --edit_len_;
        }
    }
    if (kp[platform::KeyDelete]) {
        if (edit_select_all_) { edit_len_ = edit_cur_ = 0; edit_select_all_ = false; }
        else if (edit_cur_ < edit_len_) {
            std::memmove(buf + edit_cur_, buf + edit_cur_ + 1, static_cast<usize>(edit_len_ - edit_cur_ - 1));
            --edit_len_;
        }
    }
    if (kp[platform::KeyLeft]) { edit_select_all_ = false; if (edit_cur_ > 0) --edit_cur_; }
    if (kp[platform::KeyRight]) { edit_select_all_ = false; if (edit_cur_ < edit_len_) ++edit_cur_; }
    if (kp[platform::KeyHome]) { edit_select_all_ = false; edit_cur_ = 0; }
    if (kp[platform::KeyEnd]) { edit_select_all_ = false; edit_cur_ = edit_len_; }
    buf[edit_len_] = 0;
    // Dibujo: caja activa, selección y cursor parpadeante.
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    dl.fill_round_rect(r, rad, 0xFF101217u);
    dl.round_rect(r, rad, st_.accent, 1);
    const int fw = st_.fw();
    const int tx = r.x + 8, ty = r.y + r.h / 2 - font_center(st_.font);
    if (edit_select_all_ && edit_len_ > 0) dl.fill_rect({tx - 1, r.y + 4, edit_len_ * fw + 2, r.h - 8}, st_.accent_dim);
    dl.text(tx, ty, buf, st_.text_strong, st_.font, edit_len_);
    if (std::fmod(time_, 1.0) < 0.6) dl.fill_rect({tx + edit_cur_ * fw, r.y + 5, 1, r.h - 10}, st_.accent_hover);
    if (commit) { edit_id_ = 0; return true; }
    (void)id;
    return false;
}

// ---- Sliders ----------------------------------------------------------------------------------------------
bool Context::slider_core(const char* label, float* v, float lo, float hi, bool is_int, const char* f, const char* unit) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    const int n = display_len(label);
    const int lw = n > 0 ? r.w * st_.label_frac / 100 : 0;
    const Rect tr{r.x + lw, r.y, r.w - lw, r.h};
    if (n > 0) label_left({r.x, r.y, lw - 6, r.h}, label, n, st_.text);
    bool changed = false;
    // --- Modo edición (valor tecleado) ---
    if (edit_id_ == id) {
        item_add(id, tr);
        if (text_edit(id, tr, edit_buf_, static_cast<int>(sizeof edit_buf_))) {
            char* end = nullptr;
            const double x = std::strtod(edit_buf_, &end);
            if (end != edit_buf_ && std::isfinite(x)) {
                float nv = clamp_(static_cast<float>(x), lo, hi);
                if (is_int) nv = std::round(nv);
                changed = nv != *v;
                *v = nv;
            }
        }
        return changed;
    }
    const bool hov = item_add(id, tr);
    const int km = 4;                                    // margen interior de la pista
    const float span = static_cast<float>(max_(1, tr.w - 2 * km));
    auto enter_edit = [&](float restore) {
        *v = restore;
        edit_id_ = id;
        std::snprintf(edit_buf_, sizeof edit_buf_, is_int ? "%.0f" : "%g", static_cast<double>(*v));
        edit_len_ = edit_cur_ = static_cast<int>(std::strlen(edit_buf_));
        edit_select_all_ = true;
        edit_commit_ = edit_cancel_ = false;
        edit_alive_ = true;                              // vivo desde este cuadro (end_frame no lo cancela)
        edit_rect_ = tr;
        clear_active();
    };
    if (hov && !click_consumed_ && in_.mouse_pressed[0]) {
        if (in_.ctrl) { enter_edit(*v); return false; }
        if (in_.double_click && last_click_id_ == id) { const float old = last_click_value_; changed = old != *v; enter_edit(old); return changed; }
        last_click_id_ = id;
        last_click_value_ = *v;
        set_active(id);
        if (focus_id_ != id) focus_visible_ = false;
        focus_id_ = id;
    }
    if (active_id_ == id) {
        active_alive_ = true;
        if (in_.mouse_down[0] || in_.mouse_pressed[0]) {
            // Ajuste fino (Mayús): acumulador continuo propio (f) + bandera "estaba en modo
            // fino" (i) en una ranura salada. Sin él, slider_int redondeaba cada cuadro el
            // incremento sub-unidad (p.ej. 0.05 por píxel) y el valor NUNCA se movía.
            const Id fid = id ^ 0xF17E5A1DE7ACC0DEull;
            Slot* fs = slot(fid);
            if (fs->key != fid) { fs->key = fid; fs->f = *v; fs->i = 0; }
            float nv;
            if (in_.shift) {
                if (!fs->i || in_.mouse_pressed[0]) fs->f = *v;          // entra en modo fino
                fs->i = 1;
                fs->f = clamp_(fs->f + static_cast<float>(in_.mouse_dx) * (hi - lo) / span * 0.1f, lo, hi);
                nv = fs->f;
            } else {
                fs->i = 0;
                nv = lo + (static_cast<float>(in_.mouse_x - tr.x - km) + 0.5f) / span * (hi - lo);
            }
            nv = clamp_(nv, lo, hi);
            if (is_int) nv = std::round(nv);
            if (nv != *v) { *v = nv; changed = true; }
        }
        if (!in_.mouse_down[0]) clear_active();
    }
    if (focus_visible_ && focus_id_ == id && !edit_id_) {
        const float step = is_int ? max_(1.0f, std::round((hi - lo) * (in_.shift ? 0.001f : 0.01f))) : (hi - lo) * (in_.shift ? 0.001f : 0.01f);
        float nv = *v;
        if (in_.key_pressed[platform::KeyLeft]) nv -= step;
        if (in_.key_pressed[platform::KeyRight]) nv += step;
        nv = clamp_(nv, lo, hi);
        if (nv != *v) { *v = nv; changed = true; }
        if (in_.key_pressed[platform::KeyEnter]) enter_edit(*v);
    }
    // --- Dibujo ---
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    const bool act = active_id_ == id;
    dl.fill_round_rect(tr, rad, act ? st_.frame_active : (hov ? st_.frame_hover : st_.frame));
    const float t = hi > lo ? clamp_((*v - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
    const int kx = tr.x + km + static_cast<int>(t * span);
    const u32 fill = act || hov ? 0xFF24548Cu : 0xFF1F4675u;
    if (lo < 0.0f && hi > 0.0f) {
        // Rango con signo: el relleno parte del cero (slider bipolar) + marca del cero.
        const int zx = tr.x + km + static_cast<int>((0.0f - lo) / (hi - lo) * span);
        if (kx != zx) dl.fill_rect({min_(kx, zx), tr.y, std::abs(kx - zx) + 1, tr.h}, fill);
        dl.fill_rect({zx, tr.y + 4, 1, tr.h - 8}, 0xFF4A5261u);
    } else if (kx > tr.x + 1) {
        dl.fill_round_rect({tr.x, tr.y, kx - tr.x + 1, tr.h}, rad, fill, 0x5u);
    }
    const int kw = max_(3, static_cast<int>(3 * st_.scale));
    char vb[96];
    const int nb = std::snprintf(vb, sizeof vb, f ? f : "%.2f", static_cast<double>(*v));
    // Unidad con espacio fino tipográfico salvo el grado ("12°", pero "250 km/h", "41 %").
    if (unit && nb > 0 && nb < 80)
        std::snprintf(vb + nb, sizeof vb - static_cast<usize>(nb), std::strncmp(unit, "°", 2) == 0 ? "%s" : " %s", unit);
    const int vw = render::text_width(vb, st_.font);
    const int tx = tr.x + (tr.w - vw) / 2;
    // Asa: barra vertical; si cae sobre el texto del valor se reduce a dos marcas (arriba y abajo)
    // para no tachar los dígitos (app: "250 km/h" con el asa sobre la "h").
    const u32 kc = act ? 0xFFFFFFFFu : (hov ? st_.accent_hover : st_.accent);
    if (kx + kw / 2 + 2 < tx || kx - kw / 2 - 2 > tx + vw) {
        dl.fill_round_rect({kx - kw / 2, tr.y + 3, kw, tr.h - 6}, kw * 0.5f, kc);
    } else {
        const int mh = max_(3, (tr.h - render::font_h(st_.font)) / 2 - 1);
        dl.fill_rect({kx - kw / 2, tr.y + 1, kw, mh}, kc);
        dl.fill_rect({kx - kw / 2, tr.y + tr.h - 1 - mh, kw, mh}, kc);
    }
    dl.text_shadow(tx, tr.y + tr.h / 2 - font_center(st_.font), vb, st_.text_strong, 0x90000000u, st_.font);
    focus_ring(id, tr);
    return changed;
}

bool Context::slider_float(const char* label, float* v, float lo, float hi, const char* f, const char* unit) {
    return slider_core(label, v, lo, hi, false, f, unit);
}
bool Context::slider_int(const char* label, int* v, int lo, int hi, const char* unit) {
    float fv = static_cast<float>(*v);
    const bool ch = slider_core(label, &fv, static_cast<float>(lo), static_cast<float>(hi), true, "%.0f", unit);
    *v = clamp_(static_cast<int>(std::lround(fv)), lo, hi);
    return ch;
}

// ---- Combo -----------------------------------------------------------------------------------------------
bool Context::combo(const char* label, int* sel, const char* const* items, int n) {
    const Id id = get_id(label);
    const Rect r = next_rect(st_.row_h);
    const int nl = display_len(label);
    const int lw = nl > 0 ? r.w * st_.label_frac / 100 : 0;
    const Rect box{r.x + lw, r.y, r.w - lw, r.h};
    if (nl > 0) label_left({r.x, r.y, lw - 6, r.h}, label, nl, st_.text);
    bool hov = false;
    bool changed = false;
    const bool open_now = popup_id_ == id;
    if (press_behavior(id, box, &hov, nullptr)) {
        if (open_now) popup_id_ = 0;
        else { popup_id_ = id; popup_opened_now_ = true; popup_scroll_ = 0; }
    }
    if (focus_visible_ && focus_id_ == id && !edit_id_ && popup_id_ != id && n > 0) {
        if (in_.key_pressed[platform::KeyUp] && *sel > 0) { --*sel; changed = true; }
        if (in_.key_pressed[platform::KeyDown] && *sel < n - 1) { ++*sel; changed = true; }
    }
    const bool open = popup_id_ == id;
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    dl.fill_round_rect(box, rad, open ? st_.frame_active : (hov ? st_.frame_hover : st_.frame));
    dl.round_rect(box, rad, open ? st_.accent : (hov ? 0xFF3E4452u : st_.border), 1);
    const int aw = static_cast<int>(18 * st_.scale);
    if (n > 0 && *sel >= 0 && *sel < n)
        label_left({box.x + 8, box.y, box.w - 8 - aw, box.h}, items[*sel], static_cast<int>(std::strlen(items[*sel])), st_.text_strong);
    {
        const float cx = static_cast<float>(box.x + box.w - aw / 2 - 2), cy = static_cast<float>(box.y) + static_cast<float>(box.h) * 0.5f;
        const float s = 3.5f * st_.scale;
        const u32 cc = open ? st_.accent : st_.text_dim;
        if (open) dl.fill_triangle({cx - s, cy + s * 0.5f}, {cx + s, cy + s * 0.5f}, {cx, cy - s * 0.6f}, cc);
        else dl.fill_triangle({cx - s, cy - s * 0.5f}, {cx + s, cy - s * 0.5f}, {cx, cy + s * 0.6f}, cc);
    }
    focus_ring(id, box);
    if (!open) return changed;

    // --- Popup (capa 1, encima de todo el panel) ---
    popup_alive_ = true;
    const int ih = st_.row_h;
    const int vis = min_(n, 12);
    const int ph = vis * ih + 8;
    int py = box.y + box.h + 3;
    if (py + ph > fb_h_ - 4) py = box.y - ph - 3;
    py = max_(py, 2);
    // Ancho: el del texto más largo (sin recortar nombres), crece hacia donde haya sitio.
    int maxg = 0;
    for (int i = 0; i < n; ++i) maxg = max_(maxg, render::utf8_count(items[i]));
    const int pw = clamp_(maxg * st_.fw() + 44, box.w, max_(box.w, fb_w_ - 8));
    int px = box.x + pw > fb_w_ - 4 ? box.x + box.w - pw : box.x;
    px = max_(px, 4);
    popup_rect_ = {px, py, pw, ph};
    if (ui_nrects_ < 16) ui_rects_[ui_nrects_++] = popup_rect_;
    DrawList& pl = layers_[1];
    pl.shadow({popup_rect_.x, popup_rect_.y + 3, popup_rect_.w, popup_rect_.h}, rad + 2, 10.0f, 0x70000000u);
    pl.fill_round_rect(popup_rect_, rad + 1, st_.popup);
    pl.round_rect(popup_rect_, rad + 1, 0xFF3A404Du, 1);
    // Rueda dentro del popup (listas largas).
    const int max_first = max_(0, n - vis);
    if (popup_rect_.contains(in_.mouse_x, in_.mouse_y) && in_.wheel != 0.0f)
        popup_scroll_ = clamp_(popup_scroll_ - in_.wheel, 0.0f, static_cast<float>(max_first));
    const int first = clamp_(static_cast<int>(popup_scroll_), 0, max_first);
    cur_layer_ = 1;
    in_popup_ = true;
    pl.push_clip({popup_rect_.x + 1, popup_rect_.y + 1, popup_rect_.w - 2, popup_rect_.h - 2});
    push_id(label);
    for (int k = 0; k < vis; ++k) {
        const int i = first + k;
        const Rect ir{popup_rect_.x + 4, popup_rect_.y + 4 + k * ih, popup_rect_.w - 8, ih};
        push_id(i);
        const Id iid = get_id("##item");
        pop_id();
        bool ih_hov = false;
        const bool clicked = press_behavior(iid, ir, &ih_hov, nullptr, false);
        --stats_.widgets;
        const bool is_sel = i == *sel;
        if (ih_hov) pl.fill_round_rect(ir, rad, st_.frame_hover);
        if (is_sel) pl.fill_rect({ir.x + 2, ir.y + 5, 2, ir.h - 10}, st_.accent);
        label_left({ir.x + 10, ir.y, ir.w - 30, ir.h}, items[i], static_cast<int>(std::strlen(items[i])), is_sel ? st_.accent_hover : (ih_hov ? st_.text_strong : st_.text));
        if (is_sel) pl.text(ir.x + ir.w - st_.fw() - 6, ir.y + ir.h / 2 - font_center(st_.font), "✓", st_.accent_hover, st_.font);
        if (clicked) { if (*sel != i) changed = true; *sel = i; popup_id_ = 0; }
    }
    pop_id();
    pl.pop_clip();
    if (max_first > 0) {                                   // indicador de desplazamiento
        const int th = max_(12, (ph - 8) * vis / n);
        const int ty = popup_rect_.y + 4 + (ph - 8 - th) * first / max_first;
        pl.fill_round_rect({popup_rect_.x + popup_rect_.w - 5, ty, 3, th}, 1.5f, st_.border);
    }
    in_popup_ = false;
    cur_layer_ = 0;
    last_id_ = id;
    last_rect_ = box;
    last_hovered_ = hov;                                   // los elementos del popup lo habían pisado
    return changed;
}

// ---- Progreso -------------------------------------------------------------------------------------------
void Context::progress(float t, const char* overlay, u32 color) {
    const Rect r = next_rect(st_.row_h - 4);
    t = clamp_(std::isfinite(t) ? t : 0.0f, 0.0f, 1.0f);
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    dl.fill_round_rect(r, rad, st_.frame);
    const int w = static_cast<int>(t * static_cast<float>(r.w));
    if (w > 0) dl.fill_round_rect({r.x, r.y, max_(w, st_.radius * 2), r.h}, rad, color ? color : st_.accent);
    if (overlay) {
        const int tw = render::text_width(overlay, st_.font);
        dl.text_shadow(r.x + (r.w - tw) / 2, r.y + r.h / 2 - font_center(st_.font), overlay, st_.text_strong, 0xA0000000u, st_.font);
    }
    item_add(0, r, false);
}

// ---- Tooltip / avisos --------------------------------------------------------------------------------------
void Context::tooltip(const char* f, ...) {
    if (!last_hovered_ || active_id_ != 0 || popup_id_ || time_ - hover_t0_ < k_tooltip_delay) return;
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(tip_, sizeof tip_, f, ap);
    va_end(ap);
    tip_pending_ = true;
}

void Context::draw_tooltip() {
    if (!tip_pending_) return;
    const int fw = st_.fw();
    const int maxg = 40;
    int lines = 0, maxw = 0;
    wrap_lines(tip_, maxg, [&](const char* p, int nb) { ++lines; maxw = max_(maxw, render::utf8_count(p, nb)); });
    if (!lines) return;
    const int pad = static_cast<int>(8 * st_.scale);
    const int lh = st_.text_h;
    const int w = maxw * fw + 2 * pad, h = lines * lh + 2 * pad - 2;
    int x = in_.mouse_x + 14, y = in_.mouse_y + 20;
    if (x + w > fb_w_ - 4) x = max_(4, in_.mouse_x - w - 8);
    if (y + h > fb_h_ - 4) y = max_(4, in_.mouse_y - h - 8);
    DrawList& dl = layers_[2];
    dl.shadow({x, y + 2, w, h}, static_cast<float>(st_.radius + 1), 8.0f, 0x80000000u);
    dl.fill_round_rect({x, y, w, h}, static_cast<float>(st_.radius + 1), st_.tooltip);
    dl.round_rect({x, y, w, h}, static_cast<float>(st_.radius + 1), 0xFF3A404Du, 1);
    int k = 0;
    wrap_lines(tip_, maxg, [&](const char* p, int nb) {
        dl.text(x + pad, y + pad + k * lh + (lh - st_.fh()) / 2 - 1, p, st_.text, st_.font, nb);
        ++k;
    });
}

void Context::toast(u32 color, const char* f, ...) {
    if (ntoasts_ == 8) { std::memmove(&toasts_[0], &toasts_[1], sizeof(Toast) * 7); --ntoasts_; }
    Toast& t = toasts_[ntoasts_++];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(t.text, sizeof t.text, f, ap);
    va_end(ap);
    t.color = color ? color : st_.accent;
    t.t0 = time_;
    t.t1 = time_ + 3.5;
}

void Context::draw_toasts() {
    int w = 0;
    for (int i = 0; i < ntoasts_; ++i) if (toasts_[i].t1 > time_) toasts_[w++] = toasts_[i];
    ntoasts_ = w;
    if (!ntoasts_) return;
    const Rect area = toast_area_.w > 0 ? toast_area_ : Rect{0, 0, fb_w_, fb_h_};
    DrawList& dl = layers_[2];
    const int h = st_.row_h + static_cast<int>(12 * st_.scale);
    int y = area.y + area.h - static_cast<int>(28 * st_.scale);
    for (int i = ntoasts_ - 1; i >= 0; --i) {
        const Toast& t = toasts_[i];
        const float age = static_cast<float>(time_ - t.t0), left = static_cast<float>(t.t1 - time_);
        const float fade = clamp_(min_(age / 0.18f, left / 0.5f), 0.0f, 1.0f);
        const int tw = render::text_width(t.text, st_.font);
        const int bw = tw + static_cast<int>(36 * st_.scale);
        const int x = area.x + (area.w - bw) / 2;
        const int slide = static_cast<int>((1.0f - clamp_(age / 0.18f, 0.0f, 1.0f)) * 12.0f);
        y -= h;
        const Rect b{x, y + slide, bw, h};
        dl.shadow({b.x, b.y + 3, b.w, b.h}, static_cast<float>(st_.radius + 2), 12.0f, mul_alpha(0x90000000u, fade));
        dl.fill_round_rect(b, static_cast<float>(st_.radius + 2), mul_alpha(0xF4202329u, fade));
        dl.round_rect(b, static_cast<float>(st_.radius + 2), mul_alpha(0xFF363B47u, fade), 1);
        const int dot = static_cast<int>(12 * st_.scale);
        dl.fill_circle(static_cast<float>(b.x + dot + 1), static_cast<float>(b.y) + b.h * 0.5f, 3.5f * st_.scale, mul_alpha(t.color, fade));
        dl.text(b.x + dot * 2, b.y + b.h / 2 - font_center(st_.font), t.text, mul_alpha(st_.text_strong, fade), st_.font);
        y -= static_cast<int>(8 * st_.scale);
    }
}

// ---- Gráficas -------------------------------------------------------------------------------------------------
// Paso "bonito" 1-2-5 × 10^k ≥ raw.
static double nice_step(double raw) {
    if (!(raw > 0)) return 1.0;
    const double e = std::pow(10.0, std::floor(std::log10(raw)));
    const double m = raw / e;
    return (m <= 1.0 ? 1.0 : m <= 2.0 ? 2.0 : m <= 5.0 ? 5.0 : 10.0) * e;
}
static void fmt_tick(char* b, usize n, double v, double step) {
    const int dec = clamp_(static_cast<int>(-std::floor(std::log10(step) + 1e-9)), 0, 5);
    if (std::fabs(v) < step * 1e-6) v = 0.0;
    std::snprintf(b, n, "%.*f", dec, v);
}

// Muestra i (0 = más antigua) de una serie circular. El offset se normaliza a [0, count):
// antes un offset negativo (o ≥ count con i grande) leía fuera del búfer (ASan).
static CFD_INLINE float series_at(const PlotSeries& S, int off, int i) {
    int k = off + i;
    if (k >= S.count) k -= S.count;
    return S.data[k];
}
static CFD_INLINE int series_off(const PlotSeries& S) {
    const int m = S.offset % S.count;
    return m < 0 ? m + S.count : m;
}

void Context::plot_lines(const char* label, const PlotSeries* series_in, int ns, int height, const char* unit, const char* f) {
    flush_row();
    // Series sin datos (data nulo o count ≤ 0) se tratan como vacías.
    PlotSeries series[8];
    ns = clamp_(ns, 0, 8);
    for (int s = 0; s < ns; ++s) {
        series[s] = series_in[s];
        if (!series[s].data || series[s].count < 0) series[s].count = 0;
    }
    const Rect r = next_rect(height);
    const Id id = get_id(label ? label : "##plot");
    const bool hov = item_add(id, r, false);
    DrawList& dl = layers_[0];
    const float rad = static_cast<float>(st_.radius);
    dl.fill_round_rect(r, rad, st_.plot_bg);
    dl.round_rect(r, rad, st_.panel_edge, 1);
    const int fw = st_.fw();
    const int ip = static_cast<int>(8 * st_.scale);
    // Cabecera: título + leyenda con valor actual.
    int hx = r.x + ip;
    const int hy = r.y + ip - 2;
    if (label) {
        hx = dl.text(hx, hy, label, st_.text_dim, st_.font, display_len(label));
        if (unit) { hx = dl.text(hx + fw / 2, hy, "[", st_.text_disabled, st_.font); hx = dl.text(hx, hy, unit, st_.text_disabled, st_.font); hx = dl.text(hx, hy, "]", st_.text_disabled, st_.font); }
    }
    // Leyenda alineada a la derecha. Si con los nombres de las series no cabe junto al título (panel estrecho),
    // se omiten los nombres (quedan la muestra de color y el valor) para no pisar el título.
    char vbs[8][48];
    int legend_w = 0;
    for (int s = 0; s < ns; ++s) {
        const PlotSeries& S = series[s];
        std::snprintf(vbs[s], sizeof vbs[s], "%s", "—");
        if (S.count > 0) {
            const float last = series_at(S, series_off(S), S.count - 1);
            std::snprintf(vbs[s], sizeof vbs[s], f ? f : "%.3f", static_cast<double>(last));
        }
        legend_w += render::text_width(vbs[s], st_.font) + 15 + fw + 6;
        if (S.name) legend_w += render::text_width(S.name, st_.font) + fw / 2;
    }
    const bool show_names = r.x + r.w - ip - legend_w >= hx + fw;
    int lx = r.x + r.w - ip;
    for (int s = ns - 1; s >= 0; --s) {
        const PlotSeries& S = series[s];
        const u32 col = S.color ? S.color : st_.series[s % 6];
        const char* vb = vbs[s];
        const int vw = render::text_width(vb, st_.font);
        lx -= vw;
        dl.text(lx, hy, vb, st_.text_strong, st_.font);
        if (S.name && show_names) {
            const int nw = render::text_width(S.name, st_.font);
            lx -= nw + fw / 2;
            dl.text(lx, hy, S.name, st_.text_dim, st_.font);
        }
        lx -= 15;
        dl.fill_round_rect({lx, hy + st_.fh() / 2 - 2, 10, 4}, 2.0f, col);
        lx -= fw + 6;
    }
    // Área de trazado.
    const int ml = fw * 6 + 4;
    const Rect pa{r.x + ml, r.y + ip + st_.fh() + 6, r.w - ml - ip, r.h - (ip + st_.fh() + 6) - ip};
    if (pa.w < 8 || pa.h < 8) return;
    float lo = 1e30f, hi = -1e30f;
    int nmax = 0;
    for (int s = 0; s < ns; ++s) {
        const PlotSeries& S = series[s];
        nmax = max_(nmax, S.count);
        for (int i = 0; i < S.count; ++i) {
            const float v = S.data[i];
            if (std::isfinite(v)) { lo = min_(lo, v); hi = max_(hi, v); }
        }
    }
    if (lo > hi) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9f) { const float m = max_(std::fabs(hi) * 0.1f, 1e-3f); lo -= m; hi += m; }
    const int nticks = clamp_(pa.h / (st_.fh() + 10), 2, 5);         // etiquetas sin solaparse
    const double step = nice_step(static_cast<double>(hi - lo) / static_cast<double>(nticks));
    const double glo = std::floor(static_cast<double>(lo) / step) * step, ghi = std::ceil(static_cast<double>(hi) / step) * step;
    const float ylo = static_cast<float>(glo), yhi = static_cast<float>(ghi);
    auto map_y = [&](float v) { return static_cast<float>(pa.y + pa.h) - (v - ylo) / (yhi - ylo) * static_cast<float>(pa.h); };
    // Rejilla y etiquetas del eje.
    char tb[32];
    for (double gv = glo; gv <= ghi + step * 0.5; gv += step) {
        const int gy = static_cast<int>(std::lround(map_y(static_cast<float>(gv))));
        const bool zero = std::fabs(gv) < step * 1e-6;
        dl.fill_rect({pa.x, gy, pa.w, 1}, zero ? 0xFF3A404Cu : st_.grid);
        fmt_tick(tb, sizeof tb, gv, step);
        const int tw = render::text_width(tb, st_.font);
        dl.text(pa.x - tw - 6, gy - font_center(st_.font), tb, st_.text_disabled, st_.font);
    }
    dl.push_clip({pa.x, pa.y - 2, pa.w + 1, pa.h + 4});
    // Series: diezmado a ≤ 2 puntos por píxel (máx/mín por columna conservan los picos).
    for (int s = 0; s < ns; ++s) {
        const PlotSeries& S = series[s];
        if (S.count < 1) continue;
        const u32 col = S.color ? S.color : st_.series[s % 6];
        scratch_.clear();
        const int off = series_off(S);
        const float dx = S.count > 1 ? static_cast<float>(pa.w) / static_cast<float>(S.count - 1) : 0.0f;
        if (S.count <= pa.w * 2) {
            for (int i = 0; i < S.count; ++i) {
                const float v = series_at(S, off, i);
                scratch_.push_back({static_cast<float>(pa.x) + dx * static_cast<float>(i), map_y(std::isfinite(v) ? v : ylo)});
            }
        } else {
            for (int px = 0; px < pa.w; ++px) {
                const int i0 = static_cast<int>(static_cast<i64>(px) * S.count / pa.w), i1 = static_cast<int>(static_cast<i64>(px + 1) * S.count / pa.w);
                float mn = 1e30f, mx = -1e30f;
                for (int i = i0; i < i1; ++i) { const float v = series_at(S, off, i); if (std::isfinite(v)) { mn = min_(mn, v); mx = max_(mx, v); } }
                if (mn > mx) continue;
                scratch_.push_back({static_cast<float>(pa.x + px) + 0.25f, map_y(mn)});
                scratch_.push_back({static_cast<float>(pa.x + px) + 0.75f, map_y(mx)});
            }
        }
        const int np = static_cast<int>(scratch_.size());
        if (s == 0 && np >= 2) dl.fill_area(scratch_.data(), np, static_cast<float>(pa.y + pa.h), with_alpha(col, 0x48), with_alpha(col, 0x00));
        dl.polyline(scratch_.data(), np, col, 1.6f * st_.scale);
        if (np >= 1) {
            const Vec2 lp = scratch_[static_cast<usize>(np - 1)];
            dl.fill_circle(lp.x, lp.y, 3.2f * st_.scale, col);
            dl.circle(lp.x, lp.y, 3.2f * st_.scale, st_.plot_bg, 1.2f);
        }
    }
    dl.pop_clip();
    // Cursor: línea vertical y valores bajo el ratón.
    if (hov && pa.contains(in_.mouse_x, in_.mouse_y) && nmax > 1) {
        const float fx = static_cast<float>(in_.mouse_x - pa.x) / static_cast<float>(pa.w);
        dl.fill_rect({in_.mouse_x, pa.y, 1, pa.h}, 0x70FFFFFFu);
        int lines = 0;
        char buf[512];
        int off = 0;
        for (int s = 0; s < ns && off < 400; ++s) {
            const PlotSeries& S = series[s];
            if (S.count < 1) continue;
            const int i = clamp_(static_cast<int>(std::lround(fx * static_cast<float>(S.count - 1))), 0, S.count - 1);
            const float v = series_at(S, series_off(S), i);
            const float px = static_cast<float>(pa.x) + static_cast<float>(i) * static_cast<float>(pa.w) / static_cast<float>(max_(1, S.count - 1));
            dl.fill_circle(px, map_y(std::isfinite(v) ? v : ylo), 3.0f * st_.scale, S.color ? S.color : st_.series[s % 6]);
            char vb[40];
            std::snprintf(vb, sizeof vb, f ? f : "%.3f", static_cast<double>(v));
            off += std::snprintf(buf + off, sizeof buf - static_cast<usize>(off), "%s%s: %s", lines ? "\n" : "", S.name ? S.name : "serie", vb);
            ++lines;
        }
        if (lines && active_id_ == 0) {
            std::snprintf(tip_, sizeof tip_, "%s", buf);
            tip_pending_ = true;
        }
    }
}

void Context::bar_chart(const char* label, const Bar* bars, int n, const char* unit, const char* f) {
    flush_row();
    DrawList& dl = layers_[0];
    if (label) {
        const Rect hr = next_rect(st_.text_h);
        const int y = hr.y + hr.h / 2 - font_center(st_.font);
        dl.text(hr.x, y, label, st_.text_dim, st_.font, display_len(label));
        if (unit) { const int uw = render::text_width(unit, st_.font); dl.text(hr.x + hr.w - uw, y, unit, st_.text_disabled, st_.font); }
    }
    if (n <= 0) return;
    float lo = 0, hi = 0;
    for (int i = 0; i < n; ++i) if (std::isfinite(bars[i].value)) { lo = min_(lo, bars[i].value); hi = max_(hi, bars[i].value); }
    if (hi - lo < 1e-9f) hi = lo + 1.0f;
    const int bh = st_.text_h + 3;
    const Rect all = next_rect(n * bh);
    item_add(get_id(label ? label : "##barras"), all, false);
    const int fw = st_.fw();
    const int lw = all.w * 36 / 100, vw = fw * 8;
    const Rect area{all.x + lw, all.y, all.w - lw - vw - 4, all.h};
    const float span = static_cast<float>(area.w);
    const int zx = area.x + static_cast<int>(std::lround((0.0f - lo) / (hi - lo) * span));
    dl.fill_round_rect({area.x, all.y, area.w, all.h}, static_cast<float>(st_.radius), 0xFF191C22u);
    for (int i = 0; i < n; ++i) {
        const Bar& B = bars[i];
        const int y = all.y + i * bh;
        const Rect row{all.x, y, all.w, bh};
        const bool hov = row.contains(in_.mouse_x, in_.mouse_y) && hoverable(row);
        if (hov) dl.fill_rect({all.x, y, all.w, bh}, 0x10FFFFFFu);
        label_left({all.x, y, lw - 6, bh}, B.label, B.label ? static_cast<int>(std::strlen(B.label)) : 0, hov ? st_.text_strong : st_.text);
        const float v = std::isfinite(B.value) ? B.value : 0.0f;
        const int vx = area.x + static_cast<int>(std::lround((v - lo) / (hi - lo) * span));
        const u32 col = B.color ? B.color : (v >= 0 ? st_.accent : st_.negative);
        const int x0 = min_(zx, vx), x1 = max_(zx, vx);
        const int bar_h = bh - 8;
        if (x1 > x0) dl.fill_round_rect({x0, y + 4, max_(x1 - x0, 2), bar_h}, 2.0f, hov ? render::lerp_color(col, 0xFFFFFFFFu, 0.2f) : col);
        char vb[40];
        std::snprintf(vb, sizeof vb, f ? f : "%+.3f", static_cast<double>(B.value));
        const int tw = render::text_width(vb, st_.font);
        dl.text(all.x + all.w - tw, y + bh / 2 - font_center(st_.font), vb, v >= 0 ? st_.text_strong : 0xFFFFB199u, st_.font);
    }
    dl.fill_rect({zx, all.y - 2, 1, all.h + 4}, 0xFF6B7280u);
}

void Context::colorbar(const char* label, render::Colormap cm, float lo, float hi, const char* unit) {
    flush_row();
    DrawList& dl = layers_[0];
    const Rect hr = next_rect(st_.text_h);
    const int ty = hr.y + hr.h / 2 - font_center(st_.font);
    if (label) dl.text(hr.x, ty, label, st_.text_dim, st_.font, display_len(label));
    if (unit) { const int uw = render::text_width(unit, st_.font); dl.text(hr.x + hr.w - uw, ty, unit, st_.text_disabled, st_.font); }
    const int bh = static_cast<int>(12 * st_.scale);
    const Rect br = next_rect(bh);
    dl.blit_scaled(render::colormap_lut(cm), 256, 1, 256, br, false);
    dl.rect(br, 0x60000000u, 1);
    // Marcas y etiquetas (5 divisiones).
    const Rect lr = next_rect(st_.text_h);
    for (int k = 0; k <= 4; ++k) {
        const int x = br.x + (br.w - 1) * k / 4;
        dl.fill_rect({x, br.y + br.h, 1, 3}, st_.text_dim);
    }
    char b0[32], b1[32], b2[32];
    std::snprintf(b0, sizeof b0, "%.3g", static_cast<double>(lo));
    std::snprintf(b1, sizeof b1, "%.3g", static_cast<double>(0.5f * (lo + hi)));
    std::snprintf(b2, sizeof b2, "%.3g", static_cast<double>(hi));
    const int y = lr.y + 2;
    dl.text(lr.x, y, b0, st_.text, st_.font);
    dl.text(lr.x + (lr.w - render::text_width(b1, st_.font)) / 2, y, b1, st_.text, st_.font);
    dl.text(lr.x + lr.w - render::text_width(b2, st_.font), y, b2, st_.text, st_.font);
    item_add(get_id(label ? label : "##colorbar"), br, false);
}

} // namespace cfd::ui
