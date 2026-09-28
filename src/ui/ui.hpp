// ============================================================================
//  ui/ui.hpp — GUI inmediata (estilo Dear ImGui) para el túnel de viento CFD.
//
//  Uso por cuadro (hilo principal):
//      ui.begin_frame(input, fb.w, fb.h, now_sec());
//      if (ui.begin_panel("panel", {fb.w - 400, 0, 400, fb.h})) {
//          if (ui.header("Túnel")) { ui.slider_float("Velocidad", &v, 50, 350, "%.0f", "km/h"); ... }
//          ui.end_panel();                       // (llamar SIEMPRE tras begin_panel)
//      }
//      ui.end_frame();
//      if (!ui.wants_mouse()) { ...cámara orbital... }
//      ...render 3D en fb...
//      ui.render(fb);                            // capas: panel → popups → tooltips/avisos (paralelo)
//
//  * Ids: FNV-1a de la etiqueta + pila de ids. "Texto##x" muestra "Texto" y usa
//    todo para el id; "Texto###x" usa sólo "###x" (la etiqueta puede cambiar).
//    Las cabeceras de sección usan un id "salado": una sección "Modelo" y un combo
//    "Modelo" no chocan. Sin NDEBUG se avisa (stderr) de ids duplicados en un cuadro.
//  * Lógica hot/active: el elemento bajo el ratón es "hot"; al pulsar pasa a
//    "active" y captura el ratón hasta soltar (arrastrar un slider fuera del
//    panel sigue funcionando y la cámara no orbita).
//  * Teclado: Tab / Mayús+Tab recorren los widgets; Espacio/Intro activan;
//    ←/→ ajustan sliders (Mayús = fino); Intro en un slider = escribir valor;
//    Esc quita el foco. Doble clic o Ctrl+clic en un slider = escribir valor.
//  * Todo el dibujo se GRABA en DrawList (sin asignaciones tras calentar) y se
//    rasteriza al final en paralelo por franjas.
//  * Escala: Style::dark(escala) — métricas × escala, fuente 16×32 si escala ≥ 1.5.
// ============================================================================
#pragma once

#include "../platform/platform.hpp"
#include "../render/colormap.hpp"
#include "../render/draw2d.hpp"
#include <cstdarg>

namespace cfd::ui {

using Id = u64;
using render::Rect;

struct Style {
    // --- Colores (0xAARRGGBB) ---
    u32 panel = 0xFF15171Cu;          // fondo del panel lateral
    u32 panel_edge = 0xFF262A32u;     // borde / línea divisoria
    u32 header = 0xFF1C1F26u;         // cabecera de sección
    u32 header_hover = 0xFF232731u;
    u32 text = 0xFFD9DDE5u;
    u32 text_dim = 0xFF8B93A2u;
    u32 text_disabled = 0xFF5A616Eu;
    u32 text_strong = 0xFFF4F6FAu;
    u32 frame = 0xFF23272Fu;          // fondo de controles
    u32 frame_hover = 0xFF2B303Au;
    u32 frame_active = 0xFF323845u;
    u32 border = 0xFF323743u;
    u32 accent = 0xFF3D9BFFu;         // azul "ingeniería"
    u32 accent_hover = 0xFF5BACFFu;
    u32 accent_dim = 0xFF1F4F85u;
    u32 accent_text = 0xFFFFFFFFu;    // texto sobre acento
    u32 warn = 0xFFFFB547u;
    u32 error = 0xFFFF5C5Cu;
    u32 success = 0xFF3DDC97u;
    u32 negative = 0xFFFF7A59u;       // barras negativas (resistencia, sustentación positiva)
    u32 popup = 0xFF1E2129u;
    u32 tooltip = 0xF0262A33u;
    u32 shadow = 0x90000000u;
    u32 separator = 0xFF262A32u;
    u32 plot_bg = 0xFF191C22u;
    u32 grid = 0xFF272B34u;
    u32 series[6] = {0xFF4AA8FFu, 0xFFFF7A45u, 0xFF3DDC97u, 0xFFC678DDu, 0xFFFFD166u, 0xFF56D4DDu};
    // --- Métricas (píxeles, ya escaladas) ---
    float scale = 1.0f;
    int pad = 12;                     // margen interior del panel
    int spacing = 5;                  // separación vertical entre widgets
    int row_h = 24;                   // alto de una fila de control
    int text_h = 18;                  // alto de una línea de texto
    int header_h = 30;
    int radius = 4;                   // redondeo de controles
    int label_frac = 42;              // % del ancho para la etiqueta en sliders/combos
    int check = 16;                   // lado de checkbox/radio
    int scrollbar_w = 6;
    int indent = 8;
    render::Font font = render::Font::Small;
    render::Font font_big = render::Font::Large;
    bool smooth_scroll = true;

    static Style dark(float scale = 1.0f);
    int fw() const { return render::font_w(font); }
    int fh() const { return render::font_h(font); }
};

// Serie para plot_lines. Búfer circular: el elemento i (0 = más antiguo) está en
// data[(offset + i) mod count] (offset se normaliza: puede ser negativo o ≥ count).
// data nulo o count ≤ 0 → serie vacía. plot_lines dibuja como mucho 8 series.
struct PlotSeries {
    const float* data = nullptr;
    int count = 0;
    int offset = 0;
    u32 color = 0;                    // 0 → paleta del estilo
    const char* name = nullptr;
};
// Barra horizontal con signo (desglose por componente).
struct Bar {
    const char* label = nullptr;
    float value = 0;
    u32 color = 0;                    // 0 → acento (positivo) / negativo
};

enum class ButtonKind : u8 { Normal, Primary, Danger, Subtle };

class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    Style& style() { return st_; }
    const Style& style() const { return st_; }
    void set_scale(float s);          // conserva los colores, recalcula métricas

    // ---- Cuadro ------------------------------------------------------------------------
    void begin_frame(const platform::Input& in, int fb_w, int fb_h, double time_s);
    void end_frame();
    void render(render::Framebuffer& fb);             // reproduce las capas en paralelo
    bool wants_mouse() const { return wants_mouse_; }
    // true si la UI está usando el teclado: editando un valor, con foco de Tab visible, o si
    // este cuadro consumió Esc (cerró un popup / canceló la edición / quitó el foco). Un clic
    // en el visor abandona el foco de teclado.
    bool wants_keyboard() const { return wants_keyboard_; }
    bool popup_open() const { return popup_id_ != 0; }

    // ---- Ids -------------------------------------------------------------------------------
    Id get_id(const char* label) const;
    void push_id(const char* s);
    void push_id(int i);
    void pop_id();

    // ---- Paneles y distribución ------------------------------------------------------------
    // Panel fijo con desplazamiento vertical (rueda o barra) cuando el contenido no cabe.
    bool begin_panel(const char* name, Rect r);
    void end_panel();
    void row(int ncols);              // los próximos ncols widgets comparten una fila
    void spacing(int px = -1);
    void separator();
    void indent(int px = -1);
    void unindent(int px = -1);
    Rect next_rect(int h);            // reserva una fila (para dibujo propio con draw())
    render::DrawList& draw() { return layers_[0]; }   // capa principal (recorte = panel)
    int content_width() const { return cur_w_; }

    // ---- Widgets -----------------------------------------------------------------------------
    void title(const char* text, const char* subtitle = nullptr);
    bool header(const char* label, bool default_open = true);            // sección plegable
    void label(const char* text, u32 color = 0);
    void text(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
    void text_colored(u32 color, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
    void text_wrapped(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
    // (extensión compatible, fase 2 / app) texto ajustado al ancho con color propio (avisos, fuentes)
    void text_wrapped_colored(u32 color, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
    void value(const char* label, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
    void value_colored(const char* label, u32 color, const char* fmt, ...) __attribute__((format(printf, 4, 5)));
    void metric(const char* label, const char* value, const char* unit = nullptr, u32 color = 0);
    bool button(const char* label, ButtonKind kind = ButtonKind::Normal);
    bool toggle_button(const char* label, bool* v);
    bool toggle(const char* label, bool* v);                             // interruptor
    bool checkbox(const char* label, bool* v);
    bool radio(const char* label, int* v, int value);
    bool segmented(const char* id, const char* const* items, int n, int* sel);
    bool slider_float(const char* label, float* v, float lo, float hi, const char* fmt = "%.2f", const char* unit = nullptr);
    bool slider_int(const char* label, int* v, int lo, int hi, const char* unit = nullptr);
    bool combo(const char* label, int* sel, const char* const* items, int n);
    void progress(float t, const char* overlay = nullptr, u32 color = 0);
    void tooltip(const char* fmt, ...) __attribute__((format(printf, 2, 3)));   // del último widget
    void plot_lines(const char* label, const PlotSeries* series, int nseries, int height,
                    const char* unit = nullptr, const char* fmt = "%.3f");
    void bar_chart(const char* label, const Bar* bars, int n, const char* unit = nullptr, const char* fmt = "%+.3f");
    void colorbar(const char* label, render::Colormap cm, float lo, float hi, const char* unit = nullptr);
    void toast(u32 color, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
    void set_toast_area(Rect r) { toast_area_ = r; }

    // ---- Consultas del último widget ---------------------------------------------------------
    bool item_hovered() const { return last_hovered_; }
    bool item_active() const { return last_id_ != 0 && active_id_ == last_id_; }
    Rect last_rect() const { return last_rect_; }
    Id active_id() const { return active_id_; }
    Id hot_id() const { return hot_id_; }
    Id focus_id() const { return focus_id_; }
    bool editing() const { return edit_id_ != 0; }

    // Almacenamiento persistente por id (estado de secciones, animaciones...).
    float* state_f(Id id, float init);
    int* state_i(Id id, int init);

    struct Stats {
        int widgets = 0;
        usize cmds = 0;
        double build_ms = 0, render_ms = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    struct PanelState;
    struct Slot { Id key; float f; int i; };
    struct Toast { char text[120]; u32 color; double t0, t1; };

    // Comportamiento de interacción
    bool hoverable(Rect r) const;
    bool item_add(Id id, Rect r, bool focusable = true);                 // devuelve hovered
    bool press_behavior(Id id, Rect r, bool* hovered, bool* held, bool focusable = true);   // clic completo
    bool key_activate(Id id) const;
    void set_active(Id id);
    void clear_active();
    void focus_ring(Id id, Rect r);
    const char* fmtv(const char* fmt, va_list ap);
    const char* fmt(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
    int display_len(const char* label) const;                            // bytes antes de "##"
    Slot* slot(Id id);
    PanelState* panel_state(Id id);
    bool slider_core(const char* label, float* v, float lo, float hi, bool is_int, const char* fmt, const char* unit);
    bool text_edit(Id id, Rect r, char* buf, int cap);                   // true al confirmar
    void draw_tooltip();
    void draw_toasts();
    void label_left(Rect r, const char* label, int nbytes, u32 color);
    void wrapped_core(u32 color, const char* s);

    Style st_;
    render::DrawList layers_[3];      // 0 panel, 1 popups, 2 tooltips y avisos
    platform::Input in_;              // copia del cuadro
    double time_ = 0, dt_ = 0;
    int fb_w_ = 0, fb_h_ = 0;
    u32 frame_ = 0;

    // Ids
    Id id_stack_[32];
    int id_depth_ = 0;
    Id hot_id_ = 0, active_id_ = 0, last_hot_ = 0;
    bool active_alive_ = false;
    Id focus_id_ = 0;
    bool focus_visible_ = false;
    bool focus_scroll_ = false;
    Id focus_list_[512];
    int focus_n_ = 0;
    Id last_id_ = 0;
    Rect last_rect_{};
    bool last_hovered_ = false;
    double hover_t0_ = 0;
    char tip_[512] = {};
    bool tip_pending_ = false;

    // Ratón
    bool ui_owns_ = false, app_owns_ = false;
    bool click_consumed_ = false;
    bool esc_consumed_ = false;
    bool wants_mouse_ = false, wants_keyboard_ = false;
    Rect ui_rects_[16];
    int ui_nrects_ = 0;
    Rect ui_rects_prev_[16];
    int ui_nrects_prev_ = 0;
    float drag_grab_ = 0;             // desplazamiento de agarre (barra de scroll)
    Id last_click_id_ = 0;
    float last_click_value_ = 0;
    bool last_click_was_slider_ = false;

    // Popup (combo)
    Id popup_id_ = 0;
    Rect popup_rect_{};
    bool popup_opened_now_ = false;
    float popup_scroll_ = 0;

    // Edición de texto (valor tecleado en sliders)
    Id edit_id_ = 0;
    char edit_buf_[64] = {};
    int edit_len_ = 0, edit_cur_ = 0;
    bool edit_select_all_ = false;
    bool edit_commit_ = false, edit_cancel_ = false;

    // Paneles / distribución
    PanelState* panels_ = nullptr;
    int npanels_ = 0;
    PanelState* cur_panel_ = nullptr;
    int cur_x_ = 0, cur_y_ = 0, cur_w_ = 0;
    int row_cols_ = 0, row_idx_ = 0, row_y_ = 0, row_maxh_ = 0;

    // Avisos
    Toast toasts_[8];
    int ntoasts_ = 0;
    Rect toast_area_{0, 0, 0, 0};

    // Almacenamiento
    Slot* slots_ = nullptr;
    static constexpr int k_slots = 2048;

    char fmtbuf_[1024];
    Stats stats_;
    double t_begin_ = 0;
    int cur_layer_ = 0;               // capa en la que se registran los widgets (1 = popup)
    bool in_popup_ = false;
    bool popup_alive_ = false;
    bool edit_alive_ = false;
    Rect edit_rect_{};
    std::vector<Vec2> scratch_;       // puntos de gráficas (reutilizado, sin asignar por cuadro)
    void flush_row();
    void ensure_visible(Rect r);
    int gap_px() const { return static_cast<int>(8.0f * st_.scale); }   // casilla → etiqueta
    Id* seen_ = nullptr;              // detector de ids duplicados (sólo sin NDEBUG)
    bool dup_warned_ = false;
};

} // namespace cfd::ui
