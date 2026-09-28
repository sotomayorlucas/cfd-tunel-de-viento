// ============================================================================
//  render/draw2d.hpp — dibujo 2D por software sobre Framebuffer (contrato del
//  módulo "ui + plataforma").
//
//  Dos formas de uso:
//   * Painter  → dibujo INMEDIATO (cada llamada escribe píxeles ya) con pila
//                de recortes (clip). Útil para overlays sencillos de la app.
//   * DrawList → GRABA comandos (POD, sin asignaciones tras el calentamiento)
//                y los reproduce después en PARALELO por franjas horizontales
//                (cada hilo del pool pinta todos los comandos recortados a su
//                franja → mismo resultado, bit a bit, que en serie). La UI
//                inmediata usa varias DrawList como capas (panel, popups, tooltips).
//
//  Colores 0xAARRGGBB (A = opacidad: 255 opaco → ruta rápida AVX2 sin mezcla).
//  Coordenadas en píxeles del framebuffer; los rectángulos son enteros y las
//  primitivas con antialias (líneas, círculos, triángulos, esquinas) usan float
//  con el CENTRO del píxel (x, y) en (x + 0.5, y + 0.5).
//
//  Texto: fuentes bitmap Terminus embebidas (render/font_data.hpp): pequeña
//  8×16 y grande 16×32, Latin-1 + ~30 glifos extra (griego Δ ρ ν ω α β θ σ Ω τ,
//  flechas, ≈ ≤ ≥ ∞ √ ≠ … – — • ▲ ▼ ▸ ▾ ◂ ▴ ✓). Entrada UTF-8: se decodifica a
//  índices de glifo (1 byte), así "¿Qué ñandú?" se ve bien. Ancho fijo → medir
//  texto = contar puntos de código (SWAR + popcount).
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "framebuffer.hpp"
#include <vector>

namespace cfd::render {

// ---- Fuentes ------------------------------------------------------------------------
enum class Font : u8 { Small = 0, Large = 1 };   // 8×16 / 16×32
enum class HAlign : u8 { Left, Center, Right };
enum class VAlign : u8 { Top, Middle, Bottom };

CFD_INLINE constexpr int font_w(Font f) { return f == Font::Small ? 8 : 16; }
CFD_INLINE constexpr int font_h(Font f) { return f == Font::Small ? 16 : 32; }
// Fila (desde arriba de la celda del glifo) del centro óptico de mayúsculas/minúsculas:
// para centrar texto verticalmente en una caja: y = centro_caja - font_center(f).
CFD_INLINE constexpr int font_center(Font f) { return f == Font::Small ? 7 : 16; }
// Línea base (primera fila bajo las letras sin descendente).
CFD_INLINE constexpr int font_baseline(Font f) { return f == Font::Small ? 12 : 26; }

// Índice de glifo (0..255) de un punto de código Unicode: Latin-1 → cp-0x20,
// extras → 224+k, equivalencias (comillas tipográficas, μ griega...) y '?' si no existe.
u8 glyph_index(char32_t cp);
// Decodifica UTF-8 (nbytes < 0 → hasta el '\0') a índices de glifo. Devuelve cuántos escribió
// (como mucho `cap`). Secuencias inválidas → '?'. No reserva memoria.
int utf8_to_glyphs(const char* s, int nbytes, u8* out, int cap);
// Nº de glifos (puntos de código) de una cadena UTF-8. SWAR: cuenta bytes que no son de
// continuación (10xxxxxx) de 8 en 8 con popcount.
int utf8_count(const char* s, int nbytes = -1);
CFD_INLINE int text_width(const char* s, Font f = Font::Small) { return utf8_count(s) * font_w(f); }
CFD_INLINE int text_width(const char* s, int nbytes, Font f) { return utf8_count(s, nbytes) * font_w(f); }
// Byte donde cortar para que quepan `max_glyphs` glifos (sin partir secuencias UTF-8).
int utf8_prefix_bytes(const char* s, int nbytes, int max_glyphs);

// ---- Utilidades de color ----------------------------------------------------------------
CFD_INLINE constexpr u32 with_alpha(u32 c, u32 a) { return (c & 0x00FFFFFFu) | (a << 24); }
CFD_INLINE constexpr u32 mul_alpha(u32 c, float k) {
    const float a = static_cast<float>(c >> 24) * k;
    const u32 ai = a <= 0.0f ? 0u : (a >= 255.0f ? 255u : static_cast<u32>(a + 0.5f));
    return with_alpha(c, ai);
}
// Interpolación por canal (incluye alfa), t ∈ [0,1].
u32 lerp_color(u32 a, u32 b, float t);
// Mezcla SWAR de un píxel: a256 ∈ [0,256] (256 = src). Canal alfa del destino = 0xFF.
CFD_INLINE u32 blend_px(u32 dst, u32 src, u32 a256) {
    const u32 ia = 256u - a256;
    const u32 rb = (((src & 0x00FF00FFu) * a256 + (dst & 0x00FF00FFu) * ia) >> 8) & 0x00FF00FFu;
    const u32 g  = (((src & 0x0000FF00u) * a256 + (dst & 0x0000FF00u) * ia) >> 8) & 0x0000FF00u;
    return 0xFF000000u | rb | g;
}
// 0..255 → 0..256 (255 ↦ 256 exacto: el color opaco no se "oscurece" 1/256).
CFD_INLINE constexpr u32 alpha256(u32 a) { return a + (a >> 7); }

// ---- Painter (inmediato) --------------------------------------------------------------------
class Painter {
public:
    Painter() = default;
    explicit Painter(Framebuffer& fb) { reset(fb); }
    void reset(Framebuffer& fb);
    Framebuffer& fb() { return *fb_; }

    // Recorte: siempre ⊆ framebuffer. push_clip intersecta con el actual (pila de 32).
    void push_clip(Rect r);
    void pop_clip();
    void set_clip(Rect r);                       // reemplaza (intersectado con el framebuffer)
    Rect clip() const { return {cx0_, cy0_, cx1_ - cx0_, cy1_ - cy0_}; }
    bool clip_empty() const { return cx0_ >= cx1_ || cy0_ >= cy1_; }

    // Rectángulos (enteros). A=255 → stores AVX2 sin leer destino; 0 < A < 255 → mezcla AVX2.
    void fill_rect(Rect r, u32 c);
    void rect(Rect r, u32 c, int thickness = 1);            // contorno
    void hline(int x0, int x1, int y, u32 c);               // [x0, x1)
    void vline(int x, int y0, int y1, u32 c);               // [y0, y1)
    // Rectángulo redondeado con esquinas antialias. corners: bit0 sup-izq, bit1 sup-der,
    // bit2 inf-izq, bit3 inf-der (los demás quedan en ángulo recto).
    void fill_round_rect(Rect r, float radius, u32 c, u32 corners = 0xF);
    void round_rect(Rect r, float radius, u32 c, int thickness = 1, u32 corners = 0xF);
    // Sombra suave (caja redondeada difuminada `blur` px). skip_inside: no pinta el interior
    // (el objeto opaco lo tapará) → coste ∝ perímetro × blur.
    void shadow(Rect r, float radius, float blur, u32 c, bool skip_inside = true);
    // Degradados (con alfa si los colores lo tienen).
    void gradient_v(Rect r, u32 top, u32 bottom);
    void gradient_h(Rect r, u32 left, u32 right);

    // Primitivas antialias (float, centro de píxel en +0.5).
    void line(float x0, float y0, float x1, float y1, u32 c, float width = 1.0f);
    // Polilínea con uniones perfectas: cobertura MÁXIMA por fila (sin doble mezcla en vértices).
    void polyline(const Vec2* pts, int n, u32 c, float width = 1.0f);
    void fill_circle(float cx, float cy, float r, u32 c);
    void circle(float cx, float cy, float r, u32 c, float width = 1.0f);
    void fill_triangle(Vec2 a, Vec2 b, Vec2 p, u32 c);
    // Área bajo una curva (x creciente) hasta y = base_y, con degradado vertical desde
    // c_top (en el punto más alto de la curva) hasta c_base; borde superior antialias.
    void fill_area(const Vec2* pts, int n, float base_y, u32 c_top, u32 c_base);

    // Imágenes ARGB. alpha=false → copia de filas; true → mezcla por píxel (alfa directo, no
    // premultiplicado). opacity multiplica el alfa.
    void blit(const u32* src, int sw, int sh, int sstride, int dx, int dy, bool alpha = false, u32 opacity = 255);
    // Escalado vecino más próximo a `dst` (p.ej. LUT 256×1 → barra de colores).
    void blit_scaled(const u32* src, int sw, int sh, int sstride, Rect dst, bool alpha = false);

    // Texto UTF-8 (nbytes < 0 → hasta '\0'). (x, y) = esquina superior izquierda de la celda.
    // Devuelve la x final (x + ancho). Glifo opaco: 1 VPMASKMOVD por fila de 8 píxeles.
    int text(int x, int y, const char* s, u32 c, Font f = Font::Small, int nbytes = -1);
    int text_glyphs(int x, int y, const u8* glyphs, int n, u32 c, Font f = Font::Small);
    // Texto con sombra de 1 px (legible sobre el render 3D).
    int text_shadow(int x, int y, const char* s, u32 c, u32 shadow = 0xC0000000u, Font f = Font::Small);
    // Alineado dentro de una caja (sin recortar a la caja: usar push_clip si hace falta).
    void text_aligned(Rect box, const char* s, u32 c, HAlign h, VAlign v = VAlign::Middle, Font f = Font::Small);

private:
    void span_fill(u32* d, int n, u32 c);              // n píxeles de color c (A decide ruta)
    void span_blend_cov(u32* d, int x0, int n, const float* cov, u32 c);
    Framebuffer* fb_ = nullptr;
    int cx0_ = 0, cy0_ = 0, cx1_ = 0, cy1_ = 0;       // recorte [x0,x1) × [y0,y1)
    Rect stack_[32];
    int depth_ = 0;
};

// ---- DrawList (grabación + reproducción paralela) -------------------------------------------
enum class CmdType : u8 {
    FillRect, Rect, RoundRect, RoundRectOutline, Shadow, GradientV, GradientH,
    Line, Polyline, FillCircle, Circle, Triangle, Blit, BlitScaled, Text, TextShadow, Area,
};

struct DrawCmd {
    CmdType type;
    u8 font = 0;
    u8 corners = 0xF;
    u8 flags = 0;
    u32 c0 = 0, c1 = 0;
    i32 clip[4];             // recorte x0, y0, x1, y1 (exclusivo)
    i32 y0, y1;              // extensión vertical (para descartar por franja)
    float f[6];              // geometría (según el tipo)
    u32 off = 0, n = 0;      // datos: glifos / puntos / índice de imagen
};

class DrawList {
public:
    DrawList();
    void clear();                                 // conserva la capacidad (sin liberar)
    bool empty() const { return cmds_.empty(); }
    usize size() const { return cmds_.size(); }

    void push_clip(Rect r);                       // intersecta con el recorte actual
    void pop_clip();
    Rect clip() const { return {clip_[0], clip_[1], clip_[2] - clip_[0], clip_[3] - clip_[1]}; }
    // Recorte base (todo lo grabado se limita a esto; por defecto "infinito", el
    // framebuffer recorta al reproducir).
    void set_base_clip(Rect r);

    void fill_rect(Rect r, u32 c);
    void rect(Rect r, u32 c, int thickness = 1);
    void fill_round_rect(Rect r, float radius, u32 c, u32 corners = 0xF);
    void round_rect(Rect r, float radius, u32 c, int thickness = 1, u32 corners = 0xF);
    void shadow(Rect r, float radius, float blur, u32 c, bool skip_inside = true);
    void gradient_v(Rect r, u32 top, u32 bottom);
    void gradient_h(Rect r, u32 left, u32 right);
    void line(float x0, float y0, float x1, float y1, u32 c, float width = 1.0f);
    void polyline(const Vec2* pts, int n, u32 c, float width = 1.0f);
    void fill_circle(float cx, float cy, float r, u32 c);
    void circle(float cx, float cy, float r, u32 c, float width = 1.0f);
    void fill_triangle(Vec2 a, Vec2 b, Vec2 p, u32 c);
    void fill_area(const Vec2* pts, int n, float base_y, u32 c_top, u32 c_base);
    // La imagen NO se copia: debe seguir viva hasta render().
    void blit(const u32* src, int sw, int sh, int sstride, int dx, int dy, bool alpha = false, u32 opacity = 255);
    void blit_scaled(const u32* src, int sw, int sh, int sstride, Rect dst, bool alpha = false);
    int text(int x, int y, const char* s, u32 c, Font f = Font::Small, int nbytes = -1);
    int text_shadow(int x, int y, const char* s, u32 c, u32 shadow = 0xC0000000u, Font f = Font::Small);
    void text_aligned(Rect box, const char* s, u32 c, HAlign h, VAlign v = VAlign::Middle, Font f = Font::Small);

    // Reproduce en un Painter (serie, respeta el recorte del Painter).
    void execute(Painter& p) const;
    // Reproduce varias listas en orden (capas) en paralelo por franjas de filas.
    // bands ≤ 0 → automático (≈ 2 franjas por hilo, mínimo 16 filas por franja).
    static void render(Framebuffer& fb, const DrawList* const* lists, int nlists, int bands = 0);
    void render(Framebuffer& fb, int bands = 0) const { const DrawList* l = this; render(fb, &l, 1, bands); }

    // Estadísticas (para el HUD de rendimiento).
    usize glyph_bytes() const { return glyphs_.size(); }

private:
    struct Image { const u32* p; int w, h, stride; };
    DrawCmd& push(CmdType t, int y0, int y1, u32 c0);
    bool culled(int x0, int y0, int x1, int y1) const {
        return x1 <= clip_[0] || y1 <= clip_[1] || x0 >= clip_[2] || y0 >= clip_[3] || clip_[0] >= clip_[2] || clip_[1] >= clip_[3];
    }
    void exec_one(Painter& p, const DrawCmd& c) const;
    std::vector<DrawCmd> cmds_;
    std::vector<u8> glyphs_;
    std::vector<Vec2> pts_;
    std::vector<Image> images_;
    i32 clip_[4];
    i32 base_[4];
    i32 stack_[32][4];
    int depth_ = 0;
};

} // namespace cfd::render
