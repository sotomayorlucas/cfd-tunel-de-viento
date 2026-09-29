// ============================================================================
//  render/flowvis.hpp — visualización del flujo (contrato público del módulo
//  "flowvis"). Implementación en render/flowvis.cpp (magnitudes, planos de
//  corte, malla coloreada, huella en el suelo, estela, sondas),
//  render/flowvis_lines.cpp (muestreador FP16, líneas de corriente, humo),
//  render/flowvis_volume.cpp (volumen de vórtices por raymarching) y
//  render/flowvis_draw.cpp (llamadas al rasterizador; separado para que los
//  cálculos enlacen sin render/raster*.cpp).
//
//  Entrada: lbm::FieldView (ρ, u en unidades de red, flags, u∞).
//  Salida: dibujo en el Framebuffer mediante render/raster.hpp
//  (draw_textured_quad, draw_polylines, draw_points, draw_ground) o, en el
//  caso del volumen, composición directa sobre fb.color respetando fb.depth.
//
//  USO TÍPICO POR CUADRO (app):
//
//      // Una vez:
//      flowvis::FlowSampler sampler;        // copia empaquetada FP16 del campo
//      flowvis::SliceView slice;            // plano de corte
//      flowvis::Streamlines lines;          // líneas de corriente
//      flowvis::Particles smoke;            // humo
//      flowvis::VortexVolume vortices;      // volumen de vórtices (Q)
//      flowvis::GroundFootprint footprint;  // huella de Cp en el suelo
//      lines.set_rakes(...); smoke.set_emitters(...);
//
//      // Cuando el solver ha escrito ρ,u nuevos (tras solver.step(k)):
//      const lbm::FieldView f = solver.field();
//      sampler.update(f);                   // ~1 ms en 256×128×96 (paralelo, AVX2+F16C)
//      slice.update(f);                     // textura del corte
//      lines.compute(sampler);              // si están activas
//      smoke.step(sampler, k);              // k = pasos de red avanzados
//      vortices.update(f);                  // si está activo (Q + ladrillos)
//      flowvis::color_mesh(mesh, f, surf);  // colores por vértice de la malla
//
//      // Render (después de la malla y el suelo, antes de la UI):
//      footprint.draw(fb, cam, spacing, offset_x, base, line);   // en lugar de draw_ground liso
//      slice.draw(fb, cam);                 // opaco → escribe profundidad; translúcido → no
//      const flowvis::TranslucentPlane tp = slice.translucent_plane();   // opacity 0 si es opaco
//      const std::span<const flowvis::TranslucentPlane> behind(&tp, 1);
//      lines.draw(fb, cam, true, behind);   // lo que quede detrás del corte se atenúa
//      smoke.draw(fb, cam, behind);
//      vortices.render(fb, cam, behind);    // último: usa la profundidad de la escena opaca
//
//  Sondas de ratón: slice.pick(cam, mx, my, hit, valor) o flowvis::probe(f, punto).
//  Leyendas: slice.effective_scale() (rango tras el automático) + quantity_info(q).
//
//  HILOS: todas las funciones de actualización son paralelas internamente
//  (cfd::pool()) y deben llamarse desde el hilo principal (no son reentrantes
//  sobre el mismo objeto). Las funciones de muestreo `const` son hilo-seguras.
//  Sin asignaciones en las rutas calientes: los buffers se redimensionan sólo
//  cuando cambian tamaños/capacidades.
//
//  NORMALIZACIÓN de magnitudes (todas adimensionales):
//    velocidades / U∞;  Cp = 2(ρ-1)/(3U∞²);  Cp0 (presión total) = Cp + ρ|u|²/U∞²
//    (= 1 en corriente libre, < 1 donde hay pérdidas: estela);
//    vorticidad |ω|·Δx/U∞  y  Q·Δx²/U∞²  (Δx = 1 celda: dependen de la resolución;
//    usar el rango automático o escalar el rango al cambiar la malla).
// ============================================================================
#pragma once

#include "../core/mem.hpp"
#include "../core/simd.hpp"
#include "../lbm/field.hpp"
#include "camera.hpp"
#include "colormap.hpp"
#include "framebuffer.hpp"
#include "mesh.hpp"
#include <span>
#include <vector>

namespace cfd::flowvis {

using render::Camera;
using render::Colormap;
using render::Framebuffer;
using render::Mesh;
using render::Rect;

// ============================================================================
//  Escalas de color
// ============================================================================
// Rango + mapa. `diverging` (sólo si lo < 0 < hi): el 0 cae exactamente en el
// centro del mapa aunque el rango sea asimétrico (p.ej. Cp ∈ [-2.5, 1]).
struct ColorScale {
    Colormap map = Colormap::Turbo;
    float lo = 0.0f, hi = 1.0f;
    bool diverging = false;
    // Se resta al valor ANTES de mapearlo (lo/hi y el pivote 0 quedan en unidades "corregidas"). La app lo usa
    // para referir Cp/Cp0 a la presión estática de referencia del túnel (Sim::rho_ref) en vez de a ρ = 1:
    // offset = 2(ρ_ref − 1)/(3u∞²). 0 = sin cambio (añadido en la fase de integración; ver docs/dev/).
    float offset = 0.0f;

    CFD_INLINE bool pivoted() const { return diverging && lo < 0.0f && hi > 0.0f; }
    // Posición normalizada t (sin sujetar) del valor v en el mapa.
    CFD_INLINE float normalize(float v) const {
        v -= offset;
        if (pivoted()) return v < 0.0f ? 0.5f - 0.5f * v / lo : 0.5f + 0.5f * v / hi;
        return (v - lo) / (hi - lo);
    }
};

// Color 0xFFRRGGBB para v (NaN → primer color del mapa). Misma aritmética (FMA) y el
// mismo redondeo (CVTSS2SI = al par más cercano) que map_color8 → resultados idénticos.
CFD_INLINE u32 map_color(const ColorScale& s, float v) {
    float t;
    v -= s.offset;
    if (s.pivoted()) t = __builtin_fmaf(v, v < 0.0f ? 127.5f / -s.lo : 127.5f / s.hi, 127.5f);
    else t = (v - s.lo) * (255.0f / (s.hi - s.lo));
    t = t > 0.0f ? (t < 255.0f ? t : 255.0f) : 0.0f;       // NaN → 0
    return render::colormap_lut(s.map)[_mm_cvtss_si32(_mm_set_ss(t))];
}
// 8 valores a la vez (FMA + VPGATHERDD sobre la LUT de 256). NaN → índice 0.
CFD_INLINE __m256i map_color8(const ColorScale& s, simd::f8 v) {
    using simd::f8;
    f8 t;
    v = v - f8(s.offset);   // (NaN − x = NaN: se conserva el índice 0 de los NaN)
    if (s.pivoted()) {
        const f8 kneg(127.5f / -s.lo), kpos(127.5f / s.hi);
        t = simd::fmadd(v, simd::select(v < f8::zero(), kneg, kpos), f8(127.5f));
    } else {
        t = (v - f8(s.lo)) * f8(255.0f / (s.hi - s.lo));
    }
    t = simd::min(simd::max(t, f8::zero()), f8(255.0f));   // max(NaN,0) = 0 (orden de operandos importa)
    return _mm256_i32gather_epi32(reinterpret_cast<const int*>(render::colormap_lut(s.map)), _mm256_cvtps_epi32(t.v), 4);
}

// ============================================================================
//  Ejes y planos translúcidos
// ============================================================================
enum class Axis : u8 { X = 0, Y = 1, Z = 2 };
inline const char* axis_name(Axis a) { return a == Axis::X ? "X" : (a == Axis::Y ? "Y" : "Z"); }

// Plano translúcido ya dibujado (p.ej. un corte con opacidad < 1, que NO escribe
// profundidad). Lo que quede DETRÁS de él visto desde la cámara se atenúa por
// (1 - opacity) al dibujarse después: líneas, humo y volumen se "ven a través" del
// corte en vez de aparecer delante. rect = extensión del plano en sus ejes (u, v)
// (mismos ejes que SliceView: X→(y,z), Y→(x,z), Z→(x,y)), en celdas.
struct TranslucentPlane {
    Axis axis = Axis::Y;
    float pos = 0.0f;
    float opacity = 0.0f;                 // 0 → sin efecto
    Vec2 lo{-1e30f, -1e30f}, hi{1e30f, 1e30f};
};
// Factor de transmisión (∏ 1-opacity) de los planos que cortan el segmento cámara→p.
float plane_transmission(const Camera& cam, Vec3 p, std::span<const TranslucentPlane> planes);

// ============================================================================
//  Magnitudes del flujo
// ============================================================================
enum class Quantity : u8 {
    Speed,       // |u|/U∞
    Ux,          // u_x/U∞ (corriente abajo; < 0 = recirculación)
    Uz,          // u_z/U∞ (vertical: upwash/downwash)
    Cp,          // coeficiente de presión estática
    Cp0,         // coeficiente de presión total (1 = sin pérdidas; estela < 1)
    Vorticity,   // |ω|·Δx/U∞
    QCriterion,  // Q·Δx²/U∞²  (> 0: núcleo de vórtice)
    Count
};

struct QuantityInfo {
    const char* name;        // nombre para la UI ("Velocidad |u|/U∞")
    const char* short_name;  // etiqueta corta para leyendas ("|u|/U∞")
    const char* description; // una frase de ayuda
    ColorScale scale;        // rango y mapa por defecto
};
const QuantityInfo& quantity_info(Quantity q);
inline const char* quantity_name(Quantity q) { return quantity_info(q).name; }
inline ColorScale default_scale(Quantity q) { return quantity_info(q).scale; }

// Valor de una magnitud en el centro de la celda (x,y,z). Al derivar se usan las
// velocidades que el solver escribe en las celdas sólidas (velocidad de pared: 0 si
// es fija, cinta/ruedas si se mueven). Celda sólida → NaN. Requiere flags != nullptr.
float cell_quantity(const lbm::FieldView& f, Quantity q, int x, int y, int z);
// Valor interpolado en un punto p (celdas): trilineal PONDERADO POR FLUIDO
// (las esquinas sólidas no cuentan). NaN si las 8 esquinas son sólidas.
float sample_quantity(const lbm::FieldView& f, Quantity q, Vec3 p);

// Sonda: todas las magnitudes en un punto (tooltip del ratón).
struct Probe {
    bool valid = false;      // dentro del dominio
    bool solid = false;      // la celda más cercana es sólida
    Vec3 pos{0, 0, 0};       // celdas
    Vec3 u{0, 0, 0};         // velocidad / U∞
    float speed = 0, cp = 0, cp0 = 0, vorticity = 0, q = 0;
};
Probe probe(const lbm::FieldView& f, Vec3 p);

// ============================================================================
//  Refinamiento local (lbm/refine.cpp): campo COMPUESTO de varias rejillas anidadas.
//  Todas las posiciones son celdas de la red BASE (el mundo de render); cada rejilla fina cubre su región propia
//  `inner` con celdas de scale·dx. Las velocidades de red son iguales en todos los niveles (escalado acústico); |ω|
//  y Q, que se miden "por celda", se reescalan a celdas de la base (×1/scale y ×1/scale²).
// ============================================================================
struct GridField {
    lbm::FieldView f;          // campos de la rejilla (sus celdas; capa fantasma con valores interpolados)
    Vec3 org{0, 0, 0};         // centro de su celda (0,0,0) en celdas de la base
    float scale = 1.0f;        // dx / dx de la base (= 2^−nivel)
    Aabb inner;                // región propia (celdas de la base): allí es la más fina
    int depth = 0;
    bool ground = false;       // su capa z = 0 es el suelo
    CFD_INLINE Vec3 to_grid(Vec3 p) const { return (p - org) * (1.0f / scale); }
};
struct MultiField {
    static constexpr int k_max = 9;
    GridField g[k_max];
    int n = 0;
    const lbm::FieldView& base() const { return g[0].f; }
    bool refined() const { return n > 1; }
    // Rejilla más fina cuya región propia contiene p (celdas de la base); 0 = la base.
    CFD_INLINE int finest(Vec3 p) const {
        int best = 0, bd = 0;
        for (int i = 1; i < n; ++i) {
            const Aabb& b = g[i].inner;
            if (g[i].depth > bd && p.x >= b.lo.x && p.x <= b.hi.x && p.y >= b.lo.y && p.y <= b.hi.y && p.z >= b.lo.z && p.z <= b.hi.z) {
                best = i;
                bd = g[i].depth;
            }
        }
        return best;
    }
};
MultiField single_field(const lbm::FieldView& f);   // una sola rejilla (red uniforme)
// Escala de una magnitud "por celda" de la rejilla g a celdas de la base (|ω|: 1/scale, Q: 1/scale²; resto 1).
float quantity_grid_scale(Quantity q, float scale);
float sample_quantity(const MultiField& m, Quantity q, Vec3 p);   // en la rejilla más fina que contiene p
Probe probe(const MultiField& m, Vec3 p);

// ============================================================================
//  FlowSampler — copia EMPAQUETADA del campo para muestreo aleatorio rápido.
//
//  Cada celda = 4 × FP16 (ux, uy, uz, ρ-1) = 8 bytes (F16C). Dos celdas
//  vecinas en X ocupan 16 bytes contiguos → la interpolación trilineal de las
//  4 componentes son 4 cargas de 128 bits + 4 VCVTPH2PS + 3 FMA en YMM, en vez
//  de 24 cargas dispersas de 3 arreglos SoA. Celdas sólidas: u = la velocidad
//  de pared que escribe el solver (0, cinta o rueda) y ρ-1 = media de sus
//  vecinas fluidas (así Cp cerca de paredes no se contamina con el ρ=1 del sólido).
//  Lo usan líneas de corriente, partículas y sondas rápidas.
// ============================================================================
class FlowSampler {
public:
    // Reempaqueta el campo (paralelo, AVX2). Llamar cada vez que el solver escribe ρ,u.
    void update(const lbm::FieldView& f);
    bool ready() const { return !cells_.empty(); }
    usize memory_bytes() const { return cells_.size() * sizeof(u64) + flags_.size(); }

    int nx = 0, ny = 0, nz = 0;
    float u_inf = 0.08f;

    // (ux, uy, uz, ρ-1) interpolados en p = (x, y, z, ·) celdas. p se sujeta al dominio.
    CFD_INLINE __m128 sample(__m128 p) const {
        const __m128 c = _mm_min_ps(_mm_max_ps(p, _mm_setzero_ps()), _mm_load_ps(hi_));
        const __m128i i = _mm_cvttps_epi32(c);
        const __m128 t = _mm_sub_ps(c, _mm_cvtepi32_ps(i));
        const usize n0 = static_cast<usize>(_mm_cvtsi128_si32(i)) +
                         sy_ * static_cast<usize>(_mm_extract_epi32(i, 1)) + sz_ * static_cast<usize>(_mm_extract_epi32(i, 2));
        const u16* b = reinterpret_cast<const u16*>(cells_.data() + n0);
        const usize oy = 4 * sy_, oz = 4 * sz_;
        const __m256 r00 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
        const __m256 r10 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + oy)));
        const __m256 r01 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + oz)));
        const __m256 r11 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + oy + oz)));
        const __m256 ty = _mm256_broadcastss_ps(_mm_permute_ps(t, 0x55));
        const __m256 tz = _mm256_broadcastss_ps(_mm_permute_ps(t, 0xAA));
        const __m256 a0 = _mm256_fmadd_ps(_mm256_sub_ps(r10, r00), ty, r00);
        const __m256 a1 = _mm256_fmadd_ps(_mm256_sub_ps(r11, r01), ty, r01);
        const __m256 a = _mm256_fmadd_ps(_mm256_sub_ps(a1, a0), tz, a0);        // [celda x0 | celda x0+1]
        const __m128 lo = _mm256_castps256_ps128(a), hi = _mm256_extractf128_ps(a, 1);
        return _mm_fmadd_ps(_mm_sub_ps(hi, lo), _mm_permute_ps(t, 0x00), lo);
    }
    // Fuera de [0, n-1] en algún eje, o NaN.
    CFD_INLINE bool outside(__m128 p) const {
        const __m128 bad = _mm_or_ps(_mm_cmp_ps(p, _mm_setzero_ps(), _CMP_NGE_UQ), _mm_cmp_ps(p, _mm_load_ps(max_), _CMP_NLE_UQ));
        return (_mm_movemask_ps(bad) & 7) != 0;
    }
    // ¿La celda más cercana a p es sólida? (p debe estar dentro)
    CFD_INLINE bool solid(__m128 p) const {
        const __m128i i = _mm_cvtps_epi32(p);   // redondeo al más cercano
        const usize n = static_cast<usize>(_mm_cvtsi128_si32(i)) + sy_ * static_cast<usize>(_mm_extract_epi32(i, 1)) +
                        sz_ * static_cast<usize>(_mm_extract_epi32(i, 2));
        return (flags_.data()[n] & lbm::kSolid) != 0;
    }
    CFD_INLINE Vec3 velocity(Vec3 p) const {
        alignas(16) float r[4];
        _mm_store_ps(r, sample(_mm_setr_ps(p.x, p.y, p.z, 0.0f)));
        return {r[0], r[1], r[2]};
    }
    // Prefetch (T0) de las 4 filas de celdas que leerá sample() en p: permite adelantar
    // los fallos de caché de accesos aleatorios (partículas) varias iteraciones antes.
    CFD_INLINE void prefetch(float x, float y, float z) const {
        const __m128 c = _mm_min_ps(_mm_max_ps(_mm_setr_ps(x, y, z, 0.0f), _mm_setzero_ps()), _mm_load_ps(hi_));
        const __m128i i = _mm_cvttps_epi32(c);
        const usize n0 = static_cast<usize>(_mm_cvtsi128_si32(i)) +
                         sy_ * static_cast<usize>(_mm_extract_epi32(i, 1)) + sz_ * static_cast<usize>(_mm_extract_epi32(i, 2));
        const char* b = reinterpret_cast<const char*>(cells_.data() + n0);
        _mm_prefetch(b, _MM_HINT_T0);
        _mm_prefetch(b + 8 * sy_, _MM_HINT_T0);
        _mm_prefetch(b + 8 * sz_, _MM_HINT_T0);
        _mm_prefetch(b + 8 * (sy_ + sz_), _MM_HINT_T0);
    }
    const u8* flags() const { return flags_.data(); }

private:
    Buffer<u64> cells_;              // 4×FP16 por celda
    Buffer<u8> flags_;               // copia de flags (instantánea coherente con cells_)
    usize sy_ = 0, sz_ = 0;
    alignas(16) float hi_[4] = {};   // mayor float < n-1 (sujeción para la trilineal: x0 ≤ n-2 siempre)
    alignas(16) float max_[4] = {};  // n-1 (límite de "dentro")
};

// Muestreador compuesto (refinamiento local): una FlowSampler por rejilla; cada muestra se toma de la rejilla más fina
// que contiene el punto (celdas de la base). Misma interfaz que FlowSampler para líneas de corriente y humo.
class MultiSampler {
public:
    void update(const MultiField& m);     // reempaqueta todas las rejillas (paralelo)
    bool ready() const { return n_ > 0 && s_[0].ready(); }
    int grids() const { return n_; }
    int nx = 0, ny = 0, nz = 0;           // red base
    float u_inf = 0.08f;
    CFD_INLINE int finest(__m128 p) const {
        alignas(16) float q[4];
        _mm_store_ps(q, p);
        int best = 0, bd = 0;
        for (int i = 1; i < n_; ++i) {
            const Aabb& b = g_[i].inner;
            if (g_[i].depth > bd && q[0] >= b.lo.x && q[0] <= b.hi.x && q[1] >= b.lo.y && q[1] <= b.hi.y && q[2] >= b.lo.z && q[2] <= b.hi.z) {
                best = i;
                bd = g_[i].depth;
            }
        }
        return best;
    }
    CFD_INLINE __m128 to_grid(int i, __m128 p) const { return _mm_mul_ps(_mm_sub_ps(p, _mm_load_ps(org_[i])), _mm_set1_ps(inv_[i])); }
    CFD_INLINE __m128 sample(__m128 p) const {
        const int i = finest(p);
        return i ? s_[i].sample(to_grid(i, p)) : s_[0].sample(p);
    }
    CFD_INLINE bool outside(__m128 p) const { return s_[0].outside(p); }
    CFD_INLINE bool solid(__m128 p) const {
        const int i = finest(p);
        return i ? s_[i].solid(to_grid(i, p)) : s_[0].solid(p);
    }
    CFD_INLINE void prefetch(float x, float y, float z) const { s_[0].prefetch(x, y, z); }
    usize memory_bytes() const { usize b = 0; for (int i = 0; i < n_; ++i) b += s_[i].memory_bytes(); return b; }

private:
    FlowSampler s_[MultiField::k_max];
    GridField g_[MultiField::k_max];
    alignas(16) float org_[MultiField::k_max][4] = {};
    float inv_[MultiField::k_max] = {};
    int n_ = 0;
};

// ============================================================================
//  Rastrillos de siembra (rakes): línea o rejilla 2D de semillas.
//  seed(i,j) = origin + du·i/(nu-1) + dv·j/(nv-1)   (celdas)
// ============================================================================
struct Rake {
    Vec3 origin{0, 0, 0};
    Vec3 du{0, 0, 0};
    Vec3 dv{0, 0, 0};
    int nu = 1, nv = 1;

    static Rake line(Vec3 a, Vec3 b, int n) { return {a, b - a, {0, 0, 0}, n, 1}; }
    static Rake grid(Vec3 corner, Vec3 du, Vec3 dv, int nu, int nv) { return {corner, du, dv, nu, nv}; }
    int count() const { return (nu > 0 ? nu : 0) * (nv > 0 ? nv : 0); }
    Vec3 seed(int k) const {
        const int i = k % nu, j = k / nu;
        const float fu = nu > 1 ? static_cast<float>(i) / static_cast<float>(nu - 1) : 0.5f;
        const float fv = nv > 1 ? static_cast<float>(j) / static_cast<float>(nv - 1) : 0.5f;
        return origin + du * fu + dv * fv;
    }
};
// Atajos para la app (obj = caja del objeto en CELDAS):
// rejilla vertical (plano Y-Z) a `gap` celdas aguas arriba del objeto, cubriendo su sección con margen.
Rake rake_upstream(const Aabb& obj, float gap, int nu, int nv, float margin = 0.15f);
// línea horizontal transversal (eje Y) a altura z aguas arriba (p.ej. cerca del suelo para ver el fondo plano).
Rake rake_floor(const Aabb& obj, float gap, float z, int n, float margin = 0.1f);
// línea vertical (eje Z) en y dado, aguas arriba (varita de humo clásica).
Rake rake_vertical(const Aabb& obj, float gap, float y, int n, float margin = 0.1f);

// ============================================================================
//  Líneas de corriente — RK4 con paso adaptativo en longitud de arco.
// ============================================================================
struct StreamlineParams {
    int max_steps = 400;          // puntos por línea (por sentido)
    float step = 0.5f;            // paso inicial (celdas)
    float min_step = 0.125f;      // límites del paso adaptativo (celdas)
    float max_step = 1.0f;
    float max_turn_deg = 10.0f;   // giro máximo entre k1 y k4 antes de reducir el paso
    float min_speed = 0.02f;      // parar si |u| < min_speed·U∞ (punto de estancamiento)
    bool both_directions = false; // integrar también hacia atrás (semillas dentro del flujo)
    Quantity color_by = Quantity::Speed;   // Speed, Ux, Uz, Cp o Cp0
    ColorScale scale = {Colormap::Turbo, 0.0f, 1.6f, false};
    float width = 1.5f;           // grosor de dibujo (píxeles)
};

class Streamlines {
public:
    StreamlineParams params;

    void set_rakes(std::span<const Rake> rakes);          // copia las semillas
    void set_seeds(std::span<const Vec3> seeds);
    // Integra todas las semillas en paralelo sobre el campo empaquetado FP16 (ruta rápida).
    void compute(const FlowSampler& s);
    void compute(const MultiSampler& s);                  // refinamiento local: la rejilla más fina en cada punto
    // Misma integración usando FieldView::velocity() en FP32 (referencia / validación).
    void compute_reference(const lbm::FieldView& f);
    // `behind`: planos translúcidos ya dibujados → los tramos detrás de ellos se atenúan.
    void draw(Framebuffer& fb, const Camera& cam, bool depth_test = true, std::span<const TranslucentPlane> behind = {}) const;

    // Salida para draw_polylines: la línea i ocupa pts[starts[i] .. starts[i]+counts[i]).
    std::span<const Vec3> points() const { return {pts_.data(), pts_.size()}; }
    std::span<const u32> colors() const { return {col_.data(), col_.size()}; }
    std::span<const u32> starts() const { return {starts_.data(), n_lines_}; }
    std::span<const u32> counts() const { return {counts_.data(), n_lines_}; }
    usize line_count() const { return n_lines_; }
    usize total_points() const;                            // suma de counts
    usize seed_count() const { return seeds_.size(); }

private:
    template <class S> void compute_impl(const S& s);
    mutable Buffer<u32> draw_col_;                        // colores atenuados (sólo con planos translúcidos)
    std::vector<Vec3> seeds_;
    Buffer<Vec3> pts_;
    Buffer<u32> col_;
    Buffer<u32> starts_, counts_;
    usize n_lines_ = 0;
};

// ============================================================================
//  Partículas de humo — sistema SoA con emisores (rakes) y reciclaje en anillo.
// ============================================================================
struct ParticleParams {
    int capacity = 300000;        // máximo de partículas vivas
    float rate = 0.0f;            // partículas emitidas por paso de red (0 → auto ≈ 0.85·capacity/max_age)
    float max_age = 0.0f;         // vida en pasos de red (0 → auto ≈ 1.25·nx/U∞)
    float jitter = 0.35f;         // dispersión aleatoria de la emisión (celdas)
    float time_scale = 1.0f;      // cámara lenta (<1) / rápida (>1)
    float max_cells_per_substep = 2.0f;   // subpasos RK2 si el avance del cuadro es grande
    int max_substeps = 4;                 // tope de subpasos (coste acotado con muchos pasos de red por cuadro;
                                          // por encima, cada subpaso avanza más de max_cells_per_substep)
    Quantity color_by = Quantity::Speed;  // Speed, Ux, Uz, Cp o Cp0
    ColorScale scale = {Colormap::Turbo, 0.0f, 1.6f, false};
    float intensity = 0.35f;      // opacidad/brillo por partícula (canal A; aditivo: suma rgb·A)
    float point_size = 2.0f;      // píxeles
    bool additive = true;
    u64 seed = 0x5EEDu;
};

class Particles {
public:
    ParticleParams params;

    struct Stats {
        u64 emitted = 0, died_outside = 0, died_solid = 0, died_age = 0, overwritten = 0;
        usize alive = 0;
        // Conservación: emitted == alive + died_outside + died_solid + died_age + overwritten
    };

    void set_emitters(std::span<const Rake> rakes);
    void reset();                                         // mata todas (mantiene emisores)
    // Avanza `lattice_steps` pasos de red (× time_scale): advección RK2, reciclaje,
    // emisión y compactación de la salida de dibujo. Paralelo.
    void step(const FlowSampler& s, float lattice_steps);
    void step(const MultiSampler& s, float lattice_steps);   // refinamiento local (pasos de la red BASE)
    void draw(Framebuffer& fb, const Camera& cam, std::span<const TranslucentPlane> behind = {}) const;

    std::span<const Vec3> points() const { return {out_pts_.data(), stats_.alive}; }
    std::span<const u32> colors() const { return {out_col_.data(), stats_.alive}; }
    const Stats& stats() const { return stats_; }
    usize capacity() const { return cap_; }
    float effective_rate() const { return rate_eff_; }
    float effective_max_age() const { return age_max_eff_; }

private:
    template <class S> void step_impl(const S& s, float lattice_steps);
    void ensure_capacity();
    usize cap_ = 0, head_ = 0;
    Buffer<float> x_, y_, z_, age_;                       // SoA (age = +inf → muerta)
    Buffer<Vec3> tmp_pts_, out_pts_;
    Buffer<u32> tmp_col_, out_col_;
    mutable Buffer<u32> draw_col_;
    Buffer<u32> chunk_cnt_;
    Buffer<u64> chunk_dead_;                              // 3 contadores por trozo
    std::vector<Vec3> seeds_;
    u64 emit_cursor_ = 0;
    double emit_acc_ = 0.0;
    float rate_eff_ = 0.0f, age_max_eff_ = 0.0f;
    u64 rng_ = 0;
    bool rng_init_ = false;
    Stats stats_;
};

// ============================================================================
//  Plano de corte (SliceView)
// ============================================================================
struct SliceParams {
    Axis axis = Axis::Y;
    float pos = 0.0f;                 // posición del plano (celdas); se interpola entre capas
    Quantity quantity = Quantity::Speed;
    ColorScale scale = {Colormap::Turbo, 0.0f, 1.6f, false};   // usar SliceView::set_quantity para cambiar ambos
    bool auto_range = false;          // rango robusto (percentiles 1%–99%) recalculado en cada update
    float opacity = 0.9f;             // alfa global al dibujar
    u32 solid_color = 0xFF2E2F33u;    // celdas sólidas: gris oscuro
    float fade_below = 0.0f;          // >0: texeles con |t| < fade (t = normalizado, centro 0.5 si divergente) → transparentes
    // LIC (Line Integral Convolution): vetas de ruido convolucionado a lo largo de la
    // velocidad EN EL PLANO → dirección del flujo sobre el corte (recirculaciones, remolinos).
    int lic = 0;                      // 0 = desactivado; 2..4 = subtexeles por celda de la textura LIC
    float lic_length = 6.0f;          // semilongitud del núcleo (celdas)
    float lic_contrast = 0.85f;       // 0..1: intensidad de las vetas sobre el color
};

// Ejes del plano: X → (u=y, v=z); Y → (u=x, v=z); Z → (u=x, v=y).
// Valores: val_w×val_h (1 por celda del plano), valor (u,v) centrado en la celda (u,v).
// Textura: tex_w×tex_h ARGB (= val_w×val_h, o ×lic con LIC), fila v; cubre el plano entero:
// esquinas del quad en [-0.5, n-0.5] (orden (u0,v0) (u1,v0) (u1,v1) (u0,v1)) → el centro del
// texel (u,v) cae en la celda (u,v) (convención de draw_textured_quad: centros en (i+½)/tw).
class SliceView {
public:
    SliceParams params;

    void set_quantity(Quantity q) { params.quantity = q; params.scale = default_scale(q); }
    void update(const lbm::FieldView& f);                 // recalcula valores + textura (paralelo, AVX2)
    // Refinamiento local: el plano (params.pos en celdas de la base) se calcula en cada rejilla que lo corta con su
    // resolución y se compone en una textura con la de la rejilla más fina que lo cruza (hasta ~4 M texeles).
    // ground_layers: cada rejilla apoyada en el suelo usa su PRIMERA capa de fluido (huella en el suelo).
    void update(const MultiField& m, bool ground_layers = false);
    float texels_per_cell() const { return res_; }        // texeles por celda de la base (1 sin refinamiento)
    void draw(Framebuffer& fb, const Camera& cam) const;  // draw_textured_quad (depth_write si opaco)

    const u32* texture() const { return tex_.data(); }
    int tex_w() const { return lw_; }
    int tex_h() const { return lh_; }
    int val_w() const { return tw_; }
    int val_h() const { return th_; }
    const float* values() const { return vals_.data(); }  // valor por celda del plano (NaN = sólido), val_w×val_h
    void corners(Vec3 out[4]) const;
    ColorScale effective_scale() const { return eff_; }  // tras el rango automático
    float value_min() const { return vmin_; }             // extremos reales del plano (sólo valores finitos: sin NaN ni ±inf)
    float value_max() const { return vmax_; }
    float plane_pos() const { return pos_; }              // posición efectiva (sujeta; NaN → 0)
    // Eje del ÚLTIMO update(). params.axis/pos/quantity/scale/auto_range/fade_below/lic* se aplican
    // en update(); hasta entonces draw/corners/pick/value_at* siguen describiendo el corte calculado.
    // Sólo params.opacity es de dibujo.
    Axis axis() const { return axis_; }
    // Descriptor para atenuar lo que se dibuje detrás (opacity = 0 si es opaco: ya escribe profundidad).
    TranslucentPlane translucent_plane() const;

    // Valor bilineal en coordenadas del plano (u, v en celdas). NaN fuera o en sólido.
    float value_at(float u, float v) const;
    // Punto 3D (celdas) → valor en el plano (ignora la coordenada normal).
    float value_at_world(Vec3 p) const;
    // Rayo desde la cámara por el píxel (sx, sy) → punto del plano y valor. false si no corta.
    bool pick(const Camera& cam, float sx, float sy, Vec3& hit, float& value) const;

private:
    void build_lic(const lbm::FieldView& f, int k0, int k1, float t);
    void lic_convolve();                               // convolución LIC sobre vel2_ (ya calculada)
    void plane_values(const lbm::FieldView& f, Axis ax, float pos, Quantity q, float* out, int tw, int th);
    void compute_range();                              // eff_, vmin_, vmax_ de vals_
    void colorize();                                   // textura de vals_ (sin LIC)
    float value_tex(float u, float v) const;           // valor bilineal en coordenadas de texel
    Buffer<float> vals_;
    Buffer<float> gbuf_[MultiField::k_max];            // (refinamiento) valores del plano en cada rejilla
    float res_ = 1.0f;                                 // texeles por celda de la base
    Buffer<u32> tex_;
    Buffer<float> scratch_;
    Buffer<float> vel2_;               // LIC: velocidad en el plano (u, v) por celda
    Buffer<float> lic_;                // LIC: intensidad por subtexel
    Buffer<u8> noise_;                 // LIC: ruido blanco fijo por subtexel
    int tw_ = 0, th_ = 0, n_axis_ = 0;
    int lw_ = 0, lh_ = 0, noise_w_ = 0, noise_h_ = 0;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    float pos_ = 0.0f;
    float vmin_ = 0.0f, vmax_ = 0.0f;
    float fade_ = 0.0f;                // fade_below horneado en la textura
    Axis axis_ = Axis::Y;              // eje del corte calculado
    ColorScale eff_;
};

// Rayo de cámara ∩ plano eje = pos (celdas). false si es paralelo o queda detrás.
bool pick_plane(const Camera& cam, float sx, float sy, Axis axis, float pos, Vec3& hit);

// ============================================================================
//  Huella en el suelo: magnitud en la primera capa fluida sobre el suelo
//  (muestra la succión bajo el fondo plano: ¡efecto suelo!).
// ============================================================================
struct GroundParams {
    Quantity quantity = Quantity::Cp;
    ColorScale scale = {Colormap::CoolWarm, -2.5f, 1.0f, true};
    bool auto_range = false;
    float z = -1.0f;              // capa (celdas); < 0 → automática (primera capa sin suelo, normalmente 1)
};
class GroundFootprint {
public:
    GroundParams params;
    void update(const lbm::FieldView& f);
    void update(const MultiField& m);                     // refinamiento: primera capa de fluido de cada rejilla con suelo
    // Textura tw×th: u ↔ X (de extent.lo.x a extent.hi.x), v ↔ Y (de lo.y a hi.y).
    const u32* texture() const { return slice_.texture(); }
    int tex_w() const { return slice_.tex_w(); }
    int tex_h() const { return slice_.tex_h(); }
    Aabb extent() const;                                  // en celdas, z = plano de la pared (0.5)
    const SliceView& slice() const { return slice_; }
    // draw_ground con la textura de la huella.
    void draw(Framebuffer& fb, const Camera& cam, float spacing, float offset_x, u32 base, u32 line) const;

private:
    SliceView slice_;
    int nx_ = 0, ny_ = 0;
};

// ============================================================================
//  Estela (wake survey): plano X aguas abajo con Cp0 + integrales de pérdida.
//  ∫(1-Cp0) dA ≈ área de resistencia de perfil (celdas²);  ∫(v²+w²)/U∞² dA =
//  energía cinética transversal (resistencia inducida por vórtices).
//  Multiplicar por Δx² para m² (SCx ≈ suma de ambas, aproximación de Betz).
// ============================================================================
struct WakeStats {
    float x = 0.0f;               // posición del plano (celdas)
    float loss_area = 0.0f;       // ∫(1 - Cp0) dA  sobre celdas con Cp0 < 1 (celdas²)
    float crossflow_area = 0.0f;  // ∫(v²+w²)/U∞² dA (celdas²)
    float min_cp0 = 1.0f;
    Vec2 centroid{0, 0};          // centroide (y, z) de la pérdida
    int cells = 0;                // celdas con pérdida significativa (Cp0 < 0.98)
};
WakeStats wake_survey(const lbm::FieldView& f, float x_cells);
// Parámetros de corte listos para ver la estela (eje X, Cp0, Turbo [0,1]).
SliceParams wake_slice_params(float x_cells);

// ============================================================================
//  Colores de la malla (presión en superficie, etc.)
// ============================================================================
enum class SurfaceMode : u8 {
    Cp,          // Cp muestreado una celda hacia fuera a lo largo de la normal
    Speed,       // |u|/U∞ cerca de la pared (a `offset` celdas)
    Component,   // color por grupo/pieza (group_colors o paleta por id)
    Solid,       // color fijo
};
struct SurfaceParams {
    SurfaceMode mode = SurfaceMode::Cp;
    ColorScale scale = {Colormap::CoolWarm, -2.5f, 1.0f, true};
    float offset = 1.0f;                  // celdas hacia fuera a lo largo de la normal
    u32 solid_color = 0xFFB8BCC4u;
    std::span<const u32> group_colors;    // 256 entradas indexadas por id de grupo (opcional)
};
const char* surface_mode_name(SurfaceMode m);
// Rellena mesh.color (redimensiona si hace falta). Paralelo sobre vértices.
void color_mesh(Mesh& mesh, const lbm::FieldView& f, const SurfaceParams& p);
// Refinamiento local: cada vértice se muestrea en la rejilla más fina que lo contiene, a `offset` celdas DE ESA rejilla.
void color_mesh(Mesh& mesh, const MultiField& m, const SurfaceParams& p);

// ============================================================================
//  Volumen de vórtices — Q (o |ω|) en rejilla (opcionalmente 2× reducida),
//  ladrillos 8³ min/max para saltar espacio vacío, raymarching por píxel
//  (opcionalmente a media resolución + reescalado bilineal consciente de la
//  profundidad), composición front-to-back, terminación temprana y parada en
//  la profundidad de la escena opaca (los vórtices envuelven al coche).
// ============================================================================
enum class VolumeField : u8 { QCriterion, Vorticity };
enum class VolumeColor : u8 {
    Streamwise,   // color por u_x/U∞ (típico de imágenes CFD de F1)
    Magnitude,    // color por intensidad del campo (Inferno)
};
enum class VolumeStyle : u8 {
    Surface,      // isosuperficies (umbral) iluminadas y semitransparentes (estilo "Q-iso" de F1)
    Cloud,        // nube volumétrica: opacidad proporcional a la intensidad
};
inline const char* volume_field_name(VolumeField f) { return f == VolumeField::QCriterion ? "Criterio Q" : "Vorticidad |ω|"; }
inline const char* volume_color_name(VolumeColor c) { return c == VolumeColor::Streamwise ? "Velocidad axial u_x/U∞" : "Intensidad"; }
inline const char* volume_style_name(VolumeStyle s) { return s == VolumeStyle::Surface ? "Isosuperficies" : "Nube volumétrica"; }

struct VolumeParams {
    VolumeField field = VolumeField::QCriterion;
    VolumeStyle style = VolumeStyle::Surface;
    int downsample = 2;               // 1 = resolución completa, 2 = rejilla 2× más gruesa (promedio 2³)
    float full = 0.015f;              // valor normalizado que satura la cuantización (densidad 255). Cambiarlo exige update()
    float threshold = 0.0015f;        // umbral/isovalor (normalizado). Se aplica en render(): cambiarlo NO exige update()
    float density = 0.35f;            // Cloud: opacidad por celda de recorrido a densidad máxima (0..1)
    float surface_alpha = 0.72f;      // Surface: opacidad de cada capa de isosuperficie
    VolumeColor color_by = VolumeColor::Streamwise;
    ColorScale color_scale = {Colormap::Turbo, 0.0f, 1.3f, false};   // para Streamwise (u_x/U∞); cambiarlo exige update()
    float step = 0.5f;                // paso de raymarching (vóxeles)
    bool half_res = true;             // raymarching a media resolución + reescalado
    bool shading = true;              // iluminación (Surface: normal por gradiente; Cloud: derivada hacia la luz)
    bool hide_near_wall = true;       // anula Q/|ω| en celdas fluidas con un vecino sólido (ruido de escalera en paredes)
    Vec3 light_dir = normalize(Vec3(-0.4f, -0.5f, 0.75f));
};
// En update() se aplican: field, downsample, full, color_by, color_scale, hide_near_wall.
// En render() (sin recalcular): style, threshold, density, surface_alpha, step, half_res, shading, light_dir.
class VortexVolume {
public:
    VolumeParams params;
    void update(const lbm::FieldView& f);                 // Q/|ω| + cuantización + ladrillos (paralelo)
    // Refinamiento local: rejilla de volumen de la base (con downsample); cada vóxel promedia las celdas de la rejilla más
    // fina que contiene su centro (Q/|ω| calculados en cada rejilla con su dx y pasados a unidades de la base).
    void update(const MultiField& m);
    // Compone sobre fb dentro de cam.vp ∩ framebuffer. `behind`: hasta 4 planos translúcidos ya dibujados
    // (el tramo del rayo detrás de cada uno pesa ×(1-opacity)).
    void render(Framebuffer& fb, const Camera& cam, std::span<const TranslucentPlane> behind = {});
    struct Stats { int vx = 0, vy = 0, vz = 0; usize bricks = 0, bricks_nonempty = 0; float max_value = 0; };
    const Stats& stats() const { return stats_; }

private:
    void count_bricks();
    void build_bricks(float vmax);
    u32 iso_index() const;
    Buffer<u8> dens_, col_, brick_;
    Buffer<float> gq_[MultiField::k_max], gu_[MultiField::k_max];   // (refinamiento) Q/|ω| y u_x por celda de cada rejilla
    Buffer<float> scratch_;
    Buffer<u32> low_;                  // RGBA premultiplicado (media resolución)
    Buffer<float> low_depth_;          // profundidad de vista del primer aporte
    Buffer<u16> vpad_;                 // fila vertical interpolada (reescalado)
    int vx_ = 0, vy_ = 0, vz_ = 0, ds_ = 2;
    int bx_ = 0, by_ = 0, bz_ = 0;
    int occ_lo_[3] = {0, 0, 0}, occ_hi_[3] = {-1, -1, -1};   // caja (en ladrillos) de los ladrillos ocupados
    float full_ = 0.015f;              // params.full usado en el último update (escala de dens_)
    bool col_ux_ = true;               // col_ calculado como u_x/U∞ (Streamwise) o como densidad (Magnitude)
    Colormap col_map_ = Colormap::Turbo;
    float alpha_lut_[256] = {};
    float rgb_lut_[256 * 4] = {};
    Stats stats_;
};

// ============================================================================
//  Utilidades internas compartidas entre los .cpp del módulo (no usar en la app).
// ============================================================================
namespace detail {
// Valores de la magnitud q para toda la fila (y,z): out[0..nx). NaN en sólidos.
void quantity_row(const lbm::FieldView& f, Quantity q, int y, int z, float* CFD_RESTRICT out);
// Rango robusto (percentiles plo/phi) de n valores ignorando NaN e ±inf (valores no finitos:
// p.ej. un solver que diverge). false si no hay valores finitos. plo ≤ 0 y phi ≥ 1 → mín/máx
// exactos sin histograma.
bool robust_range(const float* v, usize n, float plo, float phi, float& lo, float& hi);
// NaN en las celdas de la fila (y,z) que son sólidas o tienen un vecino 6-conexo sólido
// (VolumeParams::hide_near_wall). Expuesta para los tests.
void mask_near_wall(const lbm::FieldView& f, int y, int z, float* CFD_RESTRICT out);
}

} // namespace cfd::flowvis
