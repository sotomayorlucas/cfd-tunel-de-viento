// ============================================================================
//  render/raster.cpp — rasterizador por software ("render core"): índice del
//  módulo y postprocesos en espacio de pantalla.
//
//  Organización del módulo:
//    raster.hpp            contrato público (+ extensiones: fxaa, RasterStats,
//                          MeshStyle::debug_overdraw)
//    raster_internal.hpp   piezas compartidas: cámara aplanada, binning por tiles
//                          (ordenación por conteo), recorrido de triángulos en
//                          punto fijo 28.4 (filas de 8 / sellos 4×2), SWAR, LUTs
//    raster_mesh.cpp       draw_mesh (vértices AVX2 → binning → tiles con Hi-Z y
//                          sombreado diferido)
//    raster_prims.cpp      líneas, polilíneas, puntos, caja, flechas
//    raster_fill.cpp       cuadrilátero texturizado, suelo
//    raster.cpp            screen_space_edges (SSAO barato) y fxaa
//
//  Hilos: todas las funciones usan el pool internamente y comparten buffers de
//  trabajo persistentes → llamarlas desde UN solo hilo a la vez.
// ============================================================================
#include "raster_internal.hpp"

namespace cfd::render {
namespace rz {

struct PostScratch {
    Buffer<u32> src;     // copia del color (FXAA)
    Buffer<u8> luma;
};
static PostScratch& post_scratch() { static PostScratch s; return s; }

// Oclusión ambiental en espacio de pantalla (SSAO barato) + oscurecimiento de bordes.
// Para cada píxel con profundidad finita se comparan 12 vecinos (anillo de 3 px + cruz de 7 px):
// un vecino más cercano en un 0.4 %–30 % de la profundidad ocluye (más lejos = silueta, no
// cuenta: evita halos). ao ∈ [0,1] → color · (1 - strength·ao).
inline constexpr int k_ao_n = 12;
inline constexpr int k_ao_dx[k_ao_n] = {-3, 3, 0, 0, -2, 2, -2, 2, -7, 7, 0, 0};
inline constexpr int k_ao_dy[k_ao_n] = {0, 0, -3, 3, -2, -2, 2, 2, 0, 0, -7, 7};
inline constexpr int k_ao_r = 7;
CFD_INLINE float ao_term(float zc, float zs) {
    const float d = (zc - zs) / zc;
    return clamp_((d - 0.004f) * 40.0f, 0.0f, 1.0f) * clamp_(1.5f - d * 5.0f, 0.0f, 1.0f);
}

// ---- FXAA ------------------------------------------------------------------------------------
CFD_INLINE float luma_of(u32 c) {
    return static_cast<float>(((c >> 16) & 255) + 2 * ((c >> 8) & 255) + (c & 255)) * 0.25f;
}
// Muestra bilineal (en coordenadas de píxel con centro en i+0.5) de la copia, sujeta al rect.
CFD_INLINE void sample_rgb(const u32* src, int W, int H, float px, float py, float out[3]) {
    float x = clamp_(px - 0.5f, 0.0f, static_cast<float>(W - 1)), y = clamp_(py - 0.5f, 0.0f, static_cast<float>(H - 1));
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = min_(x0 + 1, W - 1), y1 = min_(y0 + 1, H - 1);
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const u32 c00 = src[y0 * W + x0], c10 = src[y0 * W + x1], c01 = src[y1 * W + x0], c11 = src[y1 * W + x1];
    for (int k = 0; k < 3; ++k) {
        const int sh = 16 - 8 * k;
        const float a = static_cast<float>((c00 >> sh) & 255), b = static_cast<float>((c10 >> sh) & 255);
        const float c = static_cast<float>((c01 >> sh) & 255), d = static_cast<float>((c11 >> sh) & 255);
        out[k] = (a + (b - a) * fx) * (1.0f - fy) + (c + (d - c) * fx) * fy;
    }
}
// FXAA (Lottes, variante "consola"/FXAA 2): dirección del borde por luma diagonal,
// dos muestras a lo largo del borde (A) y cuatro (B); B si sigue dentro del rango local.
static u32 fxaa_pixel(const u32* src, const u8* L, usize ls, int W, int H, int x, int y) {
    const u8* r0 = L + static_cast<usize>(y - 1) * ls;
    const u8* r1 = L + static_cast<usize>(y) * ls;
    const u8* r2 = L + static_cast<usize>(y + 1) * ls;
    const float lNW = r0[x - 1], lNE = r0[x + 1];
    const float lSW = r2[x - 1], lSE = r2[x + 1];
    const float lM = r1[x];
    const float lmin = min_(lM, min_(min_(lNW, lNE), min_(lSW, lSE)));
    const float lmax = max_(lM, max_(max_(lNW, lNE), max_(lSW, lSE)));
    float dx = -((lNW + lNE) - (lSW + lSE));
    float dy = ((lNW + lSW) - (lNE + lSE));
    const float reduce = max_((lNW + lNE + lSW + lSE) * (0.25f * 0.125f), 255.0f / 128.0f);
    const float rcp = 1.0f / (min_(std::fabs(dx), std::fabs(dy)) + reduce);
    dx = clamp_(dx * rcp, -8.0f, 8.0f);
    dy = clamp_(dy * rcp, -8.0f, 8.0f);
    const float px = static_cast<float>(x) + 0.5f, py = static_cast<float>(y) + 0.5f;
    float a0[3], a1[3], b0[3], b1[3];
    sample_rgb(src, W, H, px + dx * (1.0f / 3.0f - 0.5f), py + dy * (1.0f / 3.0f - 0.5f), a0);
    sample_rgb(src, W, H, px + dx * (2.0f / 3.0f - 0.5f), py + dy * (2.0f / 3.0f - 0.5f), a1);
    sample_rgb(src, W, H, px - dx * 0.5f, py - dy * 0.5f, b0);
    sample_rgb(src, W, H, px + dx * 0.5f, py + dy * 0.5f, b1);
    float A[3], B[3];
    for (int k = 0; k < 3; ++k) { A[k] = 0.5f * (a0[k] + a1[k]); B[k] = 0.5f * A[k] + 0.25f * (b0[k] + b1[k]); }
    const float lB = (B[0] + 2.0f * B[1] + B[2]) * 0.25f;
    const float* R = (lB < lmin || lB > lmax) ? A : B;
    return rgb(static_cast<u32>(R[0] + 0.5f), static_cast<u32>(R[1] + 0.5f), static_cast<u32>(R[2] + 0.5f));
}

} // namespace rz

void screen_space_edges(Framebuffer& fb, Rect vp, float strength) {
    using namespace rz;
    if (fb.color.empty() || fb.depth.empty() || !(strength > 0.0f)) return;
    const Clip c = clip_of(fb, vp);
    if (c.empty()) return;
    strength = min_(strength, 1.0f);
    const float kmul = strength * 1.5f / static_cast<float>(k_ao_n);
    const usize stride = static_cast<usize>(fb.stride);
    parallel_for(c.y0, c.y1 + 1, 8, [&](i64 lo, i64 hi) {
        const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
        // rz::k_inf calificado: en el build unity draw2d.cpp define otro k_inf en un espacio anónimo
        // de cfd::render y con `using namespace rz` el nombre sin calificar sería ambiguo.
        const __m256 inf = _mm256_set1_ps(rz::k_inf);
        for (i64 yy = lo; yy < hi; ++yy) {
            const int y = static_cast<int>(yy);
            const float* zr[k_ao_n];
            for (int s = 0; s < k_ao_n; ++s) zr[s] = fb.depth.data() + static_cast<usize>(clamp_(y + k_ao_dy[s], c.y0, c.y1)) * stride;
            const float* zc_row = fb.depth.data() + static_cast<usize>(y) * stride;
            u32* crow = fb.color.data() + static_cast<usize>(y) * stride;
            auto scalar_px = [&](int x) {
                const float zc = zc_row[x];
                if (!(zc < rz::k_inf)) return;
                float occ = 0.0f;
                for (int s = 0; s < k_ao_n; ++s) {
                    const float zs = zr[s][clamp_(x + k_ao_dx[s], c.x0, c.x1)];
                    if (zs < rz::k_inf) occ += ao_term(zc, zs);
                }
                const float k = 1.0f - min_(occ * kmul, strength);
                const u32 w = static_cast<u32>(clamp_(k, 0.0f, 1.0f) * 256.0f);
                const u32 col = crow[x];
                crow[x] = 0xFF000000u | ((((col & 0x00FF00FFu) * w) >> 8) & 0x00FF00FFu) | ((((col & 0x0000FF00u) * w) >> 8) & 0x0000FF00u);
            };
            int x = c.x0;
            // cabeza escalar hasta que el bloque de 8 y sus vecinos quepan en la tijera
            const int xs = max_((c.x0 + k_ao_r + 7) & ~7, c.x0);
            for (; x < min_(xs, c.x1 + 1); ++x) scalar_px(x);
            for (; x + 7 + k_ao_r <= c.x1; x += 8) {
                const __m256 zc = _mm256_load_ps(zc_row + x);
                const int valid = _mm256_movemask_ps(_mm256_cmp_ps(zc, inf, _CMP_LT_OQ));
                if (!valid) continue;
                const __m256 izc = _mm256_div_ps(one, zc);
                __m256 occ = zero;
                for (int s = 0; s < k_ao_n; ++s) {
                    const __m256 zs = _mm256_loadu_ps(zr[s] + x + k_ao_dx[s]);
                    const __m256 d = _mm256_mul_ps(_mm256_sub_ps(zc, zs), izc);   // inf en zs → -inf → 0
                    const __m256 a = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(d, _mm256_set1_ps(0.004f)), _mm256_set1_ps(40.0f)), zero), one);
                    const __m256 b = _mm256_min_ps(_mm256_max_ps(_mm256_fnmadd_ps(d, _mm256_set1_ps(5.0f), _mm256_set1_ps(1.5f)), zero), one);
                    occ = _mm256_fmadd_ps(a, b, occ);
                }
                const __m256 k = _mm256_sub_ps(one, _mm256_min_ps(_mm256_mul_ps(occ, _mm256_set1_ps(kmul)), _mm256_set1_ps(strength)));
                const __m256i dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(crow + x));
                const __m256i out = _mm256_or_si256(scale_swar(dst, alpha16(k)), _mm256_set1_epi32(static_cast<int>(0xFF000000u)));
                _mm256_store_si256(reinterpret_cast<__m256i*>(crow + x), _mm256_blendv_epi8(dst, out, mask_si(valid)));
            }
            for (; x <= c.x1; ++x) scalar_px(x);
        }
    });
}

// ---------------------------------------------------------------------------------------------
void fxaa(Framebuffer& fb, Rect vp) {
    using namespace rz;
    if (fb.color.empty()) return;
    const Clip c = clip_of(fb, vp);
    if (c.empty()) return;
    const int W = c.x1 - c.x0 + 1, H = c.y1 - c.y0 + 1;
    if (W < 3 || H < 3) return;
    PostScratch& S = post_scratch();
    const usize Wp = static_cast<usize>(W + 32);   // relleno para cargas de 32 bytes
    u32* src = grow(S.src, static_cast<usize>(W) * static_cast<usize>(H));
    u8* L = grow(S.luma, Wp * static_cast<usize>(H) + 64);
    // 1) copia + luma (R + 2G + B)/4 en u8
    parallel_for(0, H, 16, [&](i64 lo, i64 hi) {
        for (i64 yy = lo; yy < hi; ++yy) {
            const int y = static_cast<int>(yy);
            const u32* row = fb.row(c.y0 + y) + c.x0;
            u32* s = src + static_cast<usize>(y) * static_cast<usize>(W);
            u8* l = L + static_cast<usize>(y) * Wp;
            int x = 0;
            const __m256i m8 = _mm256_set1_epi32(255);
            for (; x + 8 <= W; x += 8) {
                const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(s + x), p);
                const __m256i r = _mm256_and_si256(_mm256_srli_epi32(p, 16), m8), g = _mm256_and_si256(_mm256_srli_epi32(p, 8), m8), b = _mm256_and_si256(p, m8);
                const __m256i lum = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(r, b), _mm256_slli_epi32(g, 1)), 2);
                // 8 × int32 → 8 × u8
                const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(lum), _mm256_extracti128_si256(lum, 1));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(l + x), _mm_packus_epi16(p16, p16));
            }
            for (; x < W; ++x) { s[x] = row[x]; l[x] = static_cast<u8>(luma_of(row[x])); }
        }
    });
    // 2) detección de bordes (32 píxeles por iteración, aritmética u8 saturada) + filtro escalar
    parallel_for(1, H - 1, 8, [&](i64 lo, i64 hi) {
        const __m256i k16 = _mm256_set1_epi8(12);      // umbral mínimo ≈ 0.047
        const __m256i m1f = _mm256_set1_epi8(0x1F);
        for (i64 yy = lo; yy < hi; ++yy) {
            const int y = static_cast<int>(yy);
            const u8* lm = L + static_cast<usize>(y) * Wp;
            const u8* ln = lm - Wp;
            const u8* ls = lm + Wp;
            u32* out = fb.row(c.y0 + y) + c.x0;
            for (int x = 1; x < W - 1; x += 32) {
                const __m256i M = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lm + x));
                const __m256i N = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ln + x));
                const __m256i S_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ls + x));
                const __m256i Wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lm + x - 1));
                const __m256i Ev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lm + x + 1));
                const __m256i mx = _mm256_max_epu8(_mm256_max_epu8(_mm256_max_epu8(M, N), _mm256_max_epu8(S_, Wv)), Ev);
                const __m256i mn = _mm256_min_epu8(_mm256_min_epu8(_mm256_min_epu8(M, N), _mm256_min_epu8(S_, Wv)), Ev);
                const __m256i range = _mm256_subs_epu8(mx, mn);
                // umbral = max(12, max/8): borde si range > umbral  ⇔  subs(range, umbral) ≠ 0
                const __m256i thr = _mm256_max_epu8(_mm256_and_si256(_mm256_srli_epi16(mx, 3), m1f), k16);
                const __m256i over = _mm256_subs_epu8(range, thr);
                u32 bits = ~static_cast<u32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(over, _mm256_setzero_si256())));
                const int valid = W - 1 - x;           // píxeles interiores restantes
                if (valid < 32) bits &= (1u << valid) - 1u;
                for_each_bit(bits, [&](int b) { out[x + b] = fxaa_pixel(src, L, Wp, W, H, x + b, y); });
            }
        }
    });
}

} // namespace cfd::render
