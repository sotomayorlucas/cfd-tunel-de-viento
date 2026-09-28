// ============================================================================
//  render/raster_internal.hpp — piezas internas del rasterizador por software.
//
//  NO es contrato público: sólo lo incluyen render/raster*.cpp. Todo vive en
//  el espacio de nombres cfd::render::rz para no chocar con otros módulos en
//  el build "unity" (una sola unidad de traducción).
//
//  Contenido:
//   * View        — cámara aplanada (base, focal, centro) + códigos de recorte.
//   * Bins        — reparto de primitivas a tiles 64×64 por ordenación por conteo
//                   (2 pasadas, contadores por slot → sin atómicos, orden estable),
//                   2 listas por tile opcionales (caras frontales / traseras).
//   * scan_tri*   — recorrido de un triángulo en PUNTO FIJO 28.4 con funciones de
//                   arista enteras (regla top-left exacta → estanco), 8 píxeles por
//                   iteración con AVX2: filas alineadas (RowBlk) o sellos 4×2
//                   (StampBlk) para triángulos diminutos; variante general con
//                   clasificación trivial de aristas por región (64 bits).
//   * clip_polygon— Sutherland–Hodgman en espacio de vista (plano cercano + guarda).
//   * SWAR        — mezcla/escala de colores ARGB en carriles de 16 bits (vpmullw),
//                   bilineal con 4 gathers, LUT constexpr de máscaras de carriles.
// ============================================================================
#pragma once

#include "raster.hpp"
#include "../core/simd.hpp"
#include "../core/threadpool.hpp"
#include "../core/util.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// Fuerza el inlining de lambdas pasadas a los recorridos (sintaxis GNU tras la lista de parámetros).
#define RZ_LAMBDA_INLINE __attribute__((always_inline))

namespace cfd::render::rz {

// Interruptores de experimentos A/B (sólo con -DRZ_EXPERIMENTS; en build normal son constantes 0
// y el compilador elimina las ramas). Bits: 0 filas en vez de sellos, 2 sin Hi-Z, 3 binning escalar,
// 7 puntos sin sellos, 10 vértices escalares, 11 sin orden LPT.
#ifdef RZ_EXPERIMENTS
inline int g_rz_exp = 0;
#define RZ_EXP(bit) ((::cfd::render::rz::g_rz_exp >> (bit)) & 1)
#else
#define RZ_EXP(bit) 0
#endif

// Contadores de depuración por hilo (sólo con -DRZ_STATS; en build normal no existen).
#ifdef RZ_STATS
struct RzCounters { u64 setups = 0, rows = 0, blocks = 0, covered = 0, passed = 0, early_out = 0, tris_any_pass = 0, back_setups = 0, back_pass = 0; u64 hist[9][9] = {}; u64 big = 0; u64 seg_refs = 0, seg_rows = 0, seg_blocks = 0, seg_cov = 0, seg_pass = 0; };
inline thread_local RzCounters t_rzc;
#define RZ_COUNT(field, n) (::cfd::render::rz::t_rzc.field += (n))
#else
#define RZ_COUNT(field, n) ((void)0)
#endif

// ---- Constantes del rasterizador ---------------------------------------------------------
inline constexpr int k_tile_shift = 6;                // tiles de 64×64 (16 KB color + 16 KB depth: caben en L1d;
                                                      // medido: 20-27 % más rápido que 32×32)
inline constexpr int k_tile = 1 << k_tile_shift;
inline constexpr int k_sub_shift = 4;                 // 4 bits de subpíxel (1/16 px)
inline constexpr int k_sub = 1 << k_sub_shift;
// Banda de guarda: |sx - cx|, |sy - cy| ≤ 8192 px. Con 4 bits de subpíxel las coordenadas caben en
// ±2^18 y dentro de un tile de 64 px las funciones de arista no superan 2^29 → int32 exacto.
inline constexpr float k_guard = 8192.0f;
inline constexpr u32 k_no_rect = 0xFFFFFFFFu;
inline constexpr float k_inf = std::numeric_limits<float>::infinity();

// Códigos de recorte (outcodes) por vértice.
enum : u32 { OC_LEFT = 1, OC_RIGHT = 2, OC_TOP = 4, OC_BOTTOM = 8, OC_NEAR = 16, OC_BAD = 32 };
inline constexpr u32 OC_PLANES = OC_LEFT | OC_RIGHT | OC_TOP | OC_BOTTOM | OC_NEAR;

// ---- Buffers persistentes: sólo crecen (nada de malloc por cuadro) --------------------------
template <class T>
CFD_INLINE T* grow(Buffer<T>& b, usize n) {
    if (b.size() < n) b.resize(n + n / 2 + 64);
    return b.data();
}

// ---- Tijera (scissor) inclusiva -----------------------------------------------------------
struct Clip {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    CFD_INLINE bool empty() const { return x0 > x1 || y0 > y1; }
};
CFD_INLINE Clip clip_of(const Framebuffer& fb, const Rect& r) {
    Clip c;
    c.x0 = max_(r.x, 0);
    c.y0 = max_(r.y, 0);
    c.x1 = min_(r.x + r.w, fb.w) - 1;
    c.y1 = min_(r.y + r.h, fb.h) - 1;
    return c;
}

// ---- Cámara aplanada ------------------------------------------------------------------------
// Espacio de vista: X = (p-eye)·right, Y = (p-eye)·up, Z = (p-eye)·fwd (profundidad lineal).
// Perspectiva: sx = cx + f·X/Z, sy = cy - f·Y/Z.  Ortográfica: sx = cx + s·X, sy = cy - s·Y.
struct View {
    Vec3 eye, right, up, fwd;
    float m[3][4];           // filas: X, Y, Z = m·(p,1)
    float cx, cy;            // centro del viewport (px del framebuffer)
    float f;                 // focal (persp) o escala (ortho), px por unidad
    float znear;
    bool ortho;
    Clip clip;               // viewport ∩ framebuffer
};
CFD_INLINE View make_view(const Camera& cam, const Framebuffer& fb) {
    View v;
    v.eye = cam.eye; v.right = cam.right; v.up = cam.up; v.fwd = cam.fwd;
    const Vec3 ax[3] = {cam.right, cam.up, cam.fwd};
    for (int i = 0; i < 3; ++i) {
        v.m[i][0] = ax[i].x; v.m[i][1] = ax[i].y; v.m[i][2] = ax[i].z;
        v.m[i][3] = -dot(ax[i], cam.eye);
    }
    v.cx = static_cast<float>(cam.vp.x) + 0.5f * static_cast<float>(cam.vp.w);
    v.cy = static_cast<float>(cam.vp.y) + 0.5f * static_cast<float>(cam.vp.h);
    v.ortho = cam.ortho;
    v.f = cam.ortho ? cam.ortho_scale : cam.focal;
    v.znear = cam.znear > 1e-6f ? cam.znear : 1e-6f;
    v.clip = clip_of(fb, cam.vp);
    return v;
}
CFD_INLINE Vec3 to_view(const View& v, Vec3 p) {
    return {v.m[0][0] * p.x + v.m[0][1] * p.y + v.m[0][2] * p.z + v.m[0][3],
            v.m[1][0] * p.x + v.m[1][1] * p.y + v.m[1][2] * p.z + v.m[1][3],
            v.m[2][0] * p.x + v.m[2][1] * p.y + v.m[2][2] * p.z + v.m[2][3]};
}
CFD_INLINE bool finite3(Vec3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }
// Outcode escalar (mismo criterio que la versión AVX2 de la transformación de vértices).
CFD_INLINE u32 outcode(const View& v, Vec3 c) {
    if (!finite3(c)) return OC_BAD;
    u32 oc = 0;
    if (v.ortho) {
        if (v.f * c.x < -k_guard) oc |= OC_LEFT;
        if (v.f * c.x > k_guard) oc |= OC_RIGHT;
        if (v.f * c.y > k_guard) oc |= OC_TOP;
        if (v.f * c.y < -k_guard) oc |= OC_BOTTOM;
    } else {
        const float g = k_guard * c.z;
        if (v.f * c.x < -g) oc |= OC_LEFT;
        if (v.f * c.x > g) oc |= OC_RIGHT;
        if (v.f * c.y > g) oc |= OC_TOP;
        if (v.f * c.y < -g) oc |= OC_BOTTOM;
        if (c.z < v.znear) oc |= OC_NEAR;
    }
    return oc;
}
// Distancia con signo (≥ 0 dentro) al plano de recorte p (0..4 = L,R,T,B,near).
CFD_INLINE float plane_dist(const View& v, int p, Vec3 c) {
    const float g = v.ortho ? k_guard : k_guard * c.z;
    switch (p) {
        case 0: return v.f * c.x + g;
        case 1: return g - v.f * c.x;
        case 2: return g - v.f * c.y;
        case 3: return v.f * c.y + g;
        default: return c.z - v.znear;
    }
}

// ---- Máscaras de carriles: LUT constexpr de 256 entradas (bits → máscara YMM) ----------------
struct alignas(32) MaskRow { i32 v[8]; };
inline constexpr std::array<MaskRow, 256> k_mask_lut = [] {
    std::array<MaskRow, 256> t{};
    for (int m = 0; m < 256; ++m)
        for (int i = 0; i < 8; ++i) t[static_cast<usize>(m)].v[i] = ((m >> i) & 1) ? -1 : 0;
    return t;
}();
CFD_INLINE __m256 mask_ps(int bits) {
    return _mm256_load_ps(reinterpret_cast<const float*>(k_mask_lut[static_cast<usize>(bits)].v));
}
CFD_INLINE __m256i mask_si(int bits) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(k_mask_lut[static_cast<usize>(bits)].v));
}
CFD_INLINE __m256i lane_idx() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
CFD_INLINE __m256 lane_f() { return _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7); }
CFD_INLINE __m256 lane_c() { return _mm256_setr_ps(0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f); }

// Máscara de carriles dentro de [lo, hi] para el bloque que empieza en x.
CFD_INLINE int lane_bits(int x, int lo, int hi) {   // ramas predecibles: sólo bordes
    int m = 0xFF;
    if (lo > x) m &= (0xFF << (lo - x)) & 0xFF;
    if (hi < x + 7) m &= 0xFF >> (x + 7 - hi);
    return m;
}

// ---- Colores SWAR (8 píxeles ARGB, carriles de 16 bits) ------------------------------------
// a16: peso 0..256 replicado en ambas mitades de 16 bits de cada píxel (a | a<<16).
CFD_INLINE __m256i alpha16(__m256 a01) {
    __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(a01, _mm256_set1_ps(256.0f)));
    a = _mm256_min_epi32(_mm256_max_epi32(a, _mm256_setzero_si256()), _mm256_set1_epi32(256));
    return _mm256_or_si256(a, _mm256_slli_epi32(a, 16));
}
// dst·(256-a) + src·a, por canal, /256. Sin desbordes: la suma de pesos es 256 → ≤ 255·256.
CFD_INLINE __m256i blend_swar(__m256i dst, __m256i src, __m256i a16) {
    const __m256i m = _mm256_set1_epi32(0x00FF00FF);
    const __m256i ia = _mm256_sub_epi16(_mm256_set1_epi32(0x01000100), a16);
    const __m256i rb = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_and_si256(src, m), a16),
                                        _mm256_mullo_epi16(_mm256_and_si256(dst, m), ia));
    const __m256i ag = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_srli_epi16(src, 8), a16),
                                        _mm256_mullo_epi16(_mm256_srli_epi16(dst, 8), ia));
    return _mm256_or_si256(_mm256_srli_epi16(rb, 8), _mm256_and_si256(ag, _mm256_set1_epi32(static_cast<int>(0xFF00FF00u))));
}
// c·a/256 por canal (para partículas aditivas premultiplicadas).
CFD_INLINE __m256i scale_swar(__m256i c, __m256i a16) {
    const __m256i m = _mm256_set1_epi32(0x00FF00FF);
    const __m256i rb = _mm256_mullo_epi16(_mm256_and_si256(c, m), a16);
    const __m256i ag = _mm256_mullo_epi16(_mm256_srli_epi16(c, 8), a16);
    return _mm256_or_si256(_mm256_srli_epi16(rb, 8), _mm256_and_si256(ag, _mm256_set1_epi32(static_cast<int>(0xFF00FF00u))));
}
// Empaqueta r,g,b (floats 0..255, ya sujetos) en 0xFFRRGGBB.
CFD_INLINE __m256i pack_rgb(__m256 r, __m256 g, __m256 b) {
    const __m256i ri = _mm256_cvtps_epi32(r), gi = _mm256_cvtps_epi32(g), bi = _mm256_cvtps_epi32(b);
    return _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(ri, 16), _mm256_slli_epi32(gi, 8)),
                           _mm256_or_si256(bi, _mm256_set1_epi32(static_cast<int>(0xFF000000u))));
}
CFD_INLINE __m256 clamp255(__m256 v) { return _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(255.0f)); }
CFD_INLINE __m256 chan(__m256i c, int shift) {
    return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(c, shift), _mm256_set1_epi32(255)));
}
CFD_INLINE float alpha01(u32 c) { return static_cast<float>(c >> 24) * (1.0f / 255.0f); }

// Bilineal SWAR de 8 texels a la vez (4 gathers). tx,ty en coordenadas de texel (centro en i+0.5
// ya restado). Direccionamiento "clamp to edge".
CFD_INLINE __m256i bilinear8(const u32* CFD_RESTRICT tex, int tw, int th, __m256 tx, __m256 ty) {
    tx = _mm256_min_ps(_mm256_max_ps(tx, _mm256_setzero_ps()), _mm256_set1_ps(static_cast<float>(tw - 1)));
    ty = _mm256_min_ps(_mm256_max_ps(ty, _mm256_setzero_ps()), _mm256_set1_ps(static_cast<float>(th - 1)));
    const __m256 fx0 = _mm256_floor_ps(tx), fy0 = _mm256_floor_ps(ty);
    const __m256i x0 = _mm256_cvttps_epi32(fx0), y0 = _mm256_cvttps_epi32(fy0);
    const __m256i wx = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(tx, fx0), _mm256_set1_ps(256.0f)));
    const __m256i wy = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(ty, fy0), _mm256_set1_ps(256.0f)));
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i x1 = _mm256_min_epi32(_mm256_add_epi32(x0, one), _mm256_set1_epi32(tw - 1));
    const __m256i y1 = _mm256_min_epi32(_mm256_add_epi32(y0, one), _mm256_set1_epi32(th - 1));
    const __m256i vtw = _mm256_set1_epi32(tw);
    const __m256i r0 = _mm256_mullo_epi32(y0, vtw), r1 = _mm256_mullo_epi32(y1, vtw);
    const int* T = reinterpret_cast<const int*>(tex);
    const __m256i t00 = _mm256_i32gather_epi32(T, _mm256_add_epi32(r0, x0), 4);
    const __m256i t10 = _mm256_i32gather_epi32(T, _mm256_add_epi32(r0, x1), 4);
    const __m256i t01 = _mm256_i32gather_epi32(T, _mm256_add_epi32(r1, x0), 4);
    const __m256i t11 = _mm256_i32gather_epi32(T, _mm256_add_epi32(r1, x1), 4);
    // pesos que suman exactamente 256: w11 = wx·wy/256, w10 = wx - w11, w01 = wy - w11, w00 = 256 - wx - wy + w11
    const __m256i w11 = _mm256_srli_epi32(_mm256_mullo_epi32(wx, wy), 8);
    const __m256i w10 = _mm256_sub_epi32(wx, w11), w01 = _mm256_sub_epi32(wy, w11);
    const __m256i w00 = _mm256_add_epi32(_mm256_sub_epi32(_mm256_sub_epi32(_mm256_set1_epi32(256), wx), wy), w11);
    auto rep = [](__m256i w) { return _mm256_or_si256(w, _mm256_slli_epi32(w, 16)); };
    const __m256i a00 = rep(w00), a10 = rep(w10), a01 = rep(w01), a11 = rep(w11);
    const __m256i m = _mm256_set1_epi32(0x00FF00FF);
    __m256i rb = _mm256_mullo_epi16(_mm256_and_si256(t00, m), a00);
    rb = _mm256_add_epi16(rb, _mm256_mullo_epi16(_mm256_and_si256(t10, m), a10));
    rb = _mm256_add_epi16(rb, _mm256_mullo_epi16(_mm256_and_si256(t01, m), a01));
    rb = _mm256_add_epi16(rb, _mm256_mullo_epi16(_mm256_and_si256(t11, m), a11));
    __m256i ag = _mm256_mullo_epi16(_mm256_srli_epi16(t00, 8), a00);
    ag = _mm256_add_epi16(ag, _mm256_mullo_epi16(_mm256_srli_epi16(t10, 8), a10));
    ag = _mm256_add_epi16(ag, _mm256_mullo_epi16(_mm256_srli_epi16(t01, 8), a01));
    ag = _mm256_add_epi16(ag, _mm256_mullo_epi16(_mm256_srli_epi16(t11, 8), a11));
    return _mm256_or_si256(_mm256_srli_epi16(rb, 8), _mm256_and_si256(ag, _mm256_set1_epi32(static_cast<int>(0xFF00FF00u))));
}

// ---- Binning por ordenación por conteo -----------------------------------------------------
// Pasada 1 (paralela por slot = rango contiguo de primitivas): cada primitiva calcula su
// rectángulo de tiles y suma 1 en counts[slot][tile]. Prefijo (tile-mayor, slot-menor) →
// cursores. Pasada 2: cada slot escribe sus ids en su tramo de cada tile. Resultado: para cada
// tile una lista CONTIGUA de primitivas en el orden original de envío (determinista, sin atómicos).
// Con lists = 2 cada tile tiene dos listas (bit 31 del rect: 0 = frontal, 1 = trasera) y el
// rasterizador procesa primero las caras frontales: las traseras de una malla cerrada quedan
// entonces tapadas y se descartan en el test de profundidad sin sombrearse.
struct Bins {
    int gx0 = 0, gy0 = 0;        // primer tile (coordenadas absolutas de tile)
    int ntx = 0, nty = 0, ntiles = 0;
    int lists = 1;               // listas por tile (1, o 2 = frontal/trasera)
    int nlists = 0;              // ntiles × lists
    int nslots = 1;
    Clip clip;
    Buffer<u32> counts;          // nslots × nlists  (luego cursores de escritura)
    Buffer<u32> start;           // nlists + 1
    Buffer<u32> items;
    Buffer<u32> order;           // tiles no vacíos, de más a menos cargado (LPT)
    Buffer<u32> scan_tmp;        // totales por lista (prefijo)
    int busy = 0;
    u64 total = 0;

    // Prepara la rejilla para la tijera c. Devuelve false si no hay nada que dibujar.
    bool setup(const Clip& c, i64 nprims, int slots, int lists_per_tile = 1) {
        clip = c;
        busy = 0; total = 0;
        if (c.empty() || nprims <= 0) { ntiles = 0; return false; }
        gx0 = c.x0 >> k_tile_shift; gy0 = c.y0 >> k_tile_shift;
        ntx = (c.x1 >> k_tile_shift) - gx0 + 1;
        nty = (c.y1 >> k_tile_shift) - gy0 + 1;
        if (ntx > 127 || nty > 127) { ntiles = 0; return false; }   // framebuffer > 8128 px: no soportado
        ntiles = ntx * nty;
        lists = lists_per_tile;
        nlists = ntiles * lists;
        nslots = max_(1, slots);
        u32* cnt = grow(counts, static_cast<usize>(nslots) * static_cast<usize>(nlists));
        std::memset(cnt, 0, sizeof(u32) * static_cast<usize>(nslots) * static_cast<usize>(nlists));
        grow(start, static_cast<usize>(nlists) + 1);
        grow(order, static_cast<usize>(ntiles));
        return true;
    }
    CFD_INLINE u32* slot_counts(int s) { return counts.data() + static_cast<usize>(s) * static_cast<usize>(nlists); }
    // Rectángulo de píxeles (inclusivo, ya dentro de la tijera) → rect de tiles empaquetado
    // (7 bits por coordenada; bit 31 = lista trasera).
    CFD_INLINE u32 pack_rect(int px0, int py0, int px1, int py1, bool back = false) const {
        const u32 a = static_cast<u32>((px0 >> k_tile_shift) - gx0), b = static_cast<u32>((py0 >> k_tile_shift) - gy0);
        const u32 c = static_cast<u32>((px1 >> k_tile_shift) - gx0), d = static_cast<u32>((py1 >> k_tile_shift) - gy0);
        return a | (b << 8) | (c << 16) | (d << 24) | (back ? 0x80000000u : 0u);
    }
    struct R { int x0, y0, x1, y1, base; };
    CFD_INLINE R unpack(u32 r) const {
        return {static_cast<int>(r & 127), static_cast<int>((r >> 8) & 127), static_cast<int>((r >> 16) & 127),
                static_cast<int>((r >> 24) & 127), (r >> 31) ? ntiles : 0};
    }
    CFD_INLINE void count(u32* cnt, u32 r) const {
        const R q = unpack(r);
        u32* c = cnt + q.base;
        if (q.x0 == q.x1 && q.y0 == q.y1) { ++c[q.y0 * ntx + q.x0]; return; }   // caso común: 1 tile
        for (int y = q.y0; y <= q.y1; ++y)
            for (int x = q.x0; x <= q.x1; ++x) ++c[y * ntx + x];
    }
    // Variantes con predicado por tile (keep). Un rect de UN tile se acepta sin llamar a keep (la caja
    // ya toca ese tile; si la primitiva no, sólo cuesta una referencia vacía). keep debe ser
    // determinista: count_if y emit_if tienen que decidir exactamente lo mismo para cada tile.
    template <class Keep>
    CFD_INLINE void count_if(u32* cnt, u32 r, Keep&& keep) const {
        const R q = unpack(r);
        u32* c = cnt + q.base;
        if (q.x0 == q.x1 && q.y0 == q.y1) { ++c[q.y0 * ntx + q.x0]; return; }
        for (int y = q.y0; y <= q.y1; ++y)
            for (int x = q.x0; x <= q.x1; ++x) c[y * ntx + x] += keep(x, y) ? 1u : 0u;
    }
    CFD_INLINE void emit(u32* cur, u32 r, u32 id) {
        const R q = unpack(r);
        u32* c = cur + q.base;
        u32* it = items.data();
        if (q.x0 == q.x1 && q.y0 == q.y1) { it[c[q.y0 * ntx + q.x0]++] = id; return; }
        for (int y = q.y0; y <= q.y1; ++y)
            for (int x = q.x0; x <= q.x1; ++x) it[c[y * ntx + x]++] = id;
    }
    template <class Keep>
    CFD_INLINE void emit_if(u32* cur, u32 r, u32 id, Keep&& keep) {
        const R q = unpack(r);
        u32* c = cur + q.base;
        u32* it = items.data();
        if (q.x0 == q.x1 && q.y0 == q.y1) { it[c[q.y0 * ntx + q.x0]++] = id; return; }
        for (int y = q.y0; y <= q.y1; ++y)
            for (int x = q.x0; x <= q.x1; ++x)
                if (keep(x, y)) it[c[y * ntx + x]++] = id;
    }
    // Prefijos → cursores por slot, tramos por lista y orden LPT de tiles no vacíos.
    // Prefijos en 3 fases; las fases 1 y 3 se reparten por tramos de listas entre los hilos y dentro
    // de cada tramo recorren counts (slot-mayor) de forma contigua. Recorrerlo lista-mayor saltaría
    // nlists·4 B (~4.5 KB) por acceso = una línea de caché nueva por contador (medido: ~10× más lento).
    void finalize() {
        u32* st = start.data();
        u32* tot = grow(scan_tmp, static_cast<usize>(nlists));
        u32* cnt = counts.data();
        const usize nl = static_cast<usize>(nlists);
        const int ns = nslots;
        const i64 grain = nlists > 512 ? 256 : nlists;
        parallel_for(0, nlists, grain, [&](i64 lo, i64 hi) {       // 1) totales por lista
            for (i64 t = lo; t < hi; ++t) tot[t] = 0;
            for (int s = 0; s < ns; ++s) {
                const u32* c = cnt + static_cast<usize>(s) * nl;
                for (i64 t = lo; t < hi; ++t) tot[t] += c[t];
            }
        });
        u64 run = 0;
        for (int t = 0; t < nlists; ++t) { st[t] = static_cast<u32>(run); run += tot[t]; tot[t] = st[t]; }   // 2) prefijo
        CFD_CHECK(run < 0xFFFFFFF0ull, "raster: demasiadas referencias en los tiles");
        st[nlists] = static_cast<u32>(run);
        parallel_for(0, nlists, grain, [&](i64 lo, i64 hi) {       // 3) cursores de escritura por slot
            for (int s = 0; s < ns; ++s) {
                u32* c = cnt + static_cast<usize>(s) * nl;
                for (i64 t = lo; t < hi; ++t) { const u32 n = c[t]; c[t] = tot[t]; tot[t] += n; }
            }
        });
        total = run;
        grow(items, static_cast<usize>(run) + 1);
        // Tiles no vacíos ordenados por carga descendente: planificación LPT (el tile más
        // pesado empieza primero → menos cola al final en la CPU híbrida P/E).
        u32* ord = order.data();
        busy = 0;
        for (int t = 0; t < ntiles; ++t)
            if (load(t)) ord[busy++] = static_cast<u32>(t);
        if (!RZ_EXP(11)) std::sort(ord, ord + busy, [this](u32 a, u32 b) { return load(static_cast<int>(a)) > load(static_cast<int>(b)); });
    }
    CFD_INLINE u32 load(int t) const {
        const u32* st = start.data();
        u32 n = st[t + 1] - st[t];
        if (lists == 2) n += st[t + ntiles + 1] - st[t + ntiles];
        return n;
    }
    // Rectángulo de píxeles del tile t (∩ tijera).
    CFD_INLINE void tile_rect(int t, int& x0, int& y0, int& x1, int& y1) const {
        const int tx = t % ntx + gx0, ty = t / ntx + gy0;
        x0 = max_(tx << k_tile_shift, clip.x0);
        y0 = max_(ty << k_tile_shift, clip.y0);
        x1 = min_((tx << k_tile_shift) + k_tile - 1, clip.x1);
        y1 = min_((ty << k_tile_shift) + k_tile - 1, clip.y1);
    }
};

// Número de slots de binning: suficientes para balancear P/E, no tantos como para inflar los contadores.
CFD_INLINE int bin_slots(i64 nprims, i64 per_slot = 2048) {
    const i64 nt = pool().size();
    i64 s = nprims / per_slot;
    s = s < 1 ? 1 : s;
    s = s > nt * 3 ? nt * 3 : s;
    return static_cast<int>(s > 127 ? 127 : s);
}

// ---- Triángulo en punto fijo y recorrido por región ----------------------------------------
struct TriFx {
    i32 x[3], y[3];     // 28.4, orientado con area2 > 0
    i64 area2;
    float inv_area2;
};

// ---- Bloques de 8 píxeles que recibe el callback de los recorridos ---------------------------
// Superficie de destino (color + profundidad con el mismo stride).
struct Surf {
    u32* c;
    float* z;
    usize stride;
    // Sin plano de profundidad, z apunta a un área válida que nunca se usa (evita aritmética sobre nullptr).
    CFD_INLINE static Surf of(Framebuffer& fb) {
        return {fb.color.data(), fb.depth.empty() ? reinterpret_cast<float*>(fb.color.data()) : fb.depth.data(), static_cast<usize>(fb.stride)};
    }
};
// Fila: 8 píxeles consecutivos alineados a 32 B (x múltiplo de 8).
struct RowBlk {
    u32* c; float* z; int x, y;
    CFD_INLINE RowBlk(const Surf& s, int x_, int y_) : c(s.c + static_cast<usize>(y_) * s.stride + x_), z(s.z + static_cast<usize>(y_) * s.stride + x_), x(x_), y(y_) {}
    CFD_INLINE __m256 load_z() const { return _mm256_load_ps(z); }
    CFD_INLINE void store_z(__m256 v) const { _mm256_store_ps(z, v); }
    CFD_INLINE __m256i load_c() const { return _mm256_load_si256(reinterpret_cast<const __m256i*>(c)); }
    CFD_INLINE void store_c(__m256i v) const { _mm256_store_si256(reinterpret_cast<__m256i*>(c), v); }
    CFD_INLINE __m256 px() const { return _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), lane_c()); }   // centros
    CFD_INLINE __m256 py() const { return _mm256_set1_ps(static_cast<float>(y) + 0.5f); }
};
// Sello 4×2: carriles 0-3 = fila y (x..x+3), carriles 4-7 = fila y+1. Dos accesos de 16 B por fila.
struct StampBlk {
    u32* c; float* z; usize stride; int x, y;
    CFD_INLINE StampBlk(const Surf& s, int x_, int y_) : c(s.c + static_cast<usize>(y_) * s.stride + x_), z(s.z + static_cast<usize>(y_) * s.stride + x_), stride(s.stride), x(x_), y(y_) {}
    CFD_INLINE __m256 load_z() const { return _mm256_loadu2_m128(z + stride, z); }
    CFD_INLINE void store_z(__m256 v) const { _mm256_storeu2_m128(z + stride, z, v); }
    CFD_INLINE __m256i load_c() const { return _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(c + stride), reinterpret_cast<const __m128i*>(c)); }
    CFD_INLINE void store_c(__m256i v) const { _mm256_storeu2_m128i(reinterpret_cast<__m128i*>(c + stride), reinterpret_cast<__m128i*>(c), v); }
    CFD_INLINE __m256 px() const { return _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), _mm256_setr_ps(0.5f, 1.5f, 2.5f, 3.5f, 0.5f, 1.5f, 2.5f, 3.5f)); }
    CFD_INLINE __m256 py() const { return _mm256_add_ps(_mm256_set1_ps(static_cast<float>(y)), _mm256_setr_ps(0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f)); }
};

// Recorre los píxeles cubiertos del triángulo t dentro de la región [rx0,rx1]×[ry0,ry1]
// (inclusiva, contenida en UN tile y en la tijera [xlo, xhi] en x). Llama a
//   fn(blk, bits_cobertura, l1, l2)
// para cada bloque de 8 píxeles con algún píxel cubierto; l1, l2 = coordenadas baricéntricas
// lineales en pantalla de los vértices 1 y 2 (l0 = 1 - l1 - l2).
// Regla top-left: una arista "no top-left" resta 1 → E ≥ 0 ⇔ cubierto; se prueba con el bit de
// signo del OR de las tres aristas (un único movmskps).
template <class Fn>
CFD_INLINE void scan_tri(const Surf& S, const TriFx& t, int rx0, int rx1, int ry0, int ry1, int xlo, int xhi, Fn&& fn) {
    const int bx0 = rx0 & ~7;
    const int nb = ((rx1 - bx0) >> 3) + 1;
    const i64 px0 = static_cast<i64>(bx0) * k_sub + k_sub / 2;
    const i64 py0 = static_cast<i64>(ry0) * k_sub + k_sub / 2;
    const i64 wx = static_cast<i64>(nb * 8 - 1) * k_sub, hy = static_cast<i64>(ry1 - ry0) * k_sub;
    i32 e0[3], sx[3], sy[3];
    float lb[3], ldx[3], ldy[3];
    for (int i = 0; i < 3; ++i) {
        const int a = i == 2 ? 0 : i + 1, b = i == 0 ? 2 : (i == 1 ? 0 : 1);
        const i32 A = t.y[a] - t.y[b];                 // ∂E/∂x
        const i32 B = t.x[b] - t.x[a];                 // ∂E/∂y
        const i64 E = static_cast<i64>(A) * (px0 - t.x[a]) + static_cast<i64>(B) * (py0 - t.y[a]);
        lb[i] = static_cast<float>(E) * t.inv_area2;
        ldx[i] = static_cast<float>(A * k_sub) * t.inv_area2;
        ldy[i] = static_cast<float>(B * k_sub) * t.inv_area2;
        const i64 Eb = E - static_cast<i64>(((A > 0) | ((A == 0) & (B > 0))) ^ 1);
        const i64 dA = static_cast<i64>(A) * wx, dB = static_cast<i64>(B) * hy;
        const i64 emax = Eb + (dA > 0 ? dA : 0) + (dB > 0 ? dB : 0);
        if (emax < 0) { RZ_COUNT(early_out, 1); return; }   // región entera fuera de esta arista
        const i64 emin = Eb + (dA < 0 ? dA : 0) + (dB < 0 ? dB : 0);
        if (emin >= 0) { e0[i] = 0; sx[i] = 0; sy[i] = 0; }   // arista trivialmente dentro
        else { e0[i] = static_cast<i32>(Eb); sx[i] = A * k_sub; sy[i] = B * k_sub; }
    }
    const __m256i li = lane_idx();
    __m256i ve0 = _mm256_add_epi32(_mm256_set1_epi32(e0[0]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[0]), li));
    __m256i ve1 = _mm256_add_epi32(_mm256_set1_epi32(e0[1]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[1]), li));
    __m256i ve2 = _mm256_add_epi32(_mm256_set1_epi32(e0[2]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[2]), li));
    const __m256i s80 = _mm256_set1_epi32(sx[0] * 8), s81 = _mm256_set1_epi32(sx[1] * 8), s82 = _mm256_set1_epi32(sx[2] * 8);
    const __m256i sy0 = _mm256_set1_epi32(sy[0]), sy1 = _mm256_set1_epi32(sy[1]), sy2 = _mm256_set1_epi32(sy[2]);
    const __m256 lf = lane_f();
    __m256 vl1 = _mm256_fmadd_ps(_mm256_set1_ps(ldx[1]), lf, _mm256_set1_ps(lb[1]));
    __m256 vl2 = _mm256_fmadd_ps(_mm256_set1_ps(ldx[2]), lf, _mm256_set1_ps(lb[2]));
    const __m256 d81 = _mm256_set1_ps(ldx[1] * 8.0f), d82 = _mm256_set1_ps(ldx[2] * 8.0f);
    const __m256 dy1 = _mm256_set1_ps(ldy[1]), dy2 = _mm256_set1_ps(ldy[2]);
    // Máscaras de tijera para el primer/último bloque de la fila.
    const int xl = bx0 + (nb - 1) * 8;
    const int mfirst = xlo > bx0 ? (0xFF << (xlo - bx0)) & 0xFF : 0xFF;
    const int mlast = xhi < xl + 7 ? 0xFF >> (xl + 7 - xhi) : 0xFF;
    RZ_COUNT(setups, 1); RZ_COUNT(rows, ry1 - ry0 + 1); RZ_COUNT(blocks, (ry1 - ry0 + 1) * nb);
    for (int y = ry0; y <= ry1; ++y) {
        __m256i a = ve0, b = ve1, c = ve2;
        __m256 u = vl1, v = vl2;
        for (int k = 0; k < nb; ++k) {
            const __m256i o = _mm256_or_si256(_mm256_or_si256(a, b), c);
            int bits = (~_mm256_movemask_ps(_mm256_castsi256_ps(o))) & 0xFF;
            bits &= (k == 0 ? mfirst : 0xFF) & (k == nb - 1 ? mlast : 0xFF);
            if (bits) { RZ_COUNT(covered, 1); fn(RowBlk(S, bx0 + k * 8, y), bits, u, v); }
            a = _mm256_add_epi32(a, s80); b = _mm256_add_epi32(b, s81); c = _mm256_add_epi32(c, s82);
            u = _mm256_add_ps(u, d81); v = _mm256_add_ps(v, d82);
        }
        ve0 = _mm256_add_epi32(ve0, sy0); ve1 = _mm256_add_epi32(ve1, sy1); ve2 = _mm256_add_epi32(ve2, sy2);
        vl1 = _mm256_add_ps(vl1, dy1); vl2 = _mm256_add_ps(vl2, dy2);
    }
}

// Variante rápida para triángulos pequeños: la región ES la caja de píxeles completa del
// triángulo (no la recorta ni el tile ni la tijera) → |E| < 2^22, todo en int32, sin
// clasificación de aristas ni máscaras de tijera (los carriles fuera de la caja caen fuera de
// alguna arista). Las baricéntricas salen de las propias funciones de arista (cvtdq2ps + FMA)
// y sólo en los bloques cubiertos: el bucle interno mantiene 3 registros en vez de 5.
// Lanes = (dx, dy) arbitrarios: filas de 8 (Stamp = false) o sellos 4×2 (Stamp = true).
template <bool Stamp, class Fn>
CFD_INLINE void scan_tri_small(const Surf& S, const TriFx& t, int bx0, int by0, int nbx, int nby, Fn&& fn) {
    constexpr int SW = Stamp ? 4 : 8, SH = Stamp ? 2 : 1;
    const i32 px0 = bx0 * k_sub + k_sub / 2, py0 = by0 * k_sub + k_sub / 2;
    i32 e0[3], sx[3], sy[3], cor[3];
    for (int i = 0; i < 3; ++i) {
        const int a = i == 2 ? 0 : i + 1, b = i == 0 ? 2 : (i == 1 ? 0 : 1);
        const i32 A = t.y[a] - t.y[b], B = t.x[b] - t.x[a];
        cor[i] = static_cast<i32>(((A > 0) | ((A == 0) & (B > 0))) ^ 1);   // top-left sin ramas
        e0[i] = A * (px0 - t.x[a]) + B * (py0 - t.y[a]) - cor[i];
        sx[i] = A * k_sub; sy[i] = B * k_sub;
    }
    RZ_COUNT(setups, 1); RZ_COUNT(rows, nby); RZ_COUNT(blocks, nby * nbx);
    const __m256i lx = Stamp ? _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3) : lane_idx();
    __m256i ve0 = _mm256_add_epi32(_mm256_set1_epi32(e0[0]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[0]), lx));
    __m256i ve1 = _mm256_add_epi32(_mm256_set1_epi32(e0[1]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[1]), lx));
    __m256i ve2 = _mm256_add_epi32(_mm256_set1_epi32(e0[2]), _mm256_mullo_epi32(_mm256_set1_epi32(sx[2]), lx));
    if constexpr (Stamp) {   // carriles 4-7 una fila más abajo
        const __m256i hi = _mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1);
        ve0 = _mm256_add_epi32(ve0, _mm256_and_si256(_mm256_set1_epi32(sy[0]), hi));
        ve1 = _mm256_add_epi32(ve1, _mm256_and_si256(_mm256_set1_epi32(sy[1]), hi));
        ve2 = _mm256_add_epi32(ve2, _mm256_and_si256(_mm256_set1_epi32(sy[2]), hi));
    }
    const __m256i s80 = _mm256_set1_epi32(sx[0] * SW), s81 = _mm256_set1_epi32(sx[1] * SW), s82 = _mm256_set1_epi32(sx[2] * SW);
    const __m256i sy0 = _mm256_set1_epi32(sy[0] * SH), sy1 = _mm256_set1_epi32(sy[1] * SH), sy2 = _mm256_set1_epi32(sy[2] * SH);
    const __m256 ia = _mm256_set1_ps(t.inv_area2);
    const __m256 c1 = _mm256_set1_ps(static_cast<float>(cor[1]) * t.inv_area2), c2 = _mm256_set1_ps(static_cast<float>(cor[2]) * t.inv_area2);
    for (int j = 0; j < nby; ++j) {
        __m256i a = ve0, b = ve1, c = ve2;
        for (int k = 0; k < nbx; ++k) {
            const __m256i o = _mm256_or_si256(_mm256_or_si256(a, b), c);
            const int bits = (~_mm256_movemask_ps(_mm256_castsi256_ps(o))) & 0xFF;
            if (bits) {
                RZ_COUNT(covered, 1);
                const __m256 l1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(b), ia, c1), l2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c), ia, c2);
                if constexpr (Stamp) fn(StampBlk(S, bx0 + k * SW, by0 + j * SH), bits, l1, l2);
                else fn(RowBlk(S, bx0 + k * SW, by0 + j * SH), bits, l1, l2);
            }
            a = _mm256_add_epi32(a, s80); b = _mm256_add_epi32(b, s81); c = _mm256_add_epi32(c, s82);
        }
        ve0 = _mm256_add_epi32(ve0, sy0); ve1 = _mm256_add_epi32(ve1, sy1); ve2 = _mm256_add_epi32(ve2, sy2);
    }
}

// Caja de píxeles SIN recortar (centros de píxel dentro de la caja de los vértices).
CFD_INLINE void tri_pixel_box_raw(const i32 x[3], const i32 y[3], int& px0, int& py0, int& px1, int& py1) {
    px0 = (min_(x[0], min_(x[1], x[2])) + 7) >> k_sub_shift;
    py0 = (min_(y[0], min_(y[1], y[2])) + 7) >> k_sub_shift;
    px1 = (max_(x[0], max_(x[1], x[2])) - 8) >> k_sub_shift;
    py1 = (max_(y[0], max_(y[1], y[2])) - 8) >> k_sub_shift;
}

// Recorre el triángulo en el rectángulo [cx0,cx1]×[cy0,cy1] (tile ∩ tijera). Elige:
//  * sellos 4×2 si la caja cabe entera en el rectángulo y mide ≤ 8×8 (el 90-97 % de los
//    triángulos de una malla densa): los sellos se colocan pegados al borde del tile si hace
//    falta para no solaparse NI salirse del tile (sin carreras con el hilo del tile vecino);
//  * filas de 8 alineadas en int32 si la caja cabe pero es mayor;
//  * la variante general (64 bits + clasificación trivial de aristas) si cruza el tile.
template <class Fn>
CFD_INLINE void scan_tri_in(const Surf& S, const TriFx& t, int cx0, int cy0, int cx1, int cy1, Fn&& fn) {
    int px0, py0, px1, py1;
    tri_pixel_box_raw(t.x, t.y, px0, py0, px1, py1);
    if (px0 >= cx0 && py0 >= cy0 && px1 <= cx1 && py1 <= cy1) {
        if (px0 > px1 || py0 > py1) return;
        const int w = px1 - px0 + 1, h = py1 - py0 + 1;
#ifdef RZ_STATS
        if (w <= 8 && h <= 8) ++t_rzc.hist[w][h]; else ++t_rzc.big;
#endif
        if (w <= 8 && h <= 8 && !RZ_EXP(0)) {
            const int nsx = (w + 3) >> 2, nsy = (h + 1) >> 1;
            const int sx0 = min_(px0, cx1 - 4 * nsx + 1), sy0 = min_(py0, cy1 - 2 * nsy + 1);
            if (sx0 >= cx0 && sy0 >= cy0) { scan_tri_small<true>(S, t, sx0, sy0, nsx, nsy, fn); return; }
        }
        const int bx0 = px0 & ~7;
        scan_tri_small<false>(S, t, bx0, py0, ((px1 - bx0) >> 3) + 1, h, fn);
        return;
    }
    px0 = max_(px0, cx0); py0 = max_(py0, cy0); px1 = min_(px1, cx1); py1 = min_(py1, cy1);
    if (px0 <= px1 && py0 <= py1) scan_tri(S, t, px0, px1, py0, py1, cx0, cx1, fn);
}

// Caja de píxeles (centros cubiertos posibles) de un triángulo en 28.4, ∩ tijera. false si vacía.
CFD_INLINE bool tri_pixel_box(const i32 x[3], const i32 y[3], const Clip& c, int& px0, int& py0, int& px1, int& py1) {
    const i32 xmin = min_(x[0], min_(x[1], x[2])), xmax = max_(x[0], max_(x[1], x[2]));
    const i32 ymin = min_(y[0], min_(y[1], y[2])), ymax = max_(y[0], max_(y[1], y[2]));
    // centro del píxel p en 16p+8: p ≥ ceil((min-8)/16) = (min+7)>>4 ; p ≤ floor((max-8)/16) = (max-8)>>4
    px0 = max_((xmin + 7) >> k_sub_shift, c.x0);
    py0 = max_((ymin + 7) >> k_sub_shift, c.y0);
    px1 = min_((xmax - 8) >> k_sub_shift, c.x1);
    py1 = min_((ymax - 8) >> k_sub_shift, c.y1);
    return px0 <= px1 && py0 <= py1;
}

// ---- Recorte de polígonos en espacio de vista (Sutherland–Hodgman) ---------------------------
// Vértice de recorte con hasta 8 atributos que se interpolan linealmente en 3D.
struct CVert {
    Vec3 c;          // coordenadas de vista
    float a[8];
    int orig;        // índice del vértice original (≥ 0) si no fue creado por el recorte
};
// Recorta el polígono (n vértices en `in`) contra los planos de `planes` (bits OC_*).
// La intersección se calcula SIEMPRE desde el vértice interior hacia el exterior → dos
// triángulos que comparten arista producen exactamente el mismo punto (sin grietas).
CFD_INLINE int clip_polygon(const View& v, CVert* in, int n, CVert* tmp, u32 planes, int nattr) {
    CVert* src = in;
    CVert* dst = tmp;
    const int order[5] = {4, 0, 1, 2, 3};   // primero el plano cercano
    for (int oi = 0; oi < 5 && n > 0; ++oi) {
        const int p = order[oi];
        if (!(planes & (1u << p))) continue;
        int m = 0;
        for (int i = 0; i < n; ++i) {
            const CVert& A = src[i == 0 ? n - 1 : i - 1];
            const CVert& B = src[i];
            const float da = plane_dist(v, p, A.c), db = plane_dist(v, p, B.c);
            // orig ≥ 0 ⇔ la transformación AVX2 lo dio por dentro de TODOS los planos (outcode 0) y sus
            // vecinos sin recortar usan su posición proyectada. Se mantiene dentro aunque la Z escalar
            // (otro orden de FMA, ±1 ulp) lo deje justo fuera: si no, se sustituiría por un punto
            // re-proyectado a 1/16 px del original → posible grieta con el triángulo vecino.
            const bool ina = da >= 0.0f || A.orig >= 0, inb = db >= 0.0f || B.orig >= 0;
            if (ina != inb) {
                const CVert& I = ina ? A : B;      // interior
                const CVert& O = ina ? B : A;      // exterior
                const float di = ina ? da : db, dout = ina ? db : da;
                const float t = di / (di - dout);
                CVert& r = dst[m++];
                r.c = I.c + (O.c - I.c) * t;
                for (int k = 0; k < nattr; ++k) r.a[k] = I.a[k] + (O.a[k] - I.a[k]) * t;
                r.orig = -1;
            }
            if (inb) dst[m++] = B;
        }
        n = m;
        CVert* sw = src; src = dst; dst = sw;
    }
    if (src != in)
        for (int i = 0; i < n; ++i) in[i] = src[i];
    return n;
}
// Proyección escalar de un punto de vista (Z > 0 en perspectiva) a 28.4 + q.
CFD_INLINE bool project_fx(const View& v, Vec3 c, i32& fx, i32& fy, float& q) {
    float sx, sy;
    if (v.ortho) { sx = v.cx + v.f * c.x; sy = v.cy - v.f * c.y; q = c.z; }
    else {
        const float iz = 1.0f / c.z;
        sx = v.cx + v.f * c.x * iz; sy = v.cy - v.f * c.y * iz; q = iz;
    }
    const float lim = 262000.0f / k_sub;   // |coord| < 2^18 en 28.4
    if (!(std::fabs(sx) < lim && std::fabs(sy) < lim) || !std::isfinite(q)) return false;
    fx = static_cast<i32>(std::lrintf(sx * static_cast<float>(k_sub)));
    fy = static_cast<i32>(std::lrintf(sy * static_cast<float>(k_sub)));
    return true;
}

// draw_mesh sin registrar estadísticas (lo usa draw_arrow para la cabeza cónica).
void draw_mesh_impl(Framebuffer& fb, const Camera& cam, const Mesh& mesh, const Light& light, const MeshStyle& style, bool record);

} // namespace cfd::render::rz
