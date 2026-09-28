// ============================================================================
//  platform/upscale.cpp — escalado entero del framebuffer a la ventana (AVX2).
//
//  1920×1200 → 3840×2400 = 36.9 MB escritos por cuadro: el cuello de botella es
//  el ancho de banda de memoria, no la aritmética. Por eso:
//   * cada fila fuente se lee UNA vez (7.5 KB, cabe en L1) y sus dos filas
//     destino se escriben desde los mismos registros (sin releer la primera fila
//     para copiarla: eso sería un 50 % más de tráfico);
//   * stores NO temporales (VMOVNTDQ): la imagen la lee el servidor X desde la
//     memoria compartida; escribir con RFO + ensuciar nuestra L2/L3 no aporta nada
//     (sin NT cada línea destino se lee antes de escribirse = +36.9 MB de lectura);
//   * duplicado horizontal con 2 VPERMD (8 → 16 píxeles);
//   * reparto por filas en el pool de hilos (varios núcleos = más peticiones de
//     memoria en vuelo: un solo núcleo no satura LPDDR5x).
// ============================================================================
#include "platform.hpp"

#include <cstring>

namespace cfd::platform {
namespace {

template <bool NT>
CFD_INLINE void st(u32* p, __m256i v) {
    if constexpr (NT) _mm256_stream_si256(reinterpret_cast<__m256i*>(p), v);
    else _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}

// Una fila fuente → dos filas destino (d1 puede ser nullptr si la ventana corta la última).
template <bool NT>
CFD_INLINE void row_x2(const u32* CFD_RESTRICT s, u32* CFD_RESTRICT d0, u32* CFD_RESTRICT d1, int wout) {
    const __m256i ia = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i ib = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
    int x = 0;                                          // x = píxel destino
    if (d1) {
        for (; x + 16 <= wout; x += 16) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + (x >> 1)));
            const __m256i a = _mm256_permutevar8x32_epi32(v, ia);
            const __m256i b = _mm256_permutevar8x32_epi32(v, ib);
            st<NT>(d0 + x, a); st<NT>(d0 + x + 8, b);
            st<NT>(d1 + x, a); st<NT>(d1 + x + 8, b);
        }
        for (; x < wout; ++x) d0[x] = d1[x] = s[x >> 1];
    } else {
        for (; x + 16 <= wout; x += 16) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + (x >> 1)));
            st<NT>(d0 + x, _mm256_permutevar8x32_epi32(v, ia));
            st<NT>(d0 + x + 8, _mm256_permutevar8x32_epi32(v, ib));
        }
        for (; x < wout; ++x) d0[x] = s[x >> 1];
    }
}

template <bool NT>
CFD_INLINE void copy_row(const u32* CFD_RESTRICT s, u32* CFD_RESTRICT d, int n) {
    if constexpr (NT) {
        int x = 0;
        for (; x + 8 <= n; x += 8) st<true>(d + x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + x)));
        for (; x < n; ++x) d[x] = s[x];
    } else {
        std::memcpy(d, s, static_cast<usize>(n) * 4);
    }
}

CFD_INLINE void black_row(u32* d, int n) { std::memset(d, 0, static_cast<usize>(n) * 4); }

template <bool NT>
void upscale_rows(const u32* src, int sw, int sh, int sstride, u32* dst, int dw, int dh, int dstride,
                  int scale, i64 sy0, i64 sy1) {
    const int wout = min_(dw, sw * scale);
    const int hout = min_(dh, sh * scale);
    for (i64 sy = sy0; sy < sy1; ++sy) {
        const u32* s = src + static_cast<usize>(sy) * static_cast<usize>(sstride);
        const int dy0 = static_cast<int>(sy) * scale;
        if (dy0 >= hout) break;
        u32* d0 = dst + static_cast<usize>(dy0) * static_cast<usize>(dstride);
        if (scale == 1) {
            copy_row<NT>(s, d0, wout);
        } else if (scale == 2) {
            row_x2<NT>(s, d0, dy0 + 1 < hout ? d0 + dstride : nullptr, wout);
        } else {
            for (int x = 0; x < wout; ++x) d0[x] = s[x / scale];
            for (int k = 1; k < scale && dy0 + k < hout; ++k)
                copy_row<NT>(d0, d0 + static_cast<usize>(k) * static_cast<usize>(dstride), wout);
        }
        // Margen derecho si la ventana es más ancha que fb×scale (durante un redimensionado).
        if (wout < dw)
            for (int k = 0; k < scale && dy0 + k < hout; ++k) black_row(d0 + static_cast<usize>(k) * static_cast<usize>(dstride) + wout, dw - wout);
    }
    if constexpr (NT) _mm_sfence();                 // stores NT visibles antes de avisar al servidor
}

} // namespace

void upscale_argb(const u32* src, int sw, int sh, int sstride, u32* dst, int dw, int dh, int dstride,
                  int scale, bool parallel, bool nontemporal) {
    if (!src || !dst || dw <= 0 || dh <= 0) return;
    scale = clamp_(scale, 1, 8);
    if (sw <= 0 || sh <= 0) {
        for (int y = 0; y < dh; ++y) black_row(dst + static_cast<usize>(y) * static_cast<usize>(dstride), dw);
        return;
    }
    const bool nt = nontemporal && (reinterpret_cast<uintptr_t>(dst) & 31) == 0 && (dstride & 7) == 0;
    const i64 rows = min_<i64>(sh, (dh + scale - 1) / scale);
    auto job = [&](i64 lo, i64 hi) {
        if (nt) upscale_rows<true>(src, sw, sh, sstride, dst, dw, dh, dstride, scale, lo, hi);
        else upscale_rows<false>(src, sw, sh, sstride, dst, dw, dh, dstride, scale, lo, hi);
    };
    // Grano: 8 filas fuente ≈ 16 filas destino ≈ 245 KB a 3840 px (cabe en la L2 de sobra).
    // (core/threadpool: parallel_for no admite un invocable lvalue → lambda temporal)
    if (parallel) parallel_for(0, rows, 8, [&](i64 lo, i64 hi) { job(lo, hi); });
    else job(0, rows);
    // Filas inferiores sobrantes (ventana más alta que fb×scale).
    const int hout = min_(dh, sh * scale);
    for (int y = hout; y < dh; ++y) black_row(dst + static_cast<usize>(y) * static_cast<usize>(dstride), dw);
}

} // namespace cfd::platform
