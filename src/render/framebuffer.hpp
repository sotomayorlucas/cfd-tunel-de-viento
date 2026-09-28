// ============================================================================
//  render/framebuffer.hpp — destino de todo el render por software.
//
//  color: 0xAARRGGBB (en memoria little-endian: B,G,R,A) = formato nativo de
//         una XImage ZPixmap de 24/32 bits → se copia a X11 sin convertir.
//  depth: profundidad de vista lineal (distancia a lo largo del eje de la
//         cámara, en unidades de mundo = celdas). Menor = más cerca.
//         Se limpia a +inf. Todo lo 3D hace test de profundidad contra esto.
//  Filas alineadas: el ancho en memoria (stride) es múltiplo de 16 píxeles
//  para que cada fila empiece alineada a 64 B.
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "../core/mem.hpp"
#include "../core/simd.hpp"
#include "../core/threadpool.hpp"
#include <limits>

namespace cfd::render {

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    constexpr bool contains(int px, int py) const { return px >= x && py >= y && px < x + w && py < y + h; }
};

struct Framebuffer {
    int w = 0, h = 0, stride = 0;
    Buffer<u32> color;
    Buffer<float> depth;

    void resize(int nw, int nh) {
        if (nw == w && nh == h) return;
        w = nw; h = nh;
        stride = (nw + 15) & ~15;
        color.resize(static_cast<usize>(stride) * static_cast<usize>(nh), true);
        depth.resize(static_cast<usize>(stride) * static_cast<usize>(nh), false);
        clear_depth();
    }
    CFD_INLINE u32* row(int y) { return color.data() + static_cast<usize>(y) * static_cast<usize>(stride); }
    CFD_INLINE const u32* row(int y) const { return color.data() + static_cast<usize>(y) * static_cast<usize>(stride); }
    CFD_INLINE float* drow(int y) { return depth.data() + static_cast<usize>(y) * static_cast<usize>(stride); }
    CFD_INLINE const float* drow(int y) const { return depth.data() + static_cast<usize>(y) * static_cast<usize>(stride); }

    // Limpieza con stores no temporales (no contaminan la caché) en paralelo por filas.
    void clear_color(u32 c) {
        u32* p = color.data();
        const i64 n = static_cast<i64>(color.size());
        parallel_for(0, n / 8, 1 << 14, [&](i64 lo, i64 hi) {
            const __m256i v = _mm256_set1_epi32(static_cast<int>(c));
            for (i64 i = lo; i < hi; ++i) _mm256_stream_si256(reinterpret_cast<__m256i*>(p + i * 8), v);
        });
        _mm_sfence();
    }
    void clear_depth() {
        float* p = depth.data();
        const i64 n = static_cast<i64>(depth.size());
        parallel_for(0, n / 8, 1 << 14, [&](i64 lo, i64 hi) {
            const __m256 v = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            for (i64 i = lo; i < hi; ++i) _mm256_stream_ps(p + i * 8, v);
        });
        _mm_sfence();
    }
    // Fondo con degradado vertical (cielo de túnel) dentro de un rectángulo.
    void gradient(Rect r, u32 top, u32 bottom) {
        parallel_for(r.y, r.y + r.h, 16, [&](i64 lo, i64 hi) {
            for (i64 y = lo; y < hi; ++y) {
                const u32 t = static_cast<u32>((y - r.y) * 256 / (r.h > 1 ? r.h - 1 : 1));
                const u32 c = simd::blend_argb(top, bottom, t > 256 ? 256 : t);
                u32* d = row(static_cast<int>(y)) + r.x;
                for (int x = 0; x < r.w; ++x) d[x] = c;
            }
        });
    }
};

// Utilidades de color empaquetado.
CFD_INLINE constexpr u32 rgb(u32 r, u32 g, u32 b) { return 0xFF000000u | (r << 16) | (g << 8) | b; }
CFD_INLINE constexpr u32 rgba(u32 r, u32 g, u32 b, u32 a) { return (a << 24) | (r << 16) | (g << 8) | b; }
CFD_INLINE u32 rgbf(float r, float g, float b) {
    return rgb(static_cast<u32>(saturate(r) * 255.0f + 0.5f), static_cast<u32>(saturate(g) * 255.0f + 0.5f), static_cast<u32>(saturate(b) * 255.0f + 0.5f));
}
CFD_INLINE Vec3 unpack_rgb(u32 c) {
    constexpr float k = 1.0f / 255.0f;
    return {static_cast<float>((c >> 16) & 255) * k, static_cast<float>((c >> 8) & 255) * k, static_cast<float>(c & 255) * k};
}

} // namespace cfd::render
