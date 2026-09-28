// ============================================================================
//  core/simd.hpp — envoltorios finos sobre AVX2/FMA/F16C.
//
//  f8  = 8 floats en un registro YMM, i8x = 8 int32.
//  Cero abstracción en tiempo de ejecución: todo always_inline, los
//  operadores se reducen a una instrucción. Incluye conversiones FP16
//  (vcvtph2ps / vcvtps2ph) usadas por el almacenamiento comprimido del LBM.
// ============================================================================
#pragma once

#include "config.hpp"
#include <immintrin.h>

namespace cfd::simd {

struct f8 {
    __m256 v;
    CFD_INLINE f8() = default;
    CFD_INLINE f8(__m256 x) : v(x) {}
    CFD_INLINE explicit f8(float s) : v(_mm256_set1_ps(s)) {}
    CFD_INLINE operator __m256() const { return v; }
    CFD_INLINE static f8 zero() { return _mm256_setzero_ps(); }
    CFD_INLINE static f8 load(const float* p) { return _mm256_loadu_ps(p); }
    CFD_INLINE static f8 load_a(const float* p) { return _mm256_load_ps(p); }
    CFD_INLINE void store(float* p) const { _mm256_storeu_ps(p, v); }
    CFD_INLINE void store_a(float* p) const { _mm256_store_ps(p, v); }
    CFD_INLINE void stream(float* p) const { _mm256_stream_ps(p, v); } // no-temporal: salta la caché
    // FP16 <-> FP32 (8 half = 128 bits)
    CFD_INLINE static f8 load_h(const std::uint16_t* p) { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
    CFD_INLINE void store_h(std::uint16_t* p) const { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }
    CFD_INLINE f8 operator+(f8 o) const { return _mm256_add_ps(v, o.v); }
    CFD_INLINE f8 operator-(f8 o) const { return _mm256_sub_ps(v, o.v); }
    CFD_INLINE f8 operator*(f8 o) const { return _mm256_mul_ps(v, o.v); }
    CFD_INLINE f8 operator/(f8 o) const { return _mm256_div_ps(v, o.v); }
    CFD_INLINE f8 operator-() const { return _mm256_xor_ps(v, _mm256_set1_ps(-0.0f)); }
    CFD_INLINE f8& operator+=(f8 o) { v = _mm256_add_ps(v, o.v); return *this; }
    CFD_INLINE f8& operator-=(f8 o) { v = _mm256_sub_ps(v, o.v); return *this; }
    CFD_INLINE f8& operator*=(f8 o) { v = _mm256_mul_ps(v, o.v); return *this; }
    CFD_INLINE f8 operator<(f8 o) const { return _mm256_cmp_ps(v, o.v, _CMP_LT_OQ); }
    CFD_INLINE f8 operator>(f8 o) const { return _mm256_cmp_ps(v, o.v, _CMP_GT_OQ); }
    CFD_INLINE f8 operator<=(f8 o) const { return _mm256_cmp_ps(v, o.v, _CMP_LE_OQ); }
    CFD_INLINE f8 operator>=(f8 o) const { return _mm256_cmp_ps(v, o.v, _CMP_GE_OQ); }
    CFD_INLINE f8 operator&(f8 o) const { return _mm256_and_ps(v, o.v); }
    CFD_INLINE f8 operator|(f8 o) const { return _mm256_or_ps(v, o.v); }
};

CFD_INLINE f8 fmadd(f8 a, f8 b, f8 c) { return _mm256_fmadd_ps(a, b, c); }   // a*b + c
CFD_INLINE f8 fmsub(f8 a, f8 b, f8 c) { return _mm256_fmsub_ps(a, b, c); }   // a*b - c
CFD_INLINE f8 fnmadd(f8 a, f8 b, f8 c) { return _mm256_fnmadd_ps(a, b, c); } // c - a*b
CFD_INLINE f8 min(f8 a, f8 b) { return _mm256_min_ps(a, b); }
CFD_INLINE f8 max(f8 a, f8 b) { return _mm256_max_ps(a, b); }
CFD_INLINE f8 abs(f8 a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
CFD_INLINE f8 sqrt(f8 a) { return _mm256_sqrt_ps(a); }
// select(mask, a, b): lane = mask ? a : b
CFD_INLINE f8 select(f8 mask, f8 a, f8 b) { return _mm256_blendv_ps(b, a, mask); }
// rsqrt aproximado (12 bits) + 1 paso de Newton-Raphson ≈ 23 bits.
CFD_INLINE f8 rsqrt_nr(f8 x) {
    const __m256 y = _mm256_rsqrt_ps(x);
    const __m256 hx = _mm256_mul_ps(x, _mm256_set1_ps(0.5f));
    return _mm256_mul_ps(y, _mm256_fnmadd_ps(hx, _mm256_mul_ps(y, y), _mm256_set1_ps(1.5f)));
}
// 1/x aproximado + Newton.
CFD_INLINE f8 rcp_nr(f8 x) {
    const __m256 y = _mm256_rcp_ps(x);
    return _mm256_mul_ps(y, _mm256_fnmadd_ps(x, y, _mm256_set1_ps(2.0f)));
}
CFD_INLINE int movemask(f8 m) { return _mm256_movemask_ps(m); }
// Suma horizontal de los 8 carriles.
CFD_INLINE float hsum(f8 a) {
    __m128 lo = _mm256_castps256_ps128(a.v), hi = _mm256_extractf128_ps(a.v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, sh));
}
CFD_INLINE float hmax(f8 a) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

struct i8x {
    __m256i v;
    CFD_INLINE i8x() = default;
    CFD_INLINE i8x(__m256i x) : v(x) {}
    CFD_INLINE explicit i8x(int s) : v(_mm256_set1_epi32(s)) {}
    CFD_INLINE static i8x load(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }
    CFD_INLINE void store(void* p) const { _mm256_storeu_si256(static_cast<__m256i*>(p), v); }
    CFD_INLINE i8x operator+(i8x o) const { return _mm256_add_epi32(v, o.v); }
    CFD_INLINE i8x operator-(i8x o) const { return _mm256_sub_epi32(v, o.v); }
    CFD_INLINE i8x operator*(i8x o) const { return _mm256_mullo_epi32(v, o.v); }
    CFD_INLINE i8x operator&(i8x o) const { return _mm256_and_si256(v, o.v); }
    CFD_INLINE i8x operator|(i8x o) const { return _mm256_or_si256(v, o.v); }
};
CFD_INLINE f8 to_float(i8x a) { return _mm256_cvtepi32_ps(a.v); }
CFD_INLINE i8x to_int_trunc(f8 a) { return _mm256_cvttps_epi32(a.v); }
CFD_INLINE f8 as_float(i8x a) { return _mm256_castsi256_ps(a.v); }
CFD_INLINE i8x as_int(f8 a) { return _mm256_castps_si256(a.v); }
// Expande 8 bytes (u8) a 8 int32.
CFD_INLINE i8x load_u8x8(const std::uint8_t* p) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
// Máscara float (todos 1s si el byte cumple (b & bits) != 0).
CFD_INLINE f8 mask_bits_u8(const std::uint8_t* p, int bits) {
    const __m256i b = _mm256_and_si256(load_u8x8(p).v, _mm256_set1_epi32(bits));
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(b, _mm256_setzero_si256()), _mm256_set1_epi32(-1)));
}
// Prueba rápida sobre 8 flags consecutivos: ¿alguno tiene (flag & bits)?  (SWAR sobre u64)
CFD_INLINE bool any_bits_u8x8(const std::uint8_t* p, std::uint8_t bits) {
    std::uint64_t w;
    __builtin_memcpy(&w, p, 8);
    return (w & (0x0101010101010101ull * bits)) != 0;
}

// Colores empaquetados 0xAARRGGBB: mezcla alfa de 2 píxeles a la vez con SWAR (8 bits por canal).
CFD_INLINE std::uint32_t blend_argb(std::uint32_t dst, std::uint32_t src, std::uint32_t a /*0..256*/) {
    const std::uint32_t ia = 256 - a;
    const std::uint32_t rb = (((src & 0x00FF00FFu) * a + (dst & 0x00FF00FFu) * ia) >> 8) & 0x00FF00FFu;
    const std::uint32_t g  = (((src & 0x0000FF00u) * a + (dst & 0x0000FF00u) * ia) >> 8) & 0x0000FF00u;
    return 0xFF000000u | rb | g;
}
// Suma saturada por canal (para partículas aditivas): una instrucción SSE.
CFD_INLINE std::uint32_t add_sat_argb(std::uint32_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>(_mm_cvtsi128_si32(_mm_adds_epu8(_mm_cvtsi32_si128(static_cast<int>(a)), _mm_cvtsi32_si128(static_cast<int>(b)))));
}

} // namespace cfd::simd
