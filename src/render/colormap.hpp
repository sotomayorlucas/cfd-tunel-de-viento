// ============================================================================
//  render/colormap.hpp — mapas de color científicos como LUT de 256 entradas
//  generadas EN TIEMPO DE COMPILACIÓN (constexpr) a partir de aproximaciones
//  polinómicas (Turbo: A. Mikhailov/Google; Viridis/Inferno: ajustes de mattz)
//  y de la tabla de Moreland (CoolWarm, divergente: ideal para Cp).
//  Búsqueda vectorial de 8 valores con VPGATHERDD.
// ============================================================================
#pragma once

#include "../core/simd.hpp"
#include "../geom/sdf.hpp"
#include <array>

namespace cfd::render {

enum class Colormap : u8 { Turbo, Viridis, Inferno, CoolWarm, Gray, Count };

namespace detail {
constexpr u32 pack(float r, float g, float b) {
    auto c = [](float v) { v = v < 0 ? 0 : (v > 1 ? 1 : v); return static_cast<u32>(v * 255.0f + 0.5f); };
    return 0xFF000000u | (c(r) << 16) | (c(g) << 8) | c(b);
}
constexpr u32 turbo(float x) {
    const float x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x4 * x;
    const float r = 0.13572138f + 4.61539260f * x - 42.66032258f * x2 + 132.13108234f * x3 - 152.94239396f * x4 + 59.28637943f * x5;
    const float g = 0.09140261f + 2.19418839f * x + 4.84296658f * x2 - 14.18503333f * x3 + 4.27729857f * x4 + 2.82956604f * x5;
    const float b = 0.10667330f + 12.64194608f * x - 60.58204836f * x2 + 110.36276771f * x3 - 89.90310912f * x4 + 27.34824973f * x5;
    return pack(r, g, b);
}
constexpr float poly6(float t, const float c[7]) {
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * (c[5] + t * c[6])))));
}
constexpr u32 viridis(float t) {
    constexpr float r[7] = {0.2777273272234177f, 0.1050930431085774f, -0.3308618287255563f, -4.634230498983486f, 6.228269936347081f, 4.776384997670288f, -5.435455855934631f};
    constexpr float g[7] = {0.005407344544966578f, 1.404613529898575f, 0.214847559468213f, -5.799100973351585f, 14.17993336680509f, -13.74514537774601f, 4.645852612178535f};
    constexpr float b[7] = {0.3340998053353061f, 1.384590162594685f, 0.09509516302823659f, -19.33244095627987f, 56.69055260068105f, -65.35303263337234f, 26.3124352495832f};
    return pack(poly6(t, r), poly6(t, g), poly6(t, b));
}
constexpr u32 inferno(float t) {
    constexpr float r[7] = {0.0002189403691192265f, 0.1065134194856116f, 11.60249308247187f, -41.70399613139459f, 77.162935699427f, -71.31942824499214f, 25.13112622477341f};
    constexpr float g[7] = {0.001651004631001012f, 0.5639564367884091f, -3.972853965665698f, 17.43639888205313f, -33.40235894210092f, 32.62606426397723f, -12.24266895238567f};
    constexpr float b[7] = {-0.01948089843709184f, 3.932712388889277f, -15.9423941062914f, 44.35414519872813f, -81.80730925738993f, 73.20951985803202f, -23.07032500287172f};
    return pack(poly6(t, r), poly6(t, g), poly6(t, b));
}
constexpr u32 coolwarm(float t) {
    constexpr float k[9][3] = {{59, 76, 192}, {98, 130, 234}, {141, 176, 254}, {184, 208, 249}, {221, 221, 221},
                               {245, 196, 173}, {244, 154, 123}, {222, 96, 77}, {180, 4, 38}};
    const float s = t * 8.0f;
    int i = static_cast<int>(s);
    if (i > 7) i = 7;
    const float f = s - static_cast<float>(i);
    auto L = [&](int c) { return (k[i][c] + (k[i + 1][c] - k[i][c]) * f) / 255.0f; };
    return pack(L(0), L(1), L(2));
}
template <u32 (*F)(float)>
constexpr std::array<u32, 256> make_lut() {
    std::array<u32, 256> a{};
    for (int i = 0; i < 256; ++i) a[static_cast<usize>(i)] = F(static_cast<float>(i) / 255.0f);
    return a;
}
constexpr u32 gray(float t) { return pack(t, t, t); }
alignas(64) inline constexpr std::array<u32, 256> k_luts[5] = {
    make_lut<turbo>(), make_lut<viridis>(), make_lut<inferno>(), make_lut<coolwarm>(), make_lut<gray>()};
} // namespace detail

inline const char* colormap_name(Colormap m) {
    switch (m) {
        case Colormap::Turbo: return "Turbo";
        case Colormap::Viridis: return "Viridis";
        case Colormap::Inferno: return "Inferno";
        case Colormap::CoolWarm: return "Frío-cálido";
        case Colormap::Gray: return "Grises";
        default: return "?";
    }
}
CFD_INLINE const u32* colormap_lut(Colormap m) { return detail::k_luts[static_cast<int>(m)].data(); }

// t ∈ [0,1] (se sujeta).
CFD_INLINE u32 colormap(Colormap m, float t) {
    const int i = static_cast<int>(clamp_(t, 0.0f, 1.0f) * 255.0f + 0.5f);
    return colormap_lut(m)[i];
}
// Normaliza v en [lo,hi] y mapea.
CFD_INLINE u32 colormap(Colormap m, float v, float lo, float hi) { return colormap(m, (v - lo) / (hi - lo)); }

// 8 valores de una vez: normaliza, sujeta, convierte y hace gather de la LUT.
CFD_INLINE __m256i colormap8(Colormap m, simd::f8 v, float lo, float hi) {
    const simd::f8 t = simd::min(simd::max((v - simd::f8(lo)) * simd::f8(255.0f / (hi - lo)), simd::f8::zero()), simd::f8(255.0f));
    const __m256i idx = _mm256_cvtps_epi32(t.v);
    return _mm256_i32gather_epi32(reinterpret_cast<const int*>(colormap_lut(m)), idx, 4);
}

// Paleta categórica por componente aerodinámico (desglose visual de piezas).
CFD_INLINE u32 component_color(sdf::Component c) {
    constexpr u32 pal[] = {
        0xFFB0B4BCu, // Body
        0xFF3B8BEBu, // FrontWing
        0xFFE8463Cu, // RearWing
        0xFFF08A24u, // BeamWing
        0xFF2BB673u, // Floor
        0xFF14A39Au, // Diffuser
        0xFF9B6BD6u, // Sidepods
        0xFF6A9BD8u, // Nose
        0xFF3A3A3Eu, // FrontWheels
        0xFF2E2E32u, // RearWheels
        0xFF8C8C8Cu, // Suspension
        0xFFE0C040u, // Halo
        0xFFC8CCD4u, // Object
        0xFF3B8BEBu, // WingMain
        0xFFE8463Cu, // WingFlap
        0xFF7A7F88u, // Endplate
    };
    const int i = static_cast<int>(c);
    return i < static_cast<int>(sizeof(pal) / sizeof(pal[0])) ? pal[i] : 0xFFFF00FFu;
}

} // namespace cfd::render
