// ============================================================================
//  core/util.hpp — temporizadores, RNG, FP16 escalar, trucos de bits.
// ============================================================================
#pragma once

#include "config.hpp"
#include <bit>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <x86intrin.h>

namespace cfd {

// ---- Tiempo ---------------------------------------------------------------------
// clock_gettime(CLOCK_MONOTONIC) va por vDSO: sin syscall real (~20 ns).
CFD_INLINE double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
CFD_INLINE u64 cycles() { return __rdtsc(); }   // contador de ciclos (TSC invariante)

struct Stopwatch {
    double t0 = now_sec();
    void reset() { t0 = now_sec(); }
    double elapsed() const { return now_sec() - t0; }
    double lap() { const double t = now_sec(), d = t - t0; t0 = t; return d; }
};

// Media móvil exponencial para mostrar tiempos/FPS estables.
struct Ema {
    double v = 0; bool init = false;
    double push(double x, double a = 0.1) { v = init ? v + a * (x - v) : x; init = true; return v; }
};

// ---- Aleatorios ------------------------------------------------------------------
CFD_INLINE constexpr u64 splitmix64(u64& s) {
    u64 z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// wyrand: 1 multiplicación 64x64→128. Rapidísimo y de buena calidad.
struct WyRand {
    u64 s;
    explicit constexpr WyRand(u64 seed = 0x2545F4914F6CDD1Dull) : s(seed) {}
    CFD_INLINE u64 next() {
        s += 0xA0761D6478BD642Full;
        const __uint128_t t = static_cast<__uint128_t>(s) * (s ^ 0xE7037ED1A0B428DBull);
        return static_cast<u64>(t >> 64) ^ static_cast<u64>(t);
    }
    // float en [0,1) armando la mantisa directamente (sin división).
    CFD_INLINE float uniform() { return std::bit_cast<float>(0x3F800000u | static_cast<u32>(next() >> 41)) - 1.0f; }
    CFD_INLINE float uniform(float a, float b) { return a + (b - a) * uniform(); }
    // Entero en [0, n) sin módulo (truco de Lemire: multiplicar y tomar la parte alta).
    CFD_INLINE u32 below(u32 n) { return static_cast<u32>((static_cast<u64>(static_cast<u32>(next())) * n) >> 32); }
};

// ---- FP16 escalar (F16C) ----------------------------------------------------------
CFD_INLINE u16 f32_to_f16(float f) { return static_cast<u16>(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT)); }
CFD_INLINE float f16_to_f32(u16 h) { return _cvtsh_ss(h); }

// ---- Bits ------------------------------------------------------------------------
CFD_INLINE constexpr u32 next_pow2(u32 v) { return v <= 1 ? 1u : 1u << (32 - std::countl_zero(v - 1)); }
CFD_INLINE constexpr bool is_pow2(u64 v) { return v && !(v & (v - 1)); }
CFD_INLINE constexpr u64 round_up(u64 v, u64 m) { return (v + m - 1) / m * m; }
// Itera los bits activos de una máscara: for_each_bit(m, [](int i){...}); usa ctz + borrar el bit más bajo.
template <class F>
CFD_INLINE void for_each_bit(u64 m, F&& f) { while (m) { f(std::countr_zero(m)); m &= m - 1; } }
// Entrelazado de bits (orden Morton / curva Z) de 10 bits por eje, con PDEP de BMI2.
CFD_INLINE u32 morton3(u32 x, u32 y, u32 z) {
    return _pdep_u32(x, 0x09249249u) | _pdep_u32(y, 0x12492492u) | _pdep_u32(z, 0x24924924u);
}

// Hash FNV-1a en tiempo de compilación (para ids/strings constantes).
CFD_INLINE constexpr u64 fnv1a(const char* s) {
    u64 h = 0xcbf29ce484222325ull;
    while (*s) { h ^= static_cast<unsigned char>(*s++); h *= 0x100000001b3ull; }
    return h;
}

} // namespace cfd
