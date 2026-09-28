// ============================================================================
//  core/config.hpp — macros de compilador y "trucos oscuros" de optimización.
//
//  Todo el proyecto incluye este archivo primero. Aquí viven los atributos
//  GCC/Clang que en programación competitiva se usan para exprimir el
//  compilador: forzar inlining, marcar rutas calientes/frías, pistas de
//  ramificación, alias estricto (__restrict__), suposiciones ([[assume]]),
//  prefetch explícito y alineación a línea de caché.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(__GNUC__)
#error "Este proyecto usa extensiones GCC/Clang (atributos, intrínsecos, builtins)."
#endif

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "Compilar con -march=native (se requiere AVX2 + FMA + F16C)."
#endif

// --- Inlining ---------------------------------------------------------------
#define CFD_INLINE      inline __attribute__((always_inline))
#define CFD_NOINLINE    __attribute__((noinline))
#define CFD_FLATTEN     __attribute__((flatten))          // inlinea todo el árbol de llamadas
#define CFD_HOT         __attribute__((hot))              // agrupa en .text.hot, optimiza agresivo
#define CFD_COLD        __attribute__((cold, noinline))   // fuera de la ruta caliente
#define CFD_PURE        __attribute__((pure))
#define CFD_CONST_FN    __attribute__((const))
#define CFD_RESTRICT    __restrict__
#define CFD_UNUSED      [[maybe_unused]]

// --- Pistas de ramificación --------------------------------------------------
#define CFD_LIKELY(x)   __builtin_expect(!!(x), 1)
#define CFD_UNLIKELY(x) __builtin_expect(!!(x), 0)

// C++23 [[assume(expr)]]: el optimizador puede asumir la condición (UB si es falsa).
#if defined(__has_cpp_attribute) && __has_cpp_attribute(assume)
#define CFD_ASSUME(x) [[assume(x)]]
#else
#define CFD_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#endif

#define CFD_UNREACHABLE() __builtin_unreachable()

// --- Memoria -----------------------------------------------------------------
inline constexpr std::size_t k_cache_line = 64;
#define CFD_CACHE_ALIGNED alignas(64)
#define CFD_ASSUME_ALIGNED(p, a) static_cast<decltype(p)>(__builtin_assume_aligned((p), (a)))
// rw: 0 = lectura, 1 = escritura; loc: 0 (no temporal) .. 3 (máxima localidad)
#define CFD_PREFETCH_R(p)  __builtin_prefetch((p), 0, 3)
#define CFD_PREFETCH_W(p)  __builtin_prefetch((p), 1, 3)
#define CFD_PREFETCH_NT(p) __builtin_prefetch((p), 0, 0)

// --- Diagnóstico -------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
[[noreturn]] CFD_COLD inline void cfd_fatal(const char* file, int line, const char* msg) {
    std::fprintf(stderr, "[FATAL] %s:%d: %s\n", file, line, msg);
    std::fflush(stderr);
    std::abort();
}
#define CFD_CHECK(cond, msg) do { if (CFD_UNLIKELY(!(cond))) cfd_fatal(__FILE__, __LINE__, (msg)); } while (0)

#ifndef NDEBUG
#define CFD_DASSERT(cond) CFD_CHECK((cond), "assert: " #cond)
#else
#define CFD_DASSERT(cond) ((void)0)
#endif

namespace cfd {
using i8 = std::int8_t;   using u8 = std::uint8_t;
using i16 = std::int16_t; using u16 = std::uint16_t;
using i32 = std::int32_t; using u32 = std::uint32_t;
using i64 = std::int64_t; using u64 = std::uint64_t;
using usize = std::size_t;
} // namespace cfd
