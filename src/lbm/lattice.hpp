// ============================================================================
//  lbm/lattice.hpp — conjunto de velocidades D3Q19 en tiempo de compilación.
//
//  Orden FluidX3D: dirección 0 = reposo y luego PARES OPUESTOS (1,2), (3,4), …
//  (17,18). El miembro impar i de cada par tiene c_x ≥ 0 y es el que usa el
//  streaming "Esoteric-Pull" para direccionar al vecino j_i = n + c_i.
//
//  Todas las identidades de momentos de la red se verifican con static_assert
//  sobre pesos ENTEROS (w·36) → comprobación exacta, sin tolerancias.
// ============================================================================
#pragma once

#include "../core/config.hpp"

namespace cfd::lbm::d3q19 {

inline constexpr int Q = 19;

// Velocidades discretas c_i (celdas/paso).
inline constexpr int c[Q][3] = {
    { 0, 0, 0},
    { 1, 0, 0}, {-1, 0, 0},
    { 0, 1, 0}, { 0,-1, 0},
    { 0, 0, 1}, { 0, 0,-1},
    { 1, 1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1,-1},
    { 1,-1, 0}, {-1, 1, 0},
    { 1, 0,-1}, {-1, 0, 1},
    { 0, 1,-1}, { 0,-1, 1},
};

// Pesos ×36 (enteros): 12 reposo, 2 ejes, 1 diagonales.
inline constexpr int w36[Q] = {12, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
inline constexpr float w[Q] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
};
inline constexpr double wd[Q] = {
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
};
// Dirección opuesta: con pares consecutivos, opp(i) = i ^ 1 trasladado (i impar → i+1).
inline constexpr int opp[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

inline constexpr float cs2 = 1.0f / 3.0f;   // velocidad del sonido al cuadrado

// Desplazamiento lineal del vecino n + c_i en una red nx×ny×nz (índice x + nx(y + ny z)).
CFD_INLINE constexpr i64 offset(int i, i64 nx, i64 nxny) {
    return static_cast<i64>(c[i][0]) + nx * static_cast<i64>(c[i][1]) + nxny * static_cast<i64>(c[i][2]);
}

// ---- Verificación de identidades en tiempo de compilación ---------------------------------
namespace detail {
constexpr int sum_w() { int s = 0; for (int i = 0; i < Q; ++i) s += w36[i]; return s; }
constexpr int sum_wc(int a) { int s = 0; for (int i = 0; i < Q; ++i) s += w36[i] * c[i][a]; return s; }
constexpr int sum_wcc(int a, int b) { int s = 0; for (int i = 0; i < Q; ++i) s += w36[i] * c[i][a] * c[i][b]; return s; }
constexpr int sum_wccc(int a, int b, int d) { int s = 0; for (int i = 0; i < Q; ++i) s += w36[i] * c[i][a] * c[i][b] * c[i][d]; return s; }
constexpr int sum_wcccc(int a, int b, int d, int e) {
    int s = 0; for (int i = 0; i < Q; ++i) s += w36[i] * c[i][a] * c[i][b] * c[i][d] * c[i][e]; return s;
}
constexpr bool check_opp() {
    for (int i = 0; i < Q; ++i) {
        const int o = opp[i];
        if (opp[o] != i) return false;
        for (int a = 0; a < 3; ++a) if (c[o][a] != -c[i][a]) return false;
        if (w36[o] != w36[i]) return false;
        if (i > 0 && (i & 1) && o != i + 1) return false;   // pares (i, i+1)
        if (i > 0 && (i & 1) && c[i][0] < 0) return false;    // el impar nunca apunta a -x
    }
    return true;
}
constexpr bool check_isotropy() {
    // Σ w c_a c_b = cs² δ_ab  → ×36: 12 δ_ab
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (sum_wcc(a, b) != (a == b ? 12 : 0)) return false;
    // Σ w c_a c_b c_c = 0
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int d = 0; d < 3; ++d)
                if (sum_wccc(a, b, d) != 0) return false;
    // Σ w c_a c_b c_c c_d = cs⁴ (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc)  → ×36: 4·(...)
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int d = 0; d < 3; ++d)
                for (int e = 0; e < 3; ++e) {
                    const int iso = (a == b && d == e) + (a == d && b == e) + (a == e && b == d);
                    if (sum_wcccc(a, b, d, e) != 4 * iso) return false;
                }
    return true;
}
} // namespace detail

static_assert(detail::sum_w() == 36, "D3Q19: Σ w_i = 1");
static_assert(detail::sum_wc(0) == 0 && detail::sum_wc(1) == 0 && detail::sum_wc(2) == 0, "D3Q19: Σ w_i c_i = 0");
static_assert(detail::check_isotropy(), "D3Q19: isotropía de 2º, 3er y 4º orden (Σ w c c = δ/3, Σ w c⁴ = cs⁴·(δδ+δδ+δδ))");
static_assert(detail::check_opp(), "D3Q19: tabla de opuestos / orden por pares");

} // namespace cfd::lbm::d3q19
