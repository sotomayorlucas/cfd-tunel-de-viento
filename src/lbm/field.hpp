// ============================================================================
//  lbm/field.hpp — vista de sólo lectura de los campos macroscópicos.
//
//  Contrato entre el solver (productor) y la visualización (consumidor).
//  Unidades de red: posiciones en celdas (el CENTRO de la celda (i,j,k) está
//  en (i,j,k)), velocidades en celdas/paso, densidad adimensional (ρ∞ = 1).
//  Presión de red p = ρ/3  →  Cp = (ρ - 1) / (3 · ½ u∞²) = 2(ρ-1)/(3u∞²).
//  Índice lineal: n = x + nx*(y + ny*z)   (x es el eje contiguo en memoria).
//  Celdas sólidas: el solver escribe ρ = 1 y u = velocidad de su pared (0, cinta
//  o rueda) en cada paso macro; la visualización depende de ello (gradientes, Cp).
//  Las 6 caras del dominio son siempre fronteras (entrada/salida/campo lejano).
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include <bit>

namespace cfd::lbm {

// Bits de flags por celda (u8), compartidos por solver/voxelizador/visualización.
enum CellFlag : u8 {
    kFluid      = 0,
    kSolid      = 1u << 0,   // sólido: rebote (bounce-back); solid_id[n] = grupo (1..254) o 255 = suelo
    kMoving     = 1u << 1,   // sólido con velocidad de pared ≠ 0 (ruedas girando, cinta del suelo)
    kInlet      = 1u << 2,   // frontera de equilibrio "campo lejano": ρ = 1, u = u∞ (entrada, laterales, techo)
    kOutlet     = 1u << 3,   // salida: ρ = 1, u extrapolada de la celda aguas arriba
    kNearMoving = 1u << 4,   // fluido con algún vecino kMoving → ruta escalar con corrección de pared móvil
    kSpecial    = kInlet | kOutlet | kNearMoving,   // cualquier carril con esto → ruta escalar
    kInternal5  = 1u << 5,   // RESERVADO al solver (vecino móvil = sólo la cinta del suelo)
    // ¡Usar siempre máscaras (flags & kSolid), nunca igualdades sobre flags!
};
inline constexpr u8 k_ground_id = 255;

struct FieldView {
    int nx = 0, ny = 0, nz = 0;
    const float* rho = nullptr;
    const float* ux = nullptr;
    const float* uy = nullptr;
    const float* uz = nullptr;
    const u8* flags = nullptr;
    const u8* solid_id = nullptr;
    float u_inf = 0.08f;          // velocidad de entrada en unidades de red (para Cp, normalizaciones)

    CFD_INLINE bool valid() const { return rho && ux && uy && uz && nx > 0; }
    CFD_INLINE usize index(int x, int y, int z) const {
        return static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * static_cast<usize>(z));
    }
    CFD_INLINE bool inside(Vec3 p) const {
        return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x <= nx - 1 && p.y <= ny - 1 && p.z <= nz - 1;
    }
    CFD_INLINE bool solid(int x, int y, int z) const { return flags[index(x, y, z)] & kSolid; }
    // Mayor float estrictamente menor que n-1 (x0 = n-2 como máximo: x0+1 siempre dentro).
    // Restar 1e-4 no basta: para n-1 ≥ 2048 el espaciado de float es mayor y redondea a n-1.
    static CFD_INLINE float below(int n1) { return std::bit_cast<float>(std::bit_cast<u32>(static_cast<float>(n1)) - 1u); }
    CFD_INLINE float cp_from_rho(float r) const { return 2.0f * (r - 1.0f) / (3.0f * u_inf * u_inf); }

    // Interpolación trilineal de un campo escalar en p (celdas). Fuera del dominio se sujeta al borde.
    CFD_INLINE float sample(const float* CFD_RESTRICT f, Vec3 p) const {
        const float fx = clamp_(p.x, 0.0f, below(nx - 1));
        const float fy = clamp_(p.y, 0.0f, below(ny - 1));
        const float fz = clamp_(p.z, 0.0f, below(nz - 1));
        const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), z0 = static_cast<int>(fz);
        const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
        const usize sx = 1, sy = static_cast<usize>(nx), sz = static_cast<usize>(nx) * static_cast<usize>(ny);
        const float* c = f + index(x0, y0, z0);
        const float c00 = lerp(c[0], c[sx], tx),           c10 = lerp(c[sy], c[sy + sx], tx);
        const float c01 = lerp(c[sz], c[sz + sx], tx),     c11 = lerp(c[sz + sy], c[sz + sy + sx], tx);
        return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
    }
    // Velocidad trilineal (las 3 componentes comparten pesos: 1 cálculo de índices).
    CFD_INLINE Vec3 velocity(Vec3 p) const {
        const float fx = clamp_(p.x, 0.0f, below(nx - 1));
        const float fy = clamp_(p.y, 0.0f, below(ny - 1));
        const float fz = clamp_(p.z, 0.0f, below(nz - 1));
        const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), z0 = static_cast<int>(fz);
        const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
        const usize n0 = index(x0, y0, z0);
        const usize sy = static_cast<usize>(nx), sz = static_cast<usize>(nx) * static_cast<usize>(ny);
        const usize o[8] = {n0, n0 + 1, n0 + sy, n0 + sy + 1, n0 + sz, n0 + sz + 1, n0 + sz + sy, n0 + sz + sy + 1};
        const float w[8] = {(1 - tx) * (1 - ty) * (1 - tz), tx * (1 - ty) * (1 - tz), (1 - tx) * ty * (1 - tz), tx * ty * (1 - tz),
                            (1 - tx) * (1 - ty) * tz,       tx * (1 - ty) * tz,       (1 - tx) * ty * tz,       tx * ty * tz};
        Vec3 u{0, 0, 0};
        for (int i = 0; i < 8; ++i) { u.x += w[i] * ux[o[i]]; u.y += w[i] * uy[o[i]]; u.z += w[i] * uz[o[i]]; }
        return u;
    }
    // Gradiente de velocidad por diferencias centradas en una celda interior: g[i][j] = ∂u_i/∂x_j.
    CFD_INLINE void velocity_gradient(int x, int y, int z, float g[3][3]) const {
        const int xm = x > 0 ? x - 1 : x, xp = x < nx - 1 ? x + 1 : x;
        const int ym = y > 0 ? y - 1 : y, yp = y < ny - 1 ? y + 1 : y;
        const int zm = z > 0 ? z - 1 : z, zp = z < nz - 1 ? z + 1 : z;
        const float ix = 1.0f / static_cast<float>(max_(xp - xm, 1)), iy = 1.0f / static_cast<float>(max_(yp - ym, 1)), iz = 1.0f / static_cast<float>(max_(zp - zm, 1));
        const float* U[3] = {ux, uy, uz};
        for (int i = 0; i < 3; ++i) {
            g[i][0] = (U[i][index(xp, y, z)] - U[i][index(xm, y, z)]) * ix;
            g[i][1] = (U[i][index(x, yp, z)] - U[i][index(x, ym, z)]) * iy;
            g[i][2] = (U[i][index(x, y, zp)] - U[i][index(x, y, zm)]) * iz;
        }
    }
    CFD_INLINE Vec3 vorticity(int x, int y, int z) const {
        float g[3][3];
        velocity_gradient(x, y, z, g);
        return {g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]};
    }
    // Criterio Q = ½(|Ω|² - |S|²): Q > 0 donde domina la rotación (núcleos de vórtice).
    CFD_INLINE float q_criterion(int x, int y, int z) const {
        float g[3][3];
        velocity_gradient(x, y, z, g);
        float s2 = 0, o2 = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const float s = 0.5f * (g[i][j] + g[j][i]), o = 0.5f * (g[i][j] - g[j][i]);
                s2 += s * s; o2 += o * o;
            }
        return 0.5f * (o2 - s2);
    }
};

} // namespace cfd::lbm
