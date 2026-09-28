// ============================================================================
//  geom/lattice_map.hpp — correspondencia espacio modelo (m) ↔ red LBM (celdas).
//
//  p_celdas = (p_modelo - origin) / dx      (el centro de la celda i está en i)
//  El suelo del modelo (z = 0 m) cae en z_celdas = 0.5 cuando hay suelo:
//  la capa z = 0 son celdas sólidas y la pared a mitad de camino (bounce-back
//  "halfway") queda en z = 0.5.
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "sdf.hpp"

namespace cfd {

struct LatticeMap {
    float dx = 0.04f;          // metros por celda
    Vec3 origin{0, 0, 0};      // posición (m, espacio modelo) del centro de la celda (0,0,0)

    CFD_INLINE Vec3 to_cells(Vec3 pm) const { return (pm - origin) / dx; }
    CFD_INLINE Vec3 to_model(Vec3 pc) const { return pc * dx + origin; }

    // Movimiento rígido normalizado (sdf::RigidMotion) → unidades de red:
    //   u_red(x_celdas) = v + ω × (x_celdas - c)      con v, ω en celdas/paso y 1/paso.
    struct LatticeMotion { Vec3 v{0, 0, 0}, omega{0, 0, 0}, center{0, 0, 0}; };
    CFD_INLINE LatticeMotion lattice_motion(const sdf::RigidMotion& m, float u_inf_lattice) const {
        // ω̂ [1/m] × (x_m - c_m) = ω̂·dx × (x_c - c_c)   →   ω_red = u∞ · ω̂ · dx
        return {m.v_hat * u_inf_lattice, m.omega_hat * (u_inf_lattice * dx), to_cells(m.center)};
    }
};

} // namespace cfd
