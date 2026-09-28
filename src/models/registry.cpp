// ============================================================================
//  models/registry.cpp — catálogo global de modelos, resolución de parámetros
//  y marcos de carrocería/ruedas de los coches.
// ============================================================================
#include "registry.hpp"

#include <cmath>

namespace cfd::models {

namespace detail {
namespace {
// Catálogo inmutable construido una sola vez (inicialización estática local = hilo-segura).
const std::vector<Entry>& entries() {
    static const std::vector<Entry> v = [] {
        std::vector<Entry> e;
        e.reserve(32);
        register_f1(e);        // F1 en orden cronológico
        register_objects(e);   // cuerpos y alas de referencia
        return e;
    }();
    return v;
}
} // namespace
} // namespace detail

int count() { return static_cast<int>(detail::entries().size()); }

const Info& info(int index) {
    CFD_CHECK(index >= 0 && index < count(), "models::info: índice fuera de rango");
    return detail::entries()[static_cast<usize>(index)].info;
}

int find(const std::string& id) {
    const auto& e = detail::entries();
    for (usize i = 0; i < e.size(); ++i)
        if (e[i].info.id == id) return static_cast<int>(i);
    return -1;
}

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::F1Car: return "Coche F1";
        case Kind::Wing:  return "Ala";
        case Kind::Body:  return "Cuerpo";
    }
    return "?";
}

Params resolve_params(int index, const Params& in) {
    const Info& I = info(index);
    const u32 m = I.param_mask;
    Params p = in;
    // No finitos (NaN/inf, p.ej. desde un campo de texto) → defecto: clamp_ deja pasar NaN y
    // aguas abajo acabaría en conversiones float→int indefinidas (voxelizador, LatticeMap).
    auto fin = [](float v, float def) { return std::isfinite(v) ? v : def; };
    p.yaw_deg = fin(p.yaw_deg, 0.0f);
    p.ride_front_mm = fin(p.ride_front_mm, -1.0f);
    p.ride_rear_mm = fin(p.ride_rear_mm, -1.0f);
    p.front_flap_deg = fin(p.front_flap_deg, 0.0f);
    p.rear_flap_deg = fin(p.rear_flap_deg, 0.0f);
    p.aoa_deg = fin(p.aoa_deg, -1000.0f);
    p.height_mm = fin(p.height_mm, -1.0f);
    p.flap_gap_mm = fin(p.flap_gap_mm, -1.0f);
    // Alturas de marcha (mm): < 0 → defecto; rango físico razonable.
    if (!(m & P_RideHeight) || p.ride_front_mm < 0.0f) p.ride_front_mm = I.default_ride_front_mm;
    if (!(m & P_RideHeight) || p.ride_rear_mm < 0.0f) p.ride_rear_mm = I.default_ride_rear_mm;
    p.ride_front_mm = clamp_(p.ride_front_mm, 0.0f, 400.0f);
    p.ride_rear_mm = clamp_(p.ride_rear_mm, 0.0f, 400.0f);
    p.yaw_deg = (m & P_Yaw) ? clamp_(p.yaw_deg, -30.0f, 30.0f) : 0.0f;
    p.front_flap_deg = (m & P_FrontFlap) ? clamp_(p.front_flap_deg, -20.0f, 20.0f) : 0.0f;
    p.rear_flap_deg = (m & P_RearFlap) ? clamp_(p.rear_flap_deg, -20.0f, 20.0f) : 0.0f;
    p.drs_open = (m & P_Drs) ? p.drs_open : false;
    p.wheels_rotating = (m & P_Wheels) ? p.wheels_rotating : false;
    if (!(m & P_Aoa) || p.aoa_deg < -900.0f) p.aoa_deg = I.default_aoa_deg;
    p.aoa_deg = clamp_(p.aoa_deg, -30.0f, 30.0f);
    if (!(m & P_Height) || p.height_mm < 0.0f) p.height_mm = I.default_height_mm;
    p.height_mm = clamp_(p.height_mm, 2.0f, 5000.0f);
    if (!(m & P_FlapGap) || p.flap_gap_mm < 0.0f) p.flap_gap_mm = I.default_flap_gap_mm;
    p.flap_gap_mm = clamp_(p.flap_gap_mm, 5.0f, 150.0f);
    return p;
}

bool same_geometry(int index, const Params& a, const Params& b) {
    const Params x = resolve_params(index, a), y = resolve_params(index, b);
    // wheels_rotating sólo cambia el movimiento de pared, no la forma.
    return x.yaw_deg == y.yaw_deg && x.ride_front_mm == y.ride_front_mm && x.ride_rear_mm == y.ride_rear_mm &&
           x.front_flap_deg == y.front_flap_deg && x.rear_flap_deg == y.rear_flap_deg && x.drs_open == y.drs_open &&
           x.aoa_deg == y.aoa_deg && x.height_mm == y.height_mm && x.flap_gap_mm == y.flap_gap_mm;
}

Built build(int index, const Params& params) {
    CFD_CHECK(index >= 0 && index < count(), "models::build: índice fuera de rango");
    const auto& e = detail::entries()[static_cast<usize>(index)];
    Built b;
    b.info = e.info;
    b.params = resolve_params(index, params);
    b.scene.pool.reserve(8192);
    e.build(b);                          // geometría + marcos + ejes/referencia
    // bounds_m = AABB de la escena, salvo que el constructor la haya fijado (spans_domain:
    // extensión Y = ancho exacto del dominio, ver build_airfoil2d).
    if (b.bounds_m.empty()) b.bounds_m = b.scene.bounds();
    return b;
}

// ---- Marcos de coche ----------------------------------------------------------------------
// Guiñada alrededor del eje vertical que pasa por el centro entre ejes (x = batalla/2, y = 0).
static Xform yaw_about(float xc, float yaw) {
    const Mat3 R = Mat3::rot_z(yaw);
    const Vec3 c(xc, 0.0f, 0.0f);
    return {R, c - R * c};
}

Xform car_body_frame(float ride_front_m, float ride_rear_m, float wheelbase_m, float yaw_rad) {
    // Cabeceo: el plano de referencia pasa por (0, ride_front) y (batalla, ride_rear).
    // asin (no atan) → el punto del eje trasero queda EXACTAMENTE a ride_rear.
    const float s = clamp_((ride_rear_m - ride_front_m) / max_(wheelbase_m, 1e-3f), -0.5f, 0.5f);
    const float th = std::asin(s);
    const Xform pitch{Mat3::rot_y(-th), Vec3(0.0f, 0.0f, ride_front_m)};   // rot_y(-θ): la cola sube
    return yaw_about(0.5f * wheelbase_m, yaw_rad) * pitch;
}

Xform car_wheel_frame(float wheelbase_m, float yaw_rad) { return yaw_about(0.5f * wheelbase_m, yaw_rad); }

} // namespace cfd::models
