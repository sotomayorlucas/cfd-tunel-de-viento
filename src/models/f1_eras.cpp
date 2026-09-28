// ============================================================================
//  models/f1_eras.cpp — coches de F1 por época reglamentaria.
//
//  Cada coche se construye en su marco de CARROCERÍA (x = 0 eje delantero,
//  z = 0 plano de referencia / plank) salvo las ruedas y los faldones de 1979
//  (marco de ruedas: se quedan en el suelo). Diseñados para sobrevivir a una red
//  de dx = 3 cm: espesores ≥ 3.5 cm, perfiles con espesor mínimo de 4.2 cm en la
//  parte trasera, ranuras entre elementos de 4.5 cm y 2-3 elementos por alerón.
//  Alerones: perfiles NACA invertidos (carga) con ángulos NEGATIVOS (BS arriba).
// ============================================================================
#include "f1_common.hpp"
#include "registry.hpp"


namespace cfd::models::detail {
namespace {

constexpr u32 k_car_mask = P_RideHeight | P_FrontFlap | P_RearFlap | P_Yaw | P_Wheels;
constexpr float k_drs_open = -6.0f * k_deg2rad;     // flap casi horizontal con el DRS abierto

CFD_INLINE float deg(float d) { return d * k_deg2rad; }

// Espejos retrovisores sobre soporte (grupo abierto).
void mirrors(Scene& s, Vec3 c, Vec3 base) {
    s.ellipsoid(c, Vec3(0.05f, 0.085f, 0.045f), {}, Op::Union, 0.0f, true);
    capsule(s, base, c, 0.018f, true);
}

// Lomo de la cubierta motor: de la toma de aire (x0, techo z0) al final de la cubierta (x1, techo z1), estrechándose.
// Sin él la toma de aire (un elipsoide) acababa en un escalón de ~30 cm sobre la cubierta y su estela (u ≈ 0-0.4 U
// hasta z = 1.2 m, medido a Media) dejaba el alerón trasero sin presión dinámica (SCz 0.1 m²).
void engine_spine(Scene& s, float x0, float z0, float x1, float z1, float hw0, float hw1) {
    tbox(s, x0, x1, hw0, z0 - 0.40f, z0, hw1, z1 - 0.22f, z1, 0.06f, 0.0f, false, Op::SmoothUnion, 0.10f);
}

// Piloto: casco + hombros/torso (capsula hasta el fondo del habitáculo: nada flota).
void driver(Scene& s, float x, float z, float r) {
    s.sphere(Vec3(x, 0.0f, z), r);
    capsule(s, Vec3(x + 0.04f, 0.0f, z - 0.05f), Vec3(x + 0.10f, 0.0f, z - 0.35f), 0.11f);
}

// Habitáculo + casco (grupo abierto): hueco elipsoidal y casco esférico.
void cockpit(Scene& s, float xc, float ztop, float helmet_x, float helmet_z) {
    s.ellipsoid(Vec3(xc, 0.0f, ztop), Vec3(0.42f, 0.20f, 0.16f), {}, Op::SmoothSubtract, 0.03f);
    driver(s, helmet_x, helmet_z, 0.13f);
}

Info car_info(const char* id, const char* name, const char* era, const char* desc, u32 mask, float len, float area, float wb,
              float ride_f, float ride_r, float cla, float cda, float reg_w, float fflap, float rflap, const char* src) {
    Info I;
    I.id = id; I.name = name; I.era = era; I.description = desc;
    I.kind = Kind::F1Car;
    I.param_mask = mask;
    I.ref_length_m = len; I.ref_area_m2 = area; I.wheelbase_m = wb;
    I.default_ride_front_mm = ride_f; I.default_ride_rear_mm = ride_r;
    I.default_speed_kmh = 250.0f;
    I.needs_ground = true;
    I.ref_ClA = cla; I.ref_CdA = cda;
    I.reg_width_m = reg_w;
    I.default_front_flap_deg = fflap; I.default_rear_flap_deg = rflap;
    I.ref_source = src;
    return I;
}

// =====================================================================================
//  1967 — Lotus 49: puro sin alerones
// =====================================================================================
void build_1967(Built& b) {
    constexpr float wb = 2.413f;
    CarCtx c(b.scene, b.params, wb);
    Scene& s = b.scene;
    const WheelSpec wf{0.0f, 0.762f, 0.29f, 0.20f, 0.19f, 0.05f, 0.075f};
    const WheelSpec wr{wb, 0.775f, 0.315f, 0.28f, 0.19f, 0.06f, 0.10f};

    s.begin_group("Carrocería (puro)", Component::Body);
    s.round_cone(Vec3(-0.93f, 0.0f, 0.21f), Vec3(-0.15f, 0.0f, 0.25f), 0.10f, 0.23f);                  // morro con radiador
    tbox(s, -0.35f, 1.30f, 0.23f, 0.0f, 0.47f, 0.28f, 0.0f, 0.56f, 0.12f, 0.0f, false, Op::SmoothUnion, 0.12f);   // bañera
    tbox(s, 1.20f, 1.55f, 0.27f, 0.0f, 0.52f, 0.25f, 0.02f, 0.40f, 0.10f, 0.0f, false, Op::SmoothUnion, 0.08f);   // depósito
    s.ellipsoid(Vec3(0.78f, 0.0f, 0.58f), Vec3(0.42f, 0.20f, 0.14f), {}, Op::SmoothSubtract, 0.03f);
    driver(s, 0.88f, 0.60f, 0.125f);                                                                   // piloto erguido
    capsule(s, Vec3(1.12f, 0.0f, 0.45f), Vec3(1.12f, 0.0f, 0.78f), 0.022f);                            // arco antivuelco
    mirrors(s, Vec3(0.42f, 0.31f, 0.60f), Vec3(0.46f, 0.22f, 0.50f));
    s.end_group();

    s.begin_group("Motor DFV y caja", Component::Body);
    rbox(s, Vec3(1.25f, -0.23f, 0.03f), Vec3(2.10f, 0.23f, 0.40f), 0.05f);                            // V8 portante expuesto
    tbox(s, 2.05f, 2.90f, 0.17f, 0.08f, 0.36f, 0.12f, 0.12f, 0.30f, 0.04f);                              // caja de cambios
    capsule(s, Vec3(1.95f, 0.16f, 0.38f), Vec3(2.70f, 0.20f, 0.48f), 0.03f, true);                     // escapes
    s.end_group();

    s.begin_group("Suspensión", Component::Suspension, Frame::Body);
    add_suspension_arms(c, wf, 0.18f, 0.38f, 0.10f, 0.02f, false);
    add_suspension_arms(c, wr, 0.10f, 0.30f, 0.14f, 0.02f, false);
    s.end_group();

    add_axle(c, "Ruedas delanteras", Component::FrontWheels, wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, wr);
    finish_car(b, wb);
}

// =====================================================================================
//  1979 — Lotus 79: coche-ala con faldones deslizantes
// =====================================================================================
void build_1979(Built& b) {
    constexpr float wb = 2.72f;
    CarCtx c(b.scene, b.params, wb);
    Scene& s = b.scene;
    const WheelSpec wf{0.0f, 0.85f, 0.29f, 0.28f, 0.20f, 0.05f, 0.08f};
    const WheelSpec wr{wb, 0.815f, 0.33f, 0.44f, 0.22f, 0.06f, 0.10f};

    s.begin_group("Monocasco y cubierta", Component::Body);
    tbox(s, 0.0f, 1.35f, 0.18f, 0.0f, 0.46f, 0.19f, 0.0f, 0.55f, 0.06f);
    tbox(s, 1.25f, 2.62f, 0.24f, 0.02f, 0.60f, 0.14f, 0.06f, 0.40f, 0.08f, 0.0f, false, Op::SmoothUnion, 0.08f);
    tbox(s, 2.55f, 3.25f, 0.15f, 0.10f, 0.36f, 0.10f, 0.14f, 0.30f, 0.04f, 0.0f, false, Op::SmoothUnion, 0.04f);
    s.ellipsoid(Vec3(0.85f, 0.0f, 0.56f), Vec3(0.40f, 0.16f, 0.13f), {}, Op::SmoothSubtract, 0.03f);
    driver(s, 0.95f, 0.60f, 0.125f);
    capsule(s, Vec3(1.18f, 0.0f, 0.50f), Vec3(1.18f, 0.0f, 0.76f), 0.022f);
    mirrors(s, Vec3(0.55f, 0.27f, 0.56f), Vec3(0.60f, 0.16f, 0.46f));
    s.end_group();

    s.begin_group("Morro", Component::Nose);
    tbox(s, -1.00f, 0.05f, 0.09f, 0.05f, 0.22f, 0.18f, 0.0f, 0.42f, 0.04f);
    s.end_group();

    // Pontones con perfil de ala invertida: el intradós forma con el suelo un Venturi.
    s.begin_group("Pontones (ala invertida)", Component::Sidepods);
    {
        const Vec2 pod[] = {{0.35f, 0.24f}, {0.55f, 0.13f}, {0.80f, 0.08f}, {1.10f, 0.065f}, {1.60f, 0.075f}, {2.00f, 0.16f},
                            {2.36f, 0.30f}, {2.36f, 0.40f}, {1.80f, 0.44f}, {0.80f, 0.44f}, {0.42f, 0.36f}};
        plate_xz(s, pod, 0.445f, 0.275f);
    }
    s.end_group();
    // Faldones deslizantes: siguen al suelo (marco de ruedas) y sellan el túnel por fuera.
    s.begin_group("Faldones deslizantes", Component::Sidepods, Frame::Wheels);
    rbox(s, Vec3(0.42f, 0.700f, 0.003f), Vec3(2.32f, 0.735f, 0.40f), 0.01f, true);
    s.end_group();

    s.begin_group("Alerones delanteros", Component::FrontWing);
    {
        ElemPose e[2];
        e[0] = {-0.98f, 0.12f, deg(-4.0f), 0.30f, c.eprof(0.06f, 0.40f, 0.13f, 0.30f)};
        e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, 0.22f), 0.22f, c.flap_front(30.0f), k_min_gap, 0.06f);   // calibrado (real ~16°): eje delantero en sustentación si no
        WingPlan w; w.y0 = 0.06f; w.span = 0.56f;
        add_elements(s, e, w);
        EndplateSpec ep; ep.z_min = 0.05f;
        add_endplates(c, e, w, ep);
    }
    s.end_group();

    s.begin_group("Alerón trasero", Component::RearWing);
    {
        ElemPose e[2];
        // Calibrado (real ~−6° / 22° → −2° / 10°): sobre la caja de cambios y fuera de la estela, este alerón SÍ funciona en el
        // solver (a diferencia de los modernos) y con los ángulos reales daba más carga que el de 1998.
        e[0] = {2.95f, 0.80f, deg(-2.0f), 0.40f, c.eprof(0.07f, 0.40f, 0.13f, 0.40f)};
        e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, 0.24f), 0.24f, c.flap_rear(10.0f), k_min_gap, 0.06f);
        WingPlan w; w.span = 0.52f;
        add_elements(s, e, w);
        EndplateSpec ep; ep.z_down_to = 0.58f; ep.x_pad_rear = 0.06f;
        add_endplates(c, e, w, ep);
        rbox(s, Vec3(3.05f, -0.02f, 0.28f), Vec3(3.25f, 0.02f, 0.86f), 0.015f);   // pilar central
    }
    s.end_group();

    s.begin_group("Suspensión", Component::Suspension, Frame::Body);
    add_suspension_arms(c, wf, 0.12f, 0.32f, 0.08f, 0.02f, false);
    add_suspension_arms(c, wr, 0.10f, 0.30f, 0.14f, 0.02f, false);
    s.end_group();

    add_axle(c, "Ruedas delanteras", Component::FrontWheels, wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, wr);
    finish_car(b, wb);
}

// =====================================================================================
//  Coches de fondo plano escalonado (1998-2021): piezas comunes
// =====================================================================================
struct FlatCar {
    float wb;
    WheelSpec wf, wr;
    // chasis
    float tub_front_zlo, tub_front_zhi;      // sección del monocasco en x = -0.45
    float cover_x1, cover_zlo1, cover_zhi1;  // final de la cubierta motor
    float crash_x0, crash_x1;                // caja de cambios / estructura antichoque
    float airbox_z;
    // morro
    float nose_x0, nose_zlo0, nose_zhi0, nose_hy0, nose_zlo1, nose_zhi1;
    // pontones
    float pod_x0, pod_x1, pod_hy0, pod_zlo0, pod_zhi0, pod_hy1, pod_zhi1;
    FlatFloorSpec floor;
};

void flat_body(CarCtx& c, const FlatCar& f, bool fin, float fin_x1, float fin_z1) {
    Scene& s = c.s;
    s.begin_group("Chasis y cubierta motor", Component::Body);
    tbox(s, -0.45f, 0.95f, 0.15f, f.tub_front_zlo, f.tub_front_zhi, 0.30f, 0.05f, 0.66f, 0.07f);
    tbox(s, 0.90f, f.cover_x1, 0.30f, 0.05f, 0.70f, 0.12f, f.cover_zlo1, f.cover_zhi1, 0.08f, 0.0f, false, Op::SmoothUnion, 0.10f);
    s.ellipsoid(Vec3(1.15f, 0.0f, f.airbox_z), Vec3(0.36f, 0.12f, 0.14f), {}, Op::SmoothUnion, 0.10f);
    tbox(s, f.crash_x0, f.crash_x1, 0.12f, 0.14f, 0.42f, 0.06f, 0.22f, 0.37f, 0.04f, 0.0f, false, Op::SmoothUnion, 0.05f);
    cockpit(s, 0.52f, 0.72f, 0.72f, 0.64f);
    mirrors(s, Vec3(f.pod_x0 - 0.02f, 0.46f, f.pod_zhi0 + 0.10f), Vec3(f.pod_x0 + 0.10f, 0.40f, f.pod_zhi0 - 0.04f));
    if (fin) {   // aleta de tiburón
        const Vec2 p[] = {{1.35f, 0.70f}, {1.50f, f.airbox_z + 0.08f}, {fin_x1, fin_z1}, {fin_x1 + 0.15f, fin_z1 - 0.25f}, {fin_x1 - 0.5f, 0.42f}};
        plate_xz(s, p, 0.0f, 0.0175f, false);
    }
    s.end_group();

    s.begin_group("Morro", Component::Nose);
    tbox(s, f.nose_x0, -0.40f, f.nose_hy0, f.nose_zlo0, f.nose_zhi0, 0.15f, f.nose_zlo1, f.nose_zhi1, 0.05f);
    s.end_group();

    s.begin_group("Pontones", Component::Sidepods);
    tbox(s, f.pod_x0, f.pod_x1, f.pod_hy0, f.pod_zlo0, f.pod_zhi0, f.pod_hy1, f.pod_zlo0, f.pod_zhi1, 0.10f);
    s.end_group();

    add_flat_floor(c, f.floor);
}

// Alerón delantero: plano principal continuo (y0 = 0) + flaps (continuos o sólo exteriores
// desde y_flap con valla interior) + endplates; pilones al morro alto si z_pylon_top > 0.
// Calibración (fase 2): flaps de 1998-2014 6-16° más cerrados que los reales (1998 18 → 24°, 2008 16 → 32°,
// 2011/2014 14/26 → 20/32°): con los reales el eje delantero de esos coches quedaba en sustentación (balance
// < 20 % o negativo a Media) y su carga total por debajo de la de 1998.
void flat_front_wing(CarCtx& c, float x_le, float z_le, float c_main, float c_f1, float a_f1, float c_f2, float a_f2,
                     float half_span, float y_flap, float pyl_x0, float pyl_x1, float pyl_ztop, float ep_top) {
    Scene& s = c.s;
    s.begin_group("Alerón delantero", Component::FrontWing);
    ElemPose e[3];
    int n = 1;
    e[0] = {x_le, z_le, deg(-3.0f), c_main, c.eprof(0.06f, 0.40f, 0.13f, c_main)};
    e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, c_f1), c_f1, c.flap_front(a_f1), k_min_gap, 0.06f);
    ++n;
    if (c_f2 > 0.0f) { e[2] = place_next(s, e[1], c.eprof(0.05f, 0.45f, 0.17f, c_f2), c_f2, c.flap_front(a_f2), k_min_gap, 0.06f); ++n; }
    WingPlan wm; wm.span = half_span;
    add_elements(s, std::span<const ElemPose>(e, 1), wm);
    WingPlan wf; wf.y0 = y_flap; wf.span = half_span - y_flap;
    add_elements(s, std::span<const ElemPose>(e + 1, static_cast<usize>(n - 1)), wf);
    if (y_flap > 0.0f) add_fence(c, std::span<const ElemPose>(e, static_cast<usize>(n)), y_flap);
    EndplateSpec ep; ep.z_min = 0.04f; ep.z_up_to = ep_top;
    add_endplates(c, std::span<const ElemPose>(e, static_cast<usize>(n)), wm, ep);
    if (pyl_ztop > 0.0f) add_pylons(c, pyl_x0, pyl_x1, z_le - 0.01f, pyl_ztop, 0.07f);
    s.end_group();
}

// Alerón trasero de 2 elementos (+ tercero opcional) con endplates hasta z_ep_lo, pilar central,
// DRS opcional y beam wing opcional (grupo propio).
// Calibración (fase 2, docs/FISICA.md): en 2008-2019 los ángulos van ~7° (principal) y ~5° (flap) por ENCIMA de
// los reales: el alerón trabaja en la estela del coche, con u ≈ 0.5 U y el flujo subiendo 15-20° (medido a Media).
// Así conserva algo de carga (con los reales daba SCz 0.04-0.17 m²). 1998, en cambio, va MÁS abierto (−3°, 10°/22°
// en vez de −5°, 18°/32°): su alerón alto de 3 elementos sí funciona y dejaba al 1998 por encima de 2008-2014.
void flat_rear_wing(CarCtx& c, float x_le, float z_le, float aoa_main, float c_main, float c_f1, float a_f1, float c_f2,
                    float a_f2, float half_span, float z_ep_lo, bool drs, float beam_x, float beam_z, float pyl_x, float pyl_zlo) {
    Scene& s = c.s;
    s.begin_group("Alerón trasero", Component::RearWing);
    ElemPose e[3];
    int n = 2;
    e[0] = {x_le, z_le, deg(aoa_main), c_main, c.eprof(0.08f, 0.40f, 0.14f, c_main)};
    e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, c_f1), c_f1, c.flap_rear(a_f1), k_min_gap, 0.06f);
    if (c_f2 > 0.0f) { e[2] = place_next(s, e[1], c.eprof(0.05f, 0.45f, 0.17f, c_f2), c_f2, c.flap_rear(a_f2), k_min_gap, 0.06f); ++n; }
    if (drs && c.P.drs_open) e[n - 1] = open_flap(e[n - 1], k_drs_open);
    WingPlan w; w.span = half_span;
    add_elements(s, std::span<const ElemPose>(e, static_cast<usize>(n)), w);
    EndplateSpec ep; ep.z_down_to = z_ep_lo; ep.x_pad_rear = 0.05f;
    add_endplates(c, std::span<const ElemPose>(e, static_cast<usize>(n)), w, ep);
    rbox(s, Vec3(pyl_x, -0.02f, pyl_zlo), Vec3(pyl_x + 0.16f, 0.02f, z_le + 0.03f), 0.015f);
    s.end_group();
    if (beam_x > 0.0f) {
        s.begin_group("Beam wing", Component::BeamWing);
        const ElemPose bw[1] = {{beam_x, beam_z, deg(-9.0f), 0.24f, c.eprof(0.06f, 0.40f, 0.15f, 0.24f)}};
        WingPlan wb; wb.span = half_span + 0.01f;
        add_elements(s, bw, wb);
        s.end_group();
    }
}

void flat_suspension(CarCtx& c, const WheelSpec& wf, const WheelSpec& wr) {
    c.s.begin_group("Suspensión", Component::Suspension, Frame::Body);
    add_suspension_arms(c, wf, 0.12f, 0.50f, 0.22f, 0.02f, true);
    add_suspension_arms(c, wr, 0.08f, 0.36f, 0.18f, 0.02f, false);
    c.s.end_group();
}

// =====================================================================================
//  1998 — vía estrecha, neumáticos con surcos, morro alto
// =====================================================================================
void build_1998(Built& b) {
    FlatCar f{};
    f.wb = 2.94f;
    f.wf = {0.0f, 0.745f, 0.325f, 0.30f, 0.165f, 0.05f, 0.07f};
    f.wr = {f.wb, 0.715f, 0.33f, 0.365f, 0.165f, 0.05f, 0.08f};
    f.tub_front_zlo = 0.12f; f.tub_front_zhi = 0.52f;
    f.cover_x1 = 2.95f; f.cover_zlo1 = 0.12f; f.cover_zhi1 = 0.40f;
    f.crash_x0 = 2.90f; f.crash_x1 = 3.75f; f.airbox_z = 0.80f;
    f.nose_x0 = -0.95f; f.nose_zlo0 = 0.20f; f.nose_zhi0 = 0.32f; f.nose_hy0 = 0.07f; f.nose_zlo1 = 0.16f; f.nose_zhi1 = 0.50f;
    f.pod_x0 = 0.95f; f.pod_x1 = 2.55f; f.pod_hy0 = 0.64f; f.pod_zlo0 = 0.06f; f.pod_zhi0 = 0.52f; f.pod_hy1 = 0.30f; f.pod_zhi1 = 0.34f;
    f.floor = {0.45f, 2.55f, 0.62f, 0.30f, 0.15f, 2.62f, 3.55f, 0.20f, 0.50f, 0.22f};
    CarCtx c(b.scene, b.params, f.wb);
    Scene& s = b.scene;
    flat_body(c, f, false, 0, 0);
    s.begin_group("Bargeboards", Component::Sidepods);
    { const Vec2 p[] = {{0.42f, 0.06f}, {0.92f, 0.06f}, {0.92f, 0.42f}, {0.62f, 0.42f}, {0.42f, 0.28f}}; plate_xz(s, p, 0.44f, 0.0175f); }
    s.end_group();
    flat_front_wing(c, -0.97f, 0.09f, 0.32f, 0.24f, 24.0f, 0.0f, 0.0f, 0.68f, 0.0f, -0.87f, -0.69f, 0.26f, 0.30f);
    // Alerón trasero alto de 3 elementos + plano inferior ("biplano") sobre el difusor.
    flat_rear_wing(c, 3.05f, 0.72f, -3.0f, 0.28f, 0.22f, 10.0f, 0.22f, 22.0f, 0.46f, 0.34f, false, 3.10f, 0.40f, 3.14f, 0.30f);
    flat_suspension(c, f.wf, f.wr);
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, f.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, f.wr);
    finish_car(b, f.wb);
}

// =====================================================================================
//  2008 — cumbre de los apéndices aerodinámicos
// =====================================================================================
void build_2008(Built& b) {
    FlatCar f{};
    f.wb = 3.12f;
    f.wf = {0.0f, 0.73f, 0.33f, 0.30f, 0.165f, 0.05f, 0.07f};
    f.wr = {f.wb, 0.705f, 0.33f, 0.37f, 0.165f, 0.05f, 0.08f};
    f.tub_front_zlo = 0.14f; f.tub_front_zhi = 0.58f;
    f.cover_x1 = 3.10f; f.cover_zlo1 = 0.14f; f.cover_zhi1 = 0.42f;
    f.crash_x0 = 3.05f; f.crash_x1 = 3.95f; f.airbox_z = 0.82f;
    f.nose_x0 = -1.00f; f.nose_zlo0 = 0.30f; f.nose_zhi0 = 0.40f; f.nose_hy0 = 0.06f; f.nose_zlo1 = 0.20f; f.nose_zhi1 = 0.56f;
    f.pod_x0 = 1.00f; f.pod_x1 = 2.60f; f.pod_hy0 = 0.60f; f.pod_zlo0 = 0.06f; f.pod_zhi0 = 0.56f; f.pod_hy1 = 0.28f; f.pod_zhi1 = 0.34f;
    f.floor = {0.45f, 2.70f, 0.62f, 0.30f, 0.15f, 2.78f, 3.72f, 0.175f, 0.50f, 0.22f};
    CarCtx c(b.scene, b.params, f.wb);
    Scene& s = b.scene;
    flat_body(c, f, true, 3.30f, 0.88f);
    s.begin_group("Apéndices (bargeboards, winglets, chimeneas, cuernos)", Component::Sidepods);
    {
        const Vec2 bb[] = {{0.40f, 0.06f}, {1.05f, 0.06f}, {1.05f, 0.42f}, {0.58f, 0.40f}, {0.40f, 0.26f}};
        plate_xz(s, bb, 0.40f, 0.0175f);                                                   // bargeboards
        const Vec2 sup[] = {{1.02f, 0.50f}, {1.30f, 0.50f}, {1.30f, 0.70f}, {1.02f, 0.70f}};
        plate_xz(s, sup, 0.60f, 0.0175f);                                                  // soporte del winglet
        s.wing(c.eprof(0.05f, 0.45f, 0.17f, 0.24f), Vec3(1.00f, 0.26f, 0.66f), 0.24f, 0.22f, 0.36f, deg(-8.0f), 0, 0, 0, 0, 0,
               Op::Union, 0, true);                                                        // winglets de pontón
        capsule(s, Vec3(2.15f, 0.30f, 0.34f), Vec3(2.15f, 0.30f, 0.52f), 0.045f, true);    // chimeneas
        s.wing(c.eprof(0.05f, 0.45f, 0.17f, 0.22f), Vec3(1.40f, 0.15f, 0.60f), 0.22f, 0.20f, 0.30f, deg(-6.0f), 0, 0.10f, 0, 0,
               0, Op::Union, 0, true);                                                     // cuernos ("horns")
        const Vec2 fu[] = {{2.25f, 0.07f}, {2.62f, 0.07f}, {2.62f, 0.38f}, {2.40f, 0.30f}};
        plate_xz(s, fu, 0.60f, 0.0175f);                                                   // flip-ups
    }
    s.end_group();
    flat_front_wing(c, -1.02f, 0.085f, 0.34f, 0.24f, 32.0f, 0.0f, 0.0f, 0.67f, 0.0f, -0.92f, -0.72f, 0.33f, 0.34f);
    flat_rear_wing(c, 3.25f, 0.64f, -14.0f, 0.30f, 0.24f, 32.0f, 0.0f, 0.0f, 0.45f, 0.32f, false, 3.30f, 0.36f, 3.36f, 0.30f);
    flat_suspension(c, f.wf, f.wr);
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, f.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, f.wr);
    finish_car(b, f.wb);
}

// =====================================================================================
//  2011 (2009-2013) — alerón delantero de 1.8 m, trasero estrecho y alto con DRS
// =====================================================================================
void build_2011(Built& b) {
    FlatCar f{};
    f.wb = 3.25f;
    f.wf = {0.0f, 0.76f, 0.33f, 0.27f, 0.165f, 0.05f, 0.07f};
    f.wr = {f.wb, 0.70f, 0.33f, 0.38f, 0.165f, 0.05f, 0.08f};
    f.tub_front_zlo = 0.18f; f.tub_front_zhi = 0.62f;
    f.cover_x1 = 3.20f; f.cover_zlo1 = 0.14f; f.cover_zhi1 = 0.42f;
    f.crash_x0 = 3.15f; f.crash_x1 = 4.05f; f.airbox_z = 0.82f;
    f.nose_x0 = -1.08f; f.nose_zlo0 = 0.28f; f.nose_zhi0 = 0.38f; f.nose_hy0 = 0.06f; f.nose_zlo1 = 0.26f; f.nose_zhi1 = 0.62f;
    f.pod_x0 = 1.05f; f.pod_x1 = 2.55f; f.pod_hy0 = 0.58f; f.pod_zlo0 = 0.08f; f.pod_zhi0 = 0.54f; f.pod_hy1 = 0.24f; f.pod_zhi1 = 0.32f;
    f.floor = {0.45f, 2.85f, 0.64f, 0.30f, 0.15f, 2.95f, 3.85f, 0.175f, 0.49f, 0.22f};
    CarCtx c(b.scene, b.params, f.wb);
    flat_body(c, f, true, 2.60f, 0.70f);
    flat_front_wing(c, -1.10f, 0.08f, 0.30f, 0.24f, 20.0f, 0.21f, 32.0f, 0.87f, 0.25f, -1.00f, -0.80f, 0.32f, 0.30f);
    flat_rear_wing(c, 3.40f, 0.74f, -15.0f, 0.32f, 0.25f, 34.0f, 0.0f, 0.0f, 0.35f, 0.32f, true, 3.44f, 0.38f, 3.50f, 0.30f);
    flat_suspension(c, f.wf, f.wr);
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, f.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, f.wr);
    finish_car(b, f.wb);
}

// =====================================================================================
//  2014 — inicio de la era híbrida: morro bajo "dedo", sin beam wing
// =====================================================================================
void build_2014(Built& b) {
    FlatCar f{};
    f.wb = 3.45f;
    f.wf = {0.0f, 0.76f, 0.33f, 0.27f, 0.165f, 0.05f, 0.07f};
    f.wr = {f.wb, 0.70f, 0.33f, 0.38f, 0.165f, 0.05f, 0.08f};
    f.tub_front_zlo = 0.18f; f.tub_front_zhi = 0.62f;
    f.cover_x1 = 3.40f; f.cover_zlo1 = 0.14f; f.cover_zhi1 = 0.42f;
    f.crash_x0 = 3.35f; f.crash_x1 = 4.20f; f.airbox_z = 0.84f;
    // Morro que cae hasta una punta baja (185 mm máx.) + "dedo"
    f.nose_x0 = -0.98f; f.nose_zlo0 = 0.12f; f.nose_zhi0 = 0.26f; f.nose_hy0 = 0.07f; f.nose_zlo1 = 0.20f; f.nose_zhi1 = 0.60f;
    f.pod_x0 = 1.05f; f.pod_x1 = 2.80f; f.pod_hy0 = 0.60f; f.pod_zlo0 = 0.08f; f.pod_zhi0 = 0.58f; f.pod_hy1 = 0.24f; f.pod_zhi1 = 0.32f;
    f.floor = {0.45f, 3.05f, 0.64f, 0.30f, 0.15f, 3.12f, 4.05f, 0.125f, 0.49f, 0.22f};
    CarCtx c(b.scene, b.params, f.wb);
    Scene& s = b.scene;
    flat_body(c, f, true, 2.80f, 0.72f);
    s.begin_group("Morro (dedo)", Component::Nose);
    // Dedo apoyado en el plano principal del alerón: con 1-2 cm de separación quedaba una rendija de 1 celda a Alta
    // (dx 26.6 mm) en la que la velocidad se disparaba y el cálculo divergía a 0.4 PF.
    capsule(s, Vec3(-0.96f, 0.0f, 0.16f), Vec3(-1.12f, 0.0f, 0.13f), 0.045f);
    s.end_group();
    flat_front_wing(c, -1.12f, 0.075f, 0.30f, 0.24f, 20.0f, 0.21f, 32.0f, 0.795f, 0.25f, -0.98f, -0.84f, 0.17f, 0.30f);
    flat_rear_wing(c, 3.62f, 0.76f, -13.0f, 0.28f, 0.22f, 30.0f, 0.0f, 0.0f, 0.35f, 0.34f, true, 0.0f, 0.0f, 3.70f, 0.30f);
    flat_suspension(c, f.wf, f.wr);
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, f.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, f.wr);
    finish_car(b, f.wb);
}

// =====================================================================================
//  2019 (2017-2021) — coches de 2 m, neumáticos anchos, bargeboards, halo
// =====================================================================================
void build_2019(Built& b) {
    FlatCar f{};
    f.wb = 3.6f;
    f.wf = {0.0f, 0.825f, 0.33f, 0.34f, 0.165f, 0.05f, 0.08f};
    f.wr = {f.wb, 0.775f, 0.335f, 0.44f, 0.165f, 0.05f, 0.10f};
    f.tub_front_zlo = 0.20f; f.tub_front_zhi = 0.62f;
    f.cover_x1 = 3.40f; f.cover_zlo1 = 0.14f; f.cover_zhi1 = 0.40f;
    f.crash_x0 = 3.35f; f.crash_x1 = 4.30f; f.airbox_z = 0.84f;
    f.nose_x0 = -1.10f; f.nose_zlo0 = 0.13f; f.nose_zhi0 = 0.27f; f.nose_hy0 = 0.07f; f.nose_zlo1 = 0.20f; f.nose_zhi1 = 0.60f;
    f.pod_x0 = 1.05f; f.pod_x1 = 2.90f; f.pod_hy0 = 0.62f; f.pod_zlo0 = 0.10f; f.pod_zhi0 = 0.56f; f.pod_hy1 = 0.24f; f.pod_zhi1 = 0.32f;
    f.floor = {0.45f, 3.00f, 0.78f, 0.30f, 0.15f, 3.05f, 4.20f, 0.30f, 0.53f, 0.24f};
    CarCtx c(b.scene, b.params, f.wb);
    Scene& s = b.scene;
    flat_body(c, f, true, 3.00f, 0.62f);
    s.begin_group("Bargeboards", Component::Sidepods);
    {
        const Vec2 b1[] = {{0.42f, 0.06f}, {1.20f, 0.06f}, {1.20f, 0.45f}, {0.70f, 0.48f}, {0.42f, 0.30f}};
        plate_xz(s, b1, 0.52f, 0.0175f);
        const Vec2 b2[] = {{0.55f, 0.06f}, {1.15f, 0.06f}, {1.15f, 0.28f}, {0.55f, 0.22f}};
        plate_xz(s, b2, 0.66f, 0.0175f);
    }
    s.end_group();
    flat_front_wing(c, -1.12f, 0.075f, 0.32f, 0.24f, 15.0f, 0.22f, 28.0f, 0.97f, 0.25f, -1.02f, -0.84f, 0.20f, 0.30f);
    flat_rear_wing(c, 3.80f, 0.64f, -15.0f, 0.34f, 0.26f, 34.0f, 0.0f, 0.0f, 0.485f, 0.30f, true, 3.84f, 0.37f, 3.88f, 0.30f);
    add_halo(c, 0.28f, 1.00f, 0.80f, 0.27f, 0.55f);
    flat_suspension(c, f.wf, f.wr);
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, f.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, f.wr);
    finish_car(b, f.wb);
}

// =====================================================================================
//  2022-2025 (efecto suelo) y 2026 (aero activa): túneles Venturi bajo el fondo
// =====================================================================================
struct GeCar {
    float wb;
    WheelSpec wf, wr;
    float dx;                  // desplazamiento de la parte trasera (batalla menor)
    float fw_half, rw_half;
    float nose_x0;
    const Vec2* roof; int n_roof;
    const Vec2* ramp; int n_ramp;
    float floor_yc, floor_hy, diff_hw;
    bool beam, deflectors, fences, bargeboards, xmode;
    float rw_main = -9.0f, rw_flap = 30.0f;   // alerón trasero: ángulo del plano principal (°, sdf) y del flap (°, carga +)
};

void build_ge(Built& b, const GeCar& g) {
    CarCtx c(b.scene, b.params, g.wb);
    Scene& s = b.scene;
    const float d = g.dx;

    // --- Chasis, cubierta motor, estructura trasera -----------------------------------
    s.begin_group("Chasis y cubierta motor", Component::Body);
    // Los fondos del chasis y de la cubierta quedan POR ENCIMA del techo de los túneles.
    tbox(s, -0.50f, 0.92f, 0.17f, 0.15f, 0.44f, 0.30f, 0.19f, 0.66f, 0.07f);
    tbox(s, 0.85f, 3.20f + d, 0.30f, 0.19f, 0.70f, 0.15f, 0.16f, 0.48f, 0.08f, 0.0f, false, Op::SmoothUnion, 0.10f);
    s.ellipsoid(Vec3(1.30f, 0.0f, 0.80f), Vec3(0.36f, 0.13f, 0.17f), {}, Op::SmoothUnion, 0.10f);
    engine_spine(s, 1.30f, 0.94f, 3.15f + d, 0.50f, 0.13f, 0.07f);
    tbox(s, 3.10f + d, 4.45f + d, 0.15f, 0.16f, 0.46f, 0.07f, 0.28f, 0.42f, 0.04f, 0.0f, false, Op::SmoothUnion, 0.06f);
    cockpit(s, 0.52f, 0.70f, 0.72f, 0.62f);
    mirrors(s, Vec3(0.90f, 0.50f, 0.70f), Vec3(1.00f, 0.42f, 0.58f));
    s.end_group();

    s.begin_group("Morro", Component::Nose);
    tbox(s, g.nose_x0, -0.42f, 0.09f, 0.07f, 0.20f, 0.17f, 0.15f, 0.44f, 0.06f);
    s.end_group();

    s.begin_group("Pontones", Component::Sidepods);
    tbox(s, 0.95f, 2.95f + d, 0.65f, 0.24f, 0.62f, 0.26f, 0.18f, 0.40f, 0.10f);
    if (g.bargeboards) {   // 2026: bargeboards de "in-wash" delante de los pontones
        const Vec2 p[] = {{0.45f, 0.045f}, {0.95f, 0.045f}, {0.95f, 0.38f}, {0.62f, 0.40f}, {0.45f, 0.26f}};
        plate_xz(s, p, 0.62f, 0.0175f);
    }
    s.end_group();

    // --- Fondo con túneles Venturi -------------------------------------------------------
    s.begin_group("Fondo (túneles Venturi)", Component::Floor);
    {
        sheet_xz(s, std::span<const Vec2>(g.roof, static_cast<usize>(g.n_roof)), 0.04f, g.floor_yc, g.floor_hy);   // techo
        const float xe = g.roof[g.n_roof - 1].x;
        tbox(s, 0.28f, xe + 0.03f, 0.15f, 0.0f, 0.27f, 0.14f, 0.0f, 0.22f, 0.04f);                 // quilla + plank
        // Borde del fondo: placa que baja desde el techo hasta 35 mm (sella parcialmente el túnel).
        Vec2 edge[16];
        int k = 0;
        edge[k++] = {0.90f, 0.035f};
        edge[k++] = {xe, 0.035f};
        for (int i = g.n_roof - 1; i >= 0 && k < 15; --i)
            if (g.roof[i].x >= 0.90f) edge[k++] = g.roof[i] + Vec2(0.0f, 0.03f);
        edge[k++] = {0.90f, 0.20f};
        plate_xz(s, std::span<const Vec2>(edge, static_cast<usize>(k)), g.floor_yc + g.floor_hy - 0.0175f, 0.0175f);
        if (g.fences) {
            const Vec2 fence[] = {{0.50f, 0.045f}, {1.45f, 0.045f}, {1.45f, 0.11f}, {1.00f, 0.17f}, {0.50f, 0.27f}};
            plate_xz(s, fence, 0.36f, 0.0175f);
            plate_xz(s, fence, 0.56f, 0.0175f);
        }
    }
    s.end_group();

    s.begin_group("Difusor", Component::Diffuser);
    {
        sheet_xz(s, std::span<const Vec2>(g.ramp, static_cast<usize>(g.n_ramp)), 0.04f, 0.5f * g.diff_hw, 0.5f * g.diff_hw);
        const Vec2 r0 = g.ramp[0], r1 = g.ramp[g.n_ramp - 1];
        const Vec2 wall[] = {{r0.x + 0.05f, 0.035f}, {r0.x + 0.30f, 0.035f}, {r1.x, 0.55f * r1.y}, {r1.x, r1.y + 0.04f}, {r0.x + 0.05f, r0.y + 0.04f}};
        plate_xz(s, wall, g.diff_hw - 0.0175f, 0.0175f);
        const Vec2 strake[] = {{r0.x + 0.15f, 0.045f}, {r1.x - 0.02f, 0.60f * r1.y}, {r1.x - 0.02f, r1.y + 0.02f}, {r0.x + 0.15f, r0.y + 0.04f}};
        plate_xz(s, strake, 0.20f, 0.0175f);
    }
    s.end_group();

    // --- Alerón delantero (3 elementos, anclado al morro) ---------------------------------
    s.begin_group("Alerón delantero", Component::FrontWing);
    {
        ElemPose e[3];
        const float x0 = g.nose_x0 + 0.02f;
        e[0] = {x0, 0.10f, deg(-3.0f), 0.36f, c.eprof(0.06f, 0.40f, 0.13f, 0.36f)};
        e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, 0.25f), 0.25f, c.flap_front(14.0f), k_min_gap, 0.06f);
        e[2] = place_next(s, e[1], c.eprof(0.05f, 0.45f, 0.17f, 0.23f), 0.23f, c.flap_front(28.0f), k_min_gap, 0.06f);
        if (g.xmode && c.P.drs_open) {   // 2026 modo X: flaps delanteros abiertos (casi planos)
            e[2] = open_flap(e[2], deg(-6.0f));
            e[1] = open_flap(e[1], deg(-3.0f));
        }
        WingPlan w; w.span = g.fw_half; w.zcurve = 0.05f;
        add_elements(s, e, w);
        EndplateSpec ep; ep.z_min = 0.04f;
        add_endplates(c, e, w, ep);
    }
    s.end_group();

    // --- Alerón trasero (puntas redondeadas) + DRS / modo X ---------------------------------
    s.begin_group("Alerón trasero", Component::RearWing);
    {
        ElemPose e[2];
        // Calibración (fase 2, ver docs/FISICA.md): incidencias MAYORES que las reales (plano principal −9° → g.rw_main,
        // flap 30° → g.rw_flap). En este solver el alerón trasero trabaja dentro de la estela del coche con u ≈ 0.5 U y
        // un flujo que SUBE ~15-20° (difusor + beam wing + estela): con los ángulos reales apenas daba carga (SCz 0.10
        // a Media); con +9°/+6° da 0.17.
        e[0] = {3.96f + d, 0.74f, deg(g.rw_main), 0.31f, c.eprof(0.08f, 0.40f, 0.14f, 0.31f)};
        e[1] = place_next(s, e[0], c.eprof(0.05f, 0.45f, 0.17f, 0.25f), 0.25f, c.flap_rear(g.rw_flap), k_min_gap, 0.06f);
        if (c.P.drs_open) e[1] = open_flap(e[1], g.xmode ? deg(-3.0f) : k_drs_open);
        WingPlan w; w.span = g.rw_half; w.zcurve = -0.12f;
        add_elements(s, e, w);
        EndplateSpec ep; ep.z_down_to = g.beam ? 0.36f : 0.44f; ep.round = 0.03f; ep.x_pad_rear = 0.05f;
        add_endplates(c, e, w, ep);
        rbox(s, Vec3(4.10f + d, -0.02f, 0.40f), Vec3(4.26f + d, 0.02f, 0.78f), 0.015f);   // pilón central
    }
    s.end_group();

    if (g.beam) {
        s.begin_group("Beam wing", Component::BeamWing);
        const ElemPose e[1] = {{4.02f + d, 0.41f, deg(-10.0f), 0.25f, c.eprof(0.06f, 0.40f, 0.15f, 0.25f)}};
        WingPlan w; w.span = g.rw_half + 0.01f;
        add_elements(s, e, w);
        s.end_group();
    }

    add_halo(c, 0.28f, 1.00f, 0.80f, 0.27f, 0.55f);

    s.begin_group("Suspensión", Component::Suspension, Frame::Body);
    add_suspension_arms(c, g.wf, 0.14f, 0.42f, 0.22f, 0.02f, true);
    add_suspension_arms(c, g.wr, 0.10f, 0.40f, 0.23f, 0.02f, false);
    s.end_group();

    if (g.deflectors) {
        // Deflectores de estela sobre la rueda delantera (fijos a la mangueta: marco ruedas, sin giro).
        s.begin_group("Deflectores de estela", Component::Suspension, Frame::Wheels);
        const float zt = 2.0f * g.wf.radius - g.wf.squash;
        s.wing(c.eprof(0.05f, 0.4f, 0.16f, 0.24f), Vec3(-0.17f, g.wf.half_track - 0.14f, zt + 0.01f), 0.24f, 0.20f, 0.17f,
               deg(-6.0f), 0.03f, 0, 0, 0, 0, Op::Union, 0, true);
        s.end_group();
    }

    add_axle(c, "Ruedas delanteras", Component::FrontWheels, g.wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, g.wr);
    finish_car(b, g.wb);
}

void build_2022(Built& b) {
    static constexpr Vec2 roof[] = {{0.50f, 0.24f}, {0.80f, 0.16f}, {1.20f, 0.095f}, {1.80f, 0.065f}, {2.50f, 0.06f}, {3.02f, 0.08f}};
    static constexpr Vec2 ramp[] = {{2.95f, 0.075f}, {3.40f, 0.16f}, {3.90f, 0.28f}, {4.32f, 0.36f}};
    GeCar g{};
    g.wb = 3.6f;
    g.wf = {0.0f, 0.80f, 0.36f, 0.345f, 0.235f, 0.015f, 0.075f};    // 18" con tapacubos
    g.wr = {g.wb, 0.775f, 0.36f, 0.42f, 0.235f, 0.015f, 0.08f};
    g.dx = 0.0f; g.fw_half = 0.95f; g.rw_half = 0.46f; g.nose_x0 = -1.12f;
    g.roof = roof; g.n_roof = 6; g.ramp = ramp; g.n_ramp = 4;
    g.floor_yc = 0.47f; g.floor_hy = 0.33f; g.diff_hw = 0.54f;
    g.beam = true; g.deflectors = true; g.fences = true; g.bargeboards = false; g.xmode = false;
    g.rw_main = -18.0f; g.rw_flap = 36.0f;
    build_ge(b, g);
}

void build_2026(Built& b) {
    // Fondo más plano: túneles menos profundos y difusor más pequeño.
    static constexpr Vec2 roof[] = {{0.50f, 0.17f}, {0.80f, 0.115f}, {1.20f, 0.08f}, {1.80f, 0.065f}, {2.40f, 0.065f}, {2.82f, 0.075f}};
    static constexpr Vec2 ramp[] = {{2.75f, 0.075f}, {3.20f, 0.12f}, {3.70f, 0.20f}, {4.10f, 0.26f}};
    GeCar g{};
    g.wb = 3.4f;
    g.wf = {0.0f, 0.79f, 0.3525f, 0.32f, 0.235f, 0.04f, 0.07f};     // más estrechos, sin tapacubos
    g.wr = {g.wb, 0.755f, 0.355f, 0.39f, 0.235f, 0.04f, 0.08f};
    g.dx = -0.2f; g.fw_half = 0.90f; g.rw_half = 0.46f; g.nose_x0 = -1.02f;
    g.roof = roof; g.n_roof = 6; g.ramp = ramp; g.n_ramp = 4;
    g.floor_yc = 0.45f; g.floor_hy = 0.31f; g.diff_hw = 0.52f;
    g.beam = false; g.deflectors = false; g.fences = true; g.bargeboards = true; g.xmode = true;
    g.rw_main = -16.0f; g.rw_flap = 34.0f;
    build_ge(b, g);
}

} // namespace

void register_f1(std::vector<Entry>& out) {
    out.push_back({car_info("f1_1967", "F1 1967 — Lotus 49 (sin alerones)", "1966-1967",
        "Coche \"puro\" de la última época sin alerones: carrocería en forma de cigarro con el radiador en el "
        "morro, ruedas enormes al descubierto y motor Cosworth DFV portante. La carrocería genera algo de "
        "sustentación (carga negativa) y las ruedas expuestas dominan la resistencia.",
        P_RideHeight | P_Yaw | P_Wheels, 3.95f, 1.15f, 2.413f, 100.0f, 110.0f, -0.20f, 0.75f, 0.0f, 0.0f, 0.0f,
        "estimación (Cd≈0.65, ligera sustentación; sin datos de túnel publicados)"), &build_1967});
    out.push_back({car_info("f1_1979", "F1 1979 — Lotus 79 (coche-ala)", "1977-1982",
        "Primera época de efecto suelo: los pontones son alas invertidas cuyo intradós forma con el suelo un "
        "Venturi, sellado lateralmente por faldones deslizantes que rozan el asfalto. Pequeños alerones "
        "delanteros a los lados del morro y alerón trasero sobre la caja de cambios. La carga llega sobre "
        "todo del fondo, con poca penalización en resistencia.",
        k_car_mask, 4.45f, 1.30f, 2.72f, 50.0f, 70.0f, 2.4f, 0.95f, 2.15f, 30.0f, 10.0f,
        "estimación (L/D ≈ 2.5 citada para el Lotus 79)"), &build_1979});
    out.push_back({car_info("f1_1998", "F1 1998 — vía estrecha y surcos", "1998-2008 (vía estrecha)",
        "Coches de 1.8 m con neumáticos con surcos para reducir el agarre mecánico. Morro alto con el alerón "
        "delantero colgado de pilones entre endplates, fondo plano escalonado con plank (desde 1994-95), "
        "difusor limitado y alerón trasero alto de varios elementos más un plano inferior sobre el difusor.",
        k_car_mask, 4.45f, 1.40f, 2.94f, 30.0f, 60.0f, 2.9f, 1.05f, 1.80f, 24.0f, 22.0f,
        "estimación (orden de magnitud de finales de los 90)"), &build_1998});
    out.push_back({car_info("f1_2008", "F1 2008 — cumbre de los apéndices", "2006-2008",
        "El reglamento dejaba libertad en la zona de pontones: bargeboards, winglets sobre los pontones, "
        "chimeneas, \"cuernos\", flip-ups, aleta de tiburón y morro alto en V. Alerón delantero de 1.4 m y "
        "trasero de 1 m de ancho más beam wing. Mucha carga pero estela muy sucia (difícil seguir a otro coche).",
        k_car_mask, 4.65f, 1.45f, 3.12f, 25.0f, 70.0f, 3.6f, 1.25f, 1.80f, 32.0f, 32.0f,
        "estimación (valores típicos publicados para 2008)"), &build_2008});
    out.push_back({car_info("f1_2011", "F1 2011 — alerón ancho y DRS", "2009-2013",
        "Reglamento 2009: alerón delantero de 1.8 m (ancho completo) con sección central neutra de 50 cm, "
        "alerón trasero estrecho (0.75 m) y alto, pontones limpios sin apéndices. En 2011 se prohíbe el doble "
        "difusor y llega el DRS: el flap superior se abre para reducir la resistencia en recta. Rake alto.",
        k_car_mask | P_Drs, 4.95f, 1.45f, 3.25f, 25.0f, 85.0f, 3.8f, 1.20f, 1.80f, 32.0f, 34.0f,
        "estimación (valores típicos publicados 2010-2012)"), &build_2011});
    out.push_back({car_info("f1_2014", "F1 2014 — era híbrida, morro bajo", "2014-2016",
        "Inicio de la era turbo-híbrida: morro obligatoriamente bajo (el \"dedo\"), alerón delantero reducido a "
        "1.65 m, alerón trasero menos profundo y sin beam wing, difusor más corto. Menos carga y resistencia "
        "que 2013; el DRS se mantiene.",
        k_car_mask | P_Drs, 5.05f, 1.45f, 3.45f, 30.0f, 75.0f, 3.3f, 1.10f, 1.80f, 32.0f, 30.0f,
        "estimación (valores típicos publicados 2014)"), &build_2014});
    out.push_back({car_info("f1_2019", "F1 2019 — coches anchos (2017-2021)", "2017-2021",
        "Coches de 2 m de ancho con neumáticos mucho más anchos, alerón delantero de 2 m (simplificado en "
        "2019), alerón trasero más bajo y ancho, difusor más grande que empieza antes del eje, bargeboards "
        "muy complejos y halo desde 2018. La época de mayor carga aerodinámica hasta la fecha.",
        k_car_mask | P_Drs, 5.45f, 1.55f, 3.6f, 30.0f, 100.0f, 5.0f, 1.35f, 2.00f, 28.0f, 34.0f,
        "estimación (valores típicos publicados 2019)"), &build_2019});
    out.push_back({car_info("f1_2022", "F1 2022 — efecto suelo (túneles Venturi)", "2022-2025",
        "Vuelve el efecto suelo: la mayor parte de la carga la generan dos túneles Venturi bajo el fondo "
        "(entradas tras las ruedas delanteras, garganta y un difusor enorme). Alerón delantero simplificado "
        "anclado al morro, ruedas de 18\" con tapacubos y deflectores de estela, alerón trasero curvado con "
        "puntas redondeadas y beam wing: todo pensado para reducir la estela sucia y poder seguir a otro coche.",
        k_car_mask | P_Drs, 5.6f, 1.50f, 3.6f, 30.0f, 80.0f, 4.4f, 1.15f, 2.00f, 28.0f, 36.0f,
        "estimación (datos públicos de equipos/FIA, orden de magnitud)"), &build_2022});
    out.push_back({car_info("f1_2026", "F1 2026 — aero activa (modo Z / modo X)", "2026-",
        "Coche más corto (batalla 3.4 m) y estrecho (1.9 m), neumáticos más finos, fondo más plano con "
        "túneles y difusor reducidos, sin beam wing. Aerodinámica activa: en modo Z (curva, por defecto) los "
        "flaps están cerrados; en modo X (recta, \"DRS abierto\") se abren los flaps delanteros y el trasero "
        "para reducir mucho la resistencia.",
        k_car_mask | P_Drs, 5.4f, 1.40f, 3.4f, 30.0f, 60.0f, 3.4f, 0.95f, 1.90f, 28.0f, 34.0f,
        "objetivo FIA aproximado (-30% carga, -55% resistencia en modo X)"), &build_2026});
}

} // namespace cfd::models::detail
