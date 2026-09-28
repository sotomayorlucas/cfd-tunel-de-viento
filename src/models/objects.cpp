// ============================================================================
//  models/objects.cpp — cuerpos y alas de referencia (validación y estudio).
//
//  Todos en espacio modelo (m) con el suelo potencial en z = 0 aunque el caso
//  sea de aire libre (needs_ground = false): así, si el usuario activa el suelo,
//  el objeto está a una altura física conocida (parámetro P_Height).
// ============================================================================
#include "f1_common.hpp"
#include "registry.hpp"

namespace cfd::models::detail {
namespace {

CFD_INLINE float obj_deg(float d) { return d * k_deg2rad; }   // (nombre propio: f1_eras.cpp define deg en el mismo espacio anónimo → choque en `make unity`)

// Marco "carrocería" de un objeto: guiñada alrededor del eje vertical que pasa por `c`.
Xform object_frame(Vec3 c, float yaw_deg) {
    const Mat3 R = Mat3::rot_z(yaw_deg * k_deg2rad);
    return {R, c - R * c};
}

Info obj_info(const char* id, const char* name, const char* desc, Kind k, u32 mask, float len, float area, bool ground,
              float speed, float cla, float cda, const char* src) {
    Info I;
    I.id = id; I.name = name; I.era = "Referencia"; I.description = desc;
    I.kind = k; I.param_mask = mask;
    I.ref_length_m = len; I.ref_area_m2 = area;
    I.needs_ground = ground;
    I.default_speed_kmh = speed;
    I.ref_ClA = cla; I.ref_CdA = cda;
    I.ref_source = src;
    return I;
}

// ---- Esfera ------------------------------------------------------------------------------
void build_sphere(Built& b) {
    Scene& s = b.scene;
    const float r = 0.5f, zc = b.params.height_mm * 1e-3f + r;
    s.begin_group("Esfera", Component::Object, Frame::Fixed);
    s.sphere(Vec3(0.0f, 0.0f, zc), r);
    s.end_group();
    s.set_frames(Xform{}, Xform{});
    b.moment_ref_m = Vec3(0.0f, 0.0f, zc);
}

// ---- Cilindro finito transversal -------------------------------------------------------------
void build_cylinder(Built& b) {
    Scene& s = b.scene;
    const float r = 0.25f, hw = 1.0f, zc = b.params.height_mm * 1e-3f + r;
    s.begin_group("Cilindro", Component::Object, Frame::Body);
    s.cylinder_y(Vec3(0.0f, 0.0f, zc), r, hw, 0.01f);
    s.end_group();
    s.set_frames(object_frame(Vec3(0.0f, 0.0f, zc), b.params.yaw_deg), Xform{});
    b.moment_ref_m = Vec3(0.0f, 0.0f, zc);
}

// ---- Cubo ----------------------------------------------------------------------------------
void build_cube(Built& b) {
    Scene& s = b.scene;
    const float h = 0.5f, zc = b.params.height_mm * 1e-3f + h;
    s.begin_group("Cubo", Component::Object, Frame::Body);
    s.round_box(Vec3(0.0f, 0.0f, zc), Vec3(h), 0.004f);
    s.end_group();
    s.set_frames(object_frame(Vec3(0.0f, 0.0f, zc), b.params.yaw_deg), Xform{});
    b.moment_ref_m = Vec3(0.0f, 0.0f, zc);
}

// ---- Cuerpo de Ahmed (25°) ----------------------------------------------------------------------
// L = 1044 mm, W = 389 mm, H = 288 mm, radio frontal 100 mm, luneta de 222 mm a 25°,
// 4 patas de Ø30 mm, 50 mm de altura libre (Ahmed, Ramm & Faltin 1984).
void build_ahmed(Built& b) {
    Scene& s = b.scene;
    constexpr float L = 1.044f, W = 0.389f, H = 0.288f, R = 0.100f, gap = 0.050f;
    constexpr float slant_len = 0.222f, slant = 25.0f * k_deg2rad;
    const float z0 = gap, z1 = gap + H;
    s.begin_group("Cuerpo de Ahmed", Component::Body, Frame::Body);
    // Caja exacta primero (define una AABB ajustada del grupo); después el radio frontal como
    // intersección con dos cajas redondeadas (redondeo en planta y en alzado) prolongadas hacia
    // atrás y hacia fuera para que la trasera y las aristas longitudinales queden vivas.
    s.round_box(Vec3(0.5f * L, 0.0f, 0.5f * (z0 + z1)), Vec3(0.5f * L, 0.5f * W, 0.5f * H), 0.0f);
    s.round_box(Vec3(0.5f * (L + 0.4f), 0.0f, 0.5f * (z0 + z1)), Vec3(0.5f * (L + 0.4f), 0.5f * W, 0.5f * H + 0.3f), R, {},
                Op::Intersect);
    s.round_box(Vec3(0.5f * (L + 0.4f), 0.0f, 0.5f * (z0 + z1)), Vec3(0.5f * (L + 0.4f), 0.5f * W + 0.3f, 0.5f * H), R, {},
                Op::Intersect);
    // Trasera recta (x ≤ L), laterales/techo/suelo y luneta inclinada 25° (normales hacia fuera).
    const float xs = L - slant_len * std::cos(slant);                      // inicio de la luneta
    const Vec3 ns = normalize(Vec3(std::sin(slant), 0.0f, std::cos(slant)));
    const Vec4 pl[] = {{ns.x, 0.0f, ns.z, dot(ns, Vec3(xs, 0.0f, z1))}};
    s.convex(s.planes(pl), Op::Intersect);
    s.end_group();
    s.begin_group("Patas", Component::Suspension, Frame::Body);
    const Mat3 Rx = Mat3::rot_x(0.5f * k_pi);                                   // cilindro vertical
    for (float xl : {0.163f, 0.633f})
        s.add(sdf::PrimType::Cylinder, Xform{Rx, Vec3(xl, 0.1195f, 0.5f * (gap + 0.02f) - 0.002f)}, {0.015f, 0.5f * (gap + 0.02f) + 0.002f, 0.002f},
              Op::Union, 0.0f, true);
    s.end_group();
    const Vec3 c(0.5f * L, 0.0f, 0.0f);
    s.set_frames(object_frame(c, b.params.yaw_deg), Xform{});
    b.moment_ref_m = c;
}

// ---- Alas en aire libre -------------------------------------------------------------------------
// Rectangular, cuerda c, envergadura total `span` (+ `overhang` por cada lado), AoA alrededor del
// 25% de cuerda; el punto más bajo queda a `height`.
void free_wing(Built& b, float m, float p, float t, float chord, float span, const char* name, float overhang = 0.0f) {
    Scene& s = b.scene;
    const float aoa = obj_deg(b.params.aoa_deg);
    const Handle h = naca4_floor(s, m, p, t, 0.0f, false, 24);
    ElemPose e{0.0f, 0.0f, aoa, chord, h};
    // BA tal que el 25% de cuerda quede en x = 0.25c (rotación alrededor del cuarto de cuerda).
    e.x = 0.25f * chord - 0.25f * chord * std::cos(aoa);
    e.z = 0.25f * chord * std::sin(aoa);
    float x0, z0, x1, z1;
    elem_box(s, e, x0, z0, x1, z1);
    e.z += b.params.height_mm * 1e-3f - z0;
    s.begin_group(name, Component::WingMain, Frame::Fixed);
    const ElemPose el[1] = {e};
    WingPlan w; w.span = 0.5f * span + overhang;
    add_elements(s, el, w);
    s.end_group();
    s.set_frames(Xform{}, Xform{});
    b.moment_ref_m = Vec3(e.x + 0.25f * chord * std::cos(aoa), 0.0f, e.z - 0.25f * chord * std::sin(aoa));   // 25% de cuerda
}

void build_naca0012(Built& b) { free_wing(b, 0.0f, 0.0f, 0.12f, 1.0f, 3.0f, "Ala NACA 0012"); }
void build_naca4412(Built& b) { free_wing(b, 0.04f, 0.40f, 0.12f, 1.0f, 3.0f, "Ala NACA 4412"); }

// Pseudo-2D (Info::spans_domain): bounds_m.y es EXACTAMENTE el ancho del dominio (1 m) y la
// geometría sobresale 0.1 m por cada lado, así el perfil llega a las paredes laterales sin
// rendija. Antes bounds_m era la AABB de la escena, que la primitiva Wing amplía 0.02·cuerda:
// una app que hiciese ny·dx = extensión Y de bounds_m obtenía 1.04 m para un ala de 1 m y dejaba
// 2 cm de fluido junto a cada pared (≈ 1 celda a dx = 3 cm) → flujo de punta en un caso "2D".
void build_airfoil2d(Built& b) {
    constexpr float span = 1.0f, overhang = 0.10f;
    free_wing(b, 0.04f, 0.40f, 0.12f, 1.0f, span, "Perfil NACA 4412 (2D)", overhang);
    b.bounds_m = b.scene.bounds();
    b.bounds_m.lo.y = -0.5f * span;
    b.bounds_m.hi.y = 0.5f * span;
}

// ---- Ala invertida de 2 elementos en efecto suelo -------------------------------------------------
// Estudio clásico (Zerihan & Zhang 2000-2003): la carga crece al bajar la altura h hasta un
// máximo cerca de h/c ≈ 0.1 y luego cae (la capa límite del intradós se separa y el
// flujo bajo el ala se bloquea). P_Aoa en convención F1: + = BA abajo = más carga.
void build_f1_wing_ge(Built& b) {
    Scene& s = b.scene;
    const Params& P = b.params;
    constexpr float cm = 0.50f, cf = 0.25f, span = 1.2f;
    const Handle hm = naca4_floor(s, 0.08f, 0.40f, 0.12f, 0.0f, true, 20);
    const Handle hf = naca4_floor(s, 0.06f, 0.45f, 0.14f, 0.0f, true, 18);
    ElemPose e[2];
    e[0] = {0.0f, 0.0f, obj_deg(-P.aoa_deg), cm, hm};
    e[1] = place_next(s, e[0], hf, cf, obj_deg(-(20.0f + P.front_flap_deg + P.aoa_deg)), P.flap_gap_mm * 1e-3f, 0.05f);
    float x0, z0, x1, z1;
    elems_box(s, e, x0, z0, x1, z1);
    const float dz = P.height_mm * 1e-3f - z0;
    for (ElemPose& q : e) q.z += dz;
    z0 += dz; z1 += dz;
    WingPlan w; w.span = 0.5f * span;
    s.begin_group("Plano principal", Component::WingMain, Frame::Fixed);
    add_elements(s, std::span<const ElemPose>(e, 1), w);
    s.end_group();
    s.begin_group("Flap", Component::WingFlap, Frame::Fixed);
    add_elements(s, std::span<const ElemPose>(e + 1, 1), w);
    s.end_group();
    s.begin_group("Endplates", Component::Endplate, Frame::Fixed);
    const float zl = max_(z0 - 0.02f, 0.5f * z0);   // nunca toca el suelo
    rbox(s, Vec3(x0 - 0.03f, 0.5f * span - 0.005f, zl), Vec3(x1 + 0.03f, 0.5f * span + 0.02f, z1 + 0.05f), 0.006f, true);
    s.end_group();
    s.set_frames(Xform{}, Xform{});
    // 25% de la cuerda del plano principal (como en las alas libres; antes se ponía a nivel del
    // suelo, z = 0: el momento de cabeceo cambiaba en D·h respecto al convenio de las otras alas).
    b.moment_ref_m = Vec3(e[0].x + 0.25f * cm * std::cos(e[0].aoa), 0.0f, e[0].z - 0.25f * cm * std::sin(e[0].aoa));
}

// ---- Turismo tipo DrivAer (fastback) -----------------------------------------------------------------
void build_road_car(Built& b) {
    constexpr float wb = 2.786f;
    CarCtx c(b.scene, b.params, wb);
    Scene& s = b.scene;
    const WheelSpec wf{0.0f, 0.78f, 0.316f, 0.225f, 0.20f, 0.03f, 0.06f};
    const WheelSpec wr{wb, 0.78f, 0.316f, 0.225f, 0.20f, 0.03f, 0.06f};
    s.begin_group("Carrocería", Component::Body);
    {
        // Habitáculo fastback (perfil lateral) redondeado por un elipsoide: primero, para que la
        // intersección sólo le afecte a él.
        const Vec2 cab[] = {{0.70f, 0.52f}, {1.62f, 1.24f}, {2.30f, 1.27f}, {3.72f, 0.78f}, {3.72f, 0.50f}};
        s.extrude_xz(s.polygon(cab), 0.0f, 0.70f);
        s.ellipsoid(Vec3(1.95f, 0.0f, 0.45f), Vec3(2.4f, 0.86f, 0.95f), {}, Op::Intersect);
        // Cuerpo inferior (capó bajo delante, zaga alta).
        tbox(s, -0.84f, 3.77f, 0.86f, 0.0f, 0.52f, 0.88f, 0.0f, 0.72f, 0.16f, 0.0f, false, Op::SmoothUnion, 0.12f);
        // Pasos de rueda.
        for (const WheelSpec* w : {&wf, &wr}) {
            const Vec3 hub = hub_body(c, *w);
            s.cylinder_y(Vec3(hub.x, 0.86f, hub.z), w->radius + 0.05f, 0.16f, 0.02f, Op::SmoothSubtract, 0.03f, true);
        }
        s.ellipsoid(Vec3(0.95f, 0.90f, 0.80f), Vec3(0.07f, 0.10f, 0.06f), {}, Op::Union, 0.0f, true);    // retrovisores
        capsule(s, Vec3(1.00f, 0.66f, 0.66f), Vec3(0.95f, 0.88f, 0.79f), 0.022f, true);
    }
    s.end_group();
    add_axle(c, "Ruedas delanteras", Component::FrontWheels, wf);
    add_axle(c, "Ruedas traseras", Component::RearWheels, wr);
    finish_car(b, wb);
}

} // namespace

void register_objects(std::vector<Entry>& out) {
    {
        Info I = obj_info("sphere", "Esfera (D = 1 m)",
            "Esfera lisa en aire libre: caso clásico de validación. Cd ≈ 0.47 en régimen subcrítico (Re < 2·10⁵) "
            "y ≈ 0.1-0.2 tras la crisis de resistencia; una red gruesa con LES suele dar valores intermedios.",
            Kind::Body, P_Height, 1.0f, 0.7853982f, false, 100.0f, 0.0f, 0.37f, "Cd = 0.47 subcrítico (Hoerner)");
        I.default_height_mm = 1500.0f;
        out.push_back({I, &build_sphere});
    }
    {
        Info I = obj_info("cylinder", "Cilindro finito transversal (D = 0.5 m, L = 2 m)",
            "Cilindro de extremos planos con el eje perpendicular al flujo (L/D = 4). Muestra la calle de vórtices "
            "de Kármán y el efecto de los extremos libres (Cd menor que el del cilindro infinito, ≈ 1.2).",
            Kind::Body, P_Height | P_Yaw, 2.0f, 1.0f, false, 100.0f, 0.0f, 0.70f, "Cd ≈ 0.7 para L/D = 4 subcrítico (Hoerner)");
        I.default_height_mm = 1500.0f;
        out.push_back({I, &build_cylinder});
    }
    {
        Info I = obj_info("cube", "Cubo (1 m)",
            "Cubo de aristas vivas con una cara al flujo: separación fija en las aristas, casi independiente del "
            "Reynolds. Con guiñada se estudia la variación de la resistencia con el ángulo.",
            Kind::Body, P_Height | P_Yaw, 1.0f, 1.0f, false, 100.0f, 0.0f, 1.05f, "Cd ≈ 1.05 (Hoerner)");
        I.default_height_mm = 1500.0f;
        out.push_back({I, &build_cube});
    }
    {
        Info I = obj_info("ahmed_25", "Cuerpo de Ahmed (luneta 25°)",
            "Cuerpo de referencia de automoción (Ahmed 1984): 1044 × 389 × 288 mm, frontal redondeado y luneta de "
            "25°, sobre 4 patas a 50 mm del suelo. Con 25° el flujo se reengancha en la luneta y aparecen dos "
            "fuertes vórtices longitudinales: el caso de máxima resistencia (Cd ≈ 0.285).",
            Kind::Body, P_Yaw, 1.044f, 0.1120f, true, 144.0f, -0.034f, 0.0319f, "Ahmed et al. 1984 (Cd 0.285), Cl ≈ 0.3 aprox.");
        out.push_back({I, &build_ahmed});
    }
    {
        Info I = obj_info("naca0012_wing", "Ala NACA 0012 (c = 1 m, b = 3 m)",
            "Ala rectangular simétrica de alargamiento 3 en aire libre. Útil para ver la sustentación que crece "
            "con el ángulo de ataque, los torbellinos de punta y la resistencia inducida. Perfil 2D: Cl ≈ 0.11/°.",
            Kind::Wing, P_Aoa | P_Height, 3.0f, 3.0f, false, 150.0f, -1.05f, 0.074f,
            "teoría de ala finita (Helmbold) a 6°");
        I.default_aoa_deg = 6.0f; I.default_height_mm = 1500.0f;
        out.push_back({I, &build_naca0012});
    }
    {
        Info I = obj_info("naca4412_wing", "Ala NACA 4412 (c = 1 m, b = 3 m)",
            "Ala rectangular con perfil combado 4412 (alargamiento 3): sustentación a 0° (ángulo de sustentación "
            "nula ≈ -4°) y entrada en pérdida gradual desde el borde de salida hacia 14-16°.",
            Kind::Wing, P_Aoa | P_Height, 3.0f, 3.0f, false, 150.0f, -1.41f, 0.11f, "teoría de ala finita a 4°");
        I.default_aoa_deg = 4.0f; I.default_height_mm = 1500.0f;
        out.push_back({I, &build_naca4412});
    }
    {
        Info I = obj_info("f1_wing_ge", "Ala F1 invertida de 2 elementos en efecto suelo",
            "Plano principal + flap invertidos con endplates a una altura h del suelo móvil. Al bajar h la carga "
            "crece (efecto Venturi bajo el ala) hasta un máximo cerca de h/c ≈ 0.1; más abajo la capa límite del "
            "intradós se separa y la carga cae. Ángulo en convención F1 (+ = más carga); flap y ranura ajustables.",
            Kind::Wing, P_Aoa | P_Height | P_FlapGap | P_FrontFlap, 1.2f, 0.90f, true, 180.0f, 2.4f, 0.20f,
            "orden de magnitud de Zerihan & Zhang (2 elementos, h/c ≈ 0.2)");
        I.default_aoa_deg = 4.0f; I.default_height_mm = 100.0f; I.default_flap_gap_mm = 25.0f;
        I.default_front_flap_deg = 20.0f;
        out.push_back({I, &build_f1_wing_ge});
    }
    {
        Info I = obj_info("airfoil_2d", "Perfil NACA 4412 de envergadura completa (pseudo-2D)",
            "Perfil que atraviesa todo el ancho del túnel (sin puntas): aproxima el flujo bidimensional para "
            "estudiar la curva Cl-α y la entrada en pérdida. La app debe ajustar el ancho del dominio a la "
            "envergadura del modelo (spans_domain) y usar laterales de deslizamiento o periódicos.",
            Kind::Wing, P_Aoa | P_Height, 1.0f, 1.0f, false, 150.0f, -0.88f, 0.008f, "Abbott & von Doenhoff, Re = 3·10⁶, 4°");
        I.default_aoa_deg = 4.0f; I.default_height_mm = 2000.0f; I.spans_domain = true;
        out.push_back({I, &build_airfoil2d});
    }
    {
        Info I = obj_info("road_car", "Turismo fastback (tipo DrivAer)",
            "Turismo genérico de 4.6 m con zaga fastback, bajos lisos y ruedas girando, inspirado en el modelo "
            "DrivAer (TU München). Sirve de comparación: resistencia muy baja frente a un F1 y ligera sustentación.",
            Kind::Body, P_RideHeight | P_Yaw | P_Wheels, 4.61f, 2.16f, true, 140.0f, -0.2f, 0.54f, "DrivAer fastback Cd ≈ 0.25");
        I.wheelbase_m = 2.786f; I.default_ride_front_mm = 150.0f; I.default_ride_rear_mm = 150.0f;
        out.push_back({I, &build_road_car});
    }
}

} // namespace cfd::models::detail
