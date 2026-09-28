// ============================================================================
//  models/model.hpp — catálogo de modelos paramétricos (contrato público).
//
//  Cada modelo construye una sdf::Scene en ESPACIO MODELO (metros):
//    * suelo en z = 0, aire hacia +X, el vehículo mira hacia -X,
//    * coches: eje delantero en x = 0 (punto de contacto), eje trasero en
//      x = +batalla, simétricos respecto a y = 0.
//  La altura/inclinación (rake) y la guiñada se aplican con los marcos
//  (Scene::set_frames): la carrocería (Frame::Body) sube y cabecea, las
//  ruedas (Frame::Wheels) se quedan en el suelo.
//
//  Implementación: models/registry.cpp (catálogo), models/f1_*.cpp (épocas F1),
//  models/objects.cpp (cuerpos y alas de referencia).
// ============================================================================
#pragma once

#include "../geom/sdf.hpp"
#include <string>
#include <vector>

namespace cfd::models {

enum class Kind : u8 { F1Car, Wing, Body };

// Qué parámetros tienen sentido para el modelo (la UI sólo muestra esos).
enum ParamBit : u32 {
    P_RideHeight = 1u << 0,   // ride_front_mm / ride_rear_mm
    P_FrontFlap  = 1u << 1,
    P_RearFlap   = 1u << 2,
    P_Drs        = 1u << 3,   // DRS (2011-2025) o modo X de aero activa (2026)
    P_Yaw        = 1u << 4,
    P_Aoa        = 1u << 5,   // alas / perfiles
    P_Height     = 1u << 6,   // altura sobre el suelo de un ala (efecto suelo)
    P_FlapGap    = 1u << 7,
    P_Wheels     = 1u << 8,   // ruedas girando sí/no
};

struct Params {
    float yaw_deg = 0.0f;
    // Coche
    float ride_front_mm = -1.0f;   // < 0 → valor por defecto del modelo
    float ride_rear_mm = -1.0f;
    float front_flap_deg = 0.0f;   // incremento sobre el ángulo de flap por defecto de la época
    float rear_flap_deg = 0.0f;    //   (+ = flap más cerrado/inclinado = más carga y más resistencia)
                                   // En el ala de 2 elementos en efecto suelo, front_flap_deg = flap.
    bool drs_open = false;         // 2026: modo X (alerones delantero y trasero abiertos)
    bool wheels_rotating = true;
    // Alas / objetos
    float aoa_deg = -1000.0f;      // < -900 → valor por defecto del modelo
    float height_mm = -1.0f;       // altura del punto más bajo del ala sobre el suelo (< 0 → defecto)
    float flap_gap_mm = -1.0f;
};

struct Info {
    std::string id;           // "f1_2022", "naca0012", ...
    std::string name;         // "F1 2022 — efecto suelo (Venturi)"
    std::string era;          // "2022-2025"
    std::string description;  // 2-4 frases en español: qué caracteriza esta época aerodinámicamente
    Kind kind = Kind::Body;
    u32 param_mask = 0;
    float ref_length_m = 1.0f;     // longitud característica (dimensiona el dominio)
    float ref_area_m2 = 1.0f;      // área de referencia para Cl/Cd (frontal en coches, planta en alas)
    float wheelbase_m = 0.0f;      // 0 si no es coche
    float default_ride_front_mm = 0.0f, default_ride_rear_mm = 0.0f;
    float default_aoa_deg = 0.0f, default_height_mm = 0.0f, default_flap_gap_mm = 0.0f;
    float default_speed_kmh = 250.0f;
    bool needs_ground = true;      // los coches y alas en efecto suelo sí; esfera/ala libre no
    // Valores de referencia reales aproximados (para comparar en la UI; 0 = desconocido):
    float ref_ClA = 0.0f, ref_CdA = 0.0f;   // SCz y SCx típicos en m² (carga positiva = hacia abajo)
    // --- Extensiones compatibles (módulo models) -----------------------------------------
    float reg_width_m = 0.0f;      // anchura máxima reglamentaria de la época (0 = libre / no aplica)
    float default_front_flap_deg = 0.0f;   // ángulo absoluto de flap de la época (informativo, + = más carga)
    float default_rear_flap_deg = 0.0f;
    // true → el modelo debe atravesar TODO el ancho (Y) del túnel (perfil pseudo-2D): la app
    // debe ajustar ny·dx a la extensión Y de bounds_m y usar laterales de deslizamiento/periódicos.
    // En estos modelos bounds_m.y es EXACTAMENTE ese ancho y la geometría sobresale (0.1 m por
    // lado): la sección es sólida hasta la pared sea cual sea el origen de la red.
    bool spans_domain = false;
    const char* ref_source = "";   // de dónde salen ref_ClA/ref_CdA (texto corto, español)
};

struct Built {
    sdf::Scene scene;              // marcos ya aplicados según Params
    Info info;
    Params params;                 // parámetros efectivos (con defectos resueltos)
    Aabb bounds_m;                 // AABB en espacio modelo (= scene.bounds(); con spans_domain,
                                   //   en Y = ancho exacto del dominio, la geometría sobresale)
    Vec3 front_axle_m{0, 0, 0};    // puntos de contacto de los ejes (para el balance aero)
    Vec3 rear_axle_m{0, 0, 0};
    Vec3 moment_ref_m{0, 0, 0};    // referencia para momentos: coches = centro entre ejes a nivel del
                                   //   suelo; alas = 25% de la cuerda (principal); cuerpos = centro
};

// Catálogo global (orden estable: primero F1 cronológico, luego objetos de referencia).
int count();
const Info& info(int index);
int find(const std::string& id);   // -1 si no existe
Built build(int index, const Params& params);

// Utilidad común: marco de carrocería para un coche con alturas del plano de referencia
// (plank) en el eje delantero/trasero y guiñada (rad) alrededor del centro entre ejes.
Xform car_body_frame(float ride_front_m, float ride_rear_m, float wheelbase_m, float yaw_rad);
Xform car_wheel_frame(float wheelbase_m, float yaw_rad);

// --- Extensiones compatibles ---------------------------------------------------------------
// Parámetros efectivos (defectos del modelo resueltos) sin construir la escena.
Params resolve_params(int index, const Params& params);
// ¿Cambian la geometría dos juegos de parámetros? (para no re-voxelizar sin necesidad).
bool same_geometry(int index, const Params& a, const Params& b);
const char* kind_name(Kind k);     // "Coche F1", "Ala", "Cuerpo"

} // namespace cfd::models
