// ============================================================================
//  models/f1_common.hpp — utilidades INTERNAS del módulo models (no es contrato).
//
//  Piezas reutilizables para construir coches y objetos como sdf::Scene:
//    * caché de perfiles NACA por construcción (un handle por perfil distinto),
//    * colocación 2D de alas multi-elemento con ranura (gap) exacta, calculada
//      con la distancia polígono-polígono de los perfiles reales,
//    * DRS / modo X: giro del flap alrededor de su borde de salida,
//    * ruedas con movimiento rígido (rodadura sobre la cinta), suspensión,
//      halo, cajas ahusadas por extremos, placas extruidas...
//
//  Convenciones (marco carrocería de un coche): x = 0 eje delantero, x = batalla
//  eje trasero, z = 0 plano de referencia (plank), +Y lateral; el coche mira a -X.
//  Ángulos de ala en la convención de sdf::Scene::wing (+ = borde de ataque ARRIBA):
//  un ala de F1 que genera carga lleva ángulos NEGATIVOS (borde de salida arriba).
// ============================================================================
#pragma once

#include "model.hpp"
#include <array>
#include <span>

namespace cfd::models::detail {

using sdf::Component;
using sdf::Frame;
using sdf::Op;
using sdf::Prim;
using sdf::RigidMotion;
using sdf::Scene;
using Handle = sdf::DataHandle;

// Puntos por cara de los perfiles NACA. 40 (valor por defecto de Scene::naca4) es excesivo
// para redes de 2-4 cm: con 14 (espaciado coseno) el polígono se separa del perfil fino
// ≤ 1.0 mm en una cuerda de 0.35 m (medido; 0.03·dx a 3 cm; con 40: 0.13 mm) y la distancia
// al polígono (O(aristas)) es más barata.
// Medido (tests/test_models_bench.cpp, mínimo de 5, P-core fijado): -7..-23% de coste de eval
// en la región de los alerones y construcción de los coches 5× más rápida (colocación de
// ranuras O(n²)). Se puede cambiar con -DCFD_MODELS_PROFILE_PTS=n.
#ifndef CFD_MODELS_PROFILE_PTS
#define CFD_MODELS_PROFILE_PTS 14
#endif
inline constexpr int k_profile_pts = CFD_MODELS_PROFILE_PTS;

// Resolución mínima de diseño: todo elemento debe sobrevivir a dx = 3 cm.
//  * Una placa inclinada θ sólo queda 4-conexa (sin fugas para las velocidades diagonales
//    de D3Q19) si su espesor vertical ≥ dx·(1 + tan θ): con el engrosamiento 0.12·dx del
//    voxelizador eso pide ≥ 3.5 cm en flaps a 30°.
//  * Una ranura sólo deja un canal de fluido conexo si su anchura ≥ ~1.3·dx + 2·0.12·dx.
inline constexpr float k_min_thick = 0.035f;   // espesor mínimo de placas (m)
inline constexpr float k_min_gap = 0.045f;     // ranura mínima entre elementos (m)
inline constexpr float k_min_elem_thick = 0.042f;   // espesor mínimo de un perfil desde el 12% de cuerda
// (barrido medido a dx = 3 cm: 2.4 cm → ~60% de cortes con fuga, 3.0 cm → 5-30%, 3.6 cm → 0; 4.2 cm = margen)
// Solape en el plano de simetría de las extrusiones espejadas que acaban justo en y = 0 (alas con
// raíz en y0 = 0, láminas de difusor): sin él las dos mitades sólo se tocan y el SDF vale 0 (no
// < 0) en TODO su interior sobre el plano y = 0 → lámina "fuera" para el mallador (d < 0) o una
// voxelización sin engrosamiento. Con 1 cm el interior del plano queda a ≤ -1 cm.
inline constexpr float k_sym_overlap = 0.01f;

// NACA 4 dígitos con espesor mínimo `tf` (fracción de cuerda) desde el 12% de la cuerda hasta
// el borde de salida (BS romo), con rampa parabólica hacia el BA (morro redondo): el perfil no
// se "deshilacha" al voxelizar en redes gruesas. n = puntos por cara.
Handle naca4_floor(Scene& s, float m, float p, float t, float tf, bool inverted, int n = k_profile_pts);

// ---- Caché de perfiles ------------------------------------------------------------------
struct ProfileCache {
    struct Entry { float m, p, t, tf; bool inv; Handle h; };
    std::array<Entry, 24> e{};
    int n = 0;
    Handle get(Scene& s, float m, float p, float t, bool inverted, float tf = 0.0f);
};

// ---- Elementos de ala en 2D (plano XZ de la sección raíz) -------------------------------
struct ElemPose {
    float x = 0, z = 0;      // borde de ataque (m)
    float aoa = 0;           // rad, + = BA arriba (F1: negativo)
    float chord = 0.3f;      // m
    Handle prof{};
};
CFD_INLINE Vec2 elem_te(const ElemPose& e) {
    return {e.x + e.chord * std::cos(e.aoa), e.z - e.chord * std::sin(e.aoa)};
}
// Distancia con signo de un punto (x,z) al perfil del elemento (m).
float elem_dist(const Scene& s, const ElemPose& e, Vec2 p);
// Distancia mínima (con signo) entre dos elementos (vértice-arista en ambos sentidos: exacta
// para polígonos disjuntos).
float elem_elem_dist(const Scene& s, const ElemPose& a, const ElemPose& b);
// Coloca el siguiente elemento: BA a `overlap`·cuerda_prev por delante del BS del anterior y
// a la altura que deja una ranura mínima EXACTA = gap (bisección).
ElemPose place_next(const Scene& s, const ElemPose& prev, Handle prof, float chord, float aoa, float gap,
                    float overlap);
// Gira el elemento alrededor de su borde de salida hasta el ángulo `aoa_new` (DRS / modo X).
ElemPose rotate_about_te(const ElemPose& e, float aoa_new);
// DRS / modo X: ABRE el flap (gira sobre su BS) hasta `aoa_open` sólo si está más cerrado
// (aoa < aoa_open; convención sdf: F1 = ángulos negativos). Si el usuario ya lo abrió más con
// un incremento de flap negativo, no se toca: girarlo "hasta" aoa_open le AÑADIRÍA incidencia y
// bajaría su BA sobre el elemento anterior cerrando la ranura (medido: 45 → 12.6 mm en el modo X
// de 2026 con flap -20°; el DRS de 2014 con flap -20° pasaba el flap de -4° a -6°).
CFD_INLINE ElemPose open_flap(const ElemPose& e, float aoa_open) {
    return e.aoa < aoa_open ? rotate_about_te(e, aoa_open) : e;
}
// Caja 2D (x0,z0,x1,z1) de los vértices reales del perfil colocado.
void elem_box(const Scene& s, const ElemPose& e, float& x0, float& z0, float& x1, float& z1);

// Ala multi-elemento: poses + parámetros de planta comunes (misma flecha/diedro/curvatura
// para todos → la ranura se conserva a lo largo de la envergadura).
struct WingPlan {
    float y0 = 0;            // raíz (0 = continua a través del plano de simetría)
    float span = 0.5f;       // semienvergadura desde y0 (con espejo)
    float sweep = 0, dihedral = 0, zcurve = 0, xcurve = 0;
    float taper = 1.0f;      // cuerda punta / raíz
};
void add_elements(Scene& s, std::span<const ElemPose> el, const WingPlan& w, Op op = Op::Union, float k = 0.0f);
// Desplazamiento (dx, dz) del BA en la punta respecto a la raíz (para colocar endplates).
CFD_INLINE Vec2 plan_tip_offset(const WingPlan& w) { return {w.sweep + w.xcurve, w.dihedral + w.zcurve}; }

// ---- Primitivas cómodas -------------------------------------------------------------------
// Caja ahusada definida por extremos en X: (medio ancho, fondo, techo) en x0 y en x1.
Prim& tbox(Scene& s, float x0, float x1, float hy0, float zlo0, float zhi0, float hy1, float zlo1, float zhi1,
           float round, float yc = 0.0f, bool mirror = false, Op op = Op::Union, float k = 0.0f);
// Caja redondeada por esquinas.
Prim& rbox(Scene& s, Vec3 lo, Vec3 hi, float round, bool mirror = false, Op op = Op::Union, float k = 0.0f);
Prim& capsule(Scene& s, Vec3 a, Vec3 b, float r, bool mirror = false, Op op = Op::Union, float k = 0.0f);
// Placa vertical (plano XZ) extruida en Y alrededor de yc con medio espesor ht.
Prim& plate_xz(Scene& s, std::span<const Vec2> poly, float yc, float ht, bool mirror = true, Op op = Op::Union,
               float k = 0.0f);
// Lámina gruesa que sigue una polilínea inferior z(x) (piso/túnel/difusor), extruida en Y.
Prim& sheet_xz(Scene& s, std::span<const Vec2> bottom, float thick, float yc, float hy, bool mirror = true,
               Op op = Op::Union, float k = 0.0f);

// ---- Coche ----------------------------------------------------------------------------------
struct CarCtx {
    Scene& s;
    const Params& P;         // parámetros efectivos (defectos resueltos)
    float wb;                // batalla (m)
    float rf, rr;            // alturas del plank en eje del./tras. (m)
    ProfileCache pc;
    Xform body0;             // marco carrocería con guiñada 0 (para convertir puntos)
    CarCtx(Scene& sc, const Params& p, float wheelbase);
    Handle prof(float m, float p, float t, bool inverted = true) { return pc.get(s, m, p, t, inverted); }
    // Perfil invertido de F1 para un elemento de cuerda `chord`: espesor mínimo absoluto
    // k_min_elem_thick en la parte trasera (ver naca4_floor).
    Handle eprof(float m, float p, float t, float chord) {
        return pc.get(s, m, p, t, true, min_(t, k_min_elem_thick / chord));
    }
    // Punto del espacio modelo (guiñada 0) → marco carrocería (para unir suspensión y bujes).
    Vec3 to_body(Vec3 pm) const { return body0.inverse_apply(pm); }
    float flap_front(float base_deg) const { return -(base_deg + P.front_flap_deg) * k_deg2rad; }
    float flap_rear(float base_deg) const { return -(base_deg + P.rear_flap_deg) * k_deg2rad; }
};

struct WheelSpec {
    float x = 0;             // eje (m, marco ruedas)
    float half_track = 0.8f; // semivía al centro de la banda
    float radius = 0.33f;
    float width = 0.35f;
    float rim_radius = 0.2f; // radio del hueco de llanta (0 = sin hueco)
    float rim_depth = 0.04f; // profundidad del hueco exterior (tapacubos 2022: ~0.015)
    float round = 0.07f;     // redondeo del flanco
    float squash = 0.005f;   // penetración en el suelo (sin rendija bajo el neumático)
};
// Añade un grupo con las dos ruedas del eje (espejo), marco Wheels y rodadura si procede.
int add_axle(CarCtx& c, const char* name, Component comp, const WheelSpec& w);
// Centro del buje en marco carrocería.
Vec3 hub_body(const CarCtx& c, const WheelSpec& w, float dy = 0.0f);
// Suspensión de un eje: brazos superior e inferior (+ bieleta) desde el chasis al buje.
void add_suspension_arms(CarCtx& c, const WheelSpec& w, float y_in, float z_up_in, float z_lo_in, float r,
                         bool pushrod);
// Halo (2018+): pilar central + arco en U hasta los anclajes traseros.
void add_halo(CarCtx& c, float x_front, float x_rear, float z_top, float half_w, float z_mount);
// Ala multi-elemento genérica en coordenadas de carrocería con endplates opcionales.
struct EndplateSpec {
    bool enable = true;
    float thick = 0.035f;
    float x_pad_front = 0.03f, x_pad_rear = 0.04f;
    float z_pad_lo = 0.03f, z_pad_hi = 0.05f;
    float z_min = -1e9f;     // recorte inferior absoluto (p.ej. altura mínima del endplate)
    float z_down_to = 1e9f;  // prolongar hacia abajo hasta esta z (endplates traseros hasta el beam wing)
    float z_up_to = -1e9f;   // prolongar hacia arriba hasta esta z
    float round = 0.012f;
};
void add_endplates(CarCtx& c, std::span<const ElemPose> el, const WingPlan& w, const EndplateSpec& ep);
// Caja 2D de un conjunto de elementos (sección raíz).
void elems_box(const Scene& s, std::span<const ElemPose> el, float& x0, float& z0, float& x1, float& z1);
// Valla vertical interior en y = yc que abarca los elementos (flaps sólo exteriores 2009-2021).
void add_fence(CarCtx& c, std::span<const ElemPose> el, float yc, float pad = 0.02f);
// Pilones del morro al alerón delantero (morro alto): placas en y = ±yc entre z0 y z1.
void add_pylons(CarCtx& c, float x0, float x1, float z0, float z1, float yc);

// Fondo plano escalonado (1995-2021): plank + plano de referencia central y plano escalonado
// 50 mm más alto; difusor con rampa, paredes laterales y separador central.
struct FlatFloorSpec {
    float x0 = 0.45f, x1 = 2.6f;       // placa escalonada
    float hw = 0.62f;                  // semiancho de la placa escalonada
    float plank_x0 = 0.3f, plank_hw = 0.15f;
    float xd0 = 2.7f, xd1 = 3.6f;      // difusor
    float z_exit = 0.18f;              // altura de salida del difusor (sobre la referencia)
    float hwd = 0.50f;                 // semiancho del difusor
    float strake_y = 0.22f;            // separador interior (0 = sin él)
};
void add_flat_floor(CarCtx& c, const FlatFloorSpec& f);

// Finaliza un coche: marcos, ejes, referencia de momentos.
void finish_car(Built& b, float wheelbase);

} // namespace cfd::models::detail
