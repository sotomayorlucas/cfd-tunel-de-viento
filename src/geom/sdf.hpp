// ============================================================================
//  geom/sdf.hpp — escena de funciones de distancia con signo (SDF).
//
//  Representación única de la geometría: la usan el voxelizador (celdas
//  sólidas del LBM), el mallador (marching cubes para el render) y los
//  modelos (coches F1 por reglamento, alas, cuerpos de referencia).
//
//  Espacio "modelo": metros, suelo en z = 0, el aire fluye hacia +X,
//  el vehículo mira hacia -X, Y lateral (+Y = derecha del piloto).
//
//  Estructura:  Escena = unión de GRUPOS.  Grupo = secuencia CSG de PRIMITIVAS
//  (unión / unión suave / resta / resta suave / intersección), evaluada en orden.
//  Cada grupo tiene: componente aerodinámico (para desglosar fuerzas), marco
//  (carrocería, ruedas o fijo), movimiento rígido (ruedas que giran, suelo
//  móvil) y AABB en espacio modelo para descartar rápido.
//  El id de grupo (1..254) es lo que el voxelizador guarda por celda sólida.
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include <span>
#include <string>
#include <vector>

namespace cfd::sdf {

enum class PrimType : u8 {
    Sphere,        // p0 = radio
    Ellipsoid,     // p0..2 = semiejes (aprox. de iq, cota)
    RoundBox,      // p0..2 = semiejes, p3 = radio de redondeo (incluido en los semiejes)
    TaperBox,      // caja redondeada que varía a lo largo de X local ∈ [-p0, p0]:
                   //   p1,p2 = semiancho Y / semialto Z en x=-p0;  p3,p4 = idem en x=+p0
                   //   p5,p6 = centro Z en x=-p0 / x=+p0 (permite morros inclinados)
                   //   p7 = redondeo
    RoundCone,     // de a=(p0,p1,p2) a b=(p3,p4,p5), radios p6 (en a) y p7 (en b). Cápsula si iguales.
    Cylinder,      // eje Y local: p0 = radio, p1 = semiancho (Y), p2 = redondeo de aristas
    Torus,         // eje Y local: p0 = radio mayor, p1 = radio menor
    Wing,          // perfil alar extruido en Y local ∈ [0, p2]:
                   //   p0 = cuerda raíz (y=0), p1 = cuerda punta (y=span), p2 = envergadura
                   //   p3 = flecha (x del borde de ataque en la punta), p4 = diedro (z del BA en la punta)
                   //   p5 = torsión en la punta (rad, sobre el 25% de cuerda)
                   //   p6 = curvatura z cuadrática (z += p6*s²), p7 = curvatura x cuadrática (x += p7*s²)
                   //   data = perfil (polígono en coordenadas de cuerda unitaria, BA en (0,0), BS en (1,0))
    ExtrudePoly,   // polígono en plano (X,Z) local extruido en Y: p0 = semiancho, data = polígono (m)
    ConvexPlanes,  // intersección de semiespacios: data = n planos (nx,ny,nz,d): dist = max(n·p - d)
};

enum class Op : u8 {
    Union,         // min(d, e)
    SmoothUnion,   // smin(d, e, k)
    Subtract,      // max(d, -e)
    SmoothSubtract,// smax(d, -e, k)
    Intersect,     // max(d, e)
};

enum class Frame : u8 {
    Body,     // carrocería: sigue altura/inclinación (rake) y guiñada
    Wheels,   // ruedas: sólo guiñada (se quedan apoyadas en el suelo)
    Fixed,    // espacio modelo directo (objetos de prueba, suelo)
};

// Movimiento rígido NORMALIZADO por la velocidad libre U∞:
//   u_pared(x) / U∞ = v_hat + omega_hat × (x - center)       (x, center en metros de espacio modelo)
// Rueda de radio R rodando (túnel con cinta):  omega_hat = (0, -1/R, 0), center = eje.
// Cinta del suelo: v_hat = (1, 0, 0).
struct RigidMotion {
    Vec3 v_hat{0, 0, 0};
    Vec3 omega_hat{0, 0, 0};
    Vec3 center{0, 0, 0};
    constexpr bool moving() const { return length2(v_hat) + length2(omega_hat) > 0.0f; }
    constexpr Vec3 velocity_at(Vec3 x) const { return v_hat + cross(omega_hat, x - center); }
};

// Componentes aerodinámicos (para el desglose de carga/resistencia).
enum class Component : u8 {
    Body = 0, FrontWing, RearWing, BeamWing, Floor, Diffuser, Sidepods, Nose,
    FrontWheels, RearWheels, Suspension, Halo, Object, WingMain, WingFlap, Endplate,
    Count
};
const char* component_name(Component c);   // nombre en español para la UI

// Referencia a datos extra en Scene::pool (fuera de Scene: GCC no permite usar como
// argumento por defecto un tipo anidado con inicializadores antes de completar la clase).
struct DataHandle { u32 off = 0, n = 0; };

struct Prim {
    PrimType type = PrimType::Sphere;
    Op op = Op::Union;
    bool mirror_y = false;   // evaluar en (x, |y|, z) del marco → pieza simétrica izquierda/derecha
    float k = 0.0f;          // radio de mezcla suave (m) para SmoothUnion/SmoothSubtract
    Xform xf;                // local → marco del grupo
    Xform inv;               // marco → local (cacheado)
    float p[8] = {};
    u32 data_off = 0, data_n = 0;   // datos extra en Scene::pool (aristas precalculadas / planos)
    Aabb box_frame;          // AABB en el marco del grupo (ya incluye el espejo si aplica)
    Aabb box_model;          // AABB en espacio modelo (recalculado en set_frames)
};

struct Group {
    std::string name;
    Component component = Component::Body;
    Frame frame = Frame::Body;
    RigidMotion motion;      // movimiento en espacio MODELO (lo recalcula set_frames para ruedas con guiñada)
    RigidMotion motion_frame;// movimiento tal como se definió (en el marco)
    u32 first = 0, count = 0;// rango en Scene::prims
    Aabb box_model;
};

class Scene {
public:
    // ---- Perfiles y polígonos (devuelven un "handle" = índice en pool) -------------
    using Handle = DataHandle;
    // NACA 4 dígitos: m = combadura máx (p.ej. 0.06), p = posición (0.4), t = espesor (0.12).
    // inverted=true → combadura hacia abajo (ala de F1 que genera carga). n puntos por cara.
    Handle naca4(float m, float p, float t, bool inverted, int n = 40);
    Handle profile(std::span<const Vec2> pts);    // polígono cerrado en coords de cuerda unitaria
    Handle polygon(std::span<const Vec2> pts);    // polígono cerrado en metros (para ExtrudePoly)
    Handle planes(std::span<const Vec4> planes);  // (nx,ny,nz,d) — normales hacia fuera

    // ---- Grupos ------------------------------------------------------------------
    int begin_group(const char* name, Component c, Frame f = Frame::Body, RigidMotion m = {});
    void end_group();

    // ---- Primitivas (se añaden al grupo abierto) ------------------------------------
    // Todas aceptan una transformación local→marco y la operación CSG.
    Prim& add(PrimType t, const Xform& xf, std::initializer_list<float> params, Op op = Op::Union,
              float k = 0.0f, bool mirror_y = false, Handle data = {});
    // Atajos frecuentes:
    Prim& sphere(Vec3 c, float r, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& ellipsoid(Vec3 c, Vec3 radii, const Mat3& R = {}, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& round_box(Vec3 c, Vec3 half, float round, const Mat3& R = {}, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& taper_box(Vec3 c, float half_len, Vec2 half_yz_front, Vec2 half_yz_rear, float zc_front, float zc_rear,
                    float round, const Mat3& R = {}, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& round_cone(Vec3 a, Vec3 b, float ra, float rb, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& cylinder_y(Vec3 c, float radius, float half_width, float round, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& torus_y(Vec3 c, float R, float r, Op op = Op::Union, float k = 0, bool mirror = false);
    // Ala: `le_root` = borde de ataque en la raíz, se extiende hacia +Y (o -Y si span < 0 → se usa espejo interno).
    // aoa_rad: ángulo de ataque geométrico (positivo = borde de ataque ARRIBA, rotación sobre el BA en Y).
    Prim& wing(Handle prof, Vec3 le_root, float chord_root, float chord_tip, float span, float aoa_rad,
               float sweep = 0, float dihedral = 0, float twist = 0, float zcurve = 0, float xcurve = 0,
               Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& extrude_xz(Handle poly, float y_center, float half_width, Op op = Op::Union, float k = 0, bool mirror = false);
    Prim& convex(Handle planes, Op op = Op::Union, float k = 0, bool mirror = false);

    // ---- Marcos -----------------------------------------------------------------------
    // body: marco carrocería → modelo;  wheels: marco ruedas → modelo.  Recalcula AABBs/movimientos.
    void set_frames(const Xform& body, const Xform& wheels);
    const Xform& body_frame() const { return frame_[0]; }
    const Xform& wheel_frame() const { return frame_[1]; }

    // ---- Evaluación (hilo-segura, const) ----------------------------------------------
    // Distancia con signo en espacio modelo (m). group_id ← 1 + índice del grupo más cercano (0 si vacío).
    float eval(Vec3 p, int* group_id = nullptr) const;
    // Distancia de un único grupo (índice 0-based) en espacio modelo.
    float eval_group(int group_index, Vec3 p) const;
    Vec3 normal(Vec3 p, float h = 1e-3f) const;       // gradiente normalizado (tetraedro, 4 evals)
    Aabb bounds() const;                               // unión de AABB de grupos (espacio modelo)

    const std::vector<Group>& groups() const { return groups_; }
    const std::vector<Prim>& prims() const { return prims_; }
    std::vector<Group>& groups() { return groups_; }
    usize group_count() const { return groups_.size(); }
    void clear();

    std::vector<float> pool;   // datos extra (aristas de polígonos precalculadas, planos)

private:
    float eval_prim(const Prim& pr, Vec3 pf) const;   // pf = punto en el marco del grupo
    Aabb local_bounds(const Prim& pr) const;
    void finalize_prim(Prim& pr, Frame f);
    std::vector<Group> groups_;
    std::vector<Prim> prims_;
    Xform frame_[2];
    int open_group_ = -1;
};

// ---- Primitivas sueltas (inline, reutilizables en tests/otros módulos) -------------------
CFD_INLINE float smin(float a, float b, float k) {
    if (k <= 0.0f) return min_(a, b);
    const float h = max_(k - std::fabs(a - b), 0.0f) / k;
    return min_(a, b) - h * h * k * 0.25f;
}
CFD_INLINE float smax(float a, float b, float k) { return -smin(-a, -b, k); }
// Distancia de extrusión (iq): combina distancia 2D con distancia al intervalo en el eje extruido.
CFD_INLINE float extrude(float d2, float dh) {
    const float mx = max_(d2, dh);
    const float a = max_(d2, 0.0f), b = max_(dh, 0.0f);
    return min_(mx, 0.0f) + std::sqrt(a * a + b * b);
}
// Polígono 2D con aristas precalculadas: por arista (vx, vy, ex, ey, 1/|e|²).
float sd_polygon_edges(const float* edges, u32 n, Vec2 p);

} // namespace cfd::sdf
