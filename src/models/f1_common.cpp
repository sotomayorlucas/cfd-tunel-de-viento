// ============================================================================
//  models/f1_common.cpp — piezas comunes de coches y objetos (ver f1_common.hpp).
// ============================================================================
#include "f1_common.hpp"

#include <cmath>

namespace cfd::models::detail {

Handle naca4_floor(Scene& s, float m, float p, float t, float tf, bool inverted, int n) {
    n = max_(n, 8);
    Vec2 up[65], lo[65];
    CFD_CHECK(n <= 64, "naca4_floor: n <= 64");
    for (int i = 0; i <= n; ++i) {
        const float x = 0.5f * (1.0f - std::cos(k_pi * static_cast<float>(i) / static_cast<float>(n)));   // espaciado coseno
        float yt = 5.0f * t * (0.2969f * std::sqrt(x) - 0.1260f * x - 0.3516f * x * x + 0.2843f * x * x * x - 0.1015f * x * x * x * x);
        // Suelo de espesor: pleno desde el 12% de cuerda, con rampa parabólica (√x) hacia el BA
        // → el morro sigue siendo redondo pero ya no hay un tramo delgado que se "rompa" en la red.
        if (tf > 0.0f) yt = max_(yt, 0.5f * tf * std::sqrt(min_(x / 0.12f, 1.0f)));
        float yc = 0, dyc = 0;
        if (m > 0 && p > 0 && p < 1) {
            if (x < p) { yc = m / (p * p) * (2 * p * x - x * x);                 dyc = 2 * m / (p * p) * (p - x); }
            else       { yc = m / sq(1 - p) * ((1 - 2 * p) + 2 * p * x - x * x); dyc = 2 * m / sq(1 - p) * (p - x); }
        }
        const float th = std::atan(dyc);
        up[i] = {x - yt * std::sin(th), yc + yt * std::cos(th)};
        lo[i] = {x + yt * std::sin(th), yc - yt * std::cos(th)};
    }
    // Extradós BS→BA, intradós BA→BS (el BA no se duplica; el BS romo tiene dos vértices).
    Vec2 poly[130];
    int k = 0;
    for (int i = n; i >= 0; --i) poly[k++] = up[i];
    for (int i = 1; i <= n; ++i) poly[k++] = lo[i];
    if (tf <= 0.0f) --k;                  // BS cerrado: el último punto del intradós coincide
    if (inverted) for (int i = 0; i < k; ++i) poly[i].y = -poly[i].y;
    return s.profile(std::span<const Vec2>(poly, static_cast<usize>(k)));
}

Handle ProfileCache::get(Scene& s, float m, float p, float t, bool inverted, float tf) {
    for (int i = 0; i < n; ++i) {
        const Entry& x = e[static_cast<usize>(i)];
        if (x.m == m && x.p == p && x.t == t && x.tf == tf && x.inv == inverted) return x.h;
    }
    const Handle h = naca4_floor(s, m, p, t, tf, inverted, k_profile_pts);
    if (n < static_cast<int>(e.size())) e[static_cast<usize>(n++)] = {m, p, t, tf, inverted, h};
    return h;
}

// ---------------------------------------------------------------------------------------
float elem_dist(const Scene& s, const ElemPose& e, Vec2 p) {
    const float dx = p.x - e.x, dz = p.y - e.z;
    const float c = std::cos(e.aoa), sn = std::sin(e.aoa);
    // local = R^T (p - BA), R = rot_y(aoa) restringida a (x,z)
    const float lx = c * dx - sn * dz, lz = sn * dx + c * dz;
    const float ic = 1.0f / e.chord;
    return sdf::sd_polygon_edges(s.pool.data() + e.prof.off, e.prof.n, Vec2(lx * ic, lz * ic)) * e.chord;
}

namespace {
template <class F>
void for_each_vertex(const Scene& s, const ElemPose& e, F&& f) {
    const float c = std::cos(e.aoa), sn = std::sin(e.aoa);
    const float* E = s.pool.data() + e.prof.off;
    for (u32 i = 0; i < e.prof.n; ++i) {
        const float u = E[5 * i] * e.chord, v = E[5 * i + 1] * e.chord;
        f(Vec2(e.x + u * c + v * sn, e.z - u * sn + v * c));
    }
}
} // namespace

float elem_elem_dist(const Scene& s, const ElemPose& a, const ElemPose& b) {
    float d = 1e30f;
    for_each_vertex(s, a, [&](Vec2 v) { d = min_(d, elem_dist(s, b, v)); });
    for_each_vertex(s, b, [&](Vec2 v) { d = min_(d, elem_dist(s, a, v)); });
    return d;
}

void elem_box(const Scene& s, const ElemPose& e, float& x0, float& z0, float& x1, float& z1) {
    x0 = z0 = 1e30f; x1 = z1 = -1e30f;
    for_each_vertex(s, e, [&](Vec2 v) { x0 = min_(x0, v.x); x1 = max_(x1, v.x); z0 = min_(z0, v.y); z1 = max_(z1, v.y); });
}

ElemPose place_next(const Scene& s, const ElemPose& prev, Handle prof, float chord, float aoa, float gap, float overlap) {
    const Vec2 te = elem_te(prev);
    ElemPose n{te.x - overlap * prev.chord, te.y, aoa, chord, prof};
    // f(h) = distancia mínima - gap, creciente en h (el elemento sube alejándose del BS previo).
    // Regula falsi de Illinois: convergencia superlineal (~6-8 evaluaciones de f, cada una
    // O(n²) sobre los vértices de ambos perfiles) frente a las 48 de una bisección.
    auto f = [&](float h) { n.z = te.y + h; return elem_elem_dist(s, prev, n) - gap; };
    float lo = 0.0f, hi = 0.6f, flo = f(lo), fhi = f(hi);
    for (int i = 0; i < 40 && flo > 0.0f; ++i) { hi = lo; fhi = flo; lo -= 0.02f; flo = f(lo); }   // horquilla con solape
    int side = 0;
    for (int i = 0; i < 40 && hi - lo > 1e-6f; ++i) {
        const float h = (lo * fhi - hi * flo) / (fhi - flo);
        const float fh = f(h);
        if (std::fabs(fh) < 1e-6f) { lo = hi = h; break; }
        if (fh < 0.0f) { lo = h; flo = fh; if (side == -1) fhi *= 0.5f; side = -1; }
        else           { hi = h; fhi = fh; if (side == +1) flo *= 0.5f; side = +1; }
    }
    n.z = te.y + hi;
    return n;
}

ElemPose rotate_about_te(const ElemPose& e, float aoa_new) {
    const Vec2 te = elem_te(e);
    ElemPose r = e;
    r.aoa = aoa_new;
    r.x = te.x - e.chord * std::cos(aoa_new);
    r.z = te.y + e.chord * std::sin(aoa_new);
    return r;
}

void add_elements(Scene& s, std::span<const ElemPose> el, const WingPlan& w, Op op, float k) {
    // Raíz en el plano de simetría: la mitad espejada empieza k_sym_overlap por debajo de y = 0
    // (misma punta; la ley de flecha/curvatura se estira ese 1-2% de envergadura: invisible).
    const float y0 = w.y0 > 0.0f ? w.y0 : -k_sym_overlap;
    const float span = w.span + (w.y0 - y0);
    for (const ElemPose& e : el)
        s.wing(e.prof, Vec3(e.x, y0, e.z), e.chord, e.chord * w.taper, span, e.aoa, w.sweep, w.dihedral, 0.0f,
               w.zcurve, w.xcurve, op, k, true);
}

// ---------------------------------------------------------------------------------------
Prim& tbox(Scene& s, float x0, float x1, float hy0, float zlo0, float zhi0, float hy1, float zlo1, float zhi1,
           float round, float yc, bool mirror, Op op, float k) {
    const float hl = 0.5f * (x1 - x0);
    return s.taper_box(Vec3(0.5f * (x0 + x1), yc, 0.0f), hl, Vec2(hy0, 0.5f * (zhi0 - zlo0)), Vec2(hy1, 0.5f * (zhi1 - zlo1)),
                       0.5f * (zlo0 + zhi0), 0.5f * (zlo1 + zhi1), round, Mat3{}, op, k, mirror);
}

Prim& rbox(Scene& s, Vec3 lo, Vec3 hi, float round, bool mirror, Op op, float k) {
    return s.round_box((lo + hi) * 0.5f, (hi - lo) * 0.5f, round, Mat3{}, op, k, mirror);
}

Prim& capsule(Scene& s, Vec3 a, Vec3 b, float r, bool mirror, Op op, float k) {
    return s.round_cone(a, b, r, r, op, k, mirror);
}

namespace {
// Extrusión espejada que acaba justo en y = 0: se prolonga k_sym_overlap al otro lado (ver
// k_sym_overlap). Misma cara exterior (yc + hy).
CFD_INLINE void sym_overlap(bool mirror, float& yc, float& hy) {
    if (mirror && std::fabs(yc - hy) < 1e-6f) { yc -= 0.5f * k_sym_overlap; hy += 0.5f * k_sym_overlap; }
}
} // namespace

Prim& plate_xz(Scene& s, std::span<const Vec2> poly, float yc, float ht, bool mirror, Op op, float k) {
    sym_overlap(mirror, yc, ht);
    return s.extrude_xz(s.polygon(poly), yc, ht, op, k, mirror);
}

Prim& sheet_xz(Scene& s, std::span<const Vec2> bottom, float thick, float yc, float hy, bool mirror, Op op, float k) {
    Vec2 pts[64];
    const usize n = bottom.size();
    CFD_CHECK(n >= 2 && n <= 32, "sheet_xz: 2..32 puntos");
    for (usize i = 0; i < n; ++i) pts[i] = bottom[i];
    for (usize i = 0; i < n; ++i) pts[n + i] = bottom[n - 1 - i] + Vec2(0.0f, thick);
    sym_overlap(mirror, yc, hy);
    return s.extrude_xz(s.polygon(std::span<const Vec2>(pts, 2 * n)), yc, hy, op, k, mirror);
}

// ---------------------------------------------------------------------------------------
CarCtx::CarCtx(Scene& sc, const Params& p, float wheelbase)
    : s(sc), P(p), wb(wheelbase), rf(p.ride_front_mm * 1e-3f), rr(p.ride_rear_mm * 1e-3f),
      body0(car_body_frame(p.ride_front_mm * 1e-3f, p.ride_rear_mm * 1e-3f, wheelbase, 0.0f)) {}

int add_axle(CarCtx& c, const char* name, Component comp, const WheelSpec& w) {
    const float zc = w.radius - w.squash;
    RigidMotion m{};
    if (c.P.wheels_rotating) { m.omega_hat = Vec3(0.0f, -1.0f / w.radius, 0.0f); m.center = Vec3(w.x, 0.0f, zc); }
    const int g = c.s.begin_group(name, comp, Frame::Wheels, m);
    c.s.cylinder_y(Vec3(w.x, w.half_track, zc), w.radius, 0.5f * w.width, w.round, Op::Union, 0.0f, true);
    if (w.rim_radius > 0.0f && w.rim_depth > 0.0f) {
        // Hueco de llanta en la cara exterior (espejo → ambos lados).
        const float d = w.rim_depth;
        c.s.cylinder_y(Vec3(w.x, w.half_track + 0.5f * w.width, zc), w.rim_radius, d, 0.01f, Op::Subtract, 0.0f, true);
        // buje
        c.s.cylinder_y(Vec3(w.x, w.half_track + 0.5f * w.width - d, zc), 0.35f * w.rim_radius, 0.5f * d, 0.01f,
                       Op::Union, 0.0f, true);
    }
    c.s.end_group();
    return g;
}

Vec3 hub_body(const CarCtx& c, const WheelSpec& w, float dy) {
    return c.to_body(Vec3(w.x, w.half_track + dy, w.radius - w.squash));
}

void add_suspension_arms(CarCtx& c, const WheelSpec& w, float y_in, float z_up_in, float z_lo_in, float r, bool pushrod) {
    // Punto de anclaje en la mangueta: ligeramente dentro de la cara interior de la rueda.
    const float y_out = w.half_track - 0.5f * w.width + 0.06f;
    const Vec3 hub = hub_body(c, w, y_out - w.half_track);
    const float zr = 0.55f * w.radius;
    const Vec3 up_o = hub + Vec3(0.0f, 0.0f, zr * 0.8f), lo_o = hub - Vec3(0.0f, 0.0f, zr * 0.75f);
    capsule(c.s, Vec3(hub.x - 0.10f, y_in, z_up_in), up_o, r, true);
    capsule(c.s, Vec3(hub.x + 0.10f, y_in, z_up_in), up_o, r, true);
    capsule(c.s, Vec3(hub.x - 0.12f, y_in, z_lo_in), lo_o, r, true);
    capsule(c.s, Vec3(hub.x + 0.12f, y_in, z_lo_in), lo_o, r, true);
    if (pushrod) capsule(c.s, lo_o + Vec3(0.0f, -0.03f, 0.02f), Vec3(hub.x, y_in, z_up_in + 0.06f), r, true);
}

void add_halo(CarCtx& c, float xf, float xr, float zt, float hw, float zm) {
    Scene& s = c.s;
    s.begin_group("Halo", Component::Halo, Frame::Body);
    const float r = 0.024f;
    const Vec3 apex(xf, 0.0f, zt);
    capsule(s, Vec3(xf - 0.12f, 0.0f, zm - 0.02f), apex, r + 0.004f, false);                     // pilar central
    const Vec3 a1(xf + 0.16f, hw * 0.82f, zt + 0.02f);
    capsule(s, apex, a1, r, true, Op::SmoothUnion, 0.03f);                                       // arco frontal
    const Vec3 a2(xr, hw, zt + 0.01f);
    capsule(s, a1, a2, r, true, Op::SmoothUnion, 0.03f);                                         // laterales
    capsule(s, a2, Vec3(xr + 0.08f, hw * 0.9f, zm), r, true, Op::SmoothUnion, 0.03f);            // patas traseras
    s.end_group();
}

void elems_box(const Scene& s, std::span<const ElemPose> el, float& x0, float& z0, float& x1, float& z1) {
    x0 = z0 = 1e30f; x1 = z1 = -1e30f;
    for (const ElemPose& e : el) {
        float a, b, cc, d;
        elem_box(s, e, a, b, cc, d);
        x0 = min_(x0, a); z0 = min_(z0, b); x1 = max_(x1, cc); z1 = max_(z1, d);
    }
}

void add_endplates(CarCtx& c, std::span<const ElemPose> el, const WingPlan& w, const EndplateSpec& ep) {
    if (!ep.enable || el.empty()) return;
    float x0, z0, x1, z1;
    elems_box(c.s, el, x0, z0, x1, z1);
    const Vec2 off = plan_tip_offset(w);
    x0 += off.x; x1 += off.x; z0 += off.y; z1 += off.y;
    const float y = w.y0 + w.span;
    const float zl = max_(min_(z0 - ep.z_pad_lo, ep.z_down_to), ep.z_min);
    const float zh = max_(z1 + ep.z_pad_hi, ep.z_up_to);
    rbox(c.s, Vec3(x0 - ep.x_pad_front, y - 0.25f * ep.thick, zl), Vec3(x1 + ep.x_pad_rear, y + 0.75f * ep.thick, zh),
         ep.round, true);
}

void add_fence(CarCtx& c, std::span<const ElemPose> el, float yc, float pad) {
    float x0, z0, x1, z1;
    elems_box(c.s, el, x0, z0, x1, z1);
    rbox(c.s, Vec3(x0 - pad, yc - 0.0175f, z0 - pad), Vec3(x1 + pad, yc + 0.0175f, z1 + pad), 0.01f, true);
}

void add_pylons(CarCtx& c, float x0, float x1, float z0, float z1, float yc) {
    rbox(c.s, Vec3(x0, yc - 0.0175f, z0), Vec3(x1, yc + 0.0175f, z1), 0.012f, true);
}

void add_flat_floor(CarCtx& c, const FlatFloorSpec& f) {
    Scene& s = c.s;
    s.begin_group("Fondo plano escalonado", Component::Floor);
    // Plank + plano de referencia central (z = 0) y plano escalonado (+50 mm), 4 cm de espesor.
    tbox(s, f.plank_x0, f.xd0 + 0.05f, f.plank_hw, 0.0f, 0.10f, f.plank_hw, 0.0f, 0.10f, 0.02f);
    rbox(s, Vec3(f.x0, -f.hw, 0.05f), Vec3(f.x1, f.hw, 0.09f), 0.015f);
    if (f.xd0 > f.x1 + 0.02f)   // estrechamiento delante de las ruedas traseras hasta el difusor
        tbox(s, f.x1 - 0.05f, f.xd0 + 0.05f, f.hw, 0.05f, 0.09f, f.hwd, 0.05f, 0.09f, 0.015f);
    s.end_group();

    s.begin_group("Difusor", Component::Diffuser);
    const float L = f.xd1 - f.xd0, h = f.z_exit - 0.05f;
    const Vec2 ramp[] = {{f.xd0, 0.05f}, {f.xd0 + 0.30f * L, 0.05f + 0.18f * h}, {f.xd0 + 0.65f * L, 0.05f + 0.60f * h}, {f.xd1, f.z_exit}};
    sheet_xz(s, ramp, 0.04f, 0.5f * f.hwd, 0.5f * f.hwd);
    const Vec2 wall[] = {{f.xd0, 0.035f}, {f.xd0 + 0.25f * L, 0.035f}, {f.xd1, 0.05f + 0.45f * h}, {f.xd1, f.z_exit + 0.05f}, {f.xd0, 0.12f}};
    plate_xz(s, wall, f.hwd - 0.0175f, 0.0175f);
    if (f.strake_y > 0.0f) {
        const Vec2 st[] = {{f.xd0 + 0.1f * L, 0.04f}, {f.xd1, 0.05f + 0.5f * h}, {f.xd1, f.z_exit + 0.04f}, {f.xd0 + 0.1f * L, 0.10f}};
        plate_xz(s, st, f.strake_y, 0.0175f);
    }
    s.end_group();
}

void finish_car(Built& b, float wb) {
    const Params& P = b.params;
    const float yaw = P.yaw_deg * k_deg2rad;
    const Xform body = car_body_frame(P.ride_front_mm * 1e-3f, P.ride_rear_mm * 1e-3f, wb, yaw);
    const Xform wheels = car_wheel_frame(wb, yaw);
    b.scene.set_frames(body, wheels);
    b.front_axle_m = wheels.apply(Vec3(0.0f, 0.0f, 0.0f));
    b.rear_axle_m = wheels.apply(Vec3(wb, 0.0f, 0.0f));
    b.moment_ref_m = wheels.apply(Vec3(0.5f * wb, 0.0f, 0.0f));
}

} // namespace cfd::models::detail
