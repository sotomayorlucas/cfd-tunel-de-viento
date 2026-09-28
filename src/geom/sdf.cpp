#include "sdf.hpp"

#include <algorithm>
#include <cmath>

namespace cfd::sdf {

const char* component_name(Component c) {
    switch (c) {
        case Component::Body:        return "Carrocería";
        case Component::FrontWing:   return "Alerón delantero";
        case Component::RearWing:    return "Alerón trasero";
        case Component::BeamWing:    return "Beam wing";
        case Component::Floor:       return "Fondo plano";
        case Component::Diffuser:    return "Difusor";
        case Component::Sidepods:    return "Pontones";
        case Component::Nose:        return "Morro";
        case Component::FrontWheels: return "Ruedas del.";
        case Component::RearWheels:  return "Ruedas tras.";
        case Component::Suspension:  return "Suspensión";
        case Component::Halo:        return "Halo";
        case Component::Object:      return "Objeto";
        case Component::WingMain:    return "Plano principal";
        case Component::WingFlap:    return "Flap";
        case Component::Endplate:    return "Endplate";
        default:                     return "?";
    }
}

// ---------------------------------------------------------------------------------------
float sd_polygon_edges(const float* CFD_RESTRICT E, u32 n, Vec2 p) {
    float d = 1e30f;
    bool inside = false;
    for (u32 i = 0; i < n; ++i, E += 5) {
        const float vx = E[0], vy = E[1], ex = E[2], ey = E[3], il2 = E[4];
        const float wx = p.x - vx, wy = p.y - vy;
        const float t = clamp_((wx * ex + wy * ey) * il2, 0.0f, 1.0f);
        const float bx = wx - ex * t, by = wy - ey * t;
        d = min_(d, bx * bx + by * by);
        // Regla de paridad de iq sin ramas: invierte si (c1 && c2 && c3) || (!c1 && !c2 && !c3)
        const bool c1 = p.y >= vy, c2 = p.y < vy + ey, c3 = ex * wy > ey * wx;
        inside ^= (c1 == c2) & (c2 == c3);
    }
    const float r = std::sqrt(d);
    return inside ? -r : r;
}

namespace {

void push_edges(std::vector<float>& pool, std::span<const Vec2> v) {
    const usize n = v.size();
    for (usize i = 0; i < n; ++i) {
        const Vec2 a = v[i], b = v[(i + n - 1) % n];      // e = v[j] - v[i], j = i-1
        const Vec2 e = b - a;
        const float l2 = dot(e, e);
        pool.push_back(a.x); pool.push_back(a.y);
        pool.push_back(e.x); pool.push_back(e.y);
        pool.push_back(l2 > 0 ? 1.0f / l2 : 0.0f);
    }
}

CFD_INLINE float length3(float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); }

float sd_round_cone(Vec3 p, Vec3 a, Vec3 b, float r1, float r2) {
    const Vec3 ba = b - a;
    const float l2 = dot(ba, ba);
    if (l2 < 1e-12f) return length(p - a) - max_(r1, r2);
    const float rr = r1 - r2;
    const float a2 = l2 - rr * rr;
    const float il2 = 1.0f / l2;
    const Vec3 pa = p - a;
    const float y = dot(pa, ba);
    const float z = y - l2;
    const Vec3 xv = pa * l2 - ba * y;
    const float x2 = dot(xv, xv);
    const float y2 = y * y * l2;
    const float z2 = z * z * l2;
    const float k = (rr > 0 ? 1.0f : (rr < 0 ? -1.0f : 0.0f)) * rr * rr * x2;
    if ((z > 0 ? 1.0f : (z < 0 ? -1.0f : 0.0f)) * a2 * z2 > k) return std::sqrt(x2 + z2) * il2 - r2;
    if ((y > 0 ? 1.0f : (y < 0 ? -1.0f : 0.0f)) * a2 * y2 < k) return std::sqrt(x2 + y2) * il2 - r1;
    return (std::sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

} // namespace

// ---------------------------------------------------------------------------------------
Scene::Handle Scene::naca4(float m, float p, float t, bool inverted, int n) {
    n = max_(n, 8);
    std::vector<Vec2> up, lo;
    up.reserve(static_cast<usize>(n) + 1); lo.reserve(static_cast<usize>(n) + 1);
    for (int i = 0; i <= n; ++i) {
        const float x = 0.5f * (1.0f - std::cos(k_pi * static_cast<float>(i) / static_cast<float>(n)));  // espaciado coseno
        const float yt = 5.0f * t * (0.2969f * std::sqrt(x) - 0.1260f * x - 0.3516f * x * x + 0.2843f * x * x * x - 0.1036f * x * x * x * x);
        float yc = 0, dyc = 0;
        if (m > 0 && p > 0 && p < 1) {
            if (x < p) { yc = m / (p * p) * (2 * p * x - x * x);                  dyc = 2 * m / (p * p) * (p - x); }
            else       { yc = m / sq(1 - p) * ((1 - 2 * p) + 2 * p * x - x * x);  dyc = 2 * m / sq(1 - p) * (p - x); }
        }
        const float th = std::atan(dyc);
        up.push_back({x - yt * std::sin(th), yc + yt * std::cos(th)});
        lo.push_back({x + yt * std::sin(th), yc - yt * std::cos(th)});
    }
    // Polígono: extradós del BS al BA, intradós del BA al BS (sin duplicar extremos).
    std::vector<Vec2> poly;
    for (int i = n; i >= 0; --i) poly.push_back(up[static_cast<usize>(i)]);
    for (int i = 1; i < n; ++i) poly.push_back(lo[static_cast<usize>(i)]);
    if (inverted) for (auto& q : poly) q.y = -q.y;
    return profile(poly);
}

Scene::Handle Scene::profile(std::span<const Vec2> pts) {
    Handle h{static_cast<u32>(pool.size()), static_cast<u32>(pts.size())};
    push_edges(pool, pts);
    return h;
}
Scene::Handle Scene::polygon(std::span<const Vec2> pts) { return profile(pts); }

Scene::Handle Scene::planes(std::span<const Vec4> pl) {
    Handle h{static_cast<u32>(pool.size()), static_cast<u32>(pl.size())};
    for (const Vec4& q : pl) {
        const Vec3 nn = normalize(q.xyz());
        const float s = length(q.xyz());
        pool.push_back(nn.x); pool.push_back(nn.y); pool.push_back(nn.z);
        pool.push_back(s > 0 ? q.w / s : q.w);
    }
    return h;
}

// ---------------------------------------------------------------------------------------
int Scene::begin_group(const char* name, Component c, Frame f, RigidMotion m) {
    CFD_CHECK(open_group_ < 0, "begin_group: ya hay un grupo abierto");
    CFD_CHECK(groups_.size() < 254, "demasiados grupos (máx 254)");
    Group g;
    g.name = name;
    g.component = c;
    g.frame = f;
    g.motion_frame = m;
    g.motion = m;
    g.first = static_cast<u32>(prims_.size());
    groups_.push_back(g);
    open_group_ = static_cast<int>(groups_.size()) - 1;
    return open_group_;
}

void Scene::end_group() {
    CFD_CHECK(open_group_ >= 0, "end_group sin begin_group");
    open_group_ = -1;
    set_frames(frame_[0], frame_[1]);   // recalcula cajas del grupo recién cerrado (barato)
}

Prim& Scene::add(PrimType t, const Xform& xf, std::initializer_list<float> params, Op op, float k, bool mirror_y, Handle data) {
    CFD_CHECK(open_group_ >= 0, "add(): no hay grupo abierto");
    Prim pr;
    pr.type = t; pr.op = op; pr.k = k; pr.mirror_y = mirror_y;
    pr.xf = xf; pr.inv = xf.inverse();
    int i = 0;
    for (float v : params) { if (i < 8) pr.p[i++] = v; }
    pr.data_off = data.off; pr.data_n = data.n;
    finalize_prim(pr, groups_[static_cast<usize>(open_group_)].frame);
    prims_.push_back(pr);
    groups_[static_cast<usize>(open_group_)].count++;
    return prims_.back();
}

Aabb Scene::local_bounds(const Prim& pr) const {
    const float* p = pr.p;
    Aabb b;
    switch (pr.type) {
        case PrimType::Sphere:    b = {Vec3(-p[0]), Vec3(p[0])}; break;
        case PrimType::Ellipsoid: b = {Vec3(-p[0], -p[1], -p[2]), Vec3(p[0], p[1], p[2])}; break;
        case PrimType::RoundBox:  b = {Vec3(-p[0], -p[1], -p[2]), Vec3(p[0], p[1], p[2])}; break;
        case PrimType::TaperBox: {
            const float hy = max_(p[1], p[3]);
            b = {Vec3(-p[0], -hy, min_(p[5] - p[2], p[6] - p[4])), Vec3(p[0], hy, max_(p[5] + p[2], p[6] + p[4]))};
            break;
        }
        case PrimType::RoundCone: {
            const float r = max_(p[6], p[7]);
            b.grow(Vec3(p[0], p[1], p[2])); b.grow(Vec3(p[3], p[4], p[5]));
            b = b.expanded(r);
            break;
        }
        case PrimType::Cylinder: b = {Vec3(-p[0], -p[1], -p[0]), Vec3(p[0], p[1], p[0])}; break;
        case PrimType::Torus:    b = {Vec3(-(p[0] + p[1]), -p[1], -(p[0] + p[1])), Vec3(p[0] + p[1], p[1], p[0] + p[1])}; break;
        case PrimType::Wing: {
            float xmin = 1e30f, xmax = -1e30f, zmin = 1e30f, zmax = -1e30f;
            const float* E = pool.data() + pr.data_off;
            for (u32 i = 0; i < pr.data_n; ++i) {
                xmin = min_(xmin, E[5 * i]); xmax = max_(xmax, E[5 * i]);
                zmin = min_(zmin, E[5 * i + 1]); zmax = max_(zmax, E[5 * i + 1]);
            }
            for (int si = 0; si <= 16; ++si) {
                const float s = static_cast<float>(si) / 16.0f;
                const float c = lerp(p[0], p[1], s);
                const float xle = p[3] * s + p[7] * s * s, zle = p[4] * s + p[6] * s * s, tw = p[5] * s;
                const float ct = std::cos(tw), st = std::sin(tw);
                for (int k = 0; k < 4; ++k) {
                    const float ux = ((k & 1) ? xmax : xmin) - 0.25f, uz = (k & 2) ? zmax : zmin;
                    const float rx = ux * ct + uz * st + 0.25f, rz = -ux * st + uz * ct;   // rot_y(tw) sobre 25% cuerda
                    b.grow(Vec3(xle + c * rx, s * p[2], zle + c * rz));
                }
            }
            b = b.expanded(0.02f * max_(p[0], p[1]));
            break;
        }
        case PrimType::ExtrudePoly: {
            const float* E = pool.data() + pr.data_off;
            for (u32 i = 0; i < pr.data_n; ++i) { b.grow(Vec3(E[5 * i], -p[0], E[5 * i + 1])); b.grow(Vec3(E[5 * i], p[0], E[5 * i + 1])); }
            break;
        }
        case PrimType::ConvexPlanes: {
            // Vértices = intersecciones de ternas de planos (+ caja de ±100 m) que cumplen todos.
            std::vector<Vec4> pl;
            const float* P = pool.data() + pr.data_off;
            for (u32 i = 0; i < pr.data_n; ++i) pl.push_back({P[4 * i], P[4 * i + 1], P[4 * i + 2], P[4 * i + 3]});
            const float B = 100.0f;
            pl.push_back({1, 0, 0, B}); pl.push_back({-1, 0, 0, B}); pl.push_back({0, 1, 0, B});
            pl.push_back({0, -1, 0, B}); pl.push_back({0, 0, 1, B}); pl.push_back({0, 0, -1, B});
            const usize n = pl.size();
            for (usize i = 0; i < n; ++i)
                for (usize j = i + 1; j < n; ++j)
                    for (usize k = j + 1; k < n; ++k) {
                        const Vec3 a = pl[i].xyz(), bb = pl[j].xyz(), c = pl[k].xyz();
                        const float det = dot(a, cross(bb, c));
                        if (std::fabs(det) < 1e-9f) continue;
                        const Vec3 v = (cross(bb, c) * pl[i].w + cross(c, a) * pl[j].w + cross(a, bb) * pl[k].w) / det;
                        bool ok = true;
                        for (usize m = 0; m < n && ok; ++m) ok = dot(pl[m].xyz(), v) - pl[m].w <= 1e-4f;
                        if (ok) b.grow(v);
                    }
            break;
        }
    }
    return b;
}

void Scene::finalize_prim(Prim& pr, Frame) {
    Aabb bf = local_bounds(pr).transformed(pr.xf);
    if (pr.mirror_y) {                  // mitad y ≥ 0 (se evalúa con |y|)
        bf.lo.y = max_(bf.lo.y, 0.0f);
        bf.hi.y = max_(bf.hi.y, 0.0f);
    }
    pr.box_frame = bf;
}

void Scene::set_frames(const Xform& body, const Xform& wheels) {
    frame_[0] = body;
    frame_[1] = wheels;
    for (Group& g : groups_) {
        const Xform F = g.frame == Frame::Body ? frame_[0] : (g.frame == Frame::Wheels ? frame_[1] : Xform{});
        g.box_model = Aabb{};
        for (u32 i = g.first; i < g.first + g.count; ++i) {
            Prim& pr = prims_[i];
            Aabb full = pr.box_frame;
            if (pr.mirror_y) { full.lo.y = -pr.box_frame.hi.y; }
            pr.box_model = full.transformed(F);
            if (pr.op == Op::Union || pr.op == Op::SmoothUnion)
                g.box_model.grow(pr.box_model.expanded(pr.op == Op::SmoothUnion ? 0.25f * pr.k + 1e-4f : 1e-4f));
        }
        // Movimiento rígido: marco → modelo.
        g.motion.v_hat = F.apply_dir(g.motion_frame.v_hat);
        g.motion.omega_hat = F.apply_dir(g.motion_frame.omega_hat);
        g.motion.center = F.apply(g.motion_frame.center);
    }
}

// ---------------------------------------------------------------------------------------
Prim& Scene::sphere(Vec3 c, float r, Op op, float k, bool mirror) {
    return add(PrimType::Sphere, Xform{Mat3{}, c}, {r}, op, k, mirror);
}
Prim& Scene::ellipsoid(Vec3 c, Vec3 radii, const Mat3& R, Op op, float k, bool mirror) {
    return add(PrimType::Ellipsoid, Xform{R, c}, {radii.x, radii.y, radii.z}, op, k, mirror);
}
Prim& Scene::round_box(Vec3 c, Vec3 half, float round, const Mat3& R, Op op, float k, bool mirror) {
    return add(PrimType::RoundBox, Xform{R, c}, {half.x, half.y, half.z, round}, op, k, mirror);
}
Prim& Scene::taper_box(Vec3 c, float half_len, Vec2 hf, Vec2 hr, float zcf, float zcr, float round, const Mat3& R, Op op, float k, bool mirror) {
    return add(PrimType::TaperBox, Xform{R, c}, {half_len, hf.x, hf.y, hr.x, hr.y, zcf, zcr, round}, op, k, mirror);
}
Prim& Scene::round_cone(Vec3 a, Vec3 b, float ra, float rb, Op op, float k, bool mirror) {
    return add(PrimType::RoundCone, Xform{}, {a.x, a.y, a.z, b.x, b.y, b.z, ra, rb}, op, k, mirror);
}
Prim& Scene::cylinder_y(Vec3 c, float radius, float half_width, float round, Op op, float k, bool mirror) {
    return add(PrimType::Cylinder, Xform{Mat3{}, c}, {radius, half_width, round}, op, k, mirror);
}
Prim& Scene::torus_y(Vec3 c, float R, float r, Op op, float k, bool mirror) {
    return add(PrimType::Torus, Xform{Mat3{}, c}, {R, r}, op, k, mirror);
}
Prim& Scene::wing(Handle prof, Vec3 le_root, float cr, float ct, float span, float aoa, float sweep, float dihedral,
                  float twist, float zcurve, float xcurve, Op op, float k, bool mirror) {
    Mat3 R = Mat3::rot_y(aoa);
    if (span < 0) {                       // reflexión en Y (det = -1; la inversa sigue siendo la traspuesta)
        Mat3 F; F.m[1][1] = -1.0f;
        R = R * F;
        span = -span;
    }
    return add(PrimType::Wing, Xform{R, le_root}, {cr, ct, span, sweep, dihedral, twist, zcurve, xcurve}, op, k, mirror, prof);
}
Prim& Scene::extrude_xz(Handle poly, float y_center, float half_width, Op op, float k, bool mirror) {
    return add(PrimType::ExtrudePoly, Xform{Mat3{}, Vec3(0, y_center, 0)}, {half_width}, op, k, mirror, poly);
}
Prim& Scene::convex(Handle pl, Op op, float k, bool mirror) {
    return add(PrimType::ConvexPlanes, Xform{}, {}, op, k, mirror, pl);
}

// ---------------------------------------------------------------------------------------
float Scene::eval_prim(const Prim& pr, Vec3 pf) const {
    const Vec3 q = pr.inv.apply(pf);
    const float* p = pr.p;
    switch (pr.type) {
        case PrimType::Sphere: return length(q) - p[0];
        case PrimType::Ellipsoid: {
            const float k0 = length3(q.x / p[0], q.y / p[1], q.z / p[2]);
            const float k1 = length3(q.x / (p[0] * p[0]), q.y / (p[1] * p[1]), q.z / (p[2] * p[2]));
            return k1 > 1e-12f ? k0 * (k0 - 1.0f) / k1 : -min_(p[0], min_(p[1], p[2]));
        }
        case PrimType::RoundBox: {
            const float r = p[3];
            const float qx = std::fabs(q.x) - p[0] + r, qy = std::fabs(q.y) - p[1] + r, qz = std::fabs(q.z) - p[2] + r;
            return length3(max_(qx, 0.0f), max_(qy, 0.0f), max_(qz, 0.0f)) + min_(max_(qx, max_(qy, qz)), 0.0f) - r;
        }
        case PrimType::TaperBox: {
            const float hx = p[0];
            const float t = clamp_((q.x + hx) / (2.0f * hx), 0.0f, 1.0f);
            const float hy = lerp(p[1], p[3], t), hz = lerp(p[2], p[4], t), zc = lerp(p[5], p[6], t), r = p[7];
            const float qx = std::fabs(q.x) - hx + r, qy = std::fabs(q.y) - hy + r, qz = std::fabs(q.z - zc) - hz + r;
            return length3(max_(qx, 0.0f), max_(qy, 0.0f), max_(qz, 0.0f)) + min_(max_(qx, max_(qy, qz)), 0.0f) - r;
        }
        case PrimType::RoundCone: return sd_round_cone(q, Vec3(p[0], p[1], p[2]), Vec3(p[3], p[4], p[5]), p[6], p[7]);
        case PrimType::Cylinder: {
            const float rb = p[2];
            const float dx = std::sqrt(q.x * q.x + q.z * q.z) - (p[0] - rb);
            const float dy = std::fabs(q.y) - (p[1] - rb);
            return min_(max_(dx, dy), 0.0f) + std::sqrt(sq(max_(dx, 0.0f)) + sq(max_(dy, 0.0f))) - rb;
        }
        case PrimType::Torus: {
            const float a = std::sqrt(q.x * q.x + q.z * q.z) - p[0];
            return std::sqrt(a * a + q.y * q.y) - p[1];
        }
        case PrimType::Wing: {
            const float span = p[2];
            const float s = clamp_(q.y / span, 0.0f, 1.0f);
            const float c = lerp(p[0], p[1], s);
            const float xle = p[3] * s + p[7] * s * s, zle = p[4] * s + p[6] * s * s;
            float ux = (q.x - xle) / c, uz = (q.z - zle) / c;
            if (p[5] != 0.0f) {                       // torsión: rotación inversa sobre el 25% de cuerda
                const float tw = p[5] * s, ct = std::cos(tw), st = std::sin(tw);
                const float rx = ux - 0.25f;
                ux = rx * ct - uz * st + 0.25f;
                uz = rx * st + uz * ct;
            }
            const float d2 = sd_polygon_edges(pool.data() + pr.data_off, pr.data_n, Vec2(ux, uz)) * c;
            const float dh = std::fabs(q.y - 0.5f * span) - 0.5f * span;
            return extrude(d2, dh);
        }
        case PrimType::ExtrudePoly: {
            const float d2 = sd_polygon_edges(pool.data() + pr.data_off, pr.data_n, Vec2(q.x, q.z));
            return extrude(d2, std::fabs(q.y) - p[0]);
        }
        case PrimType::ConvexPlanes: {
            const float* P = pool.data() + pr.data_off;
            float d = -1e30f;
            for (u32 i = 0; i < pr.data_n; ++i, P += 4) d = max_(d, P[0] * q.x + P[1] * q.y + P[2] * q.z - P[3]);
            return d;
        }
    }
    CFD_UNREACHABLE();
}

float Scene::eval_group(int gi, Vec3 p) const {
    const Group& g = groups_[static_cast<usize>(gi)];
    Vec3 pf = p;
    if (g.frame == Frame::Body) pf = frame_[0].inverse_apply(p);
    else if (g.frame == Frame::Wheels) pf = frame_[1].inverse_apply(p);
    const Vec3 pm{pf.x, std::fabs(pf.y), pf.z};
    float d = 1e30f;
    for (u32 i = g.first, e = g.first + g.count; i < e; ++i) {
        const Prim& pr = prims_[i];
        const Vec3 q = pr.mirror_y ? pm : pf;
        const float bd = pr.box_frame.distance(q);
        switch (pr.op) {
            case Op::Union:          if (bd >= d) continue; d = min_(d, eval_prim(pr, q)); break;
            case Op::SmoothUnion:    if (bd >= d + pr.k) continue; d = smin(d, eval_prim(pr, q), pr.k); break;
            case Op::Subtract:       if (bd > max_(-d, 0.0f)) continue; d = max_(d, -eval_prim(pr, q)); break;   // d>0: sólo se descarta fuera de la caja restada
            case Op::SmoothSubtract: if (bd > max_(pr.k - d, 0.0f)) continue; d = smax(d, -eval_prim(pr, q), pr.k); break;
            case Op::Intersect:      d = max_(d, eval_prim(pr, q)); break;
        }
    }
    return d;
}

float Scene::eval(Vec3 p, int* group_id) const {
    float best = 1e30f;
    int bi = 0;
    const int n = static_cast<int>(groups_.size());
    for (int gi = 0; gi < n; ++gi) {
        if (groups_[static_cast<usize>(gi)].box_model.distance(p) >= best) continue;
        const float d = eval_group(gi, p);
        if (d < best) { best = d; bi = gi + 1; }
    }
    if (group_id) *group_id = bi;
    return best;
}

Vec3 Scene::normal(Vec3 p, float h) const {
    const Vec3 k0{1, -1, -1}, k1{-1, -1, 1}, k2{-1, 1, -1}, k3{1, 1, 1};
    const Vec3 n = k0 * eval(p + k0 * h) + k1 * eval(p + k1 * h) + k2 * eval(p + k2 * h) + k3 * eval(p + k3 * h);
    return normalize(n);
}

Aabb Scene::bounds() const {
    Aabb b;
    for (const Group& g : groups_) if (!g.box_model.empty()) b.grow(g.box_model);
    return b;
}

void Scene::clear() {
    groups_.clear();
    prims_.clear();
    pool.clear();
    frame_[0] = frame_[1] = Xform{};
    open_group_ = -1;
}

} // namespace cfd::sdf
