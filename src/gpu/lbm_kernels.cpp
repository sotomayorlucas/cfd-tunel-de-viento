// ============================================================================
//  gpu/lbm_kernels.cpp — compute shaders del LBM generados con el DSL SPIR-V.
//
//  Port 1:1 de lbm/solver.cpp (mismas fórmulas, mismo orden de operaciones
//  salvo la contracción en FMA que decida el compilador de la GPU). Cada
//  bloque cita la función de la CPU que reproduce.
// ============================================================================
#include "lbm_kernels.hpp"
#include "spirv.hpp"
#include "../lbm/lattice.hpp"

#include <cmath>
#include <cstdlib>

namespace cfd::gpu::lbmk {

using namespace spv;
namespace L = lbm::d3q19;

namespace {

constexpr u32 kSolidBit = 1u << 0, kMovingBit = 1u << 1, kInletBit = 1u << 2, kOutletBit = 1u << 3, kNearMovingBit = 1u << 4,
              kNearWallBit = 1u << 6;
constexpr u32 kGroundId = 255;
constexpr float kScale = 32768.0f, kInvScale = 1.0f / 32768.0f;

// ---- Álgebra de la colisión (= solver.cpp: moments, noneq, one_minus_omega, wall_omc, collide) ------
// Plantillas sobre el tipo de valor: F (una celda por hilo) o F2 (dos celdas por hilo en paralelo, como
// el f8x2 de la CPU: dos cadenas independientes que se escriben a la vez empaquetadas en f16×2).
struct F2 {
    F a, b;
    F2() = default;
    F2(F x, F y) : a(x), b(y) {}
    F2(float s) : a(s), b(s) {}   // NOLINT: difusión de una constante
};
inline F2 operator+(F2 x, F2 y) { return {x.a + y.a, x.b + y.b}; }
inline F2 operator-(F2 x, F2 y) { return {x.a - y.a, x.b - y.b}; }
inline F2 operator*(F2 x, F2 y) { return {x.a * y.a, x.b * y.b}; }
inline F2 operator/(F2 x, F2 y) { return {x.a / y.a, x.b / y.b}; }
inline F2 operator-(F2 x) { return {-x.a, -x.b}; }
inline F2& operator+=(F2& x, F2 y) { x = x + y; return x; }
inline F2& operator-=(F2& x, F2 y) { x = x - y; return x; }
inline F2 fma(F2 x, F2 y, F2 z) { return {fma(x.a, y.a, z.a), fma(x.b, y.b, z.b)}; }
inline F2 sqrt(F2 x) { return {sqrt(x.a), sqrt(x.b)}; }
inline F2 fmax(F2 x, F2 y) { return {fmax(x.a, y.a), fmax(x.b, y.b)}; }
struct B2 { B a, b; };
inline F2 select(B2 c, F2 x, F2 y) { return {select(c.a, x.a, y.a), select(c.b, x.b, y.b)}; }
inline B2 operator!(B2 c) { return {!c.a, !c.b}; }

template <class V> struct Mom { V drho, jx, jy, jz, pxx, pyy, pzz, pxy, pxz, pyz; };

template <class V, class Ld>
Mom<V> moments(const Ld& ld) {
    Mom<V> m, g;
    V a = ld(1), b = ld(2);
    V s = a + b, d = a - b;
    m.drho = ld(0) + s; m.jx = d; m.pxx = s;
    a = ld(3); b = ld(4); s = a + b; d = a - b;
    m.drho += s; m.jy = d; m.pyy = s;
    a = ld(5); b = ld(6); s = a + b; d = a - b;
    m.drho += s; m.jz = d; m.pzz = s;
    a = ld(7); b = ld(8); s = a + b; d = a - b;
    g.drho = s; g.jx = d; g.jy = d; g.pxx = s; g.pyy = s; m.pxy = s;
    a = ld(9); b = ld(10); s = a + b; d = a - b;
    g.drho += s; g.jx += d; g.jz = d; g.pxx += s; g.pzz = s; m.pxz = s;
    a = ld(11); b = ld(12); s = a + b; d = a - b;
    g.drho += s; g.jy += d; g.jz += d; g.pyy += s; g.pzz += s; m.pyz = s;
    a = ld(13); b = ld(14); s = a + b; d = a - b;
    g.drho += s; g.jx += d; g.jy -= d; g.pxx += s; g.pyy += s; m.pxy -= s;
    a = ld(15); b = ld(16); s = a + b; d = a - b;
    g.drho += s; g.jx += d; g.jz -= d; g.pxx += s; g.pzz += s; m.pxz -= s;
    a = ld(17); b = ld(18); s = a + b; d = a - b;
    g.drho += s; g.jy += d; g.jz -= d; g.pyy += s; g.pzz += s; m.pyz -= s;
    m.drho += g.drho; m.jx += g.jx; m.jy += g.jy; m.jz += g.jz;
    m.pxx += g.pxx; m.pyy += g.pyy; m.pzz += g.pzz;
    return m;
}

template <class V> struct Neq { V xx, yy, zz, xy, xz, yz; };

template <class V>
Neq<V> noneq(const Mom<V>& m, V drho, V rho, V ux, V uy, V uz, float Sc) {
    const V d3 = drho * V(1.0f / 3.0f);
    const V rx = rho * ux, ry = rho * uy, rz = rho * uz;
    Neq<V> q;
    if (Sc == 1.0f) {
        q.xx = fma(-rx, ux, m.pxx - d3);
        q.yy = fma(-ry, uy, m.pyy - d3);
        q.zz = fma(-rz, uz, m.pzz - d3);
        q.xy = fma(-rx, uy, m.pxy);
        q.xz = fma(-rx, uz, m.pxz);
        q.yz = fma(-ry, uz, m.pyz);
    } else {
        const V is(1.0f / Sc);
        q.xx = fma(-rx, ux, fma(m.pxx, is, -d3));
        q.yy = fma(-ry, uy, fma(m.pyy, is, -d3));
        q.zz = fma(-rz, uz, fma(m.pzz, is, -d3));
        q.xy = fma(-rx, uy, m.pxy * is);
        q.xz = fma(-rx, uz, m.pxz * is);
        q.yz = fma(-ry, uz, m.pyz * is);
    }
    return q;
}

template <class V>
V one_minus_omega(const Neq<V>& q, V inv_rho, V tau0, V tau0sq, V K) {
    const V dxy = fma(q.xx, q.xx, q.yy * q.yy);
    const V dz = fma(q.zz, q.zz, V(2.0f) * (q.xy * q.xy));
    const V oo = V(2.0f) * fma(q.xz, q.xz, q.yz * q.yz);
    const V qq = (dxy + dz) + oo;
    const V den = tau0 + sqrt(fma(K * sqrt(qq), inv_rho, tau0sq));
    return V(1.0f) - V(2.0f) / den;
}

template <class V>
V wall_omc(V u2, V tau0, V C3, V omc_s, V ff) {
    const V s1 = sqrt(u2);
    const V s2 = sqrt(s1);
    const V u34 = sqrt(s1 * s2);
    const V tw = fmax(fma(C3, u34, V(0.5f)), tau0);
    const V ts = V(1.0f) / (V(1.0f) - omc_s);
    const V tf = fma(ff, ts - V(0.5f), V(0.5f));
    return V(1.0f) - V(1.0f) / fmax(tw, tf);
}

template <class V, class Ld, class St>
void collide(bool reg, bool rr, float Sc, const Ld& ld, const St& st, V drho, V rho, V ux, V uy, V uz, const Neq<V>& q, V omc, V omcb, V om3) {
    const V ux3 = V(3.0f) * ux, uy3 = V(3.0f) * uy, uz3 = V(3.0f) * uz;
    const V c3 = -fma(ux3, ux, fma(uy3, uy, uz3 * uz));
    const V hr = V(0.5f) * rho;
    const float W0 = L::w[0] * Sc, W1 = L::w[1] * Sc, W2 = L::w[7] * Sc;
    const V r1 = V(W1) * rho, h1 = V(W1) * hr, d1 = V(W1) * drho;
    const V r2 = V(W2) * rho, h2 = V(W2) * hr, d2 = V(W2) * drho;
    if (reg || rr) {
        // Traza de Π^neq (volumen) relajada con omcb, plegada en el término constante (= solver.cpp collide).
        const V mxx = V(4.5f) * q.xx, myy = V(4.5f) * q.yy, mzz = V(4.5f) * q.zz;
        const V mxy = V(9.0f) * q.xy, mxz = V(9.0f) * q.xz, myz = V(9.0f) * q.yz;
        const V trm = (mxx + myy + mzz) * V(1.0f / 3.0f);
        const V rx = mxx - trm, ry = myy - trm, rz = mzz - trm;
        const V tb = omcb * trm;
        const V d2b = fma(V(W2), tb, d2);
        st(0, fma(-V(W0), tb, V(W0) * fma(hr, c3, drho)));
        const V rxy = rx + ry, rxz = rx + rz, ryz = ry + rz;
        if (!rr) {
            auto pair = [&](int i, V a, V R, float W, V rW, V hW, V dW) {
                const V t = fma(hW, fma(a, a, c3), dW);
                const V ra = rW * a;
                const V pr = V(W) * R;
                st(i, fma(omc, pr, t + ra));
                st(i + 1, fma(omc, pr, t - ra));
            };
            pair(1, ux3, rx, W1, r1, h1, d1);
            pair(3, uy3, ry, W1, r1, h1, d1);
            pair(5, uz3, rz, W1, r1, h1, d1);
            pair(7, ux3 + uy3, rxy + mxy, W2, r2, h2, d2b);
            pair(9, ux3 + uz3, rxz + mxz, W2, r2, h2, d2b);
            pair(11, uy3 + uz3, ryz + myz, W2, r2, h2, d2b);
            pair(13, ux3 - uy3, rxy - mxy, W2, r2, h2, d2b);
            pair(15, ux3 - uz3, rxz - mxz, W2, r2, h2, d2b);
            pair(17, uy3 - uz3, ryz - myz, W2, r2, h2, d2b);
        } else {
            // Regularización recursiva: no equilibrio de 3er orden reconstruido de Π^neq, forma par ± impar
            // (= solver.cpp collide, Collision::Recursive).
            const V ux2 = ux + ux, uy2 = uy + uy, uz2 = uz + uz;
            const V k9 = V(9.0f * W2);
            const V nxxy = k9 * fma(ux2, q.xy, uy * q.xx), nxyy = k9 * fma(uy2, q.xy, ux * q.yy);
            const V nxzz = k9 * fma(uz2, q.xz, ux * q.zz), nxxz = k9 * fma(ux2, q.xz, uz * q.xx);
            const V nyzz = k9 * fma(uz2, q.yz, uy * q.zz), nyyz = k9 * fma(uy2, q.yz, uz * q.yy);
            auto pair = [&](int i, V a, V R, float W, V rW, V hW, V dW, V p3) {
                const V t = fma(hW, fma(a, a, c3), dW);
                const V ra = rW * a;
                const V pr = V(W) * R;
                const V E = fma(omc, pr, t);
                const V O = fma(om3, p3, ra);
                st(i, E + O);
                st(i + 1, E - O);
            };
            const V m2 = V(-2.0f);
            pair(1, ux3, rx, W1, r1, h1, d1, m2 * (nxyy + nxzz));
            pair(3, uy3, ry, W1, r1, h1, d1, m2 * (nxxy + nyzz));
            pair(5, uz3, rz, W1, r1, h1, d1, m2 * (nyyz + nxxz));
            pair(7, ux3 + uy3, rxy + mxy, W2, r2, h2, d2b, nxxy + nxyy);
            pair(9, ux3 + uz3, rxz + mxz, W2, r2, h2, d2b, nxzz + nxxz);
            pair(11, uy3 + uz3, ryz + myz, W2, r2, h2, d2b, nyzz + nyyz);
            pair(13, ux3 - uy3, rxy - mxy, W2, r2, h2, d2b, nxyy - nxxy);
            pair(15, ux3 - uz3, rxz - mxz, W2, r2, h2, d2b, nxzz - nxxz);
            pair(17, uy3 - uz3, ryz - myz, W2, r2, h2, d2b, nyzz - nyyz);
        }
    } else {
        auto pair = [&](int i, V a, V rW, V hW, V dW) {
            const V fi = ld(i), fj = ld(i + 1);
            const V t = fma(hW, fma(a, a, c3), dW);
            const V ra = rW * a;
            const V e1 = t + ra, e2 = t - ra;
            st(i, fma(omc, fi - e1, e1));
            st(i + 1, fma(omc, fj - e2, e2));
        };
        {
            const V f0 = ld(0);
            const V e0 = V(W0) * fma(hr, c3, drho);
            st(0, fma(omc, f0 - e0, e0));
        }
        pair(1, ux3, r1, h1, d1);
        pair(3, uy3, r1, h1, d1);
        pair(5, uz3, r1, h1, d1);
        pair(7, ux3 + uy3, r2, h2, d2);
        pair(9, ux3 + uz3, r2, h2, d2);
        pair(11, uy3 + uz3, r2, h2, d2);
        pair(13, ux3 - uy3, r2, h2, d2);
        pair(15, ux3 - uz3, r2, h2, d2);
        pair(17, uy3 - uz3, r2, h2, d2);
    }
}

// τ0(x) de la esponja calculado en el shader (= Solver::Impl::build_tau, mismas operaciones en float):
// una tabla en memoria añadía una segunda ida y vuelta DEPENDIENTE por hilo (medido: −16 %).
struct Tau { F t0, t0sq, om0; };
Tau tau_at(Kernel& k, const Buf& par, U x, int nx) {
    const F fx = to_f(x);
    const F nu = k.ldf(par, U(kPNu)), xs = k.ldf(par, U(kPXs)), numax = k.ldf(par, U(kPNuMax));
    const F t = clamp((fx - xs) / (F(float(nx - 1)) - xs), F(0.0f), F(1.0f));
    const F sm = t * t * (F(3.0f) - F(2.0f) * t);
    const B in = (fx > xs) && (k.ldf(par, U(kPSponge)) > F(0.5f));
    const F v = select(in, nu + (numax - nu) * sm * sm, nu);
    Tau r;
    r.t0 = F(3.0f) * v + F(0.5f);
    r.t0sq = r.t0 * r.t0;
    r.om0 = F(1.0f) - F(2.0f) / (r.t0 + r.t0);
    return r;
}

// ---- Direccionamiento Esoteric-Pull (= Solver::Impl::step_ptrs) --------------------------------
// Población k del paso de paridad p: se CARGA de la ranura ls[k] en n + lo[k] y el post-colisión
// se GUARDA en la ranura ss[k] en n + so[k].
struct Addr { int ls[19], ss[19]; i64 lo[19], so[19]; };
Addr addr(const Spec& s, int p) {
    Addr a{};
    a.ls[0] = 0; a.lo[0] = 0;
    for (int i = 1; i < 19; i += 2) {
        if (p) { a.ls[i] = i; a.lo[i] = 0; a.ls[i + 1] = i + 1; a.lo[i + 1] = s.off[i]; }
        else { a.ls[i] = i + 1; a.lo[i] = 0; a.ls[i + 1] = i; a.lo[i + 1] = s.off[i]; }
    }
    a.ss[0] = 0; a.so[0] = 0;
    for (int i = 1; i < 19; i += 2) {
        a.ss[i] = a.ls[i + 1]; a.so[i] = a.lo[i + 1];
        a.ss[i + 1] = a.ls[i]; a.so[i + 1] = a.lo[i];
    }
    return a;
}
// Índice dentro de la ranura: P + n + o (P ≥ |o|: nunca negativo).
U slot_index(const Spec& s, U n, i64 o) { return n + U(static_cast<u32>(static_cast<i64>(s.P) + o)); }
// n + d con d con signo (aritmética módulo 2^32).
U offset_cell(U n, i64 d) { return d == 0 ? n : n + U(static_cast<u32>(d)); }

struct Vel { F x, y, z; };
// Velocidad de pared del id en el punto p (= wall_velocity con la tabla ya escalada por la rampa r):
// z ≤ contact_z → cinta (u_g, 0, 0); el suelo (255) tiene contact_z = +∞ en la tabla.
Vel wall_vel(Kernel& k, const Buf& mot, U id, F px, F py, F pz, F r, F ug) {
    const U b = id * U(kMotionWords);
    auto m = [&](u32 j) { return k.ldf(mot, b + U(j)); };
    const F rx = px - m(6), ry = py - m(7), rz = pz - m(8);
    const F ox = m(3), oy = m(4), oz = m(5);
    F wx = (m(0) + (oy * rz - oz * ry)) * r;
    F wy = (m(1) + (oz * rx - ox * rz)) * r;
    F wz = (m(2) + (ox * ry - oy * rx)) * r;
    const B cz = pz <= m(9);
    return {select(cz, ug, wx), select(cz, F(0.0f), wy), select(cz, F(0.0f), wz)};
}
B conserve_of(Kernel& k, const Buf& mot, U id) { return k.ldf(mot, id * U(kMotionWords) + U(10u)) > F(0.5f); }

KernelOptions opts(const Spec& s, u32 wg) {
    KernelOptions o;
    o.local_size = wg;
    static const int fm = std::getenv("CFD_GPU_FMODE") ? std::atoi(std::getenv("CFD_GPU_FMODE")) : 3;
    o.rte16 = s.rte16 && (fm & 1);
    o.denorm16 = s.denorm16 && (fm & 2);
    return o;
}

} // namespace

// ================================================================================================
//  Kernel de celdas: un paso completo (= rows_kernel → block_vec / block_scalar / block_skip_macro)
// ================================================================================================
namespace {

struct StepRes {   // recursos comunes de los kernels de celdas
    Buf flags, sid, mac, mot, par, bad;
    F u_in, rr, ug, K, C3, floor_, omcb;
};

StepRes step_resources(Kernel& k, bool macro) {
    StepRes r;
    r.flags = k.buffer(kBindFlags, Elem::U8, true);
    r.sid = k.buffer(kBindSid, Elem::U8, true);
    if (macro) r.mac = k.buffer(kBindMacro, Elem::F32, false, true);
    r.mot = k.buffer(kBindMotion, Elem::F32, true);
    r.par = k.buffer(kBindParams, Elem::F32, true);
    r.bad = k.buffer(kBindBad, Elem::U32);
    k.push_constants(kPushBytes / 4, 0);
    const U step = k.pc_u(0);
    const U pb = U(kParamHeader) + step * U(kParamPerStep);
    r.u_in = k.ldf(r.par, pb + U(kSUin)); r.rr = k.ldf(r.par, pb + U(kSR)); r.ug = k.ldf(r.par, pb + U(kSUg));
    r.K = k.ldf(r.par, U(kPK)); r.C3 = k.ldf(r.par, U(kPWallC3)); r.floor_ = k.ldf(r.par, U(kPWallFloor));
    r.omcb = k.ldf(r.par, U(kPOmcb));
    return r;
}

i64 A_lo(const Spec& s, int p, int i) { return addr(s, p).lo[i]; }

// Experimentos de docs/opt/gpu.md (CFD_GPU_EXP): patrón de memoria puro y variantes.
std::vector<u32> gen_step_experiment(const Spec& s, int p, int mode) {
    Kernel k(opts(s, s.wg));
    const Elem E = s.fp16 ? Elem::F16 : Elem::F32;
    const Addr A = addr(s, p);
    Buf d[19];
    const bool packed = mode == 4;
    for (int q = 0; q < 19; ++q) d[q] = k.buffer(kBindSlot + q, packed ? Elem::U32 : E);
    const Buf flags = k.buffer(kBindFlags, Elem::U8, true);
    k.push_constants(kPushBytes / 4, 0);
    const U n = (k.workgroup_id(1) * U(s.gx) + k.workgroup_id(0)) * U(s.wg) + k.local_index();
    static const int nf = std::getenv("CFD_GPU_FMA") ? std::atoi(std::getenv("CFD_GPU_FMA")) : 0;
    if (mode == 1 || mode == 3) {   // 1: sólo el patrón Esoteric-Pull; 3: + nf multiplicaciones por población
        k.if_(n < U(static_cast<u32>(s.N)), [&] {
            F f[19];
            for (int i = 0; i < 19; ++i) f[i] = k.ldf(d[A.ls[i]], slot_index(s, n, A.lo[i]));
            for (int j = 0; j < (mode == 3 ? nf : 0); ++j)
                for (int i = 0; i < 19; ++i) f[i] = f[i] * F(1.0001f);
            for (int i = 0; i < 19; ++i) k.stf(d[A.ss[i]], slot_index(s, n, A.so[i]), f[i]);
        });
    } else if (mode == 2) {   // cargas dentro de un if sobre el flag (dependencia de control → latencia en serie)
        const U fl = k.ldu(flags, n);
        k.if_(!bit(fl, 0), [&] {
            F f[19];
            for (int i = 0; i < 19; ++i) f[i] = k.ldf(d[A.ls[i]], slot_index(s, n, A.lo[i]));
            F sum = f[0];
            for (int i = 1; i < 19; ++i) sum = sum + f[i];
            for (int i = 0; i < 19; ++i) k.stf(d[A.ss[i]], slot_index(s, n, A.so[i]), f[i] + sum * F(1e-6f));
        });
    } else if (mode == 5) {   // 5: patrón real del modo par (ubicaciones, impares con V2 + 2 escrituras f16) + nf mult.
        Buf W[19], H[19];
        for (int q = 0; q < 19; ++q) { W[q] = k.buffer(kBindLoc + q, Elem::U32, false, false, false); H[q] = k.buffer(kBindLocAlt + q, Elem::F16, false, false, false); }
        const U m = n, nA = n + n;
        k.if_(nA < U(static_cast<u32>(s.N)), [&] {
            F a[19], b[19];
            for (int i = 0; i < 19; ++i) {
                if (A.lo[i] & 1) { F t0, t1; unpack_half2(k.ldu(W[i], m), t0, a[i]); unpack_half2(k.ldu(W[i], m + U(1u)), b[i], t1); }
                else unpack_half2(k.ldu(W[i], m), a[i], b[i]);
            }
            for (int j = 0; j < nf; ++j)
                for (int i = 0; i < 19; ++i) { a[i] = a[i] * F(1.0001f); b[i] = b[i] * F(1.0001f); }
            if (std::getenv("CFD_GPU_XFLAG")) {   // + una carga independiente (flags) usada al final
                const Buf f32 = k.buffer(kBindFlags32, Elem::U32, true);
                const F fl = to_f(k.ldu(f32, m >> U(1u)) & U(1u));
                for (int i = 0; i < 19; ++i) { a[i] = a[i] + fl; b[i] = b[i] + fl; }
            }
            for (int i = 0; i < 19; ++i) {
                const int o = L::opp[i];
                if (A.lo[o] & 1) { k.stf(H[o], nA + U(1u), a[i]); k.stf(H[o], nA + U(2u), b[i]); }
                else k.stu(W[o], m, pack_half2(a[i], b[i]));
            }
        });
    } else {   // 4: 2 celdas por hilo con palabras u32 (f16×2) alineadas + nf multiplicaciones (cota del empaquetado)
        k.if_(n < U(static_cast<u32>(s.N / 2)), [&] {
            F lo[19], hi[19];
            for (int i = 0; i < 19; ++i) unpack_half2(k.ldu(d[A.ls[i]], n + U(static_cast<u32>(s.P / 2))), lo[i], hi[i]);
            for (int j = 0; j < nf; ++j)
                for (int i = 0; i < 19; ++i) { lo[i] = lo[i] * F(1.0001f); hi[i] = hi[i] * F(1.0001f); }
            for (int i = 0; i < 19; ++i) k.stu(d[A.ss[i]], n + U(static_cast<u32>(s.P / 2)), pack_half2(lo[i], hi[i]));
        });
    }
    return k.finish();
}

// ---- Una celda por hilo (accesos de 16/32 bits por población) --------------------------------------------
std::vector<u32> gen_step_single(const Spec& s, int p, bool macro) {
    Kernel k(opts(s, s.wg));
    const Elem E = s.fp16 ? Elem::F16 : Elem::F32;
    const float Sc = s.fp16 ? kScale : 1.0f;
    Buf d[19];   // d[i] = ubicación de la población i en este paso (base del descriptor = ranura + P + desplazamiento)
    for (int q = 0; q < 19; ++q) d[q] = k.buffer(kBindLoc + q, E, false, false, false);   // alias entre sí: sin Restrict
    const StepRes R = step_resources(k, macro);
    const Addr A = addr(s, p);
    const U n = (k.workgroup_id(1) * U(s.gx) + k.workgroup_id(0)) * U(s.wg) + k.local_index();
    auto odd = [&](int i) { return s.fp16 && (A.lo[i] & 1); };   // base alineada a 4 B: +1 elemento
    const U n1 = n + U(1u);
    const U x = n % U(static_cast<u32>(s.nx));
    auto coords = [&](F& fx, F& fy, F& fz) {
        const U r = n / U(static_cast<u32>(s.nx));
        fx = to_f(x);
        fy = to_f(r % U(static_cast<u32>(s.ny)));
        fz = to_f(r / U(static_cast<u32>(s.ny)));
    };
    VarB vbad = k.varb(B{k.c_bool(false)});
    // Un bloque sin ramas por celda (salvo el relleno n ≥ N del último grupo y la ruta rara de paredes
    // móviles): las 19 cargas NO dependen del flag (dentro de un if sobre el flag cada carga esperaba al byte
    // de flags → latencia de DRAM en serie: 709 → 1123 MLUPS en el patrón de memoria puro).
    k.if_(n < U(static_cast<u32>(s.N)), [&] {
        F fv[19];
        for (int i = 0; i < 19; ++i) fv[i] = k.ldf(d[i], odd(i) ? n1 : n);
        const U fl = k.ldu(R.flags, n);
        const B solid = bit(fl, 0);
        // (Ladd de paredes móviles: ya sumado por el kernel de nodos al escribir la población entrante.)
        const Mom<F> m = moments<F>([&](int i) { return fv[i]; });
        const F drho_raw = s.fp16 ? m.drho * F(kInvScale) : m.drho;
        const F rho_raw = F(1.0f) + drho_raw;
        const F inv = F(1.0f) / rho_raw;
        const F invj = s.fp16 ? inv * F(kInvScale) : inv;
        // u "cruda" (0 en sólidos): la salida copia la de la celda x−1 por desplazamiento de subgrupo (x contiguo
        // en los carriles; nx múltiplo de 8 y subgrupo ≥ 8 → x−1 siempre en el mismo subgrupo que x = nx−1).
        const F ux0 = select(solid, F(0.0f), m.jx * invj), uy0 = select(solid, F(0.0f), m.jy * invj), uz0 = select(solid, F(0.0f), m.jz * invj);
        const F pux = k.sg_shuffle_up(ux0, 1), puy = k.sg_shuffle_up(uy0, 1), puz = k.sg_shuffle_up(uz0, 1);
        const B inlet = bit(fl, 2), outlet = bit(fl, 3);
        const B eq = any_bits(fl, kInletBit | kOutletBit);
        const F ux = select(inlet, R.u_in, select(outlet, pux, ux0));
        const F uy = select(inlet, F(0.0f), select(outlet, puy, uy0));
        const F uz = select(inlet, F(0.0f), select(outlet, puz, uz0));
        const F rho = select(eq, F(1.0f), rho_raw);
        const F drho = select(eq, F(0.0f), drho_raw);
        Neq<F> q = noneq<F>(m, drho, rho, ux, uy, uz, Sc);
        // Fronteras de equilibrio: (1−ω) = 0 → f̃eq exacto; Π^neq a 0 (lo cargado en las caras puede ser relleno).
        q.xx = select(eq, F(0.0f), q.xx); q.yy = select(eq, F(0.0f), q.yy); q.zz = select(eq, F(0.0f), q.zz);
        q.xy = select(eq, F(0.0f), q.xy); q.xz = select(eq, F(0.0f), q.xz); q.yz = select(eq, F(0.0f), q.yz);
        const Tau T = tau_at(k, R.par, x, s.nx);
        F omc = select(R.K > F(0.0f), one_minus_omega<F>(q, inv, T.t0, T.t0sq, R.K), T.om0);
        if (s.wall) omc = select(bit(fl, 6), wall_omc<F>(fma(ux, ux, fma(uy, uy, uz * uz)), T.t0, R.C3, omc, R.floor_), omc);
        const F omcb = select(eq, F(0.0f), s.bulk ? R.omcb : omc);
        omc = select(eq, F(0.0f), omc);
        // Capa junto a los cuerpos (bit interno 7 = solver.cpp kLayer): sin el término de 3er orden de la RR.
        const F om3 = select(bit(fl, 7), F(0.0f), omc);
        vbad.set((ngt(rho, F(0.2f)) || nlt(rho, F(5.0f))) && !solid);
        if (macro) {
            // Sólidos: ρ = 1 y u = velocidad de su pared (= macro_solid / block_skip_macro).
            VarF wx = k.var(F(0.0f)), wy = k.var(F(0.0f)), wz = k.var(F(0.0f));
            k.if_(bit(fl, 1), [&] {
                F fx, fy, fz;
                coords(fx, fy, fz);
                const Vel w = wall_vel(k, R.mot, k.ldu(R.sid, n), fx, fy, fz, R.rr, R.ug);
                wx.set(w.x); wy.set(w.y); wz.set(w.z);
            });
            const U mb = bits_u(k.ldf(R.par, U(kPMacroBase)));
            k.stf(R.mac, mb + n, select(solid, F(1.0f), rho));
            k.stf(R.mac, mb + U(static_cast<u32>(s.N)) + n, select(solid, wx.get(), ux));
            k.stf(R.mac, mb + U(static_cast<u32>(2 * s.N)) + n, select(solid, wy.get(), uy));
            k.stf(R.mac, mb + U(static_cast<u32>(3 * s.N)) + n, select(solid, wz.get(), uz));
        }
        // Escritura sólo en celdas no sólidas: una rama DESPUÉS de las cargas no serializa latencia (nada
        // espera a los stores). Con un select por población las 19 cargas quedaban vivas durante la colisión
        // (7:25 derrames de registros, −30 %).
        k.if_(!solid, [&] {
            auto ld = [&](int i) { return fv[i]; };
            // La población i se escribe en la ubicación de la opuesta (conjunto de acceso de Esoteric-Pull).
            auto st = [&](int i, F v) { const int o = L::opp[i]; k.stf(d[o], odd(o) ? n1 : n, v); };
            collide<F>(s.regularized, s.rr, Sc, ld, st, drho, rho, ux, uy, uz, q, omc, omcb, om3);
        });
    });
    k.if_(k.sg_any(vbad.get()), [&] { k.if_(k.sg_elect(), [&] { k.atomic_or(R.bad, U(0u), U(1u)); }); });
    return k.finish();
}

// ---- Dos celdas contiguas en x por hilo (modo par) ------------------------------------------------------
// Cada hilo procesa las celdas A = 2m y B = 2m+1 (nx par → nunca cruzan una fila). Con FP16S la población de
// ambas va en UNA palabra u32 (f16×2) cuando el desplazamiento es par (14 de las 19): la mitad de mensajes de
// memoria que con un f16 por hilo, que era el límite real (medido: con palabras u32 las FMAs no cuestan nada;
// con un f16 por carril cada operación extra costaba −20 %). Desplazamientos impares (c_x = ±1): dos palabras
// contiguas en la carga (el compilador las fusiona en un mensaje d32 V2) y dos escrituras de 16 bits.
// Grupos "puros" (sin geometría): los flags salen de las coordenadas → sin cargas de flags/sid dependientes.
std::vector<u32> gen_step_pair(const Spec& s, int p, bool macro) {
    Kernel k(opts(s, s.wg));
    const bool h = s.fp16;
    const float Sc = h ? kScale : 1.0f;
    // W[i]/H[i] = ubicación de la población i (vista u32 y f16 en FP16S; f32 en FP32), base ya desplazada.
    Buf W[19], H[19];
    for (int q = 0; q < 19; ++q) {
        W[q] = k.buffer(kBindLoc + q, h ? Elem::U32 : Elem::F32, false, false, false);   // alias: sin Restrict
        if (h) H[q] = k.buffer(kBindLocAlt + q, Elem::F16, false, false, false);
    }
    const Buf flags32 = k.buffer(kBindFlags32, Elem::U32, true);
    const Buf cls = k.buffer(kBindClass, Elem::U32, true);
    const StepRes R = step_resources(k, macro);
    static const int px = std::getenv("CFD_GPU_PX") ? std::atoi(std::getenv("CFD_GPU_PX")) : 0;   // experimentos
    const U wgid = k.workgroup_id(1) * U(s.gx) + k.workgroup_id(0);
    const U m = wgid * U(s.wg) + k.local_index();
    const U nA = m + m;
    const U nB = nA + U(1u);
    const U row = nA / U(static_cast<u32>(s.nx));
    const U xA = nA - row * U(static_cast<u32>(s.nx));
    auto coordsA = [&](F& fx, F& fy, F& fz) {
        fx = to_f(xA);
        fy = to_f(row % U(static_cast<u32>(s.ny)));
        fz = to_f(row / U(static_cast<u32>(s.ny)));
    };
    auto coordsB = [&](F& fx, F& fy, F& fz) { coordsA(fx, fy, fz); fx = fx + F(1.0f); };
    VarB vbad = k.varb(B{k.c_bool(false)});

    // Carga de la población i para A y B desde su ubicación (base alineada; desplazamiento impar → +1 elemento).
    auto oddl = [&](int i) { return (A_lo(s, p, i) & 1) != 0; };
    auto load = [&](int i, F& a, F& b) {
        if (!h) {
            a = k.ldf(W[i], nA);
            b = k.ldf(W[i], nB);
        } else if (!oddl(i)) {
            unpack_half2(k.ldu(W[i], m), a, b);
        } else {   // A = mitad alta de la palabra m, B = mitad baja de la palabra m+1
            F lo0, hi1;
            unpack_half2(k.ldu(W[i], m), lo0, a);
            unpack_half2(k.ldu(W[i], m + U(1u)), b, hi1);
        }
    };
    auto store2 = [&](int j, F a, F b) {   // ambas fluidas: empaquetada si el desplazamiento es par
        if (!h) {
            k.stf(W[j], nA, a);
            k.stf(W[j], nB, b);
        } else if (!oddl(j)) {
            k.stu(W[j], m, pack_half2(a, b));
        } else {
            k.stf(H[j], nA + U(1u), a);
            k.stf(H[j], nB + U(1u), b);
        }
    };
    auto store1 = [&](int j, U cell, F v) {   // una sola celda (pares mixtos sólido/fluido, ruta rara)
        if (h) k.stf(H[j], oddl(j) ? cell + U(1u) : cell, v);
        else k.stf(W[j], cell, v);
    };
    // Flags deducidos de las coordenadas (= expected_flags en la CPU).
    auto coord_flags = [&](U x, U y, U z) -> U {
        const bool gnd = s.ground != 0;
        const B zlo = z == U(0u), zhi = z == U(static_cast<u32>(s.nz - 1));
        const B far = (y == U(0u)) || (y == U(static_cast<u32>(s.ny - 1))) || zhi || (gnd ? B{k.c_bool(false)} : zlo);
        const B face = (x == U(0u)) || (x == U(static_cast<u32>(s.nx - 1))) || far;
        const U fface = select(far || (x != U(static_cast<u32>(s.nx - 1))), U(kInletBit), U(kOutletBit));
        U inter = U(0u);
        if (s.ground == 2) inter = select(z == U(1u), U(kNearMovingBit | (1u << 5)), U(0u));
        else if (s.ground == 1 && s.wall) inter = select(z == U(1u), U(kNearWallBit), U(0u));
        U f = select(face, fface, inter);
        if (gnd) f = select(zlo, U(s.ground == 2 ? (kSolidBit | kMovingBit) : kSolidBit), f);
        return f;
    };

    auto body = [&](bool pure) {
        F fa[19], fb[19];
        for (int i = 0; i < 19; ++i) load(i, fa[i], fb[i]);
        U flA, flB;
        if (pure) {
            const U y = row % U(static_cast<u32>(s.ny)), z = row / U(static_cast<u32>(s.ny));
            flA = coord_flags(xA, y, z);
            flB = coord_flags(xA + U(1u), y, z);
        } else {
            // Flags de A y B: una palabra u32 con 4 flags (el par está en la mitad (m & 1)).
            const U fw = (px & 1) ? U(0u) : k.ldu(flags32, m >> U(1u)) >> ((m & U(1u)) << U(4u));
            flA = fw & U(0xFFu);
            flB = (fw >> U(8u)) & U(0xFFu);
        }
        const B2 solid{bit(flA, 0), bit(flB, 0)};
        // (Ladd de paredes móviles: ya sumado por el kernel de nodos al escribir la población entrante.)
        const Mom<F2> mm = moments<F2>([&](int i) { return F2(fa[i], fb[i]); });
        const F2 drho_raw = h ? mm.drho * F2(kInvScale) : mm.drho;
        const F2 rho_raw = F2(1.0f) + drho_raw;
        const F2 inv = F2(1.0f) / rho_raw;
        const F2 invj = h ? inv * F2(kInvScale) : inv;
        const F2 zero(0.0f);
        const F2 u0x = select(solid, zero, mm.jx * invj), u0y = select(solid, zero, mm.jy * invj), u0z = select(solid, zero, mm.jz * invj);
        // Salida (x = nx−1, siempre la celda B): u de la celda x−1 = la A del mismo hilo (sin subgrupo).
        const B2 inlet{bit(flA, 2), bit(flB, 2)};
        const B outB = bit(flB, 3);
        const B2 eq{any_bits(flA, kInletBit | kOutletBit), any_bits(flB, kInletBit | kOutletBit)};
        const F2 ux = select(inlet, F2(R.u_in, R.u_in), F2(u0x.a, select(outB, u0x.a, u0x.b)));
        const F2 uy = select(inlet, zero, F2(u0y.a, select(outB, u0y.a, u0y.b)));
        const F2 uz = select(inlet, zero, F2(u0z.a, select(outB, u0z.a, u0z.b)));
        const F2 rho = select(eq, F2(1.0f), rho_raw);
        const F2 drho = select(eq, zero, drho_raw);
        Neq<F2> q = noneq<F2>(mm, drho, rho, ux, uy, uz, Sc);
        q.xx = select(eq, zero, q.xx); q.yy = select(eq, zero, q.yy); q.zz = select(eq, zero, q.zz);
        q.xy = select(eq, zero, q.xy); q.xz = select(eq, zero, q.xz); q.yz = select(eq, zero, q.yz);
        const Tau TA = tau_at(k, R.par, xA, s.nx), TB = tau_at(k, R.par, xA + U(1u), s.nx);
        const F2 t0(TA.t0, TB.t0), t0sq(TA.t0sq, TB.t0sq), om0(TA.om0, TB.om0);
        const B kpos = R.K > F(0.0f);
        F2 omc = select(B2{kpos, kpos}, one_minus_omega<F2>(q, inv, t0, t0sq, F2(R.K, R.K)), om0);
        if (s.wall)
            omc = select(B2{bit(flA, 6), bit(flB, 6)}, wall_omc<F2>(fma(ux, ux, fma(uy, uy, uz * uz)), t0, F2(R.C3, R.C3), omc, F2(R.floor_, R.floor_)), omc);
        const F2 omcb = select(eq, zero, s.bulk ? F2(R.omcb, R.omcb) : omc);
        omc = select(eq, zero, omc);
        const F2 om3 = select(B2{bit(flA, 7), bit(flB, 7)}, zero, omc);   // capa junto a los cuerpos (kLayer)
        const B badA = (ngt(rho.a, F(0.2f)) || nlt(rho.a, F(5.0f))) && !solid.a;
        const B badB = (ngt(rho.b, F(0.2f)) || nlt(rho.b, F(5.0f))) && !solid.b;
        vbad.set(badA || badB);
        if (macro) {
            const U mb = bits_u(k.ldf(R.par, U(kPMacroBase)));
            auto macro_cell = [&](U cell, U fl, B sol, F r, F x, F y, F z, bool isB) {
                F wx = F(0.0f), wy = F(0.0f), wz = F(0.0f);
                if (pure) {   // único sólido posible: el suelo (cinta a u_g si se mueve)
                    wx = select(bit(fl, 1), R.ug, F(0.0f));
                } else {
                    VarF vx = k.var(F(0.0f)), vy = k.var(F(0.0f)), vz = k.var(F(0.0f));
                    k.if_(bit(fl, 1), [&] {
                        F fx, fy, fz;
                        if (isB) coordsB(fx, fy, fz); else coordsA(fx, fy, fz);
                        const Vel w = wall_vel(k, R.mot, k.ldu(R.sid, cell), fx, fy, fz, R.rr, R.ug);
                        vx.set(w.x); vy.set(w.y); vz.set(w.z);
                    });
                    wx = vx.get(); wy = vy.get(); wz = vz.get();
                }
                k.stf(R.mac, mb + cell, select(sol, F(1.0f), r));
                k.stf(R.mac, mb + U(static_cast<u32>(s.N)) + cell, select(sol, wx, x));
                k.stf(R.mac, mb + U(static_cast<u32>(2 * s.N)) + cell, select(sol, wy, y));
                k.stf(R.mac, mb + U(static_cast<u32>(3 * s.N)) + cell, select(sol, wz, z));
            };
            macro_cell(nA, flA, solid.a, rho.a, ux.a, uy.a, uz.a, false);
            macro_cell(nB, flB, solid.b, rho.b, ux.b, uy.b, uz.b, true);
        }
        auto ld = [&](int i) { return F2(fa[i], fb[i]); };
        // Ruta rápida: ambas fluidas → escritura empaquetada. Pares mixtos (junto a sólidos): cada celda fluida
        // escribe sus 19 poblaciones sola (la otra mitad de cada palabra es de un sólido y no se toca). En los
        // grupos puros los sólidos son capas z = 0 completas: un par nunca es mixto.
        const B both = !solid.a && !solid.b;
        static const int one_collide = std::getenv("CFD_GPU_3COLL") ? 0 : 1;
        if (one_collide) {
            // UNA sola copia de la colisión (código más corto: medido +x % frente a 3 copias): cada población se
            // escribe empaquetada si el par es fluido y, si no, sólo en la mitad de la celda fluida.
            collide<F2>(s.regularized, s.rr, Sc, ld, [&](int i, F2 v) {
                const int j = L::opp[i];
                if (pure) { store2(j, v.a, v.b); return; }
                k.if_else(both, [&] { store2(j, v.a, v.b); }, [&] {
                    k.if_(!solid.a, [&] { store1(j, nA, v.a); });
                    k.if_(!solid.b, [&] { store1(j, nB, v.b); });
                });
            }, drho, rho, ux, uy, uz, q, omc, omcb, om3);
        } else {
            k.if_else(both, [&] {
                collide<F2>(s.regularized, s.rr, Sc, ld, [&](int i, F2 v) { store2(L::opp[i], v.a, v.b); }, drho, rho, ux, uy, uz, q, omc, omcb, om3);
            }, [&] {
                if (pure) return;
                k.if_(!solid.a, [&] {
                    collide<F2>(s.regularized, s.rr, Sc, ld, [&](int i, F2 v) { store1(L::opp[i], nA, v.a); }, drho, rho, ux, uy, uz, q, omc, omcb, om3);
                });
                k.if_(!solid.b, [&] {
                    collide<F2>(s.regularized, s.rr, Sc, ld, [&](int i, F2 v) { store1(L::opp[i], nB, v.b); }, drho, rho, ux, uy, uz, q, omc, omcb, om3);
                });
            });
        }
    };
    k.if_(nA < U(static_cast<u32>(s.N)), [&] {
        if (!s.classify) { body(false); return; }
        // Experimento (Spec::classify): rama UNIFORME por grupo de trabajo "puro" (flags de las coordenadas).
        k.if_else(k.ldu(cls, wgid) != U(0u), [&] { body(true); }, [&] { body(false); });
    });
    k.if_(k.sg_any(vbad.get()), [&] { k.if_(k.sg_elect(), [&] { k.atomic_or(R.bad, U(0u), U(1u)); }); });
    return k.finish();
}

} // namespace

u8 expected_flags(const Spec& s, int x, int y, int z) {
    const bool gnd = s.ground != 0;
    if (gnd && z == 0) return static_cast<u8>(s.ground == 2 ? (kSolidBit | kMovingBit) : kSolidBit);
    const bool far = y == 0 || y == s.ny - 1 || z == s.nz - 1 || (!gnd && z == 0);
    if (x == 0 || x == s.nx - 1 || far) return static_cast<u8>((x == s.nx - 1 && !far) ? kOutletBit : kInletBit);
    if (s.ground == 2 && z == 1) return static_cast<u8>(kNearMovingBit | (1u << 5));
    if (s.ground == 1 && s.wall && z == 1) return static_cast<u8>(kNearWallBit);
    return 0;
}

void load_slot(const Spec& s, int parity, int i, int* slot, i64* off) {
    const Addr A = addr(s, parity);
    *slot = A.ls[i];
    *off = A.lo[i];
}

void location(const Spec& s, int parity, int i, int* slot, u64* elem) {
    const Addr A = addr(s, parity);
    i64 e = static_cast<i64>(s.P) + A.lo[i];
    if (s.fp16 && (A.lo[i] & 1)) e -= 1;
    *slot = A.ls[i];
    *elem = static_cast<u64>(e);
}

std::vector<u32> gen_step(const Spec& s, int p, bool macro) {
    static const int exp_mode = std::getenv("CFD_GPU_EXP") ? std::atoi(std::getenv("CFD_GPU_EXP")) : 0;
    if (exp_mode) return gen_step_experiment(s, p, exp_mode);
    return s.pair ? gen_step_pair(s, p, macro) : gen_step_single(s, p, macro);
}

// ================================================================================================
//  Kernel de nodos de pared (= Solver::Impl::boundary_pass, un hilo por nodo)
// ================================================================================================
std::vector<u32> gen_boundary(const Spec& s, int p) {
    Kernel k(opts(s, s.wg_nodes));
    const Elem E = s.fp16 ? Elem::F16 : Elem::F32;
    const float isc = s.fp16 ? kInvScale : 1.0f, sc = s.fp16 ? kScale : 1.0f;
    Buf d[19];
    for (int q = 0; q < 19; ++q) d[q] = k.buffer(kBindSlot + q, E);
    const Buf flags = k.buffer(kBindFlags, Elem::U8, true);
    const Buf sid = k.buffer(kBindSid, Elem::U8, true);
    const Buf mot = k.buffer(kBindMotion, Elem::F32, true);
    const Buf par = k.buffer(kBindParams, Elem::F32, true);
    const Buf nodes = k.buffer(kBindNodes, Elem::U32, true);
    const Buf recs = k.buffer(kBindRecs, Elem::F32, false, true);
    k.push_constants(kPushBytes / 4, 0);

    const Addr C = addr(s, p), Nx = addr(s, p ^ 1);   // post-colisión del paso tt / cargas del paso tt+1
    const U step = k.pc_u(0);
    const U pb = U(kParamHeader) + step * U(kParamPerStep);
    const F rr = k.ldf(par, pb + U(kSR)), ug = k.ldf(par, pb + U(kSUg));
    const F rr1 = k.ldf(par, pb + U(kSR1)), ug1 = k.ldf(par, pb + U(kSUg1));
    const U nn = k.pc_u(1);   // nº de nodos
    const U i = k.global_id(0);

    k.if_(i < nn, [&] {
        const U b = i * U(kNodeWords);
        auto w = [&](u32 j) { return k.ldu(nodes, b + U(j)); };
        const U n = w(0), mask = w(1), xy = w(2), zk = w(3);
        const F fx = to_f(xy & U(0xFFFFu)), fy = to_f(xy >> U(16u)), fz = to_f(zk & U(0xFFFFu));
        const U kind = zk >> U(16u);
        const F nrx = bits_f(w(4)), nry = bits_f(w(5)), nrz = bits_f(w(6));
        const F yw = bits_f(w(7)), yw17 = bits_f(w(8)), sgeo = bits_f(w(9));
        U qw[5];
        for (u32 j = 0; j < 5; ++j) qw[j] = w(10 + j);
        auto qcode = [&](int kk) { return ext(qw[kk / 4], static_cast<u32>(8 * (kk % 4)), 8); };
        const U ids = w(15);
        U idslot[kSlots];
        for (u32 j = 0; j < kSlots; ++j) idslot[j] = ext(ids, 8 * j, 8);
        F g[19];
        for (int kk = 0; kk < 19; ++kk) g[kk] = k.ldf(d[C.ss[kk]], slot_index(s, n, C.so[kk]));
        auto ldn_idx = [&](int kk) { return slot_index(s, n, Nx.lo[kk]); };
        auto cf = [&](int kk, int a) { return F(float(L::c[kk][a])); };

        // ---- Destino SIN deslizamiento de cada enlace (Bouzidi lineal) ----
        VarF v0[19];
        if (s.interp) {
            for (int kk = 1; kk < 19; ++kk) {
                v0[kk] = k.var(F(0.0f));
                k.if_(bit(mask, static_cast<u32>(kk)), [&] {
                    const U sidx = offset_cell(n, s.off[kk]);
                    const B skip = bit(k.ldu(flags, sidx), 1) && (k.ldu(sid, sidx) == U(kGroundId));
                    k.if_(!skip, [&] {
                        const int kb = L::opp[kk];
                        const U qc = qcode(kk);
                        const F q = to_f(qc) * F(1.0f / 254.0f);
                        const F Av = g[kk];
                        k.if_else(qc == U(127u), [&] { v0[kk].set(Av); }, [&] {
                            const B nsol = !bit(k.ldu(flags, offset_cell(n, -s.off[kk])), 0);
                            k.if_else(q < F(0.5f) && nsol, [&] {
                                const F Cv = k.ldf(d[Nx.ls[kk]], ldn_idx(kk));
                                v0[kk].set(fma(F(2.0f) * q, Av - Cv, Cv));
                            }, [&] {
                                const F h = F(0.5f) / fmax(q, F(0.5f));
                                v0[kk].set(fma(h, Av - g[kb], g[kb]));
                            });
                        });
                    });
                });
            }
        }

        // ---- Modelo Slip: imposición exacta de la tensión de pared (Werner-Wengle) ----
        VarF usx = k.var(F(0.0f)), usy = k.var(F(0.0f)), usz = k.var(F(0.0f)), mcorr = k.var(F(0.0f));
        VarB slip_on = k.varb(B{k.c_bool(false)});
        if (s.slip) {
            k.if_(bit(kind, 1), [&] {
                F r = g[0], jx = F(0.0f), jy = F(0.0f), jz = F(0.0f);
                for (int kk = 1; kk < 19; ++kk) {
                    r = r + g[kk];
                    if (L::c[kk][0]) jx = L::c[kk][0] > 0 ? jx + g[kk] : jx - g[kk];
                    if (L::c[kk][1]) jy = L::c[kk][1] > 0 ? jy + g[kk] : jy - g[kk];
                    if (L::c[kk][2]) jz = L::c[kk][2] > 0 ? jz + g[kk] : jz - g[kk];
                }
                const F rho = F(1.0f) + r * F(isc);
                const F sc_u = F(isc) / rho;
                const F ux = jx * sc_u, uy = jy * sc_u, uz = jz * sc_u;
                const F un = ux * nrx + uy * nry + uz * nrz;
                const F utx = ux - nrx * un, uty = uy - nry * un, utz = uz - nrz * un;
                const F ut2 = utx * utx + uty * uty + utz * utz;
                k.if_(ut2 > F(1e-14f), [&] {
                    const F um = sqrt(ut2);
                    const F nu_w = k.ldf(par, U(kPNuW)), inv_nu_w = k.ldf(par, U(kPInvNuW)), a0 = k.ldf(par, U(kPWwA0));
                    const F xa = um / (a0 * yw17);
                    const F sx = sqrt(xa);
                    const F tw = fmax(nu_w * um / yw, xa * sx * sqrt(sx));
                    const F yplus = yw * sqrt(tw) * inv_nu_w;
                    const F slip_f = clamp((yplus - F(5.0f)) / F(25.0f), F(0.0f), F(1.0f)) * sgeo;
                    k.if_(slip_f > F(0.0f), [&] {
                        const F ium = F(1.0f) / um;
                        const F thx = utx * ium, thy = uty * ium, thz = utz * ium;
                        VarF G = k.var(F(0.0f)), Fns = k.var(F(0.0f)), P1 = k.var(F(0.0f)), W1 = k.var(F(0.0f));
                        VarF ax = k.var(F(0.0f)), ay = k.var(F(0.0f)), az = k.var(F(0.0f));
                        for (int kk = 1; kk < 19; ++kk) {
                            k.if_(bit(mask, static_cast<u32>(kk)), [&] {
                                k.if_(!bit(k.ldu(flags, offset_cell(n, s.off[kk])), 1), [&] {
                                    const F q = to_f(qcode(kk)) * F(1.0f / 254.0f);
                                    const F gq = select(q < F(0.5f), F(1.0f), F(0.5f) / q);
                                    const F ct = cf(kk, 0) * thx + cf(kk, 1) * thy + cf(kk, 2) * thz;
                                    const F pw = gq * F(L::w[kk]);
                                    G.set(G.get() + pw * ct * ct);
                                    P1.set(P1.get() + pw * ct);
                                    W1.set(W1.get() + pw);
                                    Fns.set(Fns.get() + (g[kk] + v0[kk].get()) * ct);
                                    ax.set(ax.get() + cf(kk, 0) * F(L::w[kk]));
                                    ay.set(ay.get() + cf(kk, 1) * F(L::w[kk]));
                                    az.set(az.get() + cf(kk, 2) * F(L::w[kk]));
                                });
                            });
                        }
                        const F w1 = W1.get(), p1 = P1.get();
                        const B hw = w1 > F(0.0f);
                        const F Gc = F(6.0f) * (G.get() - select(hw, p1 * p1 / w1, F(0.0f)));
                        const F area = F(6.0f) * sqrt(ax.get() * ax.get() + ay.get() * ay.get() + az.get() * az.get());
                        k.if_(Gc > F(1e-6f), [&] {
                            const F sv = clamp((Fns.get() * F(isc) - tw * area) / Gc, F(0.0f), slip_f * um);
                            usx.set(thx * sv); usy.set(thy * sv); usz.set(thz * sv);
                            mcorr.set(select(hw, F(6.0f) * sv * p1 / w1, F(0.0f)));
                            slip_on.set(sv > F(0.0f));
                        });
                    });
                });
            });
        }

        // ---- Ladd conservativo (implícito) y paredes móviles interpoladas: correcciones de masa ----
        VarF lcorr = k.var(F(0.0f)), ecorr = k.var(F(0.0f)), lcorr1 = k.var(F(0.0f));
        k.if_(bit(kind, 0), [&] {
            VarF Lm = k.var(F(0.0f)), Wm = k.var(F(0.0f)), Le = k.var(F(0.0f)), We = k.var(F(0.0f)), Lm1 = k.var(F(0.0f));
            for (int kk = 1; kk < 19; ++kk) {
                k.if_(bit(mask, static_cast<u32>(kk)), [&] {
                    const U sidx = offset_cell(n, s.off[kk]);
                    const U id = k.ldu(sid, sidx);
                    k.if_(bit(k.ldu(flags, sidx), 1) && conserve_of(k, mot, id), [&] {
                        const B ex = s.interp ? (id != U(kGroundId)) : B{k.c_bool(false)};
                        const Vel uw = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), select(ex, rr1, rr), select(ex, ug1, ug));
                        const F lk = F(6.0f * L::w[kk]) * (cf(kk, 0) * uw.x + cf(kk, 1) * uw.y + cf(kk, 2) * uw.z);
                        k.if_else(ex, [&] {
                            const F q = to_f(qcode(kk)) * F(1.0f / 254.0f);
                            const F gq = select(q < F(0.5f), F(1.0f), F(0.5f) / q);
                            Le.set(Le.get() + gq * lk);
                            We.set(We.get() + gq * F(L::w[kk]));
                        }, [&] {
                            Lm.set(Lm.get() + lk);
                            Wm.set(Wm.get() + F(L::w[kk]));
                            // Mismo término con la velocidad de pared del paso siguiente (Ladd que se escribe aquí).
                            const Vel u1 = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), rr1, ug1);
                            Lm1.set(Lm1.get() + F(6.0f * L::w[kk]) * (cf(kk, 0) * u1.x + cf(kk, 1) * u1.y + cf(kk, 2) * u1.z));
                        });
                    });
                });
            }
            lcorr.set(select(Wm.get() > F(0.0f), Lm.get() / Wm.get(), F(0.0f)));
            lcorr1.set(select(Wm.get() > F(0.0f), Lm1.get() / Wm.get(), F(0.0f)));
            ecorr.set(select(We.get() > F(0.0f), Le.get() / We.get(), F(0.0f)));
        });

        // ---- Por enlace: población entrante + fuerza (manométrica, Wen) acumulada por ranura de id ----
        VarF acc[kSlots][6];
        for (auto& sl : acc) for (auto& v : sl) v = k.var(F(0.0f));
        const float wref = s.gauge ? 0.0f : 1.0f, gi = s.galilean ? 1.0f : 0.0f;
        for (int kk = 1; kk < 19; ++kk) {
            k.if_(bit(mask, static_cast<u32>(kk)), [&] {
                const int kb = L::opp[kk];
                const U sidx = offset_cell(n, s.off[kk]);
                const U fs = k.ldu(flags, sidx);
                const U id = k.ldu(sid, sidx);
                const F Av = g[kk];
                const B moving = bit(fs, 1);
                VarF vin = k.var(F(0.0f)), vl = k.var(F(0.0f)), vlg = k.var(F(0.0f));
                VarF wx = k.var(F(0.0f)), wy = k.var(F(0.0f)), wz = k.var(F(0.0f));
                auto store_in = [&](F v) {   // escribe el destino y devuelve lo almacenado (redondeado a f16)
                    k.stf(d[Nx.ls[kb]], ldn_idx(kb), v);
                    return s.fp16 ? F(k.op(OpFConvert, k.tf, {k.op(OpFConvert, k.tf16, {v.id})}), 0) : v;
                };
                auto ld_in = [&]() { return k.ldf(d[Nx.ls[kb]], ldn_idx(kb)); };
                // Pared móvil con rebote IMPLÍCITO (cinta; ruedas sin rebote interpolado): la población entrante ya
                // está en su sitio (full-way); aquí se le suma el término de Ladd del paso siguiente, −Sc·l(t+1)
                // (= el +6w(c·u_w) − corrección de masa que la CPU suma al cargarla en block_scalar/Ground). Así el
                // kernel de celdas no lleva la ruta de paredes móviles (código más corto: ver docs/opt/gpu.md).
                auto ladd_next = [&](F in) {
                    const Vel u1 = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), rr1, ug1);
                    const F c1 = cf(kk, 0) * u1.x + cf(kk, 1) * u1.y + cf(kk, 2) * u1.z;
                    const F l1 = F(6.0f * L::w[kk]) * c1 - select(conserve_of(k, mot, id), lcorr1.get() * F(L::w[kk]), F(0.0f));
                    k.stf(d[Nx.ls[kb]], ldn_idx(kb), in - F(sc) * l1);
                };
                if (s.interp) {
                    k.if_else(moving && (id != U(kGroundId)), [&] {
                        // Rueda: Bouzidi + término de pared móvil g_q·6w(c·u_w) − corrección de masa.
                        const Vel uw = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), rr1, ug1);
                        const F q = to_f(qcode(kk)) * F(1.0f / 254.0f);
                        const F gq = select(q < F(0.5f), F(1.0f), F(0.5f) / q);
                        const F cw = cf(kk, 0) * uw.x + cf(kk, 1) * uw.y + cf(kk, 2) * uw.z;
                        const F mk = gq * (F(6.0f * L::w[kk]) * cw - select(conserve_of(k, mot, id), ecorr.get() * F(L::w[kk]), F(0.0f)));
                        const F in = store_in(v0[kk].get() - F(sc) * mk);
                        vin.set(in);
                        vlg.set((Av - in) * F(isc));
                        wx.set(uw.x); wy.set(uw.y); wz.set(uw.z);
                    }, [&] {
                        k.if_else(moving, [&] {
                            // Cinta: rebote implícito full-way (el destino ya guarda f_k*(n, t−1)) + Ladd en el kernel.
                            const Vel uw = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), rr, ug);
                            const F cw = cf(kk, 0) * uw.x + cf(kk, 1) * uw.y + cf(kk, 2) * uw.z;
                            const F l = F(6.0f * L::w[kk]) * cw - select(conserve_of(k, mot, id), lcorr.get() * F(L::w[kk]), F(0.0f));
                            vl.set(l); vlg.set(l);
                            const F in = ld_in();
                            vin.set(in);
                            ladd_next(in);
                            wx.set(uw.x); wy.set(uw.y); wz.set(uw.z);
                        }, [&] {
                            // Pared fija interpolada (+ deslizamiento del modelo Slip).
                            F v = v0[kk].get();
                            if (s.slip) {
                                const F q = to_f(qcode(kk)) * F(1.0f / 254.0f);
                                const F gq = select(q < F(0.5f), F(1.0f), F(0.5f) / q);
                                const F cu = cf(kk, 0) * usx.get() + cf(kk, 1) * usy.get() + cf(kk, 2) * usz.get();
                                v = v - select(slip_on.get(), F(sc) * gq * F(L::w[kk]) * (F(6.0f) * cu - mcorr.get()), F(0.0f));
                            }
                            vin.set(store_in(v));
                        });
                    });
                } else {
                    k.if_else(moving, [&] {
                        const Vel uw = wall_vel(k, mot, id, fx + cf(kk, 0), fy + cf(kk, 1), fz + cf(kk, 2), rr, ug);
                        const F cw = cf(kk, 0) * uw.x + cf(kk, 1) * uw.y + cf(kk, 2) * uw.z;
                        const F l = F(6.0f * L::w[kk]) * cw - select(conserve_of(k, mot, id), lcorr.get() * F(L::w[kk]), F(0.0f));
                        vl.set(l); vlg.set(l);
                        wx.set(uw.x); wy.set(uw.y); wz.set(uw.z);
                        const F in = ld_in();
                        vin.set(in);
                        ladd_next(in);
                    }, [&] { vin.set(ld_in()); });
                }
                const F a = Av * F(isc), bb = vin.get() * F(isc);
                const F fv = (a + bb + F(wref * 2.0f * L::w[kk])) - vl.get();
                const F lg = F(gi) * vlg.get();
                const F gx = lg * wx.get(), gy = lg * wy.get(), gz = lg * wz.get();
                const F ffx = fv * cf(kk, 0) - gx, ffy = fv * cf(kk, 1) - gy, ffz = fv * cf(kk, 2) - gz;
                // Momento propio de los enlaces móviles: −c_k × (gi·lg·u_w) (0 si la pared es fija: g = 0).
                const float cx = float(L::c[kk][0]), cy = float(L::c[kk][1]), cz = float(L::c[kk][2]);
                const F mx = -(F(cy) * gz - F(cz) * gy), my = -(F(cz) * gx - F(cx) * gz), mz = -(F(cx) * gy - F(cy) * gx);
                const F contrib[6] = {ffx, ffy, ffz, mx, my, mz};
                for (u32 sl = 0; sl < kSlots; ++sl) {
                    const B hit = id == idslot[sl];
                    for (int j = 0; j < 6; ++j) acc[sl][j].set(acc[sl][j].get() + select(hit, contrib[j], F(0.0f)));
                }
            });
        }
        // Volcado por (nodo, id): fuerza y momento Σ s×F = x_n × F + Σ(c_k × F_k)
        for (u32 sl = 0; sl < kSlots; ++sl) {
            const U rec = w(16 + sl);
            k.if_(rec != U(0xFFFFFFFFu), [&] {
                const F f0 = acc[sl][0].get(), f1 = acc[sl][1].get(), f2 = acc[sl][2].get();
                const U o = rec * U(6u);
                k.stf(recs, o, f0);
                k.stf(recs, o + U(1u), f1);
                k.stf(recs, o + U(2u), f2);
                k.stf(recs, o + U(3u), fy * f2 - fz * f1 + acc[sl][3].get());
                k.stf(recs, o + U(4u), fz * f0 - fx * f2 + acc[sl][4].get());
                k.stf(recs, o + U(5u), fx * f1 - fy * f0 + acc[sl][5].get());
            });
        }
    });
    return k.finish();
}

// ================================================================================================
//  Reducción de fuerzas: un grupo por trozo (≤ wg registros de un mismo id)
// ================================================================================================
std::vector<u32> gen_reduce(const Spec& s) {
    const u32 wg = 256;
    Kernel k(opts(s, wg));
    const Buf par = k.buffer(kBindParams, Elem::F32, true);
    const Buf recs = k.buffer(kBindRecs, Elem::F32, true);
    const Buf chunks = k.buffer(kBindChunks, Elem::U32, true);
    const Buf fout = k.buffer(kBindForces, Elem::F32, false, true);
    k.push_constants(kPushBytes / 4, 0);
    const u32 nsg = wg / s.sg;
    const Shared sh = k.shared(nsg * 6, true);
    const U step = k.pc_u(0);
    const U fo = bits_u(k.ldf(par, U(kParamHeader) + step * U(kParamPerStep) + U(kSFout)));
    const U c = k.workgroup_id(0);
    const U first = k.ldu(chunks, c * U(2u)), cnt = k.ldu(chunks, c * U(2u) + U(1u));
    const U li = k.local_index();
    const B valid = li < cnt;
    const U r = select(valid, first + li, first) * U(6u);
    const U sgid = k.subgroup_id();
    F sum[6];
    for (u32 j = 0; j < 6; ++j) sum[j] = k.sg_add(select(valid, k.ldf(recs, r + U(j)), F(0.0f)));
    k.if_(k.sg_elect(), [&] { for (u32 j = 0; j < 6; ++j) k.stf(sh, sgid * U(6u) + U(j), sum[j]); });
    k.barrier();
    k.if_(li == U(0u), [&] {
        for (u32 j = 0; j < 6; ++j) {
            F t = k.ldf(sh, U(j));
            for (u32 q = 1; q < nsg; ++q) t = t + k.ldf(sh, U(q * 6 + j));
            k.stf(fout, fo + c * U(6u) + U(j), t);
        }
    });
    return k.finish();
}

} // namespace cfd::gpu::lbmk
