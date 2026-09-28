// ============================================================================
//  tests/test_lbm.cpp — pruebas del solver LBM D3Q19 Esoteric-Pull.
//
//  Referencia independiente: LBM "A-B" (dos redes, pull ingenuo) en DOUBLE,
//  escalar, sin trucos. Los nodos sólidos de la referencia hacen una
//  "colisión de inversión" (f*_{opp k} = f_k): rebote FULL-WAY, que es lo que
//  el streaming in-place sin carreras realiza implícitamente (ver solver.cpp).
//  Mismas reglas de frontera escritas desde la especificación: entrada y
//  campo lejano en equilibrio (ρ=1, u=u∞(t)), salida en equilibrio con
//  ρ=1 y u de la celda x-1, suelo fijo/móvil, paredes móviles de Ladd,
//  esponja de viscosidad y rampa smoothstep de u∞.
//
//  Uso: build/lbm/test_lbm [filtro]   (devuelve ≠0 si algo falla)
// ============================================================================
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::lbm;
namespace L = cfd::lbm::d3q19;

static int g_fail = 0, g_pass = 0;
static const char* g_filter = nullptr;

static void report(const char* name, bool ok, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("%s %-34s %s\n", ok ? "PASS" : "FAIL", name, buf);
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}
static bool want(const char* name) { return !g_filter || std::strstr(name, g_filter); }

// ---------------------------------------------------------------------------------------------
//  Referencia A-B (double)
// ---------------------------------------------------------------------------------------------
struct RefLBM {
    enum Type : u8 { FLUID = 0, SOLID = 1, EQIN = 2, EQOUT = 3 };
    Config cfg;
    int nx = 0, ny = 0, nz = 0;
    i64 N = 0;
    std::vector<double> A, B;             // post-colisión: A = paso anterior, B = nuevo
    std::vector<u8> type, sid, nearwall;
    std::vector<double> rho, ux, uy, uz;  // macro del último paso
    std::vector<double> tau0;
    WallMotion motion[256];
    bool moving[256] = {};
    u64 t = 0;
    double F[256][3] = {};
    bool halfway = false;   // rebote half-way en sólidos FIJOS (control negativo del implícito; referencia del interpolado)
    bool gauge = false, galilean = false;   // fuerzas manométricas / término de Wen et al. (Config::force_*)
    const WallSdf* sdf = nullptr;           // rebote interpolado de Bouzidi en sólidos fijos (con halfway)
    // q del enlace n → s (misma regla y cuantización u8 que el solver: código/254).
    double link_q(int x, int y, int z, int sx, int sy, int sz) const {
        if (!sdf || !sdf->fn) return 0.5;
        const float dn = sdf->fn(sdf->ctx, Vec3(float(x), float(y), float(z))), ds = sdf->fn(sdf->ctx, Vec3(float(sx), float(sy), float(sz)));
        float q;
        if (!(dn > 0.0f) && !(ds < 0.0f)) q = 0.5f;
        else if (!(dn > 0.0f)) q = 0.0f;
        else if (!(ds < 0.0f)) q = 1.0f;
        else q = dn / (dn - ds);
        q = q < 0 ? 0 : (q > 1 ? 1 : q);
        return static_cast<double>(std::lround(q * 254.0f)) / 254.0;
    }
    // Peso del término de pared móvil de Bouzidi del enlace n → s: 1 (q < ½) o 1/(2q) (q ≥ ½).
    double link_gq(int x, int y, int z, int sx, int sy, int sz) const {
        const double q = link_q(x, y, z, sx, sy, sz);
        return q < 0.5 ? 1.0 : 0.5 / q;
    }
    // Población entrante por el enlace (n, kl) (kl = dirección de n hacia el sólido) a partir del post-colisión P.
    double bouzidi_in(const std::vector<double>& P, int x, int y, int z, int kl) const {
        const i64 n = idx(x, y, z);
        const int kb = L::opp[kl];
        const int sx = x + L::c[kl][0], sy = y + L::c[kl][1], sz = z + L::c[kl][2];
        const double q = sid[idx(sx, sy, sz)] == 255 ? 0.5 : link_q(x, y, z, sx, sy, sz);   // suelo: plano a mitad de enlace
        const double A_ = P[kl * N + n], B_ = P[kb * N + n];
        const int fx = x - L::c[kl][0], fy = y - L::c[kl][1], fz = z - L::c[kl][2];
        if (std::fabs(q - 127.0 / 254.0) < 1e-12) return A_;
        if (q < 0.5 && type[idx(fx, fy, fz)] != SOLID) {
            const double C_ = P[kl * N + idx(fx, fy, fz)];
            return 2 * q * A_ + (1 - 2 * q) * C_;
        }
        const double h = 0.5 / (q > 0.5 ? q : 0.5);
        return h * A_ + (1 - h) * B_;
    }

    i64 idx(int x, int y, int z) const { return x + static_cast<i64>(nx) * (y + static_cast<i64>(ny) * z); }
    double u_cur(u64 step) const {
        if (cfg.ramp_steps <= 0) return cfg.u_inf;
        double s = static_cast<double>(step) / cfg.ramp_steps;
        s = s < 0 ? 0 : (s > 1 ? 1 : s);
        return cfg.u_inf * s * s * (3 - 2 * s);
    }
    void wall_u(int id, int x, int y, int z, u64 step, double u[3]) const {
        u[0] = u[1] = u[2] = 0;
        const double uc = u_cur(step);
        if (id == 255) { if (cfg.ground == GroundMode::Moving) u[0] = uc; return; }
        if (!moving[id]) return;
        const double r = cfg.u_inf != 0 ? uc / cfg.u_inf : 1.0;
        const WallMotion& m = motion[id];
        const double px = x - m.center.x, py = y - m.center.y, pz = z - m.center.z;
        u[0] = r * (m.v.x + (m.omega.y * pz - m.omega.z * py));
        u[1] = r * (m.v.y + (m.omega.z * px - m.omega.x * pz));
        u[2] = r * (m.v.z + (m.omega.x * py - m.omega.y * px));
    }
    static void feq(double r, double u0, double u1, double u2, double* f) {
        const double uu = u0 * u0 + u1 * u1 + u2 * u2;
        for (int k = 0; k < L::Q; ++k) {
            const double cu = L::c[k][0] * u0 + L::c[k][1] * u1 + L::c[k][2] * u2;
            f[k] = L::wd[k] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
        }
    }
    void init(const Config& c, const u8* user_sid, const WallMotion* mot, const bool* mv) {
        cfg = c; nx = c.nx; ny = c.ny; nz = c.nz; N = static_cast<i64>(nx) * ny * nz;
        for (int i = 0; i < 256; ++i) { motion[i] = mot[i]; moving[i] = mv[i]; }
        type.assign(N, FLUID); sid.assign(N, 0);
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const i64 n = idx(x, y, z);
                    if (z == 0 && cfg.ground != GroundMode::None) { type[n] = SOLID; sid[n] = 255; continue; }
                    const bool far = y == 0 || y == ny - 1 || z == nz - 1 || z == 0;
                    if (x == 0 || far) { type[n] = EQIN; continue; }
                    if (x == nx - 1) { type[n] = EQOUT; continue; }
                    if (user_sid && user_sid[n]) { type[n] = SOLID; sid[n] = user_sid[n]; }
                }
        // Esponja (especificación): ν(x) = ν + (ν_max-ν)·s², s = smoothstep(xs, nx-1, x), ν_max = max(ν, 0.12).
        tau0.assign(nx, 0);
        const double sf = cfg.sponge_frac < 0 ? 0 : (cfg.sponge_frac > 0.9 ? 0.9 : cfg.sponge_frac);
        const int xs = static_cast<int>(static_cast<float>(nx) * (1.0f - static_cast<float>(sf)));
        const double numax = cfg.nu > 0.12 ? cfg.nu : 0.12;
        for (int x = 0; x < nx; ++x) {
            double nu = cfg.nu;
            if (cfg.sponge_frac > 0 && x > xs && nx - 1 > xs) {
                double s = static_cast<double>(x - xs) / (nx - 1 - xs);
                s = s * s * (3 - 2 * s);
                nu = cfg.nu + (numax - cfg.nu) * s * s;
            }
            tau0[x] = 3 * nu + 0.5;
        }
        // Celdas de fluido junto a una pared FIJA y sin vecinos móviles (kNearWall del solver).
        nearwall.assign(N, 0);
        for (int z = 1; z < nz - 1; ++z)
            for (int y = 1; y < ny - 1; ++y)
                for (int x = 1; x < nx - 1; ++x) {
                    const i64 n = idx(x, y, z);
                    if (type[n] != FLUID) continue;
                    bool st = false, mvn = false;
                    for (int k = 1; k < 19; ++k) {
                        const i64 m = idx(x + L::c[k][0], y + L::c[k][1], z + L::c[k][2]);
                        if (type[m] != SOLID) continue;
                        const bool mv = sid[m] == 255 ? cfg.ground == GroundMode::Moving : moving[sid[m]];
                        (mv ? mvn : st) = true;
                    }
                    nearwall[n] = st && !mvn;
                }
        A.assign(19 * N, 0); B.assign(19 * N, 0);
        rho.assign(N, 1); ux.assign(N, 0); uy.assign(N, 0); uz.assign(N, 0);
        double f0[19];
        feq(1.0, u_cur(0), 0, 0, f0);
        for (int k = 0; k < 19; ++k)
            for (i64 n = 0; n < N; ++n) A[k * N + n] = f0[k];
        t = 0;
    }
    i64 nb(int x, int y, int z, int k) const {   // n - c_k con envoltura por eje (sólo importa en fronteras)
        int xx = (x - L::c[k][0] + nx) % nx, yy = (y - L::c[k][1] + ny) % ny, zz = (z - L::c[k][2] + nz) % nz;
        return idx(xx, yy, zz);
    }
    void step() {
        const double uin = u_cur(t);
        const double K = 18.0 * std::sqrt(2.0) * cfg.cs_smag * cfg.cs_smag;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y) {
                double pu[3] = {0, 0, 0};
                for (int x = 0; x < nx; ++x) {
                    const i64 n = idx(x, y, z);
                    double fin[19];
                    for (int k = 0; k < 19; ++k) fin[k] = A[k * N + nb(x, y, z, k)];
                    if (type[n] == SOLID) {   // nodo de inversión (rebote full-way)
                        for (int k = 0; k < 19; ++k) B[L::opp[k] * N + n] = fin[k];
                        double uw[3];
                        wall_u(sid[n], x, y, z, t, uw);
                        rho[n] = 1; ux[n] = uw[0]; uy[n] = uw[1]; uz[n] = uw[2];
                        pu[0] = pu[1] = pu[2] = 0;
                        continue;
                    }
                    if (type[n] == FLUID) {
                        for (int k = 1; k < 19; ++k) {
                            const i64 s = nb(x, y, z, k);
                            if (type[s] == SOLID) {
                                const bool mv = sid[s] == 255 ? cfg.ground == GroundMode::Moving : moving[sid[s]];
                                // (v2) con halfway, también las paredes móviles que no son el suelo son explícitas
                                // (Bouzidi + término de pared móvil con peso g_q); el suelo móvil sigue full-way + Ladd.
                                const bool expl = halfway && (!mv || sid[s] != 255);
                                if (expl) fin[k] = sdf ? bouzidi_in(A, x, y, z, L::opp[k]) : A[L::opp[k] * N + n];
                                double uw[3];
                                wall_u(sid[s], x - L::c[k][0], y - L::c[k][1], z - L::c[k][2], t, uw);
                                const double gq = expl && mv ? link_gq(x, y, z, x - L::c[k][0], y - L::c[k][1], z - L::c[k][2]) : 1.0;
                                fin[k] += gq * 6 * L::wd[k] * (L::c[k][0] * uw[0] + L::c[k][1] * uw[1] + L::c[k][2] * uw[2]);
                            }
                        }
                    }
                    double r = 0, j[3] = {0, 0, 0};
                    for (int k = 0; k < 19; ++k) { r += fin[k]; for (int a = 0; a < 3; ++a) j[a] += fin[k] * L::c[k][a]; }
                    double u[3] = {j[0] / r, j[1] / r, j[2] / r};
                    const double cur[3] = {u[0], u[1], u[2]};
                    bool eq = false;
                    if (type[n] == EQIN) { r = 1; u[0] = uin; u[1] = u[2] = 0; eq = true; }
                    else if (type[n] == EQOUT) { r = 1; u[0] = pu[0]; u[1] = pu[1]; u[2] = pu[2]; eq = true; }
                    pu[0] = cur[0]; pu[1] = cur[1]; pu[2] = cur[2];
                    double fe[19];
                    feq(r, u[0], u[1], u[2], fe);
                    double post[19];
                    if (eq) {
                        for (int k = 0; k < 19; ++k) post[k] = fe[k];
                    } else {
                        double Pi[3][3] = {};
                        for (int k = 0; k < 19; ++k)
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b) Pi[a][b] += L::c[k][a] * L::c[k][b] * (fin[k] - fe[k]);
                        double q2 = 0;
                        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) q2 += Pi[a][b] * Pi[a][b];
                        const double t0 = tau0[x];
                        double tau = 0.5 * (t0 + std::sqrt(t0 * t0 + K * std::sqrt(q2) / r));
                        // (v2) Modelo Slip: Smagorinsky amortiguado en la 1.ª celda junto a paredes fijas.
                        if (cfg.wall_model == WallModel::Slip && nearwall[n]) tau = std::fmax(t0, 0.5 + 0.25 * (tau - 0.5));
                        const double om = 1.0 / tau;
                        for (int k = 0; k < 19; ++k) {
                            if (cfg.collision == Collision::BGK) {
                                post[k] = fin[k] - om * (fin[k] - fe[k]);
                            } else {
                                double qp = 0;
                                for (int a = 0; a < 3; ++a)
                                    for (int b = 0; b < 3; ++b)
                                        qp += (L::c[k][a] * L::c[k][b] - (a == b ? 1.0 / 3.0 : 0.0)) * Pi[a][b];
                                post[k] = fe[k] + (1 - om) * L::wd[k] * 4.5 * qp;
                            }
                        }
                    }
                    for (int k = 0; k < 19; ++k) B[k * N + n] = post[k];
                    rho[n] = r; ux[n] = u[0]; uy[n] = u[1]; uz[n] = u[2];
                }
            }
        std::swap(A, B);   // A = post(τ), B = post(τ-1)
        // Fuerzas: enlaces fluido n → sólido s = n + c_k:  (g_k(τ) + g_k(τ-1) - 6 w c·u_w) c_k
        for (auto& f : F) f[0] = f[1] = f[2] = 0;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const i64 s = idx(x, y, z);
                    if (type[s] != SOLID) continue;
                    double uw[3];
                    wall_u(sid[s], x, y, z, t, uw);
                    const bool mv = sid[s] == 255 ? cfg.ground == GroundMode::Moving : moving[sid[s]];
                    for (int k = 1; k < 19; ++k) {
                        const int xx = x - L::c[k][0], yy = y - L::c[k][1], zz = z - L::c[k][2];
                        if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
                        const i64 n = idx(xx, yy, zz);
                        if (type[n] != FLUID) continue;
                        const double lad = 6 * L::wd[k] * (L::c[k][0] * uw[0] + L::c[k][1] * uw[1] + L::c[k][2] * uw[2]);
                        // f_out(τ) + f_in(τ+1): full-way → f_in(τ+1) = f_out(τ-1) - Ladd; half-way → f_out(τ) (fijos) o
                        // Bouzidi − g_q·Ladd (móviles que no son el suelo).
                        const bool expl = halfway && (!mv || sid[s] != 255);
                        double fin_next;
                        if (expl) {
                            fin_next = sdf ? bouzidi_in(A, xx, yy, zz, k) : A[k * N + n];
                            if (mv) fin_next -= link_gq(xx, yy, zz, x, y, z) * lad;
                        } else {
                            fin_next = B[k * N + n] - lad;
                        }
                        const double v = A[k * N + n] + fin_next - (gauge ? 2 * L::wd[k] : 0.0);
                        const double lg = expl ? A[k * N + n] - fin_next : lad;   // f_out − f_in (Wen et al.)
                        for (int a = 0; a < 3; ++a) F[sid[s]][a] += v * L::c[k][a] - (galilean ? lg * uw[a] : 0.0);
                    }
                }
        ++t;
    }
};

// ---------------------------------------------------------------------------------------------
//  Utilidades de geometría para las pruebas
// ---------------------------------------------------------------------------------------------
static void add_sphere(std::vector<u8>& g, int nx, int ny, int nz, double cx, double cy, double cz, double R, u8 id) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const double dx = x - cx, dy = y - cy, dz = z - cz;
                if (dx * dx + dy * dy + dz * dz < R * R) g[x + static_cast<usize>(nx) * (y + static_cast<usize>(ny) * z)] = id;
            }
}
static void add_box(std::vector<u8>& g, int nx, int ny, int x0, int x1, int y0, int y1, int z0, int z1, u8 id) {
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) g[x + static_cast<usize>(nx) * (y + static_cast<usize>(ny) * z)] = id;
}

struct Diff { double drho = 0, du = 0; };
static Diff compare(const Solver& s, const RefLBM& r) {
    const FieldView v = s.field();
    Diff d;
    for (i64 n = 0; n < r.N; ++n) {
        d.drho = std::fmax(d.drho, std::fabs(v.rho[n] - r.rho[n]));
        const double du = std::fmax(std::fabs(v.ux[n] - r.ux[n]), std::fmax(std::fabs(v.uy[n] - r.uy[n]), std::fabs(v.uz[n] - r.uz[n])));
        if (!(du <= d.du)) d.du = du;   // captura NaN
    }
    return d;
}

// ---------------------------------------------------------------------------------------------
//  1. Identidades de la red (además de los static_assert de lattice.hpp)
// ---------------------------------------------------------------------------------------------
static void test_lattice() {
    double sw = 0, swc[3] = {}, swcc[3][3] = {};
    for (int k = 0; k < L::Q; ++k) {
        sw += L::w[k];
        for (int a = 0; a < 3; ++a) {
            swc[a] += L::w[k] * L::c[k][a];
            for (int b = 0; b < 3; ++b) swcc[a][b] += L::w[k] * L::c[k][a] * L::c[k][b];
        }
    }
    double err = std::fabs(sw - 1);
    for (int a = 0; a < 3; ++a) {
        err = std::fmax(err, std::fabs(swc[a]));
        for (int b = 0; b < 3; ++b) err = std::fmax(err, std::fabs(swcc[a][b] - (a == b ? 1.0 / 3.0 : 0.0)));
    }
    bool opp_ok = true;
    for (int k = 0; k < L::Q; ++k)
        for (int a = 0; a < 3; ++a) opp_ok &= L::c[L::opp[k]][a] == -L::c[k][a];
    report("1 identidades D3Q19", err < 1e-7 && opp_ok, "max err momentos %.2e, opuestos %s", err, opp_ok ? "ok" : "MAL");
}

// ---------------------------------------------------------------------------------------------
//  2. Esoteric-Pull vs referencia A-B
// ---------------------------------------------------------------------------------------------
static void run_reference_case(const char* name, Collision coll, float cs, GroundMode ground, Precision prec, double tol,
                               int ramp, bool interior255 = false, bool interp = false) {
    Config c;
    c.nx = 48; c.ny = 24; c.nz = 20;
    c.u_inf = 0.06f; c.nu = 0.02f; c.cs_smag = cs; c.collision = coll; c.precision = prec;
    c.ground = ground; c.sponge_frac = 0.2f; c.ramp_steps = ramp;
    // Esquema original (rebote implícito, fuerzas absolutas) salvo en el caso "interpolado": half-way explícito
    // en sólidos fijos (q = ½ sin distancia a la superficie) + fuerzas manométricas y galileanas. El modelo de
    // pared Slip (defecto) no actúa a este Re (y⁺ < 5 → sin deslizamiento): la referencia no lo necesita.
    c.bounce = interp ? BounceBack::Interpolated : BounceBack::Implicit;
    c.wall_model = interp ? WallModel::Slip : WallModel::None;
    c.force_gauge = interp; c.force_galilean = interp;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_box(g, c.nx, c.ny, 10, 15, 8, 14, 4, 9, 1);                         // obstáculo fijo
    // Robustez (revisión): celdas con id 255 FUERA de la capa z=0 (el contrato reserva 255 al suelo, pero un
    // voxelizador con 255 grupos lo produciría). Con suelo móvil se mueven como la cinta; sus vecinos NO pueden
    // usar el atajo vectorial de la cinta (corrección constante sólo en dirs 9/16, válida sólo para z-1).
    if (interior255) add_box(g, c.nx, c.ny, 35, 38, 3, 7, 1, 5, 255);
    for (int z = 0; z < c.nz; ++z)                                          // cilindro giratorio (eje y)
        for (int y = 6; y <= 17; ++y)
            for (int x = 0; x < c.nx; ++x) {
                const double dx = x - 28.0, dz = z - 9.0;
                if (dx * dx + dz * dz < 3.2 * 3.2) g[x + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] = 2;
            }
    WallMotion mot[256];
    bool mv[256] = {};
    mot[2].omega = Vec3(0, 0.012f, 0);
    mot[2].center = Vec3(28, 11.5f, 9);
    mv[2] = true;

    Solver s;
    s.init(c);
    // Con el rebote explícito el orden importa en el PRIMER paso: set_geometry escribe ya las entrantes
    // half-way de los sólidos fijos; si la rueda se declara móvil después, sus enlaces conservarían ese valor
    // en vez del del relleno (transitorio de 1 paso, irrelevante en la app, pero no idéntico a la referencia).
    if (interp) { s.set_wall_motion(2, mot[2]); s.set_geometry(g.data()); }
    else { s.set_geometry(g.data()); s.set_wall_motion(2, mot[2]); }
    RefLBM r;
    r.halfway = interp; r.gauge = interp; r.galilean = interp;
    r.init(c, g.data(), mot, mv);

    Diff worst;
    double fworst = 0;
    const int checkpoints[] = {1, 2, 7, 20, 51, 120, 201, 250};
    int done = 0;
    for (int cp : checkpoints) {
        s.step(cp - done, true);
        while (static_cast<int>(r.t) < cp) r.step();
        done = cp;
        const Diff d = compare(s, r);
        worst.drho = std::fmax(worst.drho, d.drho);
        if (!(d.du <= worst.du)) worst.du = d.du;
        // Fuerzas por id (obstáculo, cilindro, suelo): error relativo a la escala de la fuerza.
        for (int id : {1, 2, 255}) {
            const Vec3 fs = s.forces().force[id];
            const double sc = std::fmax(1e-3, std::sqrt(r.F[id][0] * r.F[id][0] + r.F[id][1] * r.F[id][1] + r.F[id][2] * r.F[id][2]));
            const double e = std::fmax(std::fabs(fs.x - r.F[id][0]), std::fmax(std::fabs(fs.y - r.F[id][1]), std::fabs(fs.z - r.F[id][2]))) / sc;
            if (!(e <= fworst)) fworst = e;
        }
    }
    const bool ok = worst.drho < tol && worst.du < tol && fworst < 100 * tol && !s.diverged();
    report(name, ok,
           "250 pasos: max|dρ| %.2e  max|du| %.2e  max err rel F %.2e (tol %.0e)", worst.drho, worst.du, fworst, tol);
}

// 2j: rebote interpolado de Bouzidi (q ≠ ½) contra la referencia: esfera fija fuera de la red (q en (0,1),
// ramas q < ½ y q ≥ ½, y el caso degenerado de n − c_k sólido) sobre suelo fijo, BGK y Regularizado.
struct RefSphere { float x, y, z, r; };
static float ref_sphere_sdf(const void* ctx, Vec3 p) {
    const RefSphere* e = static_cast<const RefSphere*>(ctx);
    return std::sqrt((p.x - e->x) * (p.x - e->x) + (p.y - e->y) * (p.y - e->y) + (p.z - e->z) * (p.z - e->z)) - e->r;
}
static void test_reference_bouzidi() {
    for (Collision coll : {Collision::BGK, Collision::Regularized}) {
        Config c;
        c.nx = 48; c.ny = 24; c.nz = 20;
        c.u_inf = 0.06f; c.nu = 0.02f; c.cs_smag = coll == Collision::BGK ? 0.0f : 0.16f; c.collision = coll; c.precision = Precision::FP32;
        c.ground = GroundMode::Static; c.sponge_frac = 0.2f; c.ramp_steps = 30;
        c.bounce = BounceBack::Interpolated; c.wall_model = WallModel::Slip; c.force_gauge = true; c.force_galilean = true;
        static const RefSphere sp{17.3f, 11.6f, 8.45f, 4.7f};
        const WallSdf wsdf{&ref_sphere_sdf, &sp};
        std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
        for (int z = 0; z < c.nz; ++z)
            for (int y = 0; y < c.ny; ++y)
                for (int x = 0; x < c.nx; ++x)
                    if (ref_sphere_sdf(&sp, Vec3(float(x), float(y), float(z))) < 0.0f) g[x + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] = 1;
        WallMotion mot[256];
        bool mv[256] = {};
        Solver s;
        s.init(c);
        s.set_geometry(g.data(), wsdf);
        RefLBM r;
        r.halfway = true; r.gauge = true; r.galilean = true; r.sdf = &wsdf;
        r.init(c, g.data(), mot, mv);
        Diff worst;
        double fworst = 0;
        int done = 0;
        for (int cp : {1, 2, 7, 20, 51, 120, 200}) {
            s.step(cp - done, true);
            while (static_cast<int>(r.t) < cp) r.step();
            done = cp;
            const Diff d = compare(s, r);
            worst.drho = std::fmax(worst.drho, d.drho);
            if (!(d.du <= worst.du)) worst.du = d.du;
            const Vec3 fs = s.forces().force[1];
            const double sc = std::fmax(1e-3, std::sqrt(r.F[1][0] * r.F[1][0] + r.F[1][1] * r.F[1][1] + r.F[1][2] * r.F[1][2]));
            const double e = std::fmax(std::fabs(fs.x - r.F[1][0]), std::fmax(std::fabs(fs.y - r.F[1][1]), std::fabs(fs.z - r.F[1][2]))) / sc;
            if (!(e <= fworst)) fworst = e;
        }
        const bool ok = worst.drho < 1e-5 && worst.du < 1e-5 && fworst < 1e-3 && !s.diverged();
        report(coll == Collision::BGK ? "2j EP Bouzidi (esfera) vs A-B BGK" : "2j EP Bouzidi (esfera) vs A-B Reg", ok,
               "200 pasos: max|dρ| %.2e  max|du| %.2e  max err rel F %.2e", worst.drho, worst.du, fworst);
    }
}

static void test_reference_negative() {
    Config c;
    c.nx = 48; c.ny = 24; c.nz = 20;
    c.u_inf = 0.06f; c.nu = 0.02f; c.cs_smag = 0.0f; c.collision = Collision::BGK; c.precision = Precision::FP32;
    c.ground = GroundMode::Static; c.sponge_frac = 0.2f; c.ramp_steps = 0;
    c.bounce = BounceBack::Implicit; c.wall_model = WallModel::None; c.force_gauge = false; c.force_galilean = false;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_box(g, c.nx, c.ny, 10, 15, 8, 14, 4, 9, 1);
    WallMotion mot[256];
    bool mv[256] = {};
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    RefLBM r;
    r.halfway = true;
    r.init(c, g.data(), mot, mv);
    double worst = 0;
    for (int cp : {2, 7, 20, 51}) {
        s.step(cp - static_cast<int>(s.steps()), true);
        while (static_cast<int>(r.t) < cp) r.step();
        worst = std::fmax(worst, compare(s, r).du);
    }
    report("2e control negativo (half-way)", worst > 100 * 1e-5,
           "referencia con rebote half-way DIFIERE: max|du| %.2e >> tol 1e-5 (la prueba 2 discrimina)", worst);
}

static void test_reference() {
    // Arranque impulsivo (rampa 0) = transitorio fuerte: máxima sensibilidad al esquema de rebote.
    run_reference_case("2a EP vs A-B BGK suelo movil", Collision::BGK, 0.0f, GroundMode::Moving, Precision::FP32, 1e-5, 0);
    run_reference_case("2b EP vs A-B Reg+Smag suelo fijo", Collision::Regularized, 0.16f, GroundMode::Static, Precision::FP32, 1e-5, 50);
    run_reference_case("2c EP vs A-B Reg sin suelo", Collision::Regularized, 0.0f, GroundMode::None, Precision::FP32, 1e-5, 0);
    run_reference_case("2d EP FP16S vs A-B (info)", Collision::BGK, 0.0f, GroundMode::Moving, Precision::FP16S, 2e-3, 0);
    run_reference_case("2f EP vs A-B id 255 fuera del suelo", Collision::BGK, 0.0f, GroundMode::Moving, Precision::FP32, 1e-5, 0,
                       true);
    // (revisión) FP16S + Regularizado + LES: única cobertura de noneq<Sc = 2^15> (escala plegada en Π) contra la referencia.
    run_reference_case("2g EP FP16S Reg+Smag vs A-B (info)", Collision::Regularized, 0.16f, GroundMode::Moving, Precision::FP16S, 2e-3,
                       50);
    // (fase 2) Pasada explícita de rebote (half-way en fijos y en las paredes móviles que no son el suelo, con el
    // término de pared móvil; cinta: full-way + Ladd), Smagorinsky amortiguado junto a paredes fijas (modelo Slip)
    // y fuerzas p − p∞ con el término galileano, contra la referencia half-way escrita desde la especificación.
    run_reference_case("2h EP interpolado (q=1/2) vs A-B half-way", Collision::BGK, 0.0f, GroundMode::Moving, Precision::FP32, 1e-5, 0,
                       false, true);
    run_reference_case("2i EP interpolado Reg+Smag suelo fijo", Collision::Regularized, 0.16f, GroundMode::Static, Precision::FP32, 1e-5, 50,
                       false, true);
    test_reference_bouzidi();
    test_reference_negative();
}

// ---------------------------------------------------------------------------------------------
//  3. Conservación de masa: cavidad cerrada con tapa móvil (paredes sólidas en todo el contorno)
// ---------------------------------------------------------------------------------------------
static double cavity_mass_drift(Precision prec, int steps, double* m0out) {
    Config c;
    c.nx = 40; c.ny = 40; c.nz = 40;
    c.u_inf = 0.0f; c.nu = 0.01f; c.cs_smag = 0.0f; c.collision = Collision::BGK; c.precision = prec;
    c.ground = GroundMode::None; c.sponge_frac = 0.0f; c.ramp_steps = 0;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    for (int z = 1; z <= c.nz - 2; ++z)
        for (int y = 1; y <= c.ny - 2; ++y)
            for (int x = 1; x <= c.nx - 2; ++x) {
                u8 id = 0;
                if (z == c.nz - 2) id = 2;   // tapa (cubre también los bordes superiores)
                else if (x == 1 || x == c.nx - 2 || y == 1 || y == c.ny - 2 || z == 1) id = 1;
                g[x + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] = id;
            }
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    WallMotion lid;
    lid.v = Vec3(0.05f, 0, 0);
    s.set_wall_motion(2, lid);
    const double m0 = s.total_mass();
    s.step(steps, true);
    const double m1 = s.total_mass();
    if (m0out) *m0out = m0;
    // Comprobación de que la tapa mueve el fluido (si no, la prueba sería trivial).
    const FieldView v = s.field();
    const i64 nc = 20 + 40 * (20 + 40 * 36);
    if (!(std::fabs(v.ux[nc]) > 1e-3)) return 1.0;
    return std::fabs(m1 - m0) / m0;
}
static void test_mass() {
    double m0 = 0;
    const double d32 = cavity_mass_drift(Precision::FP32, 1000, &m0);
    const double d16 = cavity_mass_drift(Precision::FP16S, 1000, nullptr);
    report("3 masa cavidad cerrada 1000 pasos", d32 < 1e-5, "FP32 deriva rel %.2e (M0=%.1f), FP16S deriva rel %.2e (info)", d32, m0, d16);
}

// ---------------------------------------------------------------------------------------------
//  4. Couette plano antisimétrico: placas en z=1 (-U/2) y z=nz-2 (+U/2), u∞ = 0.
//     Caudal neto nulo → no hace falta gradiente de presión: el perfil es lineal aunque las
//     fronteras abiertas (equilibrio con ρ=1) estén a pocos h. (Con una sola placa móvil y
//     entrada uniforme, las caras laterales abiertas inducen un gradiente adverso ~2%.)
// ---------------------------------------------------------------------------------------------
static void test_couette() {
    for (Collision coll : {Collision::BGK, Collision::Regularized}) {
        Config c;
        c.nx = 64; c.ny = 64; c.nz = 20;
        const float U = 0.05f;
        c.u_inf = 0.0f; c.nu = 1.0f / 6.0f; c.cs_smag = 0.0f; c.collision = coll; c.precision = Precision::FP32;
        c.ground = GroundMode::None; c.sponge_frac = 0.0f; c.ramp_steps = 0;
        std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
        add_box(g, c.nx, c.ny, 0, c.nx - 1, 0, c.ny - 1, 1, 1, 2);                   // placa inferior
        add_box(g, c.nx, c.ny, 0, c.nx - 1, 0, c.ny - 1, c.nz - 2, c.nz - 2, 1);     // placa superior
        // Parches centrales 16×16 de cada placa con ids propios (3 arriba, 4 abajo) y la MISMA velocidad:
        // su fuerza por intercambio de momento debe ser la tensión analítica τ = ρν U/h por el área
        // (comprobación independiente de la fórmula de fuerzas, incluido el término -6 w c·u_w de pared móvil).
        constexpr int p0 = 24, p1 = 39;
        add_box(g, c.nx, c.ny, p0, p1, p0, p1, c.nz - 2, c.nz - 2, 3);
        add_box(g, c.nx, c.ny, p0, p1, p0, p1, 1, 1, 4);
        Solver s;
        s.init(c);
        s.set_geometry(g.data());
        WallMotion m;
        m.v = Vec3(0.5f * U, 0, 0);
        s.set_wall_motion(1, m);
        s.set_wall_motion(3, m);
        m.v = Vec3(-0.5f * U, 0, 0);
        s.set_wall_motion(2, m);
        s.set_wall_motion(4, m);
        // OJO: forces_mean() promedia TODOS los pasos del último step(n); desde un arranque impulsivo el
        // transitorio (τ ~ U/√(πνt)) sesga la media +4..7 % → la media se toma sólo en los últimos 100 pasos.
        s.step(2900, false);
        s.step(100, true);
        const FieldView v = s.field();
        const double h = c.nz - 4;   // paredes en z=1.5 y z=nz-2.5
        double err = 0;
        for (int z = 2; z <= c.nz - 3; ++z) {
            const double ua = -0.5 * U + U * (z - 1.5) / h;
            const double un = v.ux[v.index(c.nx / 2, c.ny / 2, z)];
            err = std::fmax(err, std::fabs(un - ua) / U);
        }
        report(coll == Collision::BGK ? "4 Couette vs analitico BGK" : "4 Couette vs analitico Reg", err < 0.02 && !s.diverged(),
               "max err %.3f%% de U (h=%d celdas, 3000 pasos)", err * 100, static_cast<int>(h));
        // Tensión de pared: la placa superior (+U/2) es frenada por el fluido (Fx < 0); la inferior, arrastrada (+).
        // Las caras abiertas (entrada u=0 / salida de gradiente nulo) inducen un leve gradiente de presión en x
        // que suma la MISMA fuerza a ambas placas; la parte antisimétrica (|F_sup|+|F_inf|)/2 lo cancela.
        // (Referencia 1D periódica independiente en double, full-way y half-way: la fórmula es exacta.)
        const double area = static_cast<double>(p1 - p0 + 1) * (p1 - p0 + 1);
        const double tau_a = static_cast<double>(c.nu) * U / h * area;
        const double ftop = s.forces_mean().force[3].x, fbot = s.forces_mean().force[4].x;
        const double ef = std::fmax(std::fabs(ftop + tau_a), std::fabs(fbot - tau_a)) / tau_a;
        const double es = std::fabs(0.5 * (fbot - ftop) - tau_a) / tau_a;
        report(coll == Collision::BGK ? "4b Couette tension de pared BGK" : "4b Couette tension de pared Reg", ef < 0.02 && es < 0.01,
               "Fx parche sup %.5f inf %.5f vs analitico -/+%.5f (err max %.2f%%, antisim. %.2f%%)", ftop, fbot, tau_a, ef * 100,
               es * 100);
    }
}

// ---------------------------------------------------------------------------------------------
//  5/6. Esfera a Re=100 (sin suelo): Cd vs Schiller-Naumann 1.09; FP16S vs FP32
// ---------------------------------------------------------------------------------------------
static double sphere_cd(Precision prec, int nx, int nyz, double xc, int steps, double* blockage, double* secs) {
    Config c;
    c.nx = nx; c.ny = nyz; c.nz = nyz;
    const double D = 16, U = 0.05, Re = 100;
    c.u_inf = static_cast<float>(U); c.nu = static_cast<float>(U * D / Re); c.cs_smag = 0.0f;
    c.collision = Collision::Regularized; c.precision = prec; c.ground = GroundMode::None;
    c.sponge_frac = 0.12f; c.ramp_steps = 0;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_sphere(g, c.nx, c.ny, c.nz, xc, (c.ny - 1) * 0.5, (c.nz - 1) * 0.5, D * 0.5, 1);
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    const double t0 = now_sec();
    s.step(steps - 400, false);
    double fx = 0;
    const int navg = 400;
    s.step(navg, true);
    fx = s.forces_mean().force[1].x;
    if (secs) *secs = now_sec() - t0;
    const double A = 3.14159265358979 * D * D / 4;
    if (blockage) *blockage = A / (static_cast<double>(c.ny - 2) * (c.nz - 2));
    if (secs) std::printf("     (esfera D=16 en %dx%dx%d, x_c=%.0f, λ=d/H=%.3f, %.0f MLUPS)\n", nx, nyz, nyz, xc, D / (nyz - 2), s.last_mlups());
    if (s.diverged()) return -1;
    return fx / (0.5 * U * U * A);
}
static void test_sphere() {
    double blk = 0, secs = 0;
    const double cd = sphere_cd(Precision::FP32, 208, 112, 80, 4500, &blk, &secs);
    const double ref = 24.0 / 100 * (1 + 0.15 * std::pow(100.0, 0.687));
    report("5 esfera Re=100 Cd (FP32)", std::fabs(cd - ref) / ref < 0.15,
           "Cd=%.3f  Schiller-Naumann=%.3f  err=%.1f%%  bloqueo=%.1f%%  (%.1fs)", cd, ref, 100 * (cd - ref) / ref, 100 * blk, secs);
}
static void test_sphere_fp16() {
    double s16 = 0, s32 = 0;   // comparación relativa: dominio pequeño (mismo para ambas precisiones)
    const double cd16 = sphere_cd(Precision::FP16S, 160, 80, 40, 4000, nullptr, &s16);
    const double cd32 = sphere_cd(Precision::FP32, 160, 80, 40, 4000, nullptr, &s32);
    const double d = std::fabs(cd16 - cd32) / cd32;
    report("6 esfera FP16S vs FP32", d < 0.02, "Cd16=%.4f Cd32=%.4f dif=%.3f%%  (%.1fs + %.1fs)", cd16, cd32, d * 100, s16, s32);
}

// ---------------------------------------------------------------------------------------------
//  7. Estabilidad: cuerpo romo, ν=1e-5, u=0.1, Cs=0.16, Regularizado, 3000 pasos
// ---------------------------------------------------------------------------------------------
static void test_stability() {
    for (Precision prec : {Precision::FP16S, Precision::FP32}) {
        Config c;
        c.nx = 128; c.ny = 64; c.nz = 64;
        c.u_inf = 0.1f; c.nu = 1e-5f; c.cs_smag = 0.16f; c.collision = Collision::Regularized; c.precision = prec;
        c.ground = GroundMode::Moving; c.sponge_frac = 0.12f; c.ramp_steps = 300;
        std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
        add_box(g, c.nx, c.ny, 30, 45, 20, 43, 3, 22, 1);            // "coche" romo cerca del suelo
        add_sphere(g, c.nx, c.ny, c.nz, 34, 17, 5, 4.5, 2);           // rueda izquierda
        add_sphere(g, c.nx, c.ny, c.nz, 34, 46, 5, 4.5, 3);           // rueda derecha
        Solver s;
        s.init(c);
        s.set_geometry(g.data());
        WallMotion w;
        w.omega = Vec3(0, 0.1f / 4.5f, 0);
        w.center = Vec3(34, 17, 5);
        s.set_wall_motion(2, w);
        w.center = Vec3(34, 46, 5);
        s.set_wall_motion(3, w);
        const double t0 = now_sec();
        bool div = false;
        for (int i = 0; i < 30 && !div; ++i) { s.step(100, true); div = s.diverged(); }
        const FieldView v = s.field();
        double umax = 0;
        for (i64 n = 0; n < static_cast<i64>(c.nx) * c.ny * c.nz; ++n) {
            const double u2 = v.ux[n] * v.ux[n] + v.uy[n] * v.uy[n] + v.uz[n] * v.uz[n];
            if (!(u2 <= umax)) umax = u2;
        }
        umax = std::sqrt(umax);
        char nm[64];
        std::snprintf(nm, sizeof nm, "7 estabilidad nu=1e-5 %s", prec == Precision::FP32 ? "FP32" : "FP16S");
        report(nm, !div && umax < 0.35, "3000 pasos, |u|max=%.3f, Fx=%.3f, %.1f MLUPS (%.1fs)", umax, s.forces_mean().force[1].x,
               s.last_mlups(), now_sec() - t0);
    }
}

// ---------------------------------------------------------------------------------------------
//  8. Fuerzas por id: dos esferas iguales simétricas en y
// ---------------------------------------------------------------------------------------------
static void test_two_spheres() {
    Config c;
    c.nx = 128; c.ny = 96; c.nz = 64;
    c.u_inf = 0.05f; c.nu = 0.01f; c.cs_smag = 0.0f; c.collision = Collision::Regularized; c.precision = Precision::FP32;
    c.ground = GroundMode::None; c.sponge_frac = 0.12f; c.ramp_steps = 0;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    const double yc = (c.ny - 1) * 0.5, zc = (c.nz - 1) * 0.5;
    add_sphere(g, c.nx, c.ny, c.nz, 40, yc - 12, zc, 6, 1);
    add_sphere(g, c.nx, c.ny, c.nz, 40, yc + 12, zc, 6, 2);
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    s.step(1500, true);
    const Vec3 a = s.forces().force[1], b = s.forces().force[2];
    const double sc = std::fabs(a.x);
    const double e = std::fmax(std::fabs(a.x - b.x), std::fmax(std::fabs(a.y + b.y), std::fabs(a.z - b.z))) / sc;
    const bool repel = a.y < 0 && b.y > 0;   // las esferas se repelen lateralmente (flujo acelerado entre ellas... o no)
    report("8 fuerzas por id simetricas", e < 1e-3 && sc > 0, "F1=(%.4f,%.4f,%.4f) F2=(%.4f,%.4f,%.4f) asim=%.2e %s", a.x, a.y, a.z, b.x, b.y,
           b.z, e, repel ? "(repulsion)" : "(atraccion)");
}

// ---------------------------------------------------------------------------------------------
//  9. Determinismo: 1 hilo vs muchos, campos FP32 bit a bit
// ---------------------------------------------------------------------------------------------
static void test_determinism() {
    Config c;
    c.nx = 96; c.ny = 48; c.nz = 40;
    c.u_inf = 0.08f; c.nu = 1e-4f; c.cs_smag = 0.16f; c.collision = Collision::Regularized; c.precision = Precision::FP32;
    c.ground = GroundMode::Moving; c.sponge_frac = 0.12f; c.ramp_steps = 100;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_sphere(g, c.nx, c.ny, c.nz, 30, 23.5, 12, 7, 1);
    add_box(g, c.nx, c.ny, 50, 60, 15, 30, 2, 8, 2);
    auto run = [&](std::vector<float>& out, ForceSample& fo) {
        Solver s;
        s.init(c);
        s.set_geometry(g.data());
        WallMotion w; w.omega = Vec3(0, 0.005f, 0); w.center = Vec3(30, 23.5f, 12);
        s.set_wall_motion(1, w);
        s.step(150, true);
        const FieldView v = s.field();
        const usize n = static_cast<usize>(c.nx) * c.ny * c.nz;
        out.assign(v.rho, v.rho + n);
        out.insert(out.end(), v.ux, v.ux + n);
        out.insert(out.end(), v.uy, v.uy + n);
        out.insert(out.end(), v.uz, v.uz + n);
        fo = s.forces();
    };
    const int nthreads = pool().size();
    std::vector<float> a, b;
    ForceSample fa, fb;
    pool().stop();
    pool().start(1);
    run(a, fa);
    pool().stop();
    pool().start(nthreads);
    run(b, fb);
    const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    const bool fsame = std::memcmp(&fa.force, &fb.force, sizeof fa.force) == 0;
    report("9 determinismo 1 vs N hilos", same && fsame, "campos %s, fuerzas %s (1 vs %d hilos)", same ? "identicos" : "DISTINTOS",
           fsame ? "identicas" : "distintas", nthreads);
}

// ---------------------------------------------------------------------------------------------
//  10. set_geometry a mitad de simulación
// ---------------------------------------------------------------------------------------------
static void test_regeometry() {
    Config c;
    c.nx = 128; c.ny = 64; c.nz = 64;
    c.u_inf = 0.08f; c.nu = 5e-4f; c.cs_smag = 0.16f; c.collision = Collision::Regularized; c.precision = Precision::FP16S;
    c.ground = GroundMode::Moving; c.sponge_frac = 0.12f; c.ramp_steps = 200;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_sphere(g, c.nx, c.ny, c.nz, 40, 31.5, 20, 9, 1);
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    s.step(600, true);
    auto mean_ux = [&] {
        const FieldView v = s.field();
        double sum = 0; i64 cnt = 0;
        for (int z = 1; z < c.nz - 1; ++z)
            for (int y = 1; y < c.ny - 1; ++y)
                for (int x = 1; x < c.nx - 1; ++x) {
                    const usize n = v.index(x, y, z);
                    if (v.flags[n] & kSolid) continue;
                    sum += v.ux[n]; ++cnt;
                }
        return sum / cnt;
    };
    const double before = mean_ux();
    // Nueva geometría: la esfera se desplaza y aparece una caja (celdas sólido→fluido y fluido→sólido).
    std::fill(g.begin(), g.end(), 0);
    add_sphere(g, c.nx, c.ny, c.nz, 50, 31.5, 22, 9, 1);
    add_box(g, c.nx, c.ny, 70, 80, 20, 44, 1, 15, 2);
    s.set_geometry(g.data());
    s.step(1, true);
    const double just_after = mean_ux();
    s.step(600, true);
    const double after = mean_ux();
    const bool ok = !s.diverged() && just_after > 0.8 * before && after > 0.5 * c.u_inf && std::isfinite(after);
    report("10 set_geometry en marcha", ok, "<ux> antes %.4f, justo despues %.4f, +600 pasos %.4f, F2x=%.3f", before, just_after, after,
           s.forces().force[2].x);
}

// ---------------------------------------------------------------------------------------------
//  11. Cambios de estado en marcha: rampa de set_inflow, set_ground, paredes móviles, reset_flow
// ---------------------------------------------------------------------------------------------
static void test_state_changes() {
    Config c;
    c.nx = 96; c.ny = 48; c.nz = 40;
    c.u_inf = 0.06f; c.nu = 2e-4f; c.cs_smag = 0.16f; c.collision = Collision::Regularized; c.precision = Precision::FP16S;
    c.ground = GroundMode::None; c.sponge_frac = 0.12f; c.ramp_steps = 100;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_sphere(g, c.nx, c.ny, c.nz, 30, 23.5, 8, 5, 1);
    Solver s;
    s.init(c);
    s.set_geometry(g.data());
    bool ok = true;
    char why[256] = "";
    auto fail = [&](const char* m) { if (ok) std::snprintf(why, sizeof why, "%s", m); ok = false; };
    // Rampa inicial desde el reposo: smoothstep(0,100,t)·u∞.
    if (std::fabs(s.current_u_inf()) > 1e-7f) fail("u(0) != 0");
    s.step(50, true);
    if (std::fabs(s.current_u_inf() - 0.5f * c.u_inf) > 1e-6f) fail("u(50) != u/2");
    s.step(60, true);
    if (std::fabs(s.current_u_inf() - c.u_inf) > 1e-7f) fail("u(110) != u");
    // set_inflow: nueva rampa desde el valor actual.
    s.set_inflow(0.09f);
    const float u0 = s.current_u_inf();
    s.step(50, true);
    const float umid = s.current_u_inf();
    if (std::fabs(u0 - 0.06f) > 1e-6f || std::fabs(umid - 0.075f) > 1e-5f) fail("rampa de set_inflow");
    // Suelo: None → Moving → Static, con una "rueda" girando; la capa z=0 debe ser sólido 255.
    s.set_ground(GroundMode::Moving);
    WallMotion w; w.omega = Vec3(0, 0.01f, 0); w.center = Vec3(30, 23.5f, 8);
    s.set_wall_motion(1, w);
    s.step(200, true);
    {
        const FieldView v = s.field();
        const usize n = v.index(10, 10, 0);
        if (!(v.flags[n] & kSolid) || !(v.flags[n] & kMoving) || v.solid_id[n] != k_ground_id) fail("capa de suelo móvil");
        if (std::fabs(v.ux[n] - s.current_u_inf()) > 1e-6f) fail("u de la cinta en el campo macro");
    }
    s.set_ground(GroundMode::Static);
    s.clear_wall_motions();
    s.step(200, true);
    {
        const FieldView v = s.field();
        const usize n = v.index(10, 10, 0);
        if (!(v.flags[n] & kSolid) || (v.flags[n] & kMoving)) fail("suelo fijo");
        if (v.flags[v.index(30, 23, 4)] & kMoving) fail("rueda sigue móvil tras clear_wall_motions");
    }
    s.set_ground(GroundMode::None);
    s.step(200, true);
    if (s.diverged()) fail("divergió");
    const FieldView v = s.field();
    if (!(v.flags[v.index(10, 10, 0)] & kInlet)) fail("z=0 sin suelo debe ser campo lejano");
    // reset_flow: reposo y rampa reiniciada.
    s.reset_flow();
    if (s.steps() != 0 || std::fabs(s.current_u_inf()) > 1e-7f) fail("reset_flow");
    s.step(1, true);
    const FieldView v2 = s.field();
    if (std::fabs(v2.ux[v2.index(60, 20, 20)]) > 1e-3f) fail("reset_flow no deja el fluido en reposo");
    report("11 cambios de estado en marcha", ok, "%s", ok ? "rampas, suelo None/Moving/Static, paredes, reset: OK" : why);
}

// ---------------------------------------------------------------------------------------------
//  12. (revisión) Variantes de Solver::Tuning ≡ ruta por defecto, BIT A BIT
//      pair_blocks (block_vec2), prefetch, stores NT, max_threads (run_slots), FTZ/DAZ y grano:
//      ninguna estaba cubierta por la referencia A-B (todas se desactivan por defecto).
// ---------------------------------------------------------------------------------------------
static void test_tuning_variants() {
    struct Var { const char* name; Solver::Tuning t; };
    std::vector<Var> vars;
    { Solver::Tuning t; t.pair_blocks = 1; vars.push_back({"pares", t}); }
    { Solver::Tuning t; t.prefetch = 2; vars.push_back({"prefetch", t}); }
    { Solver::Tuning t; t.nt_macro = 1; vars.push_back({"NT", t}); }
    { Solver::Tuning t; t.max_threads = 3; vars.push_back({"max_threads", t}); }
    { Solver::Tuning t; t.ftz = 1; vars.push_back({"FTZ", t}); }
    { Solver::Tuning t; t.row_grain = 1; vars.push_back({"grano1", t}); }
    { Solver::Tuning t; t.row_grain = 7; t.pair_blocks = 1; t.nt_macro = 1; vars.push_back({"grano7+pares+NT", t}); }
    struct Cfg { Precision p; Collision c; float cs; };
    const Cfg cfgs[] = {{Precision::FP32, Collision::Regularized, 0.16f}, {Precision::FP16S, Collision::BGK, 0.0f},
                        {Precision::FP16S, Collision::Regularized, 0.16f}};
    bool ok = true;
    char why[256] = "";
    int nruns = 0;
    for (const Cfg& cf : cfgs) {
        Config c;
        c.nx = 96; c.ny = 40; c.nz = 32;
        c.u_inf = 0.08f; c.nu = 0.005f; c.cs_smag = cf.cs; c.collision = cf.c; c.precision = cf.p;
        c.ground = GroundMode::Moving; c.sponge_frac = 0.15f; c.ramp_steps = 30;
        std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
        add_sphere(g, c.nx, c.ny, c.nz, 30, 19.5, 12, 6, 1);                     // sólido fijo (bloques con máscara)
        add_sphere(g, c.nx, c.ny, c.nz, 55, 19.5, 5, 3.5, 2);                    // rueda girando (ruta escalar)
        auto run = [&](const Solver::Tuning& t, std::vector<float>& out, ForceSample& fo) {
            Solver s;
            s.init(c);
            s.set_tuning(t);
            s.set_geometry(g.data());
            WallMotion w; w.omega = Vec3(0, 0.08f / 3.5f, 0); w.center = Vec3(55, 19.5f, 5);
            s.set_wall_motion(2, w);
            s.step(40, false);
            s.step(20, true);
            const FieldView v = s.field();
            const usize n = static_cast<usize>(c.nx) * c.ny * c.nz;
            out.assign(v.rho, v.rho + n);
            out.insert(out.end(), v.ux, v.ux + n);
            out.insert(out.end(), v.uy, v.uy + n);
            out.insert(out.end(), v.uz, v.uz + n);
            fo = s.forces();
        };
        std::vector<float> ref, out;
        ForceSample fref, fo;
        run(Solver::Tuning{}, ref, fref);
        for (const Var& va : vars) {
            run(va.t, out, fo);
            ++nruns;
            const bool same = out.size() == ref.size() && std::memcmp(out.data(), ref.data(), ref.size() * sizeof(float)) == 0 &&
                              std::memcmp(&fo.force, &fref.force, sizeof fo.force) == 0;
            if (!same && ok) {
                std::snprintf(why, sizeof why, "variante '%s' difiere (%s %s)", va.name, cf.p == Precision::FP32 ? "FP32" : "FP16S",
                              cf.c == Collision::BGK ? "BGK" : "Reg");
                ok = false;
            }
        }
    }
    report("12 variantes de Tuning = defecto", ok, "%s", ok ? "pares/prefetch/NT/max_threads/FTZ/grano: bit a bit identicos" : why);
    (void)nruns;
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc > 1) g_filter = argv[1];
    pool().start();
    std::printf("test_lbm: %d hilos\n", pool().size());
    const double t0 = now_sec();
    if (want("1")) test_lattice();
    if (want("2")) test_reference();
    if (want("3")) test_mass();
    if (want("4")) test_couette();
    if (want("5")) test_sphere();
    if (want("6")) test_sphere_fp16();
    if (want("7")) test_stability();
    if (want("8")) test_two_spheres();
    if (want("9")) test_determinism();
    if (want("10")) test_regeometry();
    if (want("11")) test_state_changes();
    if (want("12")) test_tuning_variants();
    std::printf("\n%d PASS, %d FAIL  (%.1f s)\n", g_pass, g_fail, now_sec() - t0);
    pool().stop();
    return g_fail ? 1 : 0;
}
