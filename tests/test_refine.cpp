// ============================================================================
//  tests/test_refine.cpp — refinamiento local por bloques (lbm/refine.cpp).
//
//  [1] Flujo uniforme a través de cajas finas (1 y 2 niveles, apoyada en el suelo y flotante): sigue uniforme.
//  [2] Masa en una caja cerrada con una caja fina dentro (FP32).
//  [3] Couette plano que atraviesa la interfaz (paredes móviles arriba/abajo) frente a la solución analítica.
//  [4] Esfera a Re ≈ 100 dentro de una caja fina: Cd frente a la misma esfera en red uniforme fina y gruesa.
//  [5] Pulso acústico que cruza la interfaz: reflexión pequeña.
//  [6] Fuerzas: una barra que atraviesa la interfaz cuenta una vez (suma ≈ red uniforme fina).
//  [7] Reinicio del mismo solver refinado → uniforme: idéntico (bit a bit) a un solver uniforme nuevo.
//  [8] La iGPU rechaza limpiamente un solver con refinamiento local.
//
//  Uso: build/tests/test_refine [filtro]   (devuelve ≠0 si algo falla)
// ============================================================================
#include "lbm/solver.hpp"
#include "gpu/lbm_gpu.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::lbm;

static int g_fail = 0, g_pass = 0;
static const char* g_filter = nullptr;

static void report(const char* name, bool ok, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("%s %-40s %s\n", ok ? "PASS" : "FAIL", name, buf);
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}
static bool want(const char* name) { return !g_filter || std::strstr(name, g_filter); }

static Config base_cfg(int nx, int ny, int nz, float u, float nu) {
    Config c;
    c.nx = nx; c.ny = ny; c.nz = nz;
    c.u_inf = u;
    c.nu = nu;
    c.cs_smag = 0.10f;
    c.collision = Collision::Recursive;
    c.precision = Precision::FP32;
    c.ground = GroundMode::None;
    c.sponge_frac = 0.0f;
    c.ramp_steps = 0;
    c.wall_model = WallModel::None;
    c.bounce = BounceBack::Interpolated;
    return c;
}
static LevelBox box(int parent, int x0, int y0, int z0, int x1, int y1, int z1) {
    LevelBox b;
    b.parent = parent;
    b.lo[0] = x0; b.lo[1] = y0; b.lo[2] = z0;
    b.hi[0] = x1; b.hi[1] = y1; b.hi[2] = z1;
    return b;
}

// Máximo |u − U x̂|/U y |ρ − 1| en las celdas de fluido de todas las rejillas (campo de visualización).
static void field_dev(const Solver& s, float U, double& du, double& dr) {
    du = dr = 0;
    for (int g = 0; g < s.grids(); ++g) {
        const FieldView f = s.grid_field(g);
        for (int z = 1; z < f.nz - 1; ++z)
            for (int y = 1; y < f.ny - 1; ++y)
                for (int x = 1; x < f.nx - 1; ++x) {
                    const usize n = f.index(x, y, z);
                    if (f.flags[n] & kSolid) continue;
                    const double e = std::sqrt(double(f.ux[n] - U) * (f.ux[n] - U) + double(f.uy[n]) * f.uy[n] + double(f.uz[n]) * f.uz[n]) / U;
                    du = std::max(du, e);
                    dr = std::max(dr, std::fabs(double(f.rho[n]) - 1.0));
                }
    }
}

// ---------------------------------------------------------------------------------------------
// [1] Flujo uniforme
// ---------------------------------------------------------------------------------------------
static void test_uniform() {
    struct Case { const char* name; bool ground; int nb; LevelBox b[2]; Precision p; };
    const Case cases[] = {
        {"1a uniforme, 1 nivel, flotante", false, 1, {box(0, 20, 10, 10, 38, 21, 21)}, Precision::FP32},
        {"1b uniforme, 2 niveles, flotante", false, 2, {box(0, 16, 8, 8, 43, 23, 23), box(1, 10, 8, 8, 29, 23, 23)}, Precision::FP32},
        {"1c uniforme, 2 niveles, cinta", true, 2, {box(0, 16, 8, 1, 43, 23, 18), box(1, 10, 8, 1, 29, 23, 16)}, Precision::FP32},
        {"1d uniforme, 2 niveles, FP16S", true, 2, {box(0, 16, 8, 1, 43, 23, 18), box(1, 10, 8, 1, 29, 23, 16)}, Precision::FP16S},
    };
    for (const Case& cs : cases) {
        if (!want(cs.name)) continue;
        const float U = 0.08f;
        Config c = base_cfg(64, 32, 32, U, 1e-3f);
        c.ground = cs.ground ? GroundMode::Moving : GroundMode::None;
        c.precision = cs.p;
        c.n_boxes = cs.nb;
        for (int i = 0; i < cs.nb; ++i) c.boxes[i] = cs.b[i];
        Solver s;
        s.init(c);
        s.set_geometry(nullptr);
        for (int g = 1; g < s.grids(); ++g) s.set_grid_geometry(g, nullptr, WallSdf{});
        // Con la cinta, el relleno inicial suma dos veces el término de Ladd sobre z = 1 (artefacto del arranque de la red
        // base, también sin refinamiento): ese transitorio tarda ~900 pasos en salir por la salida.
        s.step(cs.ground ? 1200 : 400);
        double du, dr;
        field_dev(s, U, du, dr);
        const double tol = cs.p == Precision::FP32 ? 1e-3 : 3e-3;
        report(cs.name, du < tol && dr < 1e-3 && !s.diverged(), "max|du|/U = %.2e, max|dρ| = %.2e (%d rejillas)", du, dr, s.grids());
    }
}

// Geometría: caja de sólidos del usuario (inclusive) en una rejilla nx×ny×nz.
static void add_box(std::vector<u8>& g, int nx, int ny, int x0, int x1, int y0, int y1, int z0, int z1, u8 id) {
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) g[static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * z)] = id;
}

// ---------------------------------------------------------------------------------------------
// [2] Masa en una caja cerrada (paredes del usuario) con 1 y 2 niveles dentro, arranque impulsivo (chapoteo).
// ---------------------------------------------------------------------------------------------
static void test_mass() {
    for (int nb = 1; nb <= 2; ++nb) {
        const char* name = nb == 1 ? "2a masa, caja cerrada, 1 nivel" : "2b masa, caja cerrada, 2 niveles";
        if (!want(name)) continue;
        const int nx = 48, ny = 32, nz = 32;
        Config c = base_cfg(nx, ny, nz, 0.06f, 0.005f);
        c.n_boxes = nb;
        c.boxes[0] = box(0, 12, 9, 9, 33, 22, 22);
        c.boxes[1] = box(1, 8, 8, 8, 27, 19, 19);
        Solver s;
        s.init(c);
        std::vector<u8> g(static_cast<usize>(nx) * ny * nz, 0);
        add_box(g, nx, ny, 0, nx - 1, 0, ny - 1, 0, 1, 1);
        add_box(g, nx, ny, 0, nx - 1, 0, ny - 1, nz - 2, nz - 1, 1);
        add_box(g, nx, ny, 0, nx - 1, 0, 1, 0, nz - 1, 1);
        add_box(g, nx, ny, 0, nx - 1, ny - 2, ny - 1, 0, nz - 1, 1);
        add_box(g, nx, ny, 0, 1, 0, ny - 1, 0, nz - 1, 1);
        add_box(g, nx, ny, nx - 2, nx - 1, 0, ny - 1, 0, nz - 1, 1);
        s.set_geometry(g.data());
        for (int k = 1; k < s.grids(); ++k) s.set_grid_geometry(k, nullptr, WallSdf{});
        s.reset_flow();   // u = u∞ en todo el fluido: la caja cerrada chapotea (ondas y torbellinos que cruzan la interfaz)
        const double m0 = s.total_mass();
        s.step(2000, false);
        const double m1 = s.total_mass();
        s.step(2000, false);
        const double m2 = s.total_mass();
        const double d1 = (m1 - m0) / m0, d2 = (m2 - m1) / m0;
        report(name, std::fabs(d1) < 2e-5 && std::fabs(d2) < 2e-6 && !s.diverged(),
               "Δm/m: transitorio (0-2000) %+.2e, después (2000-4000) %+.2e", d1, d2);
    }
}

// ---------------------------------------------------------------------------------------------
// [3] Couette plano (placas en z a ∓U/2, u∞ = 0) con una caja fina en medio del canal: el perfil lineal cruza la
//     interfaz. Interpolación trilineal exacta para un perfil lineal → el error mide el reescalado del no equilibrio.
// ---------------------------------------------------------------------------------------------
static void test_couette() {
    for (int nb = 1; nb <= 2; ++nb) {
        const char* name = nb == 1 ? "3a Couette a traves de 1 nivel" : "3b Couette a traves de 2 niveles";
        if (!want(name)) continue;
        const int nx = 64, ny = 32, nz = 26;
        const float U = 0.05f;
        Config c = base_cfg(nx, ny, nz, 0.0f, 1.0f / 6.0f);
        c.cs_smag = 0.0f;
        c.n_boxes = nb;
        c.boxes[0] = box(0, 20, 9, 6, 42, 22, 19);
        c.boxes[1] = box(1, 12, 6, 6, 30, 21, 21);
        Solver s;
        s.init(c);
        std::vector<u8> g(static_cast<usize>(nx) * ny * nz, 0);
        add_box(g, nx, ny, 0, nx - 1, 0, ny - 1, 1, 1, 2);
        add_box(g, nx, ny, 0, nx - 1, 0, ny - 1, nz - 2, nz - 2, 1);
        s.set_geometry(g.data());
        for (int k = 1; k < s.grids(); ++k) s.set_grid_geometry(k, nullptr, WallSdf{});
        WallMotion m;
        m.v = Vec3(0.5f * U, 0, 0);
        s.set_wall_motion(1, m);
        m.v = Vec3(-0.5f * U, 0, 0);
        s.set_wall_motion(2, m);
        s.step(4000, true);
        // Perfil analítico: paredes en z = 1.5 y z = nz − 2.5 (celdas de la red base).
        const double h = nz - 4;
        double err = 0, errc = 0;
        for (int k = 0; k < s.grids(); ++k) {
            const GridInfo gi = s.grid_info(k);
            const FieldView f = s.grid_field(k);
            // Columna en el centro en x e y de cada rejilla, todas las celdas de fluido (incluida la capa fantasma).
            const int xi = f.nx / 2, yi = f.ny / 2;
            for (int z = 0; z < f.nz; ++z) {
                const usize n = f.index(xi, yi, z);
                if (f.flags[n] & kSolid) continue;
                const double z0 = gi.org.z + gi.scale * z;   // celdas de la red base
                if (z0 < 1.6 || z0 > nz - 2.6) continue;
                const double ua = -0.5 * U + U * (z0 - 1.5) / h;
                const double e = std::fabs(f.ux[n] - ua) / U;
                err = std::max(err, e);
                if (k == 0) errc = std::max(errc, e);
            }
        }
        report(name, err < 0.01 && !s.diverged(), "max err %.3f%% de U (red base %.3f%%), %d rejillas", err * 100, errc * 100, s.grids());
    }
}

// ---------------------------------------------------------------------------------------------
// Esfera con su superficie real (rebote interpolado) en cualquier rejilla: centro y radio en celdas de la red base.
// ---------------------------------------------------------------------------------------------
struct SphereSdf { Vec3 c; float r; };
static float sphere_sdf(const void* ctx, Vec3 p) { const SphereSdf* s = static_cast<const SphereSdf*>(ctx); return length(p - s->c) - s->r; }
struct GridSphere { std::vector<u8> g; SphereSdf sdf; };
// Voxeliza la esfera (c0, r0 en celdas de la red base) en la rejilla g del solver.
static void grid_sphere(const Solver& s, int k, Vec3 c0, float r0, GridSphere& out) {
    const GridInfo gi = s.grid_info(k);
    out.sdf.c = (c0 - gi.org) * (1.0f / gi.scale);
    out.sdf.r = r0 / gi.scale;
    out.g.assign(gi.cells(), 0);
    for (int z = 0; z < gi.nz; ++z)
        for (int y = 0; y < gi.ny; ++y)
            for (int x = 0; x < gi.nx; ++x)
                if (sphere_sdf(&out.sdf, Vec3(float(x), float(y), float(z))) < 0.0f)
                    out.g[static_cast<usize>(x) + static_cast<usize>(gi.nx) * (static_cast<usize>(y) + static_cast<usize>(gi.ny) * z)] = 1;
}

// Cd de una esfera a Re = 100 (u = 0.05, sin LES) con D celdas de diámetro en la red base (nx×ny×nz), opcionalmente con
// una caja fina a su alrededor (lo/hi en celdas de la red base). Media de los últimos `avg` pasos.
static double sphere_cd(int nx, int nyz, float D, float xc, const LevelBox* b, int steps, int avg, double* secs, double* mlups) {
    const float U = 0.05f;
    Config c = base_cfg(nx, nyz, nyz, U, U * D / 100.0f);
    c.cs_smag = 0.0f;
    c.sponge_frac = 0.12f;
    if (b) { c.n_boxes = 1; c.boxes[0] = *b; }
    Solver s;
    s.init(c);
    const Vec3 c0(xc, 0.5f * (nyz - 1), 0.5f * (nyz - 1));
    std::vector<GridSphere> gs(static_cast<usize>(s.grids()));
    for (int k = 0; k < s.grids(); ++k) {
        grid_sphere(s, k, c0, 0.5f * D, gs[static_cast<usize>(k)]);
        s.set_grid_geometry(k, gs[static_cast<usize>(k)].g.data(), WallSdf{&sphere_sdf, &gs[static_cast<usize>(k)].sdf});
    }
    s.reset_flow();
    const double t0 = now_sec();
    s.step(steps - avg, false);
    s.step(avg, false);
    if (secs) *secs = now_sec() - t0;
    if (mlups) *mlups = s.last_mlups();
    if (s.diverged()) return -1;
    const double A = 3.14159265358979 * D * D / 4;
    return s.forces_mean().force[1].x / (0.5 * U * U * A);
}

// ---------------------------------------------------------------------------------------------
// [4] Esfera a Re = 100 (estacionaria): D = 8 celdas en la red base con una caja fina (D = 16 en la fina) frente a la
//     misma esfera en red uniforme fina (D = 16, mismo dominio físico) y uniforme gruesa (D = 8).
// ---------------------------------------------------------------------------------------------
static void test_sphere() {
    const char* name = "4 esfera Re=100: refinada vs uniforme";
    if (!want(name)) return;
    const int nx = 96, nyz = 56;
    const float D = 8, xc = 32;
    const LevelBox b = box(0, 22, 16, 16, 60, 39, 39);
    double tr = 0, tf = 0, tg = 0, mr = 0, mf = 0, mg = 0;
    const double cd_r = sphere_cd(nx, nyz, D, xc, &b, 2400, 200, &tr, &mr);
    const double cd_g = sphere_cd(nx, nyz, D, xc, nullptr, 2400, 200, &tg, &mg);
    // Uniforme fina: la celda fina i está en i/2 − ¼ de la base (mismo dominio [−½, n−½]).
    const double cd_f = sphere_cd(2 * nx, 2 * nyz, 2 * D, 2 * xc + 0.5f, nullptr, 4800, 400, &tf, &mf);
    const double e = std::fabs(cd_r - cd_f) / cd_f;
    report(name, e < 0.03, "Cd refinada %.4f, uniforme fina %.4f (err %.2f%%), uniforme gruesa %.4f (%.2f%%); %.1f / %.1f / %.1f s",
           cd_r, cd_f, 100 * e, cd_g, 100 * (cd_g - cd_f) / cd_f, tr, tf, tg);
}

// ---------------------------------------------------------------------------------------------
// [5] Pulso acústico plano (ρ gaussiano, u = 0) que llega a la cara de una caja fina: la señal reflejada en una sonda
//     aguas arriba (diferencia con la misma simulación sin caja) debe ser pequeña frente a la amplitud incidente, y la
//     onda transmitida dentro de la caja debe coincidir con la de la red uniforme.
// ---------------------------------------------------------------------------------------------
// Escribe el equilibrio ρ(x), u = 0 en las poblaciones de la red base (FP32) antes del primer paso (paridad 0).
static void set_pulse(Solver& s, float x0, float sig, float A) {
    const ExternalView v = s.external_view();
    const Config& c = s.config();
    const i64 nx = c.nx, nxny = static_cast<i64>(c.nx) * c.ny;
    float* d = static_cast<float*>(v.ddf);
    auto slot = [&](int k, i64 n) -> float& {   // lo que el paso 0 carga como f_k de la celda n
        if (k == 0) return d[v.P + n];
        if (k & 1) return d[(static_cast<i64>(k) + 1) * v.S + v.P + n];
        return d[static_cast<i64>(k - 1) * v.S + v.P + n + v.off[k - 1]];
    };
    constexpr float w[19] = {1.f / 3, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 36, 1.f / 36, 1.f / 36,
                             1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36};
    for (i64 n = 0; n < nxny * c.nz; ++n) {
        const float x = static_cast<float>(n % nx);
        const float dr = A * std::exp(-(x - x0) * (x - x0) / (sig * sig));
        for (int k = 0; k < 19; ++k) slot(k, n) = w[k] * dr;   // f̃ = f − w = w·(ρ − 1) con u = 0
    }
}
static void test_pulse() {
    const char* name = "5 pulso acustico a traves de la interfaz";
    if (!want(name)) return;
    const int nx = 160, ny = 32, nz = 32;
    const float A = 1e-3f, x0 = 50.0f, sig = 4.0f;
    const int xp = 70, xin = 104, yc = ny / 2, zc = nz / 2;
    const int nsteps = 190;
    std::vector<double> sig_r[2], sig_i[2];
    for (int run = 0; run < 2; ++run) {
        Config c = base_cfg(nx, ny, nz, 0.0f, 1e-3f);
        c.cs_smag = 0.0f;
        if (run == 1) { c.n_boxes = 1; c.boxes[0] = box(0, 90, 6, 6, 136, 25, 25); }
        Solver s;
        s.init(c);
        s.set_geometry(nullptr);
        for (int k = 1; k < s.grids(); ++k) s.set_grid_geometry(k, nullptr, WallSdf{});
        s.reset_flow();
        set_pulse(s, x0, sig, A);
        for (int t = 0; t < nsteps; ++t) {
            s.step(1, true);
            const FieldView f = s.field();
            sig_r[run].push_back(f.rho[f.index(xp, yc, zc)] - 1.0);
            sig_i[run].push_back(f.rho[f.index(xin, yc, zc)] - 1.0);
        }
    }
    // Ventana de la reflexión en la sonda aguas arriba: tras el paso del pulso incidente y antes de que vuelva el eco de la
    // entrada (x = 0): la onda llega a la caja en ~(90 − 50)·√3 = 69 pasos y su eco a la sonda en ~104.
    // Transmitida: pico dentro de la caja frente a la red uniforme gruesa. (Con ω_b = 1 en todos los niveles la viscosidad de
    // volumen FÍSICA de la rejilla fina es la mitad → el pulso se amortigua algo menos dentro; y el dominio estrecho con
    // caras de campo lejano difracta la onda de forma distinta según la resolución: sólo una cota amplia.)
    double inc = 0, refl = 0, tr0 = 0, tr1 = 0;
    for (int t = 0; t < nsteps; ++t) {
        inc = std::max(inc, std::fabs(sig_r[0][static_cast<usize>(t)]));
        if (t >= 85 && t < 185) refl = std::max(refl, std::fabs(sig_r[1][static_cast<usize>(t)] - sig_r[0][static_cast<usize>(t)]));
        tr0 = std::max(tr0, std::fabs(sig_i[0][static_cast<usize>(t)]));
        tr1 = std::max(tr1, std::fabs(sig_i[1][static_cast<usize>(t)]));
    }
    report(name, refl < 0.01 * inc && std::fabs(tr1 / tr0 - 1.0) < 0.10, "reflejada %.2f%% del incidente (%.2e); transmitida %.3f × la uniforme",
           100 * refl / inc, inc, tr1 / tr0);
}

// ---------------------------------------------------------------------------------------------
// [6] Fuerzas de un cuerpo que atraviesa la interfaz: barra cuadrada a lo largo de x, la mitad dentro de la caja fina.
//     Cada enlace de pared debe contar una sola vez (en la rejilla más fina que lo contiene): la resistencia debe quedar
//     cerca de la de la red uniforme fina (con doble cuenta la mitad trasera contaría dos veces: ~+35 %). Límite conocido
//     (docs/FISICA.md §1.6): junto a la línea donde la interfaz corta una pared no deslizante, la plantilla trilineal
//     descarta las esquinas sólidas y la fricción local sale unos % alta (medido: barra entera dentro de la caja → 0.1 %
//     de la uniforme fina; cortada por la mitad → ~7 %).
// ---------------------------------------------------------------------------------------------
static void test_force_crossing() {
    const char* name = "6 fuerza de una barra que cruza la interfaz";
    if (!want(name)) return;
    const int nx = 112, ny = 40, nz = 40;
    const float U = 0.05f;
    double fx[3] = {0, 0, 0};
    for (int run = 0; run < 3; ++run) {
        // run 0: uniforme gruesa; 1: con caja fina (la mitad trasera de la barra dentro); 2: uniforme fina.
        const int f = run == 2 ? 2 : 1;
        Config c = base_cfg(nx * f, ny * f, nz * f, U, 0.004f * f);
        c.cs_smag = 0.0f;
        c.sponge_frac = 0.12f;
        if (run == 1) { c.n_boxes = 1; c.boxes[0] = box(0, 40, 10, 10, 78, 29, 29); }
        Solver s;
        s.init(c);
        // Barra: x ∈ [20, 60), y, z ∈ [18, 22) en celdas de la red base (caras en múltiplos enteros ±½ de la base).
        auto bar = [&](int k, std::vector<u8>& g) {
            const GridInfo gi = s.grid_info(k);
            g.assign(gi.cells(), 0);
            for (int z = 0; z < gi.nz; ++z)
                for (int y = 0; y < gi.ny; ++y)
                    for (int x = 0; x < gi.nx; ++x) {
                        const Vec3 p = run == 2 ? Vec3(x * 0.5f - 0.25f, y * 0.5f - 0.25f, z * 0.5f - 0.25f) : gi.org + Vec3(float(x), float(y), float(z)) * gi.scale;
                        if (p.x > 19.5f && p.x < 59.5f && p.y > 17.5f && p.y < 21.5f && p.z > 17.5f && p.z < 21.5f)
                            g[static_cast<usize>(x) + static_cast<usize>(gi.nx) * (static_cast<usize>(y) + static_cast<usize>(gi.ny) * z)] = 1;
                    }
        };
        std::vector<std::vector<u8>> gg(static_cast<usize>(s.grids()));
        for (int k = 0; k < s.grids(); ++k) { bar(k, gg[static_cast<usize>(k)]); s.set_grid_geometry(k, gg[static_cast<usize>(k)].data(), WallSdf{}); }
        s.reset_flow();
        s.step(1800 * f, false);
        s.step(200 * f, false);
        fx[run] = s.forces_mean().force[1].x * (run == 2 ? 0.25 : 1.0);   // uniforme fina → unidades de la gruesa
    }
    const double e = std::fabs(fx[1] - fx[2]) / fx[2];
    report(name, e < 0.10, "Fx refinada %.5f, uniforme fina %.5f (err %.1f%%), uniforme gruesa %.5f", fx[1], fx[2], 100 * e, fx[0]);
}

// ---------------------------------------------------------------------------------------------
// [7] Reinicio: el mismo objeto Solver, primero con cajas y luego sin ellas, debe comportarse exactamente como uno nuevo
//     (sin restos de las rejillas finas: propiedad de fuerzas, taps, flags de visualización...).
// ---------------------------------------------------------------------------------------------
static void test_reinit() {
    const char* name = "7 reinicio refinado -> uniforme";
    if (!want(name)) return;
    const int nx = 64, ny = 32, nz = 32;
    Config c = base_cfg(nx, ny, nz, 0.05f, 0.01f);
    std::vector<u8> g(static_cast<usize>(nx) * ny * nz, 0);
    add_box(g, nx, ny, 20, 27, 12, 19, 12, 19, 1);
    Config cr = c;
    cr.n_boxes = 1;
    cr.boxes[0] = box(0, 14, 6, 6, 40, 25, 25);
    Solver a, b;
    a.init(cr);
    a.set_geometry(g.data());
    for (int k = 1; k < a.grids(); ++k) {
        const GridInfo gi = a.grid_info(k);
        std::vector<u8> gk(gi.cells(), 0);
        a.set_grid_geometry(k, gk.data(), WallSdf{});
    }
    a.step(50);
    a.init(c);
    a.set_geometry(g.data());
    a.step(200);
    b.init(c);
    b.set_geometry(g.data());
    b.step(200);
    const Vec3 fa = a.forces_mean().force[1], fb = b.forces_mean().force[1];
    const bool same = fa.x == fb.x && fa.y == fb.y && fa.z == fb.z && a.grids() == 1;
    report(name, same, "Fx %.6e vs %.6e (rejillas %d)", static_cast<double>(fa.x), static_cast<double>(fb.x), a.grids());
}

// ---------------------------------------------------------------------------------------------
// [8] iGPU: con refinamiento local attach() debe fallar con un mensaje claro (sin tocar el solver, que sigue en la CPU).
// ---------------------------------------------------------------------------------------------
static void test_gpu_refuse() {
    const char* name = "8 la iGPU rechaza el refinamiento";
    if (!want(name)) return;
    Config c = base_cfg(64, 32, 32, 0.05f, 0.01f);
    c.n_boxes = 1;
    c.boxes[0] = box(0, 14, 6, 6, 40, 25, 25);
    Solver s;
    s.init(c);
    s.set_geometry(nullptr);
    gpu::LbmGpu g;
    std::string err;
    const bool ok = g.attach(s, &err);
    s.step(10);
    report(name, !ok && err.find("refinamiento") != std::string::npos && !s.diverged(), "attach → %s (\"%s\")", ok ? "true" : "false", err.c_str());
}

int main(int argc, char** argv) {
    if (argc > 1) g_filter = argv[1];
    pool().start();
    const double t0 = now_sec();
    test_uniform();
    test_mass();
    test_couette();
    test_sphere();
    test_pulse();
    test_force_crossing();
    test_reinit();
    test_gpu_refuse();
    std::printf("test_refine: %d PASS, %d FALLO (%.1f s)\n", g_pass, g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
