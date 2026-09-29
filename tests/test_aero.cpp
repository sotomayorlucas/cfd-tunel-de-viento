// ============================================================================
//  tests/test_aero.cpp — medida de fuerzas y física aerodinámica (fase 2, B).
//
//   1. Referencia manométrica: fluido en reposo, caja APOYADA en el suelo (y dos
//      grupos que se tocan): fuerza nula con force_gauge; sin ella, p∞·A_contacto
//      exacto (= A/3 en unidades de red) — la prueba discrimina.
//   2. Invariancia galileana: Couette plano con las placas a U_m ∓ U/2 para
//      U_m = 0 y U_m = U: la tensión de pared no depende del sistema de referencia.
//   3. Efecto Magnus: esfera giratoria a Re = 100 (α = ωD/2U = 0.5 y 1):
//      sustentación hacia el lado que se mueve con el flujo y del orden de la
//      literatura (Kim 2009; Oesterlé & Bui Dinh 1998: 0.25-0.7).
//   4. Rebote interpolado (Bouzidi): con la superficie real el Cd de una esfera
//      a Re = 100 no depende de la posición de la esfera respecto a la red
//      (la escalera sí); y la fuerza en reposo sigue siendo nula.
//   5. Coche (Sim, red pequeña): simetría (fuerza lateral ≈ 0 sin guiñada), signo
//      de la fuerza lateral con guiñada, altura de marcha efectiva (monótona,
//      rake conservado), huella de contacto de las ruedas y desglose = total.
//   6. Ley de pared (modelo Slip): placa plana a Re_x ~ 10⁶-10⁷ con la superficie
//      real fuera del centro del enlace (q ≠ ½): fricción del orden de la turbulenta
//      (el rebote no deslizante da varias veces más: la prueba discrimina).
//   7. Balance aerodinámico: reparto de la carga entre ejes por momentos (casos
//      analíticos: centro 50 %, eje delantero 100 %, resistencia alta → menos delante).
//   8. DRS (F1 2022, red pequeña): con el flap trasero abierto bajan la resistencia
//      y la carga.
//  Rápido: redes pequeñas (≈ 1 min en total en el 155H). `make test`.
// ============================================================================
#include "../src/app/app.hpp"
#include "../src/core/threadpool.hpp"
#include "../src/lbm/lattice.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace cfd;
using namespace cfd::lbm;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (cond) ++g_pass;                                           \
        else {                                                        \
            ++g_fail;                                                 \
            std::printf("  FALLO %s:%d: ", __FILE__, __LINE__);       \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static void add_box(std::vector<u8>& g, const Config& c, int x0, int x1, int y0, int y1, int z0, int z1, u8 id) {
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) g[static_cast<usize>(x) + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] = id;
}

struct SphereSdf { float xc, yc, zc, R; };
static float sphere_sdf(const void* ctx, Vec3 p) {
    const SphereSdf* s = static_cast<const SphereSdf*>(ctx);
    const float dx = p.x - s->xc, dy = p.y - s->yc, dz = p.z - s->zc;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - s->R;
}
static void add_sphere(std::vector<u8>& g, const Config& c, const SphereSdf& s, u8 id) {
    for (int z = 0; z < c.nz; ++z)
        for (int y = 0; y < c.ny; ++y)
            for (int x = 0; x < c.nx; ++x)
                if (sphere_sdf(&s, Vec3(float(x), float(y), float(z))) < 0.0f) g[static_cast<usize>(x) + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] = id;
}

// ---------------------------------------------------------------------------------------------
static void test_gauge() {
    std::printf("[1] referencia manométrica: fluido en reposo, caja apoyada en el suelo\n");
    for (int gauge = 1; gauge >= 0; --gauge) {
        for (BounceBack bb : {BounceBack::Interpolated, BounceBack::Implicit}) {
            Config c;
            c.nx = 48; c.ny = 32; c.nz = 24;
            c.u_inf = 0.0f; c.nu = 1e-3f; c.cs_smag = 0.16f; c.precision = Precision::FP32;
            c.ground = GroundMode::Static; c.sponge_frac = 0.0f; c.ramp_steps = 0;
            c.force_gauge = gauge != 0; c.bounce = bb;
            std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
            add_box(g, c, 10, 19, 8, 15, 1, 6, 1);     // caja apoyada: base 10×8 = 80 celdas² de contacto
            add_box(g, c, 20, 25, 8, 15, 1, 4, 2);     // segundo grupo pegado a la caja (cara x = 19|20 compartida)
            Solver s;
            s.init(c);
            s.set_geometry(g.data());
            s.step(20, true);
            const Vec3 f1 = s.forces().force[1], f2 = s.forces().force[2], fg = s.forces().force[255];
            if (gauge) {
                const float worst = max_(max_(length(f1), length(f2)), length(fg));
                CHECK(worst < 1e-5f, "con force_gauge (%s) la fuerza en reposo debe ser nula: |F| máx %.2e", bb == BounceBack::Implicit ? "implícito" : "interpolado", worst);
                if (bb == BounceBack::Interpolated) std::printf("    con referencia manométrica: |F| máx %.1e (cajas y suelo)\n", worst);
            } else {
                // Presión absoluta en reposo: cada enlace fluido → caja aporta 2w_k c_k; en un sólido cerrado la
                // suma es 0, así que la fuerza es MENOS la de los enlaces que faltan (los que salen de celdas no
                // fluidas: el suelo bajo la caja y la caja 2 pegada). Se enumera: F_esp = −Σ_faltan 2w_k c_k.
                namespace D = lbm::d3q19;
                Vec3 fexp{0, 0, 0};
                for (int z = 1; z <= 6; ++z)
                    for (int y = 8; y <= 15; ++y)
                        for (int x = 10; x <= 19; ++x)
                            for (int k = 1; k < D::Q; ++k) {
                                const int sx = x - D::c[k][0], sy = y - D::c[k][1], sz = z - D::c[k][2];
                                const u8 id = sz == 0 ? 255 : g[static_cast<usize>(sx) + static_cast<usize>(c.nx) * (sy + static_cast<usize>(c.ny) * sz)];
                                if (id == 1) continue;   // interior: se cancela por pares
                                if (id) fexp -= Vec3(float(D::c[k][0]), float(D::c[k][1]), float(D::c[k][2])) * (2.0f * D::w[k]);
                            }
                CHECK(length(f1 - fexp) < 1e-3f * length(fexp), "sin force_gauge: F(caja) = (%.4f, %.4f, %.4f), esperado (%.4f, %.4f, %.4f)", f1.x, f1.y, f1.z,
                      fexp.x, fexp.y, fexp.z);
                CHECK(fexp.z < -80.0f / 3.0f + 1e-3f, "la base apoyada empuja hacia abajo al menos p∞·A = 80/3 (%.4f)", fexp.z);
                if (bb == BounceBack::Interpolated)
                    std::printf("    sin referencia: F caja (%.3f, %.3f, %.3f) = −p∞·A en las caras tapadas (base de 80 celdas²) — lo que la corrección elimina\n",
                                f1.x, f1.y, f1.z);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
static void test_galilean() {
    std::printf("[2] invariancia galileana: Couette plano en dos sistemas de referencia\n");
    // Placas en z = 1 y z = nz-2 que se mueven tangencialmente a U_m ∓ U/2 (con parches centrales de id propio):
    // la tensión de pared depende SÓLO de la velocidad relativa U. Se comparan U_m = 0 y U_m = U (entrada a U_m).
    // (Un cuerpo que se "traslada" con paredes de velocidad NORMAL a su superficie no es un caso válido: los
    // vóxeles no se mueven; la app sólo usa movimientos tangentes: ruedas que giran y la cinta.)
    float tau[2][2] = {};
    for (int f = 0; f < 2; ++f) {
        const float U = 0.04f, Um = f == 0 ? 0.0f : 0.04f;
        Config c;
        c.nx = 64; c.ny = 48; c.nz = 20;
        c.u_inf = Um; c.nu = 1.0f / 6.0f; c.cs_smag = 0.0f; c.precision = Precision::FP32;
        c.ground = GroundMode::None; c.sponge_frac = 0.0f; c.ramp_steps = 0;
        std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
        add_box(g, c, 0, c.nx - 1, 0, c.ny - 1, 1, 1, 2);
        add_box(g, c, 0, c.nx - 1, 0, c.ny - 1, c.nz - 2, c.nz - 2, 1);
        add_box(g, c, 24, 39, 16, 31, c.nz - 2, c.nz - 2, 3);
        add_box(g, c, 24, 39, 16, 31, 1, 1, 4);
        Solver s;
        s.init(c);
        s.set_geometry(g.data());
        WallMotion m;
        m.v = Vec3(Um + 0.5f * U, 0, 0); s.set_wall_motion(1, m); s.set_wall_motion(3, m);
        m.v = Vec3(Um - 0.5f * U, 0, 0); s.set_wall_motion(2, m); s.set_wall_motion(4, m);
        s.step(2400, false);
        s.step(100, true);
        tau[f][0] = s.forces_mean().force[3].x;
        tau[f][1] = s.forces_mean().force[4].x;
    }
    const float ta = 1.0f / 6.0f * 0.04f / 16.0f * 256.0f;   // ρν U/h · A (h = nz - 4 = 16)
    const float e0 = 0.5f * (tau[0][1] - tau[0][0]), e1 = 0.5f * (tau[1][1] - tau[1][0]);
    std::printf("    parche: U_m = 0 → %.5f / %.5f · U_m = U → %.5f / %.5f (analítico ∓%.5f)\n", tau[0][0], tau[0][1], tau[1][0], tau[1][1], ta);
    CHECK(std::fabs(e0 - ta) / ta < 0.03f, "tensión antisimétrica en reposo %.5f vs analítica %.5f", e0, ta);
    CHECK(std::fabs(e1 - e0) / ta < 0.02f, "misma tensión en el sistema que se mueve a U (%.5f vs %.5f)", e1, e0);
}

// ---------------------------------------------------------------------------------------------
static void sphere_forces(float alpha, int D, BounceBack bb, float offset, bool use_sdf, float& cl, float& cd) {
    Config c;
    c.nx = (13 * D + 7) & ~7; c.ny = 7 * D; c.nz = 7 * D;
    c.u_inf = 0.05f; c.nu = c.u_inf * static_cast<float>(D) / 100.0f; c.cs_smag = 0.0f;
    c.precision = Precision::FP32; c.ground = GroundMode::None; c.sponge_frac = 0.12f; c.ramp_steps = 0;
    c.bounce = bb;
    const SphereSdf sp{5.0f * D + offset, 0.5f * (c.ny - 1) + 0.5f * offset, 0.5f * (c.nz - 1) + 0.3f * offset, 0.5f * D};
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    add_sphere(g, c, sp, 1);
    Solver s;
    s.init(c);
    if (use_sdf) s.set_geometry(g.data(), WallSdf{&sphere_sdf, &sp});
    else s.set_geometry(g.data());
    if (alpha != 0.0f) {
        WallMotion m;
        m.omega = Vec3(0, alpha * c.u_inf / sp.R, 0);   // ω_y > 0: superficie superior hacia +x (con el flujo)
        m.center = Vec3(sp.xc, sp.yc, sp.zc);
        s.set_wall_motion(1, m);
    }
    s.step(static_cast<int>(1.4f * c.nx / c.u_inf), false);
    const int nav = 6, per = static_cast<int>(0.3f * c.nx / c.u_inf / nav);
    double fx = 0, fz = 0;
    for (int i = 0; i < nav; ++i) { s.step(per, true); fx += s.forces_mean().force[1].x; fz += s.forces_mean().force[1].z; }
    const double q = 0.5 * c.u_inf * c.u_inf * 3.14159265 * D * D / 4.0 * nav;
    cl = static_cast<float>(fz / q);
    cd = static_cast<float>(fx / q);
    if (s.diverged()) { cl = cd = NAN; }
}

static void test_magnus() {
    std::printf("[3] efecto Magnus: esfera giratoria, Re = 100, D = 10 celdas\n");
    float cl0, cd0, cl1, cd1, cl2, cd2;
    sphere_forces(0.0f, 10, BounceBack::Interpolated, 0.0f, true, cl0, cd0);
    sphere_forces(0.5f, 10, BounceBack::Interpolated, 0.0f, true, cl1, cd1);
    sphere_forces(1.0f, 10, BounceBack::Interpolated, 0.0f, true, cl2, cd2);
    std::printf("    α = 0: CL %+.3f CD %.3f · α = 0.5: CL %+.3f CD %.3f · α = 1: CL %+.3f CD %.3f\n", cl0, cd0, cl1, cd1, cl2, cd2);
    const float cd_sn = 24.0f / 100.0f * (1.0f + 0.15f * std::pow(100.0f, 0.687f));
    CHECK(std::fabs(cl0) < 0.01f, "sin giro no hay sustentación (%.4f)", cl0);
    CHECK(std::fabs(cd0 - cd_sn) / cd_sn < 0.2f, "Cd sin giro %.3f vs Schiller-Naumann %.3f", cd0, cd_sn);
    CHECK(cl1 > 0.15f && cl1 < 0.7f, "α = 0.5: CL %.3f fuera de [0.15, 0.7] (literatura 0.25-0.55)", cl1);
    CHECK(cl2 > 0.3f && cl2 < 0.9f, "α = 1: CL %.3f fuera de [0.3, 0.9] (literatura 0.35-0.7)", cl2);
    CHECK(cl2 > cl1 && cl1 > 0.0f, "la sustentación crece con el giro y va hacia el lado que se mueve con el flujo");
    CHECK(cd2 > cd0 * 0.95f, "el giro no reduce apreciablemente la resistencia (%.3f vs %.3f)", cd2, cd0);
}

// ---------------------------------------------------------------------------------------------
static void test_interpolated() {
    std::printf("[4] rebote interpolado (Bouzidi) vs escalera: esfera Re = 100 desplazada respecto a la red\n");
    float cl, cda, cdb, cdc, cdd;
    sphere_forces(0.0f, 10, BounceBack::Interpolated, 0.25f, true, cl, cda);
    sphere_forces(0.0f, 10, BounceBack::Interpolated, 0.5f, true, cl, cdb);
    sphere_forces(0.0f, 10, BounceBack::Implicit, 0.25f, false, cl, cdc);
    sphere_forces(0.0f, 10, BounceBack::Implicit, 0.5f, false, cl, cdd);
    const float si = std::fabs(cda - cdb) / cda, ss = std::fabs(cdc - cdd) / cdc;
    const float cd_sn = 24.0f / 100.0f * (1.0f + 0.15f * std::pow(100.0f, 0.687f));
    const float ei = 0.5f * std::fabs(cda + cdb) - cd_sn, es = 0.5f * std::fabs(cdc + cdd) - cd_sn;
    std::printf("    Cd (D = 10) interpolado %.3f / %.3f (dif %.1f%%) · escalera %.3f / %.3f (dif %.1f%%) · Schiller-Naumann %.3f\n", cda, cdb, 100 * si,
                cdc, cdd, 100 * ss, cd_sn);
    CHECK(si < 0.03f, "con la superficie real el Cd casi no depende de la posición en la red (%.1f%%)", 100 * si);
    CHECK(std::fabs(ei) < std::fabs(es), "el rebote interpolado se acerca más a Schiller-Naumann que la escalera (%+.3f vs %+.3f)", ei, es);
}

// ---------------------------------------------------------------------------------------------
static app::AeroResult run_sim(app::Sim& s, float ft) {
    const int n = static_cast<int>(ft * s.ft_steps());
    for (int done = 0; done < n; done += 50) s.step(50);
    return s.res;
}

static void test_car() {
    std::printf("[5] coche en red pequeña: simetría, guiñada, altura efectiva, huella de contacto\n");
    const int m = models::find("ahmed_25");
    {
        app::Sim s;
        s.cfg.model = m;
        s.cfg.cells = 400'000;
        s.cfg.ground = GroundMode::Moving;
        s.speed_kmh = 144.0f;
        s.init();
        const app::AeroResult r0 = run_sim(s, 2.5f);
        CHECK(!s.solver.diverged() && r0.valid, "Ahmed: sin divergencias");
        CHECK(std::fabs(r0.cs) < 0.02f + 0.05f * r0.cd, "Ahmed sin guiñada: fuerza lateral ≈ 0 (CS %.4f, CD %.3f)", r0.cs, r0.cd);
        // Suma de componentes = total.
        float sz = 0, sx = 0;
        for (int i = 0; i < r0.ncomp; ++i) { sz += r0.comp[i].scz; sx += r0.comp[i].scx; }
        CHECK(std::fabs(sz - r0.scz) < 1e-4f + 1e-3f * std::fabs(r0.scz) && std::fabs(sx - r0.scx) < 1e-4f + 1e-3f * std::fabs(r0.scx),
              "desglose por componentes = total (SCz %.4f vs %.4f, SCx %.4f vs %.4f)", sz, r0.scz, sx, r0.scx);
        models::Params p = s.params;
        p.yaw_deg = 10.0f;
        s.set_params(p, 0.0f);
        s.reset_flow();
        const app::AeroResult r1 = run_sim(s, 2.5f);
        // Guiñada +10° (el morro gira hacia −y): el viento cruzado relativo empuja al coche hacia −y.
        std::printf("    Ahmed 25°: CD %.3f CS %+.4f (0°) · CD %.3f CS %+.4f (guiñada +10°)\n", r0.cd, r0.cs, r1.cd, r1.cs);
        CHECK(r1.cs < -0.05f, "guiñada +10°: fuerza lateral hacia −y (CS %.4f)", r1.cs);
    }
    {
        // Altura de marcha efectiva: f1_2022 a 30/80 mm con la red de 400 k celdas (dx ≈ 9 cm).
        app::Sim s;
        s.cfg.model = models::find("f1_2022");
        s.cfg.cells = 400'000;
        s.cfg.ground = GroundMode::Moving;
        s.init();
        // Con refinamiento local (defecto de los F1) el hueco mínimo usa la dx MÁS FINA bajo el fondo (dx_under).
        CHECK(s.dx_under > 0.0f && s.dx_under <= s.dom.dx * 1.0001f, "dx bajo el fondo %.1f mm ≤ dx de la base %.1f mm", s.dx_under * 1e3f, s.dom.dx * 1e3f);
        const float g = app::Sim::k_gap_cells * s.dx_under * 1000.0f;
        const float h = min_(s.params.ride_front_mm, s.params.ride_rear_mm);
        const float exp_front = s.params.ride_front_mm + std::pow(h * h * h * h + g * g * g * g, 0.25f) - h;
        CHECK(s.ride_limited && std::fabs(s.ride_eff_front_mm - exp_front) < 0.01f, "altura efectiva del. %.2f (esperado %.2f, g = %.1f mm)", s.ride_eff_front_mm, exp_front, g);
        CHECK(std::fabs((s.ride_eff_rear_mm - s.ride_eff_front_mm) - (s.params.ride_rear_mm - s.params.ride_front_mm)) < 0.01f, "el rake se conserva");
        CHECK(s.built.params.ride_front_mm == s.ride_eff_front_mm, "el modelo construido usa la altura efectiva");
        // Monótona en un barrido (rake fijo).
        float prev = -1.0f;
        bool mono = true;
        for (float hf = 10.0f; hf <= 120.0f; hf += 10.0f) {
            models::Params p = s.params;
            app::sweep_param_set(p, app::SweepParam::RideHeight, hf);
            s.set_params(p, 0.0f);
            mono &= s.ride_eff_front_mm > prev;
            prev = s.ride_eff_front_mm;
        }
        CHECK(mono, "la altura efectiva es monótona en la pedida");
        std::printf("    f1_2022 dx %.1f mm (bajo el fondo %.1f mm): pedida 30/80 → efectiva (h⁴+g⁴)^¼ con g = %.0f mm; barrido monótono\n", s.dom.dx * 1e3, s.dx_under * 1e3, g);
        // Huella de contacto: ninguna celda de fluido encajonada entre una rueda y el suelo en z = 1..2.
        const auto& gs = s.built.scene.groups();
        int pockets = 0;
        const usize nx = static_cast<usize>(s.dom.nx), nxny = nx * static_cast<usize>(s.dom.ny);
        for (int z = 1; z <= 2; ++z)
            for (int y = 1; y < s.dom.ny - 1; ++y)
                for (int x = 1; x < s.dom.nx - 1; ++x) {
                    const usize n = static_cast<usize>(x) + nx * static_cast<usize>(y) + nxny * static_cast<usize>(z);
                    if (s.solid[n]) continue;
                    const u8 a = s.solid[n + nxny];
                    if (!a || a > gs.size()) continue;
                    const sdf::Component cp = gs[a - 1].component;
                    const bool wheel = cp == sdf::Component::FrontWheels || cp == sdf::Component::RearWheels;
                    if (wheel && (z == 1 || s.solid[n - nxny])) ++pockets;
                }
        CHECK(pockets == 0, "huella de contacto: %d celdas de fluido encajonadas bajo las ruedas", pockets);
    }
}

// ---------------------------------------------------------------------------------------------
// Placa plana fina (2 celdas, superficie real a 0.2 celdas del centro de la última capa sólida: q ≠ ½) en
// flujo uniforme a Re_x ~ 10⁶-10⁷ (ν del aire real vía wall_nu). Fricción media de los tramos centrales
// frente a la turbulenta de placa plana cf = 0.0592·Re_x^-0.2.
struct SlabSdf { float zlo, zhi, xlo, xhi, ylo, yhi; };
static float slab_sdf(const void* ctx, Vec3 p) {
    const SlabSdf* s = static_cast<const SlabSdf*>(ctx);
    const float dz = max_(s->zlo - p.z, p.z - s->zhi), dx = max_(s->xlo - p.x, p.x - s->xhi), dy = max_(s->ylo - p.y, p.y - s->yhi);
    const float m = max_(dz, max_(dx, dy));
    if (m <= 0.0f) return m;
    const float a = max_(dz, 0.0f), b = max_(dx, 0.0f), c = max_(dy, 0.0f);
    return std::sqrt(a * a + b * b + c * c);
}
static float plate_cf(WallModel wm, float& cf_turb, float& u_half) {
    Config c;
    const int L = 192, seg = L / 6, x0 = 24;
    c.nx = L + 96; c.ny = 56; c.nz = 64;
    c.u_inf = 0.09f; c.nu = 1e-4f; c.cs_smag = 0.10f; c.precision = Precision::FP32;
    c.ground = GroundMode::None; c.sponge_frac = 0.1f; c.ramp_steps = 0;
    c.wall_model = wm; c.wall_nu = 5e-6f;
    const int zc = c.nz / 2;
    std::vector<u8> g(static_cast<usize>(c.nx) * c.ny * c.nz, 0);
    for (int z = zc; z <= zc + 1; ++z)
        for (int y = 8; y < c.ny - 8; ++y)
            for (int x = x0; x < x0 + L; ++x)
                g[static_cast<usize>(x) + static_cast<usize>(c.nx) * (y + static_cast<usize>(c.ny) * z)] =
                    static_cast<u8>((y >= 20 && y < c.ny - 20) ? 1 + (x - x0) / seg : 20);
    static SlabSdf ps;
    ps = SlabSdf{zc - 0.2f, zc + 1.2f, x0 - 0.2f, x0 + L - 0.8f, 7.5f, c.ny - 8.5f};
    Solver s;
    s.init(c);
    s.set_geometry(g.data(), WallSdf{&slab_sdf, &ps});
    s.step(3000, false);
    s.step(500, true);
    // Tramos 2..5 (lejos del borde de ataque romo y del de salida).
    double cf = 0, ct = 0;
    for (int id = 2; id <= 5; ++id) {
        const double A = 2.0 * seg * (c.ny - 40);
        cf += s.forces_mean().force[id].x / (0.5 * c.u_inf * c.u_inf * A);
        ct += 0.0592 * std::pow(c.u_inf * (id - 0.5) * seg / c.wall_nu, -0.2);
    }
    const FieldView v = s.field();
    u_half = v.ux[v.index(x0 + 3 * seg, c.ny / 2, zc + 2)] / c.u_inf;   // 1.ª celda de fluido (a ~0.8 celdas de la pared)
    cf_turb = static_cast<float>(ct / 4);
    return s.diverged() ? NAN : static_cast<float>(cf / 4);
}

static void test_wall_law() {
    std::printf("[6] ley de pared: placa plana a Re_x ~ 10⁶-10⁷ (fricción media de los tramos centrales)\n");
    float ct = 0, u_s = 0, u_n = 0;
    const float cf_s = plate_cf(WallModel::Slip, ct, u_s);
    const float cf_n = plate_cf(WallModel::None, ct, u_n);
    std::printf("    cf turbulenta %.5f · Slip %.5f (×%.2f, u₁ %.2f U) · rebote no deslizante %.5f (×%.2f, u₁ %.2f U)\n", ct, cf_s, cf_s / ct, u_s,
                cf_n, cf_n / ct, u_n);
    CHECK(cf_s > 0.3f * ct && cf_s < 2.0f * ct, "Slip: cf %.5f fuera de [0.3, 2]× la turbulenta %.5f", cf_s, ct);
    CHECK(cf_n > 1.5f * cf_s, "el rebote no deslizante da más fricción (%.5f) que la ley de pared (%.5f): la prueba discrimina", cf_n, cf_s);
    CHECK(u_s > 0.45f, "con ley de pared la 1.ª celda conserva velocidad (u %.2f U)", u_s);
}

// ---------------------------------------------------------------------------------------------
// Balance: una carga puntual D en x_p (con cualquier resistencia a cualquier altura, que también da momento
// de cabeceo) se reparte entre los contactos delantero (x = 0) y trasero (x = wb) por equilibrio de momentos.
static void test_balance() {
    std::printf("[7] balance aerodinámico (reparto de la carga entre ejes)\n");
    const Vec3 front(0, 0, 0), rear(3.0f, 0, 0), ref(1.2f, 0, 0.3f);
    auto bal = [&](float xp, float D, float drag, float zd) {
        // Carga D (hacia −z) aplicada en (xp, 0, 0) y resistencia +x aplicada a la altura zd: momento respecto a ref.
        const Vec3 Fd(0, 0, -D), Fx(drag, 0, 0);
        const Vec3 M = cross(Vec3(xp, 0, 0) - ref, Fd) + cross(Vec3(xp, 0, zd) - ref, Fx);
        return app::balance_front_pct(Fd + Fx, M, ref, front, rear);
    };
    const float b_mid = bal(1.5f, 1000, 0, 0), b_front = bal(0.0f, 1000, 0, 0), b_q = bal(0.75f, 1000, 0, 0);
    // Resistencia a 0.4 m de altura: momento de cabeceo que DESCARGA el eje delantero en ΔN = drag·zd/wb.
    const float b_drag = bal(1.5f, 1000, 300, 0.4f);
    std::printf("    carga en el centro %.1f %% · en el eje del. %.1f %% · a ¼ de batalla %.1f %% · centro + resistencia alta %.1f %%\n", b_mid,
                b_front, b_q, b_drag);
    CHECK(std::fabs(b_mid - 50.0f) < 0.01f && std::fabs(b_front - 100.0f) < 0.01f && std::fabs(b_q - 75.0f) < 0.01f, "reparto por momentos");
    CHECK(std::fabs(b_drag - (50.0f - 100.0f * 300.0f * 0.4f / 3.0f / 1000.0f)) < 0.05f, "la resistencia alta descarga el eje delantero (%.2f %%)", b_drag);
}

// DRS: con el flap trasero abierto baja la resistencia (y la carga). Red pequeña (0.8 M celdas).
static void test_drs() {
    std::printf("[8] DRS: el flap trasero abierto reduce la resistencia (F1 2022, red de 0.8 M celdas)\n");
    app::Sim s;
    s.cfg.model = models::find("f1_2022");
    s.cfg.cells = 800'000;
    s.cfg.ground = GroundMode::Moving;
    s.init();
    const app::AeroResult r0 = run_sim(s, 2.5f);
    models::Params p = s.params;
    p.drs_open = true;
    s.set_params(p, 0.0f);
    s.reset_flow();
    const app::AeroResult r1 = run_sim(s, 2.5f);
    std::printf("    cerrado: SCz %+.3f SCx %.3f · abierto: SCz %+.3f SCx %.3f (ΔSCx %+.3f m², %+.1f %%)\n", r0.scz, r0.scx, r1.scz, r1.scx, r1.scx - r0.scx,
                100.0f * (r1.scx - r0.scx) / r0.scx);
    CHECK(!s.solver.diverged() && r0.valid && r1.valid, "DRS: sin divergencias");
    CHECK(r1.scx < r0.scx - 0.02f, "DRS abierto: menos resistencia (%.3f vs %.3f)", r1.scx, r0.scx);
    CHECK(r1.scz < r0.scz, "DRS abierto: menos carga (%.3f vs %.3f)", r1.scz, r0.scz);
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    pool().start();
    const double t0 = now_sec();
    test_gauge();
    test_galilean();
    test_magnus();
    test_interpolated();
    test_car();
    test_wall_law();
    test_balance();
    test_drs();
    std::printf("test_aero: %d PASS, %d FALLO (%.1f s)\n", g_pass, g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
