// ============================================================================
//  tests/test_gpu.cpp — paridad del solver LBM en la iGPU (src/gpu) con la CPU.
//
//  Dos lbm::Solver idénticos: uno avanza en la CPU, el otro enganchado a
//  gpu::LbmGpu. Tras cientos de pasos se comparan ρ, u (todas las celdas) y las
//  fuerzas/momentos por id (último paso y media del lote).
//   [1] FP32 Regularizada (2º orden, sin viscosidad de volumen), sin suelo, rebote implícito
//   [1b] ídem con viscosidad de volumen (ω_b = 1)
//   [2] FP32 BGK (misma escena)
//   (desde [3], colisión de la app: regularización recursiva de 3er orden + viscosidad de volumen)
//   [3] FP32 escena completa: cinta móvil, rebote interpolado (Bouzidi) con SDF, modelo
//       Slip, ruedas girando impermeables con huella, dos cuerpos que se tocan (nodos
//       multi-id), rampa de arranque
//   [4] FP16S escena completa (tolerancia de FP16)
//   [5] FP32 LogLaw + rebote implícito + ruedas con Ladd en el kernel de celdas
//   [6] FP16S suelo fijo + LogLaw + interpolado
//   [7] cambios en caliente con la GPU enganchada: set_geometry, reset_flow, detach → CPU
//   [8] estabilidad: F1 2022 a resolución rápida con la configuración de la app (si hay tiempo)
//   [9] modo asíncrono (cycle: el lote siguiente se encola antes de publicar el anterior) con set_geometry en medio
//  Sin dispositivo Vulkan: PASA con una nota.
// ============================================================================
#include "../src/app/app.hpp"
#include "../src/gpu/lbm_gpu.hpp"
#include "../src/core/threadpool.hpp"
#include "../src/core/util.hpp"
#include "../src/models/model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace cfd;
using namespace cfd::lbm;

namespace {

int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FALLO: " __VA_ARGS__); std::printf("\n"); } } while (0)

// ---- Escena analítica (celdas): cuerpo redondeado + segundo cuerpo que lo toca + ruedas ----------
struct Scene {
    bool ground = true;
    bool wheels = true;
    bool second = true;
    float bx0 = 26, bx1 = 52, by0 = 14, by1 = 34, bz0 = 4, bz1 = 13, br = 3;   // caja redondeada (id 1)
    float wr = 4.5f, wy0 = 11.0f, wy1 = 37.0f, wx = 32.0f, wz = 4.9f, ww = 3.0f; // ruedas (ids 2 y 3): cilindros en y
};
float sd_rbox(const Scene& s, Vec3 p) {
    const Vec3 c{(s.bx0 + s.bx1) * 0.5f, (s.by0 + s.by1) * 0.5f, (s.bz0 + s.bz1) * 0.5f};
    const Vec3 h{(s.bx1 - s.bx0) * 0.5f - s.br, (s.by1 - s.by0) * 0.5f - s.br, (s.bz1 - s.bz0) * 0.5f - s.br};
    const Vec3 q{std::fabs(p.x - c.x) - h.x, std::fabs(p.y - c.y) - h.y, std::fabs(p.z - c.z) - h.z};
    const Vec3 qm{std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f)};
    return length(qm) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f) - s.br;
}
float sd_fin(const Scene& s, Vec3 p) {   // aleta vertical encima del cuerpo (id 4): toca al cuerpo
    const Vec3 c{s.bx1 - 6.0f, (s.by0 + s.by1) * 0.5f, s.bz1 + 3.0f};
    const Vec3 h{4.0f, 1.2f, 3.5f};
    const Vec3 q{std::fabs(p.x - c.x) - h.x, std::fabs(p.y - c.y) - h.y, std::fabs(p.z - c.z) - h.z};
    const Vec3 qm{std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f)};
    return length(qm) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
}
float sd_wheel(const Scene& s, Vec3 p, float yc) {
    const float dxz = std::sqrt((p.x - s.wx) * (p.x - s.wx) + (p.z - s.wz) * (p.z - s.wz)) - s.wr;
    const float dy = std::fabs(p.y - yc) - s.ww * 0.5f;
    const float a = std::max(dxz, 0.0f), b = std::max(dy, 0.0f);
    return std::sqrt(a * a + b * b) + std::min(std::max(dxz, dy), 0.0f);
}
float scene_sdf(const void* ctx, Vec3 p) {
    const Scene& s = *static_cast<const Scene*>(ctx);
    float d = sd_rbox(s, p);
    if (s.second) d = std::min(d, sd_fin(s, p));
    if (s.wheels) d = std::min(d, std::min(sd_wheel(s, p, s.wy0), sd_wheel(s, p, s.wy1)));
    return d;
}
std::vector<u8> voxelize(const Scene& s, int nx, int ny, int nz) {
    std::vector<u8> g(static_cast<usize>(nx) * ny * nz, 0);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const Vec3 p{float(x), float(y), float(z)};
                u8 id = 0;
                if (sd_rbox(s, p) < 0.0f) id = 1;
                else if (s.second && sd_fin(s, p) < 0.0f) id = 4;
                else if (s.wheels && sd_wheel(s, p, s.wy0) < 0.0f) id = 2;
                else if (s.wheels && sd_wheel(s, p, s.wy1) < 0.0f) id = 3;
                g[static_cast<usize>(x) + static_cast<usize>(nx) * (y + static_cast<usize>(ny) * z)] = id;
            }
    return g;
}

struct Case {
    const char* name;
    Config cfg;
    Scene scene;
    bool sdf = true;
    int steps = 240;
    double tol_u = 1e-5, tol_rho = 1e-5, tol_f = 1e-4;   // |Δu|/u∞, |Δρ|, |ΔF|/max|F|
};

void setup(Solver& s, const Case& c, const std::vector<u8>& g) {
    s.init(c.cfg);
    if (c.sdf) s.set_geometry(g.data(), WallSdf{&scene_sdf, &c.scene});
    else s.set_geometry(g.data());
    if (c.scene.wheels && c.cfg.ground == GroundMode::Moving) {
        for (u8 id : {u8(2), u8(3)}) {
            WallMotion m;
            m.omega = Vec3(0, -c.cfg.u_inf / c.scene.wr, 0);
            m.center = Vec3(c.scene.wx, id == 2 ? c.scene.wy0 : c.scene.wy1, c.scene.wz);
            m.contact_z = 1.5f;
            m.impermeable = true;
            s.set_wall_motion(id, m);
        }
    }
    s.set_moment_reference(Vec3(40, 24, 8));
}

struct Diff { double du = 0, drho = 0, df = 0, dfm = 0, fmax = 0; };

Diff compare(const Solver& a, const Solver& b) {
    Diff d;
    const FieldView fa = a.field(), fb = b.field();
    const usize N = static_cast<usize>(fa.nx) * fa.ny * fa.nz;
    for (usize n = 0; n < N; ++n) {
        d.drho = std::max(d.drho, static_cast<double>(std::fabs(fa.rho[n] - fb.rho[n])));
        d.du = std::max({d.du, static_cast<double>(std::fabs(fa.ux[n] - fb.ux[n])), static_cast<double>(std::fabs(fa.uy[n] - fb.uy[n])),
                         static_cast<double>(std::fabs(fa.uz[n] - fb.uz[n]))});
    }
    if (!(d.du == d.du)) d.du = 1e30;
    const ForceSample &la = a.forces(), &lb = b.forces(), &ma = a.forces_mean(), &mb = b.forces_mean();
    for (int id = 1; id < 256; ++id) {
        d.fmax = std::max({d.fmax, static_cast<double>(length(la.force[id])), static_cast<double>(length(ma.force[id]))});
        d.df = std::max(d.df, static_cast<double>(length(la.force[id] - lb.force[id])));
        d.dfm = std::max({d.dfm, static_cast<double>(length(ma.force[id] - mb.force[id])),
                          static_cast<double>(length(ma.moment[id] - mb.moment[id])) / 30.0});
    }
    return d;
}

void run_case(gpu::LbmGpu& G, Case c) {
    if (const char* e = std::getenv("CFD_T_ONLY")) if (std::strcmp(e, c.name) != 0) return;
    if (const char* e = std::getenv("CFD_T_STEPS")) c.steps = std::atoi(e);
    if (std::getenv("CFD_T_FP32")) { c.cfg.precision = Precision::FP32; c.tol_u = 1e-5; c.tol_rho = 1e-5; c.tol_f = 1e-4; }
    std::printf("[%s] %dx%dx%d, %s %s, %d pasos\n", c.name, c.cfg.nx, c.cfg.ny, c.cfg.nz, c.cfg.precision == Precision::FP32 ? "FP32" : "FP16S",
                c.cfg.collision == Collision::BGK ? "BGK" : (c.cfg.collision == Collision::Regularized ? "Reg" : "RR"), c.steps);
    const auto g = voxelize(c.scene, c.cfg.nx, c.cfg.ny, c.cfg.nz);
    Solver cpu, gs;
    setup(cpu, c, g);
    setup(gs, c, g);
    std::string err;
    if (!G.attach(gs, &err)) { CHECK(false, "attach: %s", err.c_str()); return; }
    // Lotes de tamaños variados (paridades distintas al empezar, búferes de comandos reutilizados).
    const int chunks[] = {1, 7, 32, 64};
    int done = 0, ci = 0;
    const double t0 = now_sec();
    while (done < c.steps) {
        int k = std::min(chunks[ci++ % 4], c.steps - done);
        if (done + k > c.steps - 40 && done < c.steps - 40) k = c.steps - 40 - done;   // el último lote = 40 (media comparable)
        if (done >= c.steps - 40) k = c.steps - done;
        cpu.step(k);
        G.step(k);
        done += k;
    }
    const double t1 = now_sec();
    const Diff d = compare(cpu, gs);
    // Sensibilidad natural del flujo: la misma simulación en la CPU con u∞·(1 + 1e-6). Si el flujo amplifica
    // perturbaciones (inestabilidades físicas), la diferencia CPU-GPU crece igual sin que haya error de código.
    Case cp = c;
    cp.cfg.u_inf *= 1.0f + 1e-6f;
    Solver ref;
    setup(ref, cp, g);
    ref.step(c.steps - 40);
    ref.step(40);
    const Diff dn = compare(cpu, ref);
    const auto& st = G.stats();
    std::printf("  |Δu|/u∞ %.2e  |Δρ| %.2e  |ΔF| %.2e / %.2e (último) · %.2e (media)  · nodos %zu (multi-id %zu, >4 ids %zu) · %.2f s\n",
                d.du / c.cfg.u_inf, d.drho, d.df, d.fmax, d.dfm, st.nodes, st.multi_id_nodes, st.overflow_nodes, t1 - t0);
    std::printf("  referencia (CPU con u∞·(1+1e-6)): |Δu|/u∞ %.2e  |Δρ| %.2e → amplificación %.1f×\n", dn.du / c.cfg.u_inf, dn.drho,
                dn.du / c.cfg.u_inf / 1e-6);
    CHECK(gs.steps() == cpu.steps(), "pasos %llu vs %llu", static_cast<unsigned long long>(gs.steps()), static_cast<unsigned long long>(cpu.steps()));
    CHECK(!gs.diverged() && !cpu.diverged(), "divergencia (cpu %d gpu %d)", cpu.diverged(), gs.diverged());
    CHECK(d.du / c.cfg.u_inf < c.tol_u, "velocidad: %.3e > %.1e", d.du / c.cfg.u_inf, c.tol_u);
    CHECK(d.drho < c.tol_rho, "densidad: %.3e > %.1e", d.drho, c.tol_rho);
    CHECK(d.df < c.tol_f * d.fmax + 1e-9 && d.dfm < c.tol_f * d.fmax + 1e-9, "fuerzas: %.3e / %.3e (tol %.1e)", std::max(d.df, d.dfm), d.fmax, c.tol_f);
    CHECK(st.overflow_nodes == 0, "nodos con más de 4 ids");
    // Masa: la de la GPU (bajada por el gancho) frente a la de la CPU.
    const double ma = cpu.total_mass(), mb = gs.total_mass();
    std::printf("  masa CPU %.6f · GPU %.6f (Δ %.2e)\n", ma, mb, std::fabs(ma - mb) / ma);
    CHECK(std::fabs(ma - mb) / ma < c.tol_rho * 10, "masa total");
    G.detach();
}

Config base_cfg(Precision p) {
    Config c;
    c.nx = 96; c.ny = 48; c.nz = 32;
    c.u_inf = 0.09f;
    c.nu = 2e-3f;
    c.cs_smag = 0.10f;
    c.precision = p;
    c.collision = Collision::Recursive;   // defecto de la app (regularización recursiva + viscosidad de volumen)
    c.bulk_omega = 1.0f;
    c.rr_wall_layer = 3;                  // capa fina: en este dominio pequeño quedan celdas con y sin término de 3er orden
    c.ground = GroundMode::Moving;
    c.wall_model = WallModel::Slip;
    c.wall_nu = 1e-5f;
    c.bounce = BounceBack::Interpolated;
    c.ramp_steps = 0;
    return c;
}

void test_hot_changes(gpu::LbmGpu& G) {
    std::printf("[7] cambios en caliente con la GPU enganchada\n");
    Case c{"7", base_cfg(Precision::FP32), Scene{}};
    const auto g = voxelize(c.scene, c.cfg.nx, c.cfg.ny, c.cfg.nz);
    Scene s2 = c.scene;
    s2.bx1 += 4; s2.bz1 += 2; s2.second = false;
    const auto g2 = voxelize(s2, c.cfg.nx, c.cfg.ny, c.cfg.nz);
    Solver cpu, gs;
    setup(cpu, c, g);
    setup(gs, c, g);
    std::string err;
    CHECK(G.attach(gs, &err), "attach");
    cpu.step(50); G.step(50);
    // Geometría nueva con el flujo en marcha (el gancho baja las poblaciones antes del rebuild).
    cpu.set_geometry(g2.data(), WallSdf{&scene_sdf, &s2});
    gs.set_geometry(g2.data(), WallSdf{&scene_sdf, &s2});
    cpu.step(30); G.step(30);
    Diff d = compare(cpu, gs);
    std::printf("  tras set_geometry: |Δu|/u∞ %.2e |Δρ| %.2e |ΔF| %.2e/%.2e\n", d.du / c.cfg.u_inf, d.drho, d.df, d.fmax);
    CHECK(d.du / c.cfg.u_inf < 1e-5 && d.drho < 1e-5, "set_geometry en caliente");
    // Viscosidad y Smagorinsky en caliente, luego lote asíncrono pendiente al llamar a reset_flow.
    cpu.set_viscosity(3e-3f); gs.set_viscosity(3e-3f);
    cpu.set_smagorinsky(0.12f); gs.set_smagorinsky(0.12f);
    cpu.step(20); G.step(20);
    G.submit(10);            // en vuelo…
    gs.reset_flow();         // …el gancho lo termina y baja el estado; reset en la CPU
    cpu.step(10); cpu.reset_flow();
    CHECK(gs.steps() == 0 && cpu.steps() == 0, "reset_flow con lote en vuelo");
    cpu.step(25); G.step(25);
    d = compare(cpu, gs);
    std::printf("  tras reset_flow: |Δu|/u∞ %.2e |Δρ| %.2e\n", d.du / c.cfg.u_inf, d.drho);
    CHECK(d.du / c.cfg.u_inf < 1e-5 && d.drho < 1e-5, "reset en caliente");
    // Desenganchar y seguir en la CPU: el estado debe continuar exactamente.
    G.detach();
    cpu.step(15); gs.step(15);
    d = compare(cpu, gs);
    std::printf("  tras detach + 15 pasos en CPU: |Δu|/u∞ %.2e |Δρ| %.2e\n", d.du / c.cfg.u_inf, d.drho);
    CHECK(d.du / c.cfg.u_inf < 1e-5 && d.drho < 1e-5, "detach");
}

void test_async(gpu::LbmGpu& G) {
    std::printf("[9] cycle() asíncrono: lotes solapados + cambio de geometría con un lote en vuelo\n");
    Case c{"9", base_cfg(Precision::FP32), Scene{}};
    c.cfg.ramp_steps = 40;   // parámetros por paso distintos en cada lote
    const auto g = voxelize(c.scene, c.cfg.nx, c.cfg.ny, c.cfg.nz);
    Scene s2 = c.scene;
    s2.bx0 -= 3; s2.second = false;
    const auto g2 = voxelize(s2, c.cfg.nx, c.cfg.ny, c.cfg.nz);
    Solver cpu, gs;
    setup(cpu, c, g);
    setup(gs, c, g);
    std::string err;
    CHECK(G.attach(gs, &err), "attach");
    const int ks[] = {5, 9, 3, 12, 7, 20, 1, 8};
    int pub = 0, sub = 0;
    for (int i = 0; i < 8; ++i) {
        const int d = G.cycle(ks[i]);
        CHECK(d >= 0, "cycle");
        pub += d; sub += ks[i];
        CHECK(static_cast<int>(gs.steps()) == pub, "pasos publicados %d vs %llu", pub, static_cast<unsigned long long>(gs.steps()));
        if (i == 4) {   // geometría nueva con un lote en vuelo: el gancho lo termina y publica
            gs.set_geometry(g2.data(), WallSdf{&scene_sdf, &s2});
            pub = static_cast<int>(gs.steps());
            cpu.step(pub - static_cast<int>(cpu.steps()));
            cpu.set_geometry(g2.data(), WallSdf{&scene_sdf, &s2});
        }
    }
    pub += G.wait();
    CHECK(pub == sub && static_cast<int>(gs.steps()) == sub, "pasos totales %d/%d", pub, sub);
    cpu.step(sub - static_cast<int>(cpu.steps()));
    const Diff d = compare(cpu, gs);
    std::printf("  %d pasos en 8 lotes solapados: |Δu|/u∞ %.2e |Δρ| %.2e (últimas fuerzas |ΔF| %.2e/%.2e)\n", sub, d.du / c.cfg.u_inf, d.drho, d.df, d.fmax);
    CHECK(d.du / c.cfg.u_inf < 1e-5 && d.drho < 1e-5 && d.df < 1e-4 * d.fmax + 1e-9, "paridad en modo asíncrono");
    G.detach();
}

void test_f1(gpu::LbmGpu& G) {
    std::printf("[8] estabilidad y fuerzas: F1 2022 a resolución rápida (configuración de la app)\n");
    using namespace cfd::app;
    const int m = models::find("f1_2022");
    if (m < 0) { std::printf("  (sin modelo f1_2022: se omite)\n"); return; }
    Sim a, b;
    for (Sim* s : {&a, &b}) {
        s->cfg.model = m;
        s->cfg.preset = Preset::Rapida;
        s->cfg.ground = GroundMode::Moving;
        s->init();
    }
    std::string err;
    CHECK(G.attach(b.solver, &err), "attach F1: %s", err.c_str());
    const int steps = 600;
    const double t0 = now_sec();
    a.solver.step(steps - 100);
    a.solver.step(100);   // misma ventana de la media de fuerzas que el último lote de la GPU
    const double t1 = now_sec();
    for (int d = 0; d < steps; d += 100) G.step(100);
    const double t2 = now_sec();
    const ForceSample &fa = a.solver.forces_mean(), &fb = b.solver.forces_mean();
    const double da = fa.total.x, db = fb.total.x, la = fa.total.z, lb = fb.total.z;
    std::printf("  %d pasos: CPU %.2f s (%.0f MLUPS) · GPU %.2f s (%.0f MLUPS de GPU) · Fx %.4f / %.4f · Fz %.4f / %.4f · div %d/%d\n", steps,
                t1 - t0, a.solver.last_mlups(), t2 - t1, G.stats().mlups, da, db, la, lb, a.solver.diverged(), b.solver.diverged());
    CHECK(!b.solver.diverged(), "la GPU divergió en el F1");
    CHECK(std::isfinite(db) && std::isfinite(lb), "fuerzas no finitas");
    const Diff d = compare(a.solver, b.solver);
    std::printf("  |Δu|/u∞ %.2e |Δρ| %.2e (FP16S, flujo turbulento: sólo orden de magnitud)\n", d.du / a.cfg.u_lat, d.drho);
    CHECK(std::fabs(da - db) < 0.05 * std::fabs(da) + 1e-3 && std::fabs(la - lb) < 0.05 * std::fabs(la) + 1e-3, "fuerzas del F1 difieren > 5 %%");
    G.detach();
}

} // namespace

int main(int argc, char** argv) {
    std::printf("== test_gpu: solver LBM en la iGPU frente a la CPU\n");
    std::string why;
    gpu::LbmGpu G;
    if (!G.init(&why)) {
        std::printf("PASA (sin dispositivo Vulkan utilizable: %s)\n", why.c_str());
        return 0;
    }
    std::printf("%s", gpu::vk::describe(G.device()).c_str());
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    pool().start();

    {   // [1] básico: proyección de 2º orden SIN viscosidad de volumen propia (esquema anterior)
        Case c{"1", base_cfg(Precision::FP32), Scene{}};
        c.cfg.collision = Collision::Regularized; c.cfg.bulk_omega = 0.0f;
        c.cfg.ground = GroundMode::None; c.cfg.bounce = BounceBack::Implicit; c.cfg.wall_model = WallModel::None;
        c.scene.wheels = false; c.scene.second = false; c.scene.bz0 = 10; c.scene.bz1 = 20; c.sdf = false;
        run_case(G, c);
    }
    {   // [1b] proyección de 2º orden + viscosidad de volumen (ω_b = 1)
        Case c{"1b", base_cfg(Precision::FP32), Scene{}};
        c.cfg.collision = Collision::Regularized;
        c.cfg.ground = GroundMode::None; c.cfg.bounce = BounceBack::Implicit; c.cfg.wall_model = WallModel::None;
        c.scene.wheels = false; c.scene.second = false; c.scene.bz0 = 10; c.scene.bz1 = 20; c.sdf = false;
        run_case(G, c);
    }
    {   // [2] BGK
        Case c{"2", base_cfg(Precision::FP32), Scene{}};
        c.cfg.collision = Collision::BGK; c.cfg.ground = GroundMode::None; c.cfg.bounce = BounceBack::Implicit; c.cfg.wall_model = WallModel::None;
        c.cfg.nu = 5e-3f;
        c.scene.wheels = false; c.scene.second = false; c.scene.bz0 = 10; c.scene.bz1 = 20; c.sdf = false;
        run_case(G, c);
    }
    {   // [3] completa FP32
        Case c{"3", base_cfg(Precision::FP32), Scene{}};
        c.cfg.ramp_steps = 60;
        run_case(G, c);
    }
    {   // [4] completa FP16S
        Case c{"4", base_cfg(Precision::FP16S), Scene{}};
        c.cfg.ramp_steps = 60;
        c.tol_u = 2e-3; c.tol_rho = 2e-4; c.tol_f = 5e-3;
        run_case(G, c);
    }
    {   // [5] LogLaw + implícito + ruedas con Ladd
        Case c{"5", base_cfg(Precision::FP32), Scene{}};
        c.cfg.bounce = BounceBack::Implicit; c.cfg.wall_model = WallModel::LogLaw;
        run_case(G, c);
    }
    {   // [6] FP16S suelo fijo + LogLaw + interpolado
        Case c{"6", base_cfg(Precision::FP16S), Scene{}};
        c.cfg.ground = GroundMode::Static; c.cfg.wall_model = WallModel::LogLaw;
        // Flujo sensible: la referencia perturbada 1e-6 crece ×100 en 240 pasos (inestabilidad física junto al
        // suelo fijo); con 120 pasos la amplificación aún es pequeña y la comparación es significativa.
        c.steps = 120;
        c.tol_u = 5e-3; c.tol_rho = 3e-4; c.tol_f = 5e-3;
        run_case(G, c);
    }
    test_hot_changes(G);
    test_async(G);
    if (!quick) test_f1(G);
    std::printf(g_fail ? "FALLA (%d)\n" : "PASA\n", g_fail);
    return g_fail ? 1 : 0;
}
