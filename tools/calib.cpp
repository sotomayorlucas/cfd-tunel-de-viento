// ============================================================================
//  tools/calib.cpp — calibración reproducible (tabla de docs/FISICA.md §5): simula un modelo con la MISMA app::Sim
//  que la aplicación (dominio, voxelización, ley de pared, refinamiento local…) y promedia las fuerzas de los
//  últimos pasos de flujo (media aritmética de forces_mean(), sin el filtro exponencial de la app).
//
//    calib --model f1_2022 [--res media] [--refine 0|1|2] [--ft 4] [--avg 2] [--param clave=valor ...]
//          [--speed km/h] [--fp32] [--threads N] [--csv fila.csv]
//
//  Claves de --param: ride_front, ride_rear, front_flap, rear_flap, drs, yaw, aoa, height, gap, wheels.
//  Salida: una línea con SCz, SCx, balance, desglose por componente, alturas efectivas, dx por nivel y tiempos.
// ============================================================================
#include "app/app.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "models/model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace cfd;
using namespace cfd::app;

int main(int argc, char** argv) {
    std::string model = "f1_2022", csv;
    Preset preset = Preset::Media;
    int refine = -1, threads = 0;
    float ft = -1.0f, avg = 2.0f, speed = -1.0f;
    bool fp32 = false;
    models::Params P;
    std::vector<std::pair<std::string, float>> kv;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = nx();
        else if (a == "--res") { if (!parse_preset(nx(), preset)) { std::fprintf(stderr, "--res desconocido\n"); return 1; } }
        else if (a == "--refine") refine = std::atoi(nx());
        else if (a == "--ft") ft = std::strtof(nx(), nullptr);
        else if (a == "--avg") avg = std::strtof(nx(), nullptr);
        else if (a == "--speed") speed = std::strtof(nx(), nullptr);
        else if (a == "--fp32") fp32 = true;
        else if (a == "--threads") threads = std::atoi(nx());
        else if (a == "--csv") csv = nx();
        else if (a == "--param") {
            const std::string t = nx();
            const usize eq = t.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "--param clave=valor\n"); return 1; }
            kv.emplace_back(t.substr(0, eq), std::strtof(t.c_str() + eq + 1, nullptr));
        } else { std::fprintf(stderr, "opción desconocida: %s\n", a.c_str()); return 1; }
    }
    const int m = models::find(model);
    if (m < 0) { std::fprintf(stderr, "modelo desconocido: %s\n", model.c_str()); return 1; }
    pool().start(threads);
    const models::Info& I = models::info(m);
    P = models::resolve_params(m, models::Params{});
    for (const auto& [k, v] : kv) {
        if (k == "ride_front") P.ride_front_mm = v;
        else if (k == "ride_rear") P.ride_rear_mm = v;
        else if (k == "front_flap") P.front_flap_deg = v;
        else if (k == "rear_flap") P.rear_flap_deg = v;
        else if (k == "drs") P.drs_open = v != 0.0f;
        else if (k == "yaw") P.yaw_deg = v;
        else if (k == "aoa") P.aoa_deg = v;
        else if (k == "height") P.height_mm = v;
        else if (k == "gap") P.flap_gap_mm = v;
        else if (k == "wheels") P.wheels_rotating = v != 0.0f;
        else { std::fprintf(stderr, "clave desconocida: %s\n", k.c_str()); return 1; }
    }
    if (ft <= 0.0f) ft = I.kind == models::Kind::F1Car || I.needs_ground ? 4.0f : 5.0f;
    Sim s;
    s.cfg.model = m;
    s.cfg.params = P;
    s.cfg.preset = preset;
    s.cfg.refine = refine;
    s.cfg.fp32 = fp32;
    s.cfg.ground = I.needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None;
    s.speed_kmh = speed > 0 ? speed : I.default_speed_kmh;
    const double t0 = now_sec();
    s.init();
    const double t_init = now_sec() - t0;
    const long total = static_cast<long>(std::lround(ft * s.ft_steps()));
    const long start = total - static_cast<long>(std::lround(avg * s.ft_steps()));
    double acc_f[256][3] = {}, acc_m[256][3] = {};
    long acc_n = 0, done = 0;
    int div = 0;
    double t_run = 0;
    const double t1 = now_sec();
    while (done < total) {
        const int k = static_cast<int>(min_(200L, total - done));
        s.solver.step(k, false);
        done += k;
        if (s.solver.diverged()) {
            ++div;
            s.nu = min_(s.nu * 3.0f, 0.02f);
            s.solver.set_viscosity(s.nu);
            s.solver.reset_flow();
            done = 0; acc_n = 0;
            std::memset(acc_f, 0, sizeof acc_f); std::memset(acc_m, 0, sizeof acc_m);
            if (div > 3) break;
            continue;
        }
        if (done > start) {
            const lbm::ForceSample& f = s.solver.forces_mean();
            for (int id = 0; id < 256; ++id)
                for (int a = 0; a < 3; ++a) {
                    acc_f[id][a] += static_cast<double>((&f.force[id].x)[a]) * k;
                    acc_m[id][a] += static_cast<double>((&f.moment[id].x)[a]) * k;
                }
            acc_n += k;
        }
    }
    t_run = now_sec() - t1;
    Vec3 f[256], mo[256];
    for (int id = 0; id < 256; ++id) {
        const double in = acc_n ? 1.0 / static_cast<double>(acc_n) : 0.0;
        f[id] = Vec3(float(acc_f[id][0] * in), float(acc_f[id][1] * in), float(acc_f[id][2] * in));
        mo[id] = Vec3(float(acc_m[id][0] * in), float(acc_m[id][1] * in), float(acc_m[id][2] * in));
    }
    const AeroResult r = compute_aero(s, f, mo);
    const double cell_upd = static_cast<double>(total);
    std::printf("[calib] %s %s refine %d%s | base %dx%dx%d dx %.1f mm", I.id.c_str(), preset_key(preset), s.refine_levels(), div ? " (DIVERGE)" : "",
                s.dom.nx, s.dom.ny, s.dom.nz, static_cast<double>(s.dom.dx * 1e3f));
    for (const LevelPlan& L : s.levels) std::printf(" | %s %.1f mm %.2f M", L.name, static_cast<double>(L.dx * 1e3f), static_cast<double>(L.cells()) * 1e-6);
    std::printf(" | total %.2f M\n", static_cast<double>(s.total_cells) * 1e-6);
    std::printf("[calib]   SCz %+.3f  SCx %.3f  CL %+.4f CD %.4f CS %+.4f  L/D %.2f  balance %.0f %%  (%ld pasos, media de %.1f PF, ν %.1e)\n",
                static_cast<double>(r.scz), static_cast<double>(r.scx), static_cast<double>(r.cl), static_cast<double>(r.cd),
                static_cast<double>(r.cs), static_cast<double>(r.ld), static_cast<double>(r.balance), done, static_cast<double>(avg),
                static_cast<double>(s.nu));
    std::printf("[calib]   comp:");
    for (int c = 0; c < r.ncomp; ++c) std::printf("  %s %+.3f/%.3f", r.comp[c].name, static_cast<double>(r.comp[c].scz), static_cast<double>(r.comp[c].scx));
    std::printf("\n");
    if (I.param_mask & models::P_RideHeight)
        std::printf("[calib]   altura pedida %.0f/%.0f mm → efectiva %.0f/%.0f mm (hueco mínimo %.0f mm, dx bajo el fondo %.1f mm)\n",
                    static_cast<double>(s.params.ride_front_mm), static_cast<double>(s.params.ride_rear_mm), static_cast<double>(s.ride_eff_front_mm),
                    static_cast<double>(s.ride_eff_rear_mm), static_cast<double>(s.ride_gap_min_mm), static_cast<double>(s.dx_under * 1e3f));
    std::printf("[calib]   tiempo: init %.1f s, %.1f s simulando (%.2f s/PF, %.1f ms/paso base, %.0f MLUPS-eq último lote)\n", t_init, t_run,
                t_run / static_cast<double>(ft), 1e3 * t_run / cell_upd, s.solver.last_mlups());
    if (!csv.empty()) {
        FILE* fp = std::fopen(csv.c_str(), "a");
        if (fp) {
            std::fprintf(fp, "%s,%s,%d,%.4f,%.4f,%.4f,%.2f,%.1f,%.1f,%.2f\n", I.id.c_str(), preset_key(preset), s.refine_levels(), static_cast<double>(r.scz),
                         static_cast<double>(r.scx), static_cast<double>(r.cl), static_cast<double>(r.balance), static_cast<double>(s.ride_eff_front_mm),
                         static_cast<double>(s.ride_eff_rear_mm), t_run);
            std::fclose(fp);
        }
    }
    return div > 3 ? 2 : 0;
}
