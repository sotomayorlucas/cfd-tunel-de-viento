// ============================================================================
//  app/cli.cpp — línea de órdenes y modos sin ventana: listado de modelos,
//  ejecución con captura PNG/CSV, barridos, benchmark, carga para PGO y
//  comprobación de estabilidad de todos los modelos.
// ============================================================================
#include "app.hpp"

#include "../core/threadpool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cfd::app {

namespace {

bool parse_float(const char* s, float& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    const float v = std::strtof(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v)) return false;
    out = v;
    return true;
}
bool parse_long(const char* s, long& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = v;
    return true;
}
bool known_param(const std::string& k) {
    static const char* const keys[] = {"ride_front", "ride_rear", "ride", "front_flap", "rear_flap", "drs", "yaw", "aoa", "height", "gap", "wheels"};
    for (const char* x : keys) if (k == x) return true;
    return false;
}

} // namespace

void print_usage(const char* argv0) {
    std::printf(
        "Túnel de viento CFD 3D (Lattice Boltzmann D3Q19, C++23, render por software)\n\n"
        "Uso: %s [opciones]\n\n"
        "  --list                     lista los modelos disponibles\n"
        "  --model <id>               modelo (defecto f1_2022)\n"
        "  --res rapida|media|alta|ultra   resolución (2.5/6/13/28 M celdas; defecto media)\n"
        "  --cells N                  presupuesto de celdas explícito (p.ej. 800000)\n"
        "  --ground none|static|moving     suelo: ninguno, fijo o cinta móvil\n"
        "  --speed <km/h>             velocidad: reescala N/kgf y fija el Reynolds real de la ley de pared\n"
        "  --param clave=valor        ride_front, ride_rear, ride (mm, conserva el rake), front_flap,\n"
        "                             rear_flap (°), drs (0/1), yaw (°), aoa (°), height (mm), gap (mm), wheels (0/1)\n"
        "  --fp32                     poblaciones en FP32 (defecto FP16S)\n"
        "  --gpu                      solver en la iGPU (Vulkan de cómputo propio; la CPU dibuja en paralelo)\n"
        "  --nu <valor>               viscosidad de red (defecto automática)\n"
        "  --cs <valor>               constante de Smagorinsky del LES (defecto 0.10; 0 = sin LES)\n"
        "  --prio fluidez|equilibrado|max   reparto del cuadro entre simulación y visualización (ventana)\n"
        "  --pause                    arrancar en pausa (ventana)\n"
        "  --wall none|log|slip       modelo de pared junto a los sólidos (ver docs/FISICA.md)\n"
        "  --bb interp|implicit|1|0   rebote interpolado (Bouzidi, superficie real) o implícito (escalera)\n"
        "  --ramp PF                  rampa de arranque de u∞ en pasos de flujo (0 = impulsivo)\n"
        "  --threads N                hilos del pool (defecto: todos menos LP-E)\n"
        "  --headless                 sin ventana (junto con --steps/--ft, --shot, --csv, --sweep)\n"
        "  --steps N | --ft F         pasos de red o pasos de flujo (L/U∞) a simular sin ventana\n"
        "  --frames N | --quit-after S    con ventana: salir tras N cuadros o S segundos\n"
        "  --spf N                    pasos de red por cuadro fijos (medidas de rendimiento; defecto adaptativo)\n"
        "  --view lateral|superior|frontal|trasera|34|bajo   cámara inicial\n"
        "  --cam x,y,z,guiñada,cabeceo,dist   cámara libre (objetivo en m del modelo, grados, distancia en m)\n"
        "  --vis a,b,...              visualización: cp, speed, component, plain, voxels, streamlines,\n"
        "                             nostreamlines, smoke, vortex, cloud, footprint, nofootprint, arrows,\n"
        "                             noarrows, comp-arrows, slice-x|slice-y|slice-z[=pos_m], noslice,\n"
        "                             q=speed|ux|uz|cp|cp0|vort|q, slice-range=a:b, opacity=f, lic,\n"
        "                             rake=vertical|floor|tips|grid, smoke-rake=..., lines=N, line-cp,\n"
        "                             q-threshold=f (Q·L²/U²; defecto 300 coche, 30 ala, 400 cuerpo), cp-range=a:b, nowire, nossao,\n"
        "                             nofxaa, nolegends, novortex, nosmoke, noprobe\n"
        "  --panel a,b                secciones abiertas: modelo,config,tunel,vis,resultados,barrido,rendimiento|todo\n"
        "  --shot out.png             captura PNG completa (con la UI) al terminar\n"
        "  --csv fuerzas.csv          historia de coeficientes (o resultados del barrido)\n"
        "  --sweep param:desde:hasta:n    barrido (aoa, height, ride, front_flap, rear_flap, yaw, gap)\n"
        "  --settle F --avg F         pasos de flujo de asentamiento / promedio por punto del barrido\n"
        "  --bench                    benchmark fijo: MLUPS y desglose del tiempo por cuadro\n"
        "  --bench-pgo                carga representativa (~30 s) para `make pgo`\n"
        "  --stability                comprueba la estabilidad de todos los modelos (con --res: sólo ese preset)\n"
        "  --size WxH                 tamaño del framebuffer (defecto 1920x1200)\n"
        "  --scale N                  escala de píxel de la ventana (defecto automática: 2 en 4K)\n"
        "  --help                     esta ayuda\n\n"
        "Controles con ventana: F1 muestra la ayuda.\n",
        argv0);
}

bool parse_cli(int argc, const char* const* argv, Options& o) {
    auto need = [&](int& i, const char* opt) -> const char* {
        if (i + 1 >= argc) { o.error = std::string("falta el valor de ") + opt; return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = nullptr;
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--list") o.list = true;
        else if (a == "--model") { if (!(v = need(i, "--model"))) return false; o.model = v; o.model_set = true; }
        else if (a == "--res") {
            if (!(v = need(i, "--res"))) return false;
            if (!parse_preset(v, o.preset)) { o.error = std::string("resolución desconocida '") + v + "' (rapida|media|alta|ultra)"; return false; }
            o.preset_set = true;
        } else if (a == "--cells") {
            long n = 0;
            if (!(v = need(i, "--cells"))) return false;
            if (!parse_long(v, n) || n < 20000) { o.error = "--cells necesita un entero ≥ 20000"; return false; }
            o.cells = static_cast<usize>(n);
        } else if (a == "--ground") {
            if (!(v = need(i, "--ground"))) return false;
            const std::string g = v;
            if (g == "none" || g == "ninguno") o.ground = 0;
            else if (g == "static" || g == "fijo") o.ground = 1;
            else if (g == "moving" || g == "cinta") o.ground = 2;
            else { o.error = "suelo desconocido '" + g + "' (none|static|moving)"; return false; }
        } else if (a == "--speed") {
            if (!(v = need(i, "--speed"))) return false;
            if (!parse_float(v, o.speed) || o.speed <= 0) { o.error = "--speed necesita un número > 0 (km/h)"; return false; }
        } else if (a == "--param") {
            if (!(v = need(i, "--param"))) return false;
            const std::string p = v;
            const usize eq = p.find('=');
            CliParam cp;
            if (eq == std::string::npos) { o.error = "--param necesita clave=valor (p.ej. aoa=6)"; return false; }
            cp.key = p.substr(0, eq);
            if (!known_param(cp.key)) { o.error = "parámetro desconocido '" + cp.key + "'"; return false; }
            if (!parse_float(p.substr(eq + 1).c_str(), cp.value)) { o.error = "valor no numérico en --param " + p; return false; }
            o.params.push_back(cp);
        } else if (a == "--fp32") o.fp32 = true;
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--nu") {
            if (!(v = need(i, "--nu"))) return false;
            if (!parse_float(v, o.nu) || o.nu <= 0 || o.nu > 0.1f) { o.error = "--nu necesita un número en (0, 0.1]"; return false; }
        } else if (a == "--cs") {
            if (!(v = need(i, "--cs"))) return false;
            if (!parse_float(v, o.cs) || o.cs < 0 || o.cs > 0.5f) { o.error = "--cs necesita un número en [0, 0.5]"; return false; }
        } else if (a == "--prio") {
            if (!(v = need(i, "--prio"))) return false;
            const std::string pz = v;
            if (pz == "fluidez") o.prio = 0;
            else if (pz == "equilibrado") o.prio = 1;
            else if (pz == "max" || pz == "maxima" || pz == "máxima") o.prio = 2;
            else { o.error = "prioridad desconocida '" + pz + "' (fluidez|equilibrado|max)"; return false; }
        } else if (a == "--pause") o.start_paused = true;
        else if (a == "--wall") {
            if (!(v = need(i, "--wall"))) return false;
            const std::string w = v;
            if (w == "none" || w == "ninguna") o.wall = 0;
            else if (w == "log" || w == "loglaw") o.wall = 1;
            else if (w == "slip" || w == "deslizamiento") o.wall = 2;
            else { o.error = "modelo de pared desconocido '" + w + "' (none|log|slip)"; return false; }
        } else if (a == "--bb") {
            if (!(v = need(i, "--bb"))) return false;
            const std::string w = v;
            if (w == "interp" || w == "bouzidi" || w == "1") o.interp_bb = 1;
            else if (w == "implicit" || w == "escalera" || w == "0") o.interp_bb = 0;
            else { o.error = "rebote desconocido '" + w + "' (interp|implicit)"; return false; }
        } else if (a == "--ramp") {
            if (!(v = need(i, "--ramp"))) return false;
            if (!parse_float(v, o.ramp_ft) || o.ramp_ft < 0 || o.ramp_ft > 10) { o.error = "--ramp necesita pasos de flujo en [0, 10]"; return false; }
        }
        else if (a == "--threads") {
            long n = 0;
            if (!(v = need(i, "--threads"))) return false;
            if (!parse_long(v, n) || n < 1 || n > 256) { o.error = "--threads necesita un entero en [1, 256]"; return false; }
            o.threads = static_cast<int>(n);
        } else if (a == "--headless") o.headless = true;
        else if (a == "--steps") {
            if (!(v = need(i, "--steps"))) return false;
            if (!parse_long(v, o.steps) || o.steps < 0) { o.error = "--steps necesita un entero ≥ 0"; return false; }
        } else if (a == "--ft") {
            if (!(v = need(i, "--ft"))) return false;
            if (!parse_float(v, o.flow_throughs) || o.flow_throughs < 0) { o.error = "--ft necesita un número ≥ 0"; return false; }
        } else if (a == "--frames") {
            if (!(v = need(i, "--frames"))) return false;
            if (!parse_long(v, o.frames) || o.frames < 1) { o.error = "--frames necesita un entero ≥ 1"; return false; }
        } else if (a == "--spf") {
            long n = 0;
            if (!(v = need(i, "--spf"))) return false;
            if (!parse_long(v, n) || n < 1 || n > 2000) { o.error = "--spf necesita un entero en [1, 2000]"; return false; }
            o.spf = static_cast<int>(n);
        } else if (a == "--quit-after") {
            if (!(v = need(i, "--quit-after"))) return false;
            if (!parse_float(v, o.quit_after) || o.quit_after <= 0) { o.error = "--quit-after necesita segundos > 0"; return false; }
        } else if (a == "--view") {
            if (!(v = need(i, "--view"))) return false;
            CamView cv;
            if (!parse_cam_view(v, cv)) { o.error = std::string("vista desconocida '") + v + "' (lateral|superior|frontal|trasera|34|bajo)"; return false; }
            o.view = static_cast<int>(cv);
        } else if (a == "--cam") {
            if (!(v = need(i, "--cam"))) return false;
            if (std::sscanf(v, "%f,%f,%f,%f,%f,%f", &o.cam[0], &o.cam[1], &o.cam[2], &o.cam[3], &o.cam[4], &o.cam[5]) != 6 || o.cam[5] <= 0) {
                o.error = "--cam necesita x,y,z,guiñada,cabeceo,distancia (objetivo en m, ángulos en grados, distancia en m)";
                return false;
            }
            o.cam_set = true;
        } else if (a == "--vis") { if (!(v = need(i, "--vis"))) return false; o.vis = v; }
        else if (a == "--panel") { if (!(v = need(i, "--panel"))) return false; o.panel = v; }
        else if (a == "--shot") { if (!(v = need(i, "--shot"))) return false; o.shot = v; }
        else if (a == "--csv") { if (!(v = need(i, "--csv"))) return false; o.csv = v; }
        else if (a == "--sweep") {
            if (!(v = need(i, "--sweep"))) return false;
            char key[32] = {};
            float f0 = 0, f1 = 0;
            int n = 0;
            if (std::sscanf(v, "%31[^:]:%f:%f:%d", key, &f0, &f1, &n) != 4 || n < 2 || n > Sweep::k_max) {
                o.error = "--sweep necesita parametro:desde:hasta:n (n entre 2 y 64), p.ej. height:25:200:4";
                return false;
            }
            if (!parse_sweep_param(key, o.sweep_param)) { o.error = std::string("parámetro de barrido desconocido '") + key + "'"; return false; }
            o.sweep = true; o.sweep_from = f0; o.sweep_to = f1; o.sweep_n = n;
        } else if (a == "--settle") {
            if (!(v = need(i, "--settle"))) return false;
            if (!parse_float(v, o.settle_ft) || o.settle_ft <= 0) { o.error = "--settle necesita un número > 0"; return false; }
        } else if (a == "--avg") {
            if (!(v = need(i, "--avg"))) return false;
            if (!parse_float(v, o.avg_ft) || o.avg_ft <= 0) { o.error = "--avg necesita un número > 0"; return false; }
        } else if (a == "--bench") o.bench = true;
        else if (a == "--bench-pgo") o.bench_pgo = true;
        else if (a == "--stability") o.stability = true;
        else if (a == "--size") {
            if (!(v = need(i, "--size"))) return false;
            if (std::sscanf(v, "%dx%d", &o.fb_w, &o.fb_h) != 2 || o.fb_w < 640 || o.fb_h < 400 || o.fb_w > 8000 || o.fb_h > 8000) {
                o.error = "--size necesita AxB (mínimo 640x400)";
                return false;
            }
        } else if (a == "--scale") {
            long n = 0;
            if (!(v = need(i, "--scale"))) return false;
            if (!parse_long(v, n) || n < 1 || n > 4) { o.error = "--scale necesita 1..4"; return false; }
            o.scale = static_cast<int>(n);
        } else {
            o.error = "opción desconocida '" + a + "' (usa --help)";
            return false;
        }
    }
    return true;
}

namespace {

void print_results(const Sim& s) {
    const AeroResult& r = s.res;
    const models::Info& I = s.built.info;
    std::printf("[cfd] %s — %s · red %dx%dx%d (%.2f M celdas) dx %.2f mm · %s · suelo %d · ν %.1e\n", I.id.c_str(), I.name.c_str(), s.dom.nx,
                s.dom.ny, s.dom.nz, static_cast<double>(s.dom.cells()) * 1e-6, static_cast<double>(s.dom.dx * 1e3f), s.cfg.fp32 ? "FP32" : "FP16S",
                static_cast<int>(s.cfg.ground), static_cast<double>(s.nu));
    std::printf("[cfd] pasos %llu (%.2f pasos de flujo) · %s (Δ %.2f %%)\n", static_cast<unsigned long long>(s.solver.steps()),
                static_cast<double>(s.flow_throughs()), s.converged ? "convergido" : "sin converger", static_cast<double>(s.conv_rel * 100.0f));
    std::printf("[cfd] CL(carga+) %+.4f  CD %.4f  CS %+.4f  SCz %+.4f m²  SCx %.4f m²  L/D %.3f", static_cast<double>(r.cl), static_cast<double>(r.cd),
                static_cast<double>(r.cs), static_cast<double>(r.scz), static_cast<double>(r.scx), static_cast<double>(r.ld));
    if (std::isfinite(r.balance) && r.cl > 0.05f) std::printf("  balance %.1f %% del.", static_cast<double>(r.balance));
    std::printf("\n[cfd] a %.0f km/h: carga %.0f N (%.0f kgf), resistencia %.0f N, potencia %.0f kW · ref. real aprox. SCz %.2f SCx %.2f\n",
                static_cast<double>(s.speed_kmh), static_cast<double>(r.down_n), static_cast<double>(r.down_kgf), static_cast<double>(r.drag_n),
                static_cast<double>(r.power_kw), static_cast<double>(I.ref_ClA), static_cast<double>(I.ref_CdA));
    if (s.is_car())
        std::printf("[cfd] altura de marcha pedida %.0f / %.0f mm → efectiva (resolución) %.0f / %.0f mm (hueco mínimo %.0f mm)%s\n",
                    static_cast<double>(s.params.ride_front_mm), static_cast<double>(s.params.ride_rear_mm), static_cast<double>(s.ride_eff_front_mm),
                    static_cast<double>(s.ride_eff_rear_mm), static_cast<double>(s.ride_gap_min_mm), s.ride_limited ? " [limitada por la red]" : "");
    for (int i = 0; i < r.ncomp; ++i) {
        int nch = 0;   // relleno por caracteres UTF-8, no por bytes ("Carrocería")
        for (const char* c = r.comp[i].name; *c; ++c) nch += (static_cast<unsigned char>(*c) & 0xC0) != 0x80;
        std::printf("[cfd]    %s%*s SCz %+.4f  SCx %+.4f\n", r.comp[i].name, max_(24 - nch, 1), "", static_cast<double>(r.comp[i].scz),
                    static_cast<double>(r.comp[i].scx));
    }
}

// Avanza la simulación `total` pasos sin ventana. Los últimos `vis_steps` pasos se hacen por cuadros
// (con la visualización actualizándose: el humo necesita integrarse); el resto va directo al solver.
bool run_steps(App& app, long total, long vis_steps, int chunk) {
    long done = 0;
    bool ok = true;
    const long direct = max_(0L, total - vis_steps);
    while (done < direct) {
        const int k = static_cast<int>(min_(static_cast<long>(chunk), direct - done));
        if (!app.sim.step(k)) { ok = false; std::fprintf(stderr, "[cfd] ¡divergencia! ν → %.2e, flujo reiniciado\n", static_cast<double>(app.sim.nu)); }
        done += k;
    }
    while (done < total) {
        const int k = static_cast<int>(min_(static_cast<long>(chunk), total - done));
        app.frame(k, false);
        done += k;
    }
    return ok;
}

long default_vis_steps(const App& app) {
    const VisSettings& v = app.view.vs;
    if (!v.smoke) return 0;
    const float u = app.sim.cfg.u_lat * max_(v.smoke_speed, 1.0f);
    return static_cast<long>(1.1f * static_cast<float>(app.sim.dom.nx) / u);
}

int run_headless(const Options& o) {
    App app;
    if (!app.setup(o, true)) return 1;
    app.fixed_steps = true;
    Sim& s = app.sim;
    const float ft = s.ft_steps();
    long total = o.steps >= 0 ? o.steps : static_cast<long>((o.flow_throughs >= 0 ? o.flow_throughs : 2.0f) * ft);
    const int chunk = clamp_(static_cast<int>(ft / 40.0f), 10, 100);
    app.steps_fixed = chunk;
    std::printf("[cfd] %s: red %dx%dx%d (%.2f M celdas), dx %.2f mm, paso de flujo = %.0f pasos → simulando %ld pasos (%.2f PF)\n",
                s.built.info.id.c_str(), s.dom.nx, s.dom.ny, s.dom.nz, static_cast<double>(s.dom.cells()) * 1e-6,
                static_cast<double>(s.dom.dx * 1e3f), static_cast<double>(ft), total, static_cast<double>(total) / static_cast<double>(ft));
    const double t0 = now_sec();
    if (o.sweep) {
        app.sweep.param = o.sweep_param;
        app.sweep.from = o.sweep_from; app.sweep.to = o.sweep_to; app.sweep.n = o.sweep_n;
        app.sweep.settle_ft = o.settle_ft; app.sweep.avg_ft = o.avg_ft;
        // Asentamiento inicial antes del primer punto (el flujo parte del reposo).
        run_steps(app, total, 0, chunk);
        if (!sweep_start(app.sweep, s)) { std::fprintf(stderr, "[cfd] el parámetro '%s' no está disponible en este modelo\n", sweep_param_key(o.sweep_param)); return 1; }
        std::printf("[cfd] barrido de %s: %d puntos de %.2f a %.2f %s (asentar %.1f + promediar %.1f PF)\n", sweep_param_name(o.sweep_param), o.sweep_n,
                    static_cast<double>(o.sweep_from), static_cast<double>(o.sweep_to), sweep_param_unit(o.sweep_param), static_cast<double>(o.settle_ft),
                    static_cast<double>(o.avg_ft));
        int last = 0;
        while (app.sweep.running) {
            if (!s.step(chunk)) { std::fprintf(stderr, "[cfd] ¡divergencia durante el barrido!\n"); sweep_stop(app.sweep, s); break; }
            sweep_after_step(app.sweep, s, chunk);
            if (app.sweep.npts != last) {
                const SweepPoint& p = app.sweep.pts[app.sweep.npts - 1];
                std::printf("[cfd]   %s = %8.2f  CL %+.4f  CD %.4f  SCz %+.4f  SCx %.4f  L/D %6.3f  bal %.1f\n", sweep_param_key(o.sweep_param),
                            static_cast<double>(p.x), static_cast<double>(p.cl), static_cast<double>(p.cd), static_cast<double>(p.scz),
                            static_cast<double>(p.scx), static_cast<double>(p.ld), static_cast<double>(p.bal));
                std::fflush(stdout);
                last = app.sweep.npts;
            }
        }
        const std::string path = o.csv.empty() ? app.auto_shot_path("barrido", "csv") : o.csv;
        const usize slash = path.find_last_of('/');
        if (slash != std::string::npos) ensure_dir(path.substr(0, slash));
        if (sweep_write_csv(app.sweep, s, path)) std::printf("[cfd] barrido → %s\n", path.c_str());
        app.sec_open[SecBarrido] = true;
    } else {
        run_steps(app, total, min_(total, default_vis_steps(app)), chunk);
        print_results(s);
        if (!o.csv.empty()) {
            if (app.write_forces_csv(o.csv)) std::printf("[cfd] CSV → %s\n", o.csv.c_str());
            else std::fprintf(stderr, "[cfd] no se pudo escribir %s\n", o.csv.c_str());
        }
    }
    const double tsim = now_sec() - t0;
    std::printf("[cfd] %.1f s de simulación (%.0f MLUPS en el último paso)\n", tsim, s.solver.last_mlups());
    if (!o.shot.empty()) {
        // Dos cuadros completos (el primero calienta la UI: alturas de secciones, desplazamiento).
        app.frame(1, true);
        app.frame(1, true);
        if (app.save_screenshot(o.shot)) std::printf("[cfd] captura → %s\n", o.shot.c_str());
        else { std::fprintf(stderr, "[cfd] no se pudo guardar %s\n", o.shot.c_str()); return 1; }
    }
    return 0;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Benchmark fijo: coche 2022, preset medio, visualización por defecto (+ humo), k pasos fijos.
int run_bench(const Options& in) {
    Options o = in;
    if (in.model == "f1_2022" && !in.preset_set && !in.cells) o.preset = Preset::Media;
    App app;
    if (!app.setup(o, true)) return 1;
    Sim& s = app.sim;
    app.fixed_steps = true;
    std::printf("[bench] %s · %dx%dx%d (%.2f M celdas) · %s · %d hilos%s\n", s.built.info.id.c_str(), s.dom.nx, s.dom.ny, s.dom.nz,
                static_cast<double>(s.dom.cells()) * 1e-6, s.cfg.fp32 ? "FP32" : "FP16S", pool().size(), s.gpu_on() ? " · solver en la iGPU" : "");
    // 1) Solver puro
    s.step(200);
    std::vector<double> ml;
    for (int r = 0; r < 5; ++r) { s.step(100); ml.push_back(s.solver.last_mlups()); }
    std::printf("[bench] solver: %.0f MLUPS (mediana de 5 × 100 pasos; kernel %.2f ms/paso, fuerzas %.3f ms/paso)\n", median(ml),
                s.solver.last_kernel_seconds() * 1e3 / 100.0, s.solver.last_force_seconds() * 1e3 / 100.0);
    // 2) Reconstrucción geométrica (antirrebote: 1 por cuadro como máximo)
    {
        std::vector<double> tb;
        models::Params p = s.params;
        // Parámetro geométrico que tenga el modelo (altura de marcha, AoA, altura, guiñada...).
        SweepParam sp = SweepParam::Yaw;
        for (int i = 0; i < k_nsweep; ++i)
            if (sweep_param_available(s.cfg.model, static_cast<SweepParam>(i))) { sp = static_cast<SweepParam>(i); break; }
        const float v0 = sweep_param_get(s.params, sp);
        for (int r = 0; r < 5; ++r) {
            sweep_param_set(p, sp, v0 + (r % 2 ? 2.0f : -2.0f));
            const double t0 = now_sec();
            s.set_params(p, 0.5f);
            tb.push_back((now_sec() - t0) * 1e3);
        }
        std::printf("[bench] reconstrucción (build+vóxeles+solver+malla 0.5dx): %.1f ms (vóx %.1f · set_geometry %.1f · malla %.1f)\n", median(tb),
                    s.times.vox, s.times.geo, s.times.mesh);
    }
    // 3) Cuadros completos
    const int ks[2] = {4, 8};
    for (int k : ks) {
        app.view.vs.smoke = true;
        app.view.dirty = true;
        for (int i = 0; i < 20; ++i) app.frame(k, true);
        std::vector<double> fr, si, vi, re, ui, me, fl, po;
        for (int i = 0; i < 60; ++i) {
            app.frame(k, true);
            fr.push_back(app.perf.last_frame_ms); si.push_back(app.perf.last_sim_ms); vi.push_back(app.perf.last_vis_ms);
            re.push_back(app.perf.last_render_ms); ui.push_back(app.perf.last_ui_ms);
            me.push_back(app.view.t_mesh); fl.push_back(app.view.t_vis); po.push_back(app.view.t_post);
        }
        const double f = median(fr);
        std::printf("[bench] k=%d pasos/cuadro: cuadro %.2f ms (%.1f FPS) = sim %.2f + vis %.2f + render %.2f (malla %.2f, flujo %.2f, post %.2f) + UI %.2f ms\n",
                    k, f, 1000.0 / f, median(si), median(vi), median(re), median(me), median(fl), median(po), median(ui));
    }
    std::printf("[bench] detalle de la actualización: muestreo %.2f · líneas %.2f · humo %.2f · color %.2f ms (último cuadro)\n",
                app.view.t_upd_sampler, app.view.t_upd_lines, app.view.t_upd_smoke, app.view.t_upd_color);
    return 0;
}

// Carga representativa para PGO: varios modelos, todas las visualizaciones, UI completa.
int run_bench_pgo(const Options& in) {
    const double t0 = now_sec();
    struct Case { const char* model; Preset p; bool fp32; };
    const Case cases[] = {{"f1_2022", Preset::Rapida, false}, {"f1_wing_ge", Preset::Rapida, false}, {"sphere", Preset::Rapida, false},
                          {"f1_1979", Preset::Rapida, true}, {"naca4412_wing", Preset::Rapida, false}};
    for (const Case& c : cases) {
        Options o = in;
        o.model = c.model;
        o.preset = c.p;
        o.fp32 = c.fp32;
        o.panel = "todo";
        App app;
        if (!app.setup(o, true)) return 1;
        app.fixed_steps = true;
        Sim& s = app.sim;
        s.step(static_cast<int>(min_(0.4f * s.ft_steps(), 1500.0f)));
        VisSettings& v = app.view.vs;
        v.lines = true; v.smoke = true; v.vortex = true; v.slice_axis = 2; v.footprint = true; v.comp_arrows = true;
        app.view.dirty = true;
        for (int i = 0; i < 40; ++i) {
            if (i == 10) { v.slice_lic = true; v.slice_q = 5; v.surf = SurfMode::Voxels; app.view.dirty = true; }
            if (i == 20) { v.slice_axis = 1; v.slice_q = 4; v.surf = SurfMode::Speed; v.vortex_cloud = true; app.view.dirty = true; }
            if (i == 30) { v.slice_axis = 3; v.surf = SurfMode::Component; v.line_rake = RakeKind::Grid; app.view.dirty = true; }
            app.frame(8, true);
        }
        // Reconstrucciones (como al arrastrar un deslizador)
        models::Params p = s.params;
        for (int r = 0; r < 3; ++r) {
            sweep_param_set(p, app.sweep.param == SweepParam::Count ? SweepParam::Yaw : app.sweep.param,
                            sweep_param_get(p, app.sweep.param == SweepParam::Count ? SweepParam::Yaw : app.sweep.param) + (r - 1) * 1.0f);
            s.set_params(p, r == 2 ? 0.5f : 1.0f);
            app.frame(8, true);
        }
        std::printf("[pgo] %s listo (%.1f s)\n", c.model, now_sec() - t0);
    }
    std::printf("[pgo] carga completa en %.1f s\n", now_sec() - t0);
    return 0;
}

// Estabilidad: cada modelo × preset, ~1 paso de flujo (o --ft), comprobando diverged().
int run_stability(const Options& in) {
    const float fts = in.flow_throughs > 0 ? in.flow_throughs : 1.0f;
    int bad = 0, total = 0;
    const double t0 = now_sec();
    for (int pi = 0; pi < k_npresets; ++pi) {
        const Preset p = static_cast<Preset>(pi);
        if (in.preset_set && p != in.preset) continue;
        for (int m = 0; m < models::count(); ++m) {
            if (in.model_set && in.model != "todos" && models::info(m).id != in.model) continue;
            Sim s;
            s.cfg.model = m;
            s.cfg.preset = p;
            s.cfg.cells = in.cells;
            s.cfg.fp32 = in.fp32;
            s.cfg.gpu = in.gpu;
            s.cfg.nu = in.nu;
            s.cfg.ground = in.ground >= 0 ? static_cast<lbm::GroundMode>(in.ground)
                                          : (models::info(m).needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None);
            if (in.params.size()) for (const CliParam& cp : in.params) if (cp.key == "drs") s.cfg.params.drs_open = cp.value != 0;
            s.init();
            const long n = static_cast<long>(fts * s.ft_steps()) + s.solver.config().ramp_steps;
            const double t1 = now_sec();
            long done = 0;
            bool div = false;
            float umax = 0;
            while (done < n) {
                const int k = static_cast<int>(min_(200L, n - done));
                if (s.gpu_on()) s.gpu->step(k);
                else s.solver.step(k, true);
                done += k;
                if (s.solver.diverged()) { div = true; break; }
            }
            if (!div) {
                const lbm::FieldView f = s.solver.field();
                const usize N = s.dom.cells();
                for (usize i = 0; i < N; ++i) umax = max_(umax, std::fabs(f.ux[i]) + std::fabs(f.uy[i]) + std::fabs(f.uz[i]));
            }
            ++total;
            bad += div;
            std::printf("[estabilidad] %-6s %-14s %4dx%4dx%4d dx %5.1f mm ν %.1e: %s tras %ld pasos (%.1f PF) · |u|₁ máx %.3f · %.0f MLUPS · %.1f s\n",
                        preset_key(p), models::info(m).id.c_str(), s.dom.nx, s.dom.ny, s.dom.nz, static_cast<double>(s.dom.dx * 1e3f),
                        static_cast<double>(s.nu), div ? "DIVERGE" : "estable", done, static_cast<double>(done) / static_cast<double>(s.ft_steps()),
                        static_cast<double>(umax), s.solver.last_mlups(), now_sec() - t1);
            std::fflush(stdout);
        }
    }
    std::printf("[estabilidad] %d/%d casos estables en %.0f s\n", total - bad, total, now_sec() - t0);
    return bad ? 2 : 0;
}

} // namespace

int run_cli(const Options& o) {
    if (o.help) { print_usage("cfd"); return 0; }
    if (o.list) {
        // Relleno por caracteres (no bytes): los nombres llevan tildes y rayas UTF-8.
        auto pad = [](const char* t, int w) {
            int n = 0;
            for (const char* c = t; *c; ++c) n += (static_cast<unsigned char>(*c) & 0xC0) != 0x80;
            std::printf("%s%*s", t, max_(w - n, 1), "");
        };
        pad("id", 16); pad("tipo", 10); pad("nombre", 54); std::printf("época\n");
        for (int i = 0; i < models::count(); ++i) {
            const models::Info& I = models::info(i);
            pad(I.id.c_str(), 16); pad(models::kind_name(I.kind), 10); pad(I.name.c_str(), 54);
            std::printf("%s\n", I.era.c_str());
        }
        return 0;
    }
    if (models::find(o.model) < 0 && !(o.stability && o.model == "todos")) {
        std::fprintf(stderr, "[cfd] modelo desconocido '%s'. Modelos disponibles:\n", o.model.c_str());
        for (int i = 0; i < models::count(); ++i) std::fprintf(stderr, "  %s\n", models::info(i).id.c_str());
        return 1;
    }
    if (o.stability) { pool().start(o.threads); return run_stability(o); }
    if (o.bench) return run_bench(o);
    if (o.bench_pgo) return run_bench_pgo(o);
    if (o.headless || o.sweep || !o.shot.empty() || o.steps >= 0 || o.flow_throughs >= 0) {
        if (!o.headless && (o.frames > 0 || o.quit_after > 0)) {
            // con ventana y salida programada: --shot guarda el último cuadro al salir; --steps/--ft
            // se ignoran (usar --headless)
        } else {
            return run_headless(o);
        }
    }
    App app;
    if (!app.setup(o, false)) return 1;
    return app.run_interactive();
}

} // namespace cfd::app
