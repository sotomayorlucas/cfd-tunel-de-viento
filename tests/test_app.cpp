// ============================================================================
//  tests/test_app.cpp — pruebas de la aplicación (integración):
//   1. Dimensionado del dominio: invariantes para los 18 modelos × 4 presets.
//   2. Balance aerodinámico con fuerzas sintéticas (100 % / 50 % / 0 %, guiñada,
//      resistencia por encima del suelo).
//   3. Línea de órdenes: casos válidos e inválidos.
//   4. Saneado de parámetros: rake limitado, la carrocería nunca atraviesa el suelo.
//   5. Corrección manométrica de fuerzas: fluido en reposo → fuerza nula por id.
//   6. Extremo a extremo sin ventana (red pequeña): F1 2022 y ala F1 en efecto
//      suelo con fuerzas finitas, resistencia > 0, simetría; el ala con carga > 0.
//      (La calibración física del coche es de tests/test_aero.cpp, ingeniero B.)
//   7. Barrido corto (3 alturas del ala en efecto suelo): puntos, restauración y CSV.
//   8. Los 18 modelos dentro de la App: cambio de modelo, todas las superficies/cortes/vistas.
//   9. Aplicación completa sin ventana (App::frame con panel, HUD y todas las
//      visualizaciones): CERO asignaciones de memoria por cuadro en régimen
//      estacionario (malloc interceptado), captura PNG no vacía, atajos de teclado.
//  Compilación: `make test` (enlaza todos los objetos salvo src/app/main.cpp).
// ============================================================================
#include "../src/app/app.hpp"
#include "../src/core/threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <dlfcn.h>
#include <execinfo.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------------------------
//  Contador de asignaciones: se interceptan malloc/calloc/realloc/aligned_alloc/posix_memalign
//  (y por tanto operator new, std::vector, Buffer<T>...) en TODO el proceso, hilos del pool
//  incluidos. Sólo cuenta mientras g_count_allocs está activo.
// ---------------------------------------------------------------------------------------------
extern "C" {
void* __libc_malloc(size_t);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void* __libc_memalign(size_t, size_t);
}
static std::atomic<bool> g_count_allocs{false};
static std::atomic<long> g_allocs{0};
// CFD_ALLOC_TRACE=1 → guarda la pila de las primeras asignaciones contadas y la imprime
// (direcciones relativas al ejecutable: `addr2line -f -C -e build/tests/test_app 0x...`).
static void* g_trace[8][16];
static int g_trace_n[8];
static bool g_trace_on = false;
static inline void count_alloc() {
    if (!g_count_allocs.load(std::memory_order_relaxed)) return;
    const long k = g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (g_trace_on && k < 8) g_trace_n[k] = backtrace(g_trace[k], 16);
}
static void print_alloc_traces() {
    if (!g_trace_on) return;
    Dl_info di{};
    dladdr(reinterpret_cast<void*>(&print_alloc_traces), &di);
    const long n = std::min(g_allocs.load(), 8L);
    for (long k = 0; k < n; ++k) {
        std::printf("    asignación %ld:", k);
        for (int i = 0; i < g_trace_n[k]; ++i)
            std::printf(" %#lx", static_cast<unsigned long>(reinterpret_cast<uintptr_t>(g_trace[k][i]) - reinterpret_cast<uintptr_t>(di.dli_fbase)));
        std::printf("\n");
    }
}
extern "C" {
void* malloc(size_t n) { count_alloc(); return __libc_malloc(n); }
void* calloc(size_t a, size_t b) { count_alloc(); return __libc_calloc(a, b); }
void* realloc(void* p, size_t n) { count_alloc(); return __libc_realloc(p, n); }
void* aligned_alloc(size_t al, size_t n) { count_alloc(); return __libc_memalign(al, n); }
void* memalign(size_t al, size_t n) { count_alloc(); return __libc_memalign(al, n); }
int posix_memalign(void** out, size_t al, size_t n) {
    count_alloc();
    void* p = __libc_memalign(al, n);
    if (!p) return 12;   // ENOMEM
    *out = p;
    return 0;
}
}

using namespace cfd;
using namespace cfd::app;

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

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// ---------------------------------------------------------------------------------------------
static void test_domain() {
    std::printf("[1] dimensionado del dominio (18 modelos × 4 presets × suelo sí/no)\n");
    double worst_ratio_lo = 10, worst_ratio_hi = 0;
    for (int m = 0; m < models::count(); ++m) {
        const models::Info& I = models::info(m);
        for (int gi = 0; gi < 2; ++gi) {
            const bool ground = gi == 0 ? I.needs_ground : !I.needs_ground;
            for (int p = 0; p < k_npresets; ++p) {
                const usize budget = preset_cells(static_cast<Preset>(p));
                const models::Params prm = sanitize_params(m, models::Params{});
                const DomainPlan d = plan_domain(m, prm, ground, budget);
                const char* id = I.id.c_str();
                CHECK(d.nx % 8 == 0 && d.nx >= 16, "%s: nx=%d no es múltiplo de 8 ≥ 16", id, d.nx);
                CHECK(d.ny >= 8 && d.nz >= 8, "%s: ny=%d nz=%d demasiado pequeños", id, d.ny, d.nz);
                const double ratio = static_cast<double>(d.cells()) / static_cast<double>(budget);
                worst_ratio_lo = std::min(worst_ratio_lo, ratio);
                worst_ratio_hi = std::max(worst_ratio_hi, ratio);
                CHECK(ratio > 0.75 && ratio < 1.3, "%s preset %d: %zu celdas para un presupuesto de %zu (×%.2f)", id, p, d.cells(), budget, ratio);
                CHECK(d.dx > 0 && std::isfinite(d.dx), "%s: dx inválido", id);
                // La caja del dominio es exactamente n·dx y el mapa pone el centro de la celda 0 en lo + dx/2.
                CHECK(near(d.box_m.size().x, static_cast<float>(d.nx) * d.dx, 1e-3f * d.dx * d.nx), "%s: caja x ≠ nx·dx", id);
                const Vec3 c0 = d.map.to_model(Vec3(0, 0, 0));
                CHECK(near(c0.x, d.box_m.lo.x + 0.5f * d.dx, 1e-4f) && near(c0.y, d.box_m.lo.y + 0.5f * d.dx, 1e-4f), "%s: origen del mapa", id);
                if (ground) CHECK(near(d.map.origin.z, -0.5f * d.dx, 1e-5f), "%s: con suelo el origen z debe ser -dx/2 (%.6f vs %.6f)", id, d.map.origin.z, -0.5f * d.dx);
                // El objeto (envolvente) cabe con ≥ 2 celdas de margen hasta las caras (salvo el suelo).
                const Vec3 lo = d.map.to_cells(d.object_m.lo), hi = d.map.to_cells(d.object_m.hi);
                CHECK(lo.x >= 2 && hi.x <= d.nx - 3, "%s: el objeto toca la entrada/salida (%.1f..%.1f de %d)", id, lo.x, hi.x, d.nx);
                if (I.spans_domain) {
                    CHECK(near(static_cast<float>(d.ny) * d.dx, I.id == "airfoil_2d" ? 1.0f : d.object_m.size().y, 1e-4f),
                          "%s: spans_domain exige ny·dx = ancho del modelo (%.5f)", id, static_cast<float>(d.ny) * d.dx);
                } else {
                    CHECK(lo.y >= 2 && hi.y <= d.ny - 3, "%s: el objeto toca los laterales (%.1f..%.1f de %d)", id, lo.y, hi.y, d.ny);
                }
                CHECK(hi.z <= d.nz - 3, "%s: el objeto toca el techo (%.1f de %d)", id, hi.z, d.nz);
                if (!ground) CHECK(lo.z >= 2, "%s: sin suelo el objeto toca la cara inferior (%.1f)", id, lo.z);
                // Suelo: z_modelo = 0 ↔ z_red = 0.5
                if (ground) CHECK(near(d.map.to_cells(Vec3(0, 0, 0)).z, 0.5f, 1e-4f), "%s: z=0 debe caer en z_red=0.5", id);
                if (I.kind == models::Kind::F1Car) {
                    CHECK(d.blockage <= 0.105f, "%s: bloqueo %.3f > 10 %%", id, d.blockage);
                    if (p >= static_cast<int>(Preset::Alta)) CHECK(d.dx <= 0.035f, "%s preset %d: dx %.4f > 3.5 cm de diseño", id, p, d.dx);
                }
                if (I.id == "f1_wing_ge" && p >= static_cast<int>(Preset::Media) && ground)
                    CHECK(d.dx <= 0.012f, "f1_wing_ge preset %d: dx %.4f > 12 mm", p, d.dx);
            }
        }
    }
    std::printf("    celdas / presupuesto: %.2f … %.2f\n", worst_ratio_lo, worst_ratio_hi);
}

// ---------------------------------------------------------------------------------------------
static void test_balance() {
    std::printf("[2] balance aerodinámico (fuerzas sintéticas)\n");
    // Coche con el eje delantero en x = 0 y el trasero en x = 3.6 (celdas o metros: da igual).
    const Vec3 front{0, 0, 0}, rear{3.6f, 0, 0}, ref{1.8f, 0, 0};
    auto load_at = [&](Vec3 p, Vec3 F) {
        const Vec3 M = cross(p - ref, F);   // momento respecto a ref
        return balance_front_pct(F, M, ref, front, rear);
    };
    const Vec3 down{0, 0, -1000.0f};
    const float b_front = load_at(front, down), b_mid = load_at({1.8f, 0.3f, 0.5f}, down), b_rear = load_at(rear, down);
    CHECK(near(b_front, 100.0f, 1e-3f), "carga en el eje delantero → 100 %% (%.4f)", b_front);
    CHECK(near(b_mid, 50.0f, 1e-3f), "carga a mitad de batalla → 50 %% (%.4f)", b_mid);
    CHECK(near(b_rear, 0.0f, 1e-3f), "carga en el eje trasero → 0 %% (%.4f)", b_rear);
    const float b_q = load_at({0.9f, 0, 0.7f}, down);
    CHECK(near(b_q, 75.0f, 1e-3f), "carga a ¼ de batalla → 75 %% (%.4f)", b_q);
    // Dos cargas: 600 delante + 400 detrás → 60 %.
    {
        const Vec3 F1v{0, 0, -600}, F2v{0, 0, -400};
        const Vec3 M = cross(front - ref, F1v) + cross(rear - ref, F2v);
        const float b = balance_front_pct(F1v + F2v, M, ref, front, rear);
        CHECK(near(b, 60.0f, 1e-3f), "600 delante + 400 detrás → 60 %% (%.4f)", b);
    }
    // Guiñada de 10° alrededor del centro: la misma carga en el eje delantero sigue dando 100 %.
    {
        const float a = 10.0f * k_deg2rad;
        const Mat3 R = Mat3::rot_z(a);
        const Vec3 fr = ref + R * (front - ref), rr = ref + R * (rear - ref);
        const Vec3 M = cross(fr - ref, down);
        const float b = balance_front_pct(down, M, ref, fr, rr);
        CHECK(near(b, 100.0f, 1e-3f), "con guiñada, carga en el eje delantero → 100 %% (%.4f)", b);
        const Vec3 pm = ref + R * Vec3(1.8f - 1.8f, 0.2f, 0.4f);
        const float b2 = balance_front_pct(down, cross(pm - ref, down), ref, fr, rr);
        CHECK(near(b2, 50.0f, 1e-3f), "con guiñada, a mitad de batalla → 50 %% (%.4f)", b2);
    }
    // Resistencia aplicada por encima del suelo: descarga el eje delantero (cabeceo hacia atrás).
    {
        const Vec3 F = down + Vec3(300.0f, 0, 0);
        const float b = load_at({1.8f, 0, 0.5f}, F);
        CHECK(b < 50.0f && b > 40.0f, "la resistencia a 0.5 de altura desplaza el balance hacia atrás (%.3f)", b);
    }
    // Carga nula → NaN (el panel muestra "—").
    CHECK(std::isnan(balance_front_pct({0, 0, 0}, {0, 0, 0}, ref, front, rear)), "sin carga → NaN");
}

// ---------------------------------------------------------------------------------------------
static bool parse(std::initializer_list<const char*> args, Options& o) {
    const char* argv[32];
    int argc = 0;
    argv[argc++] = "cfd";
    for (const char* a : args) argv[argc++] = a;
    o = Options{};
    return parse_cli(argc, argv, o);
}

static void test_cli() {
    std::printf("[3] línea de órdenes\n");
    Options o;
    CHECK(parse({}, o) && o.model == "f1_2022" && o.preset == Preset::Media, "sin argumentos");
    CHECK(parse({"--model", "sphere", "--res", "alta", "--ground", "static", "--speed", "180", "--fp32", "--threads", "6"}, o) &&
              o.model == "sphere" && o.preset == Preset::Alta && o.ground == 1 && near(o.speed, 180, 1e-6f) && o.fp32 && o.threads == 6,
          "combinación básica");
    CHECK(parse({"--param", "ride_front=25", "--param", "drs=1", "--param", "aoa=-3.5"}, o) && o.params.size() == 3 && o.params[0].key == "ride_front" &&
              near(o.params[2].value, -3.5f, 1e-6f),
          "--param");
    CHECK(parse({"--headless", "--steps", "1000", "--shot", "a.png", "--csv", "b.csv", "--view", "34", "--vis", "cp,streamlines,slice-y"}, o) &&
              o.headless && o.steps == 1000 && o.shot == "a.png" && o.csv == "b.csv" && o.view == static_cast<int>(CamView::TresCuartos) &&
              o.vis == "cp,streamlines,slice-y",
          "sin ventana con captura");
    CHECK(parse({"--sweep", "height:25:200:4", "--settle", "1", "--avg", "0.5"}, o) && o.sweep && o.sweep_param == SweepParam::Height &&
              near(o.sweep_from, 25, 1e-6f) && near(o.sweep_to, 200, 1e-6f) && o.sweep_n == 4 && near(o.settle_ft, 1.0f, 1e-6f),
          "--sweep");
    CHECK(parse({"--size", "1280x800", "--scale", "1", "--bench"}, o) && o.fb_w == 1280 && o.fb_h == 800 && o.scale == 1 && o.bench, "--size/--scale/--bench");
    CHECK(parse({"--res", "rápida"}, o) && o.preset == Preset::Rapida, "preset con tilde");
    CHECK(parse({"--cs", "0.1", "--prio", "max", "--pause", "--nu", "2e-5"}, o) && near(o.cs, 0.1f, 1e-6f) && o.prio == 2 && o.start_paused &&
              near(o.nu, 2e-5f, 1e-9f),
          "--cs/--prio/--pause/--nu");
    CHECK(parse({"--frames", "30", "--quit-after", "2.5", "--list", "--bench-pgo", "--csv", "x.csv"}, o) && o.frames == 30 &&
              near(o.quit_after, 2.5f, 1e-6f) && o.list && o.bench_pgo && o.csv == "x.csv",
          "--frames/--quit-after/--list/--bench-pgo");
    CHECK(parse({"--wall", "slip", "--bb", "implicit", "--ramp", "0"}, o) && o.wall == 2 && o.interp_bb == 0 && near(o.ramp_ft, 0.0f, 1e-9f),
          "--wall/--bb/--ramp");
    CHECK(!parse({"--wall", "rugosa"}, o) && !parse({"--ramp", "-1"}, o), "física inválida");
    CHECK(!parse({"--prio", "rapida"}, o), "prioridad desconocida");
    CHECK(!parse({"--cs", "2"}, o), "Cs fuera de rango");
    // Errores
    CHECK(!parse({"--res", "enorme"}, o) && !o.error.empty(), "preset desconocido");
    CHECK(!parse({"--ground", "hielo"}, o), "suelo desconocido");
    CHECK(!parse({"--param", "alas=3"}, o), "parámetro desconocido");
    CHECK(!parse({"--param", "aoa"}, o), "parámetro sin valor");
    CHECK(!parse({"--sweep", "aoa:0:10"}, o), "barrido incompleto");
    CHECK(!parse({"--sweep", "peso:0:10:4"}, o), "barrido de parámetro desconocido");
    CHECK(!parse({"--steps"}, o), "falta el valor");
    CHECK(!parse({"--speed", "-5"}, o), "velocidad negativa");
    CHECK(!parse({"--view", "cenital"}, o), "vista desconocida");
    CHECK(!parse({"--loquesea"}, o), "opción desconocida");
    CHECK(!parse({"--size", "10x10"}, o), "tamaño absurdo");
}

// ---------------------------------------------------------------------------------------------
static void test_sanitize() {
    std::printf("[4] saneado de parámetros (rake, holgura con el suelo)\n");
    for (int m = 0; m < models::count(); ++m) {
        const models::Info& I = models::info(m);
        if (!(I.param_mask & models::P_RideHeight)) continue;
        const ParamRanges R = param_ranges(m);
        models::Params p;
        p.ride_front_mm = 0.0f;
        p.ride_rear_mm = 400.0f;
        bool adj = false;
        const models::Params s = sanitize_params(m, p, &adj);
        const float rake = s.ride_rear_mm - s.ride_front_mm;
        CHECK(rake <= R.rake.hi + 1e-3f && rake >= R.rake.lo - 1e-3f, "%s: rake %.1f fuera de [%.0f, %.0f]", I.id.c_str(), rake, R.rake.lo, R.rake.hi);
        CHECK(adj, "%s: debe avisar de la corrección", I.id.c_str());
        // Ninguna pieza de la carrocería por debajo del suelo (muestreo del SDF en z = 1 mm).
        const models::Built b = models::build(m, s);
        bool under = false;
        for (usize g = 0; g < b.scene.groups().size() && !under; ++g) {
            const sdf::Group& G = b.scene.groups()[g];
            if (G.frame != sdf::Frame::Body || G.box_model.lo.z > 0.002f) continue;
            for (float x = G.box_model.lo.x; x <= G.box_model.hi.x && !under; x += 0.02f)
                for (float y = G.box_model.lo.y; y <= G.box_model.hi.y && !under; y += 0.02f)
                    under = b.scene.eval_group(static_cast<int>(g), Vec3(x, y, 0.001f)) < 0.0f;
        }
        CHECK(!under, "%s: la carrocería atraviesa el suelo con alturas %.0f/%.0f", I.id.c_str(), s.ride_front_mm, s.ride_rear_mm);
        // Valores por defecto: sin cambios.
        const models::Params d = models::resolve_params(m, models::Params{});
        bool adj2 = true;
        const models::Params s2 = sanitize_params(m, d, &adj2);
        CHECK(!adj2 && near(s2.ride_front_mm, d.ride_front_mm, 1e-4f), "%s: los valores por defecto no deben corregirse", I.id.c_str());
    }
}

// ---------------------------------------------------------------------------------------------
static void test_gauge() {
    // (Ingeniero B) La corrección manométrica vive ahora en el solver (lbm::Config::force_gauge); la
    // prueba exhaustiva (caja apoyada en el suelo, con y sin corrección) está en tests/test_aero.cpp.
    std::printf("[5] fuerzas manométricas: fluido en reposo sobre un coche apoyado en el suelo\n");
    Sim s;
    s.cfg.model = models::find("f1_2022");
    s.cfg.cells = 300'000;
    s.cfg.ground = lbm::GroundMode::Static;
    s.init();
    s.solver.set_inflow(0.0f);   // túnel parado
    s.solver.reset_flow();       // reposo exacto: ρ = 1, u = 0
    s.solver.step(10, true);
    Vec3 f[256], m[256];
    s.id_forces(s.solver.forces(), f, m);
    float worst = 0;
    for (int id = 1; id < 256; ++id) worst = max_(worst, length(f[id]) + length(m[id]));
    CHECK(worst < 1e-6f, "reposo → fuerza/momento residual %.2e (ruedas apoyadas, grupos que se tocan)", worst);
    std::printf("    residual máximo por id en reposo %.1e\n", worst);
}

// ---------------------------------------------------------------------------------------------
static void test_e2e() {
    std::printf("[6] extremo a extremo sin ventana (red pequeña)\n");
    struct Case { const char* id; usize cells; float ft; };
    const Case cases[] = {{"f1_2022", 1'200'000, 2.0f}, {"f1_wing_ge", 700'000, 3.0f}};
    for (const Case& c : cases) {
        Sim s;
        s.cfg.model = models::find(c.id);
        s.cfg.cells = c.cells;
        s.cfg.ground = lbm::GroundMode::Moving;
        s.init();
        const int n = static_cast<int>(c.ft * s.ft_steps());
        const double t0 = now_sec();
        bool ok = true;
        for (int done = 0; done < n; done += 50) ok &= s.step(50);
        const AeroResult& r = s.res;
        std::printf("    %-10s %dx%dx%d dx %.1f mm, %d pasos (%.1f s): CL %+.3f CD %.3f SCz %+.3f SCx %.3f bal %.1f\n", c.id, s.dom.nx, s.dom.ny,
                    s.dom.nz, static_cast<double>(s.dom.dx * 1e3f), n, now_sec() - t0, static_cast<double>(r.cl), static_cast<double>(r.cd),
                    static_cast<double>(r.scz), static_cast<double>(r.scx), static_cast<double>(r.balance));
        CHECK(ok && !s.solver.diverged(), "%s: sin divergencias", c.id);
        CHECK(r.valid && std::isfinite(r.cl) && std::isfinite(r.cd) && std::isfinite(r.cs), "%s: coeficientes finitos", c.id);
        CHECK(r.cd > 0.0f, "%s: resistencia > 0 (%.3f)", c.id, static_cast<double>(r.cd));
        // La carga del coche depende de la calibración física (ingeniero B, tests/test_aero.cpp): aquí
        // sólo se comprueba el ala en efecto suelo, cuyo signo es robusto a cualquier resolución.
        if (!s.is_car()) CHECK(r.cl > 0.0f, "%s: carga (hacia abajo) > 0 (%.3f)", c.id, static_cast<double>(r.cl));
        CHECK(std::fabs(r.cs) < 0.1f * std::fabs(r.cd) + 0.02f, "%s: fuerza lateral ~0 por simetría (%.4f)", c.id, static_cast<double>(r.cs));
        CHECK(std::isfinite(r.scz) && std::isfinite(r.scx) && std::isfinite(r.down_n) && std::isfinite(r.drag_n), "%s: fuerzas físicas finitas", c.id);
        // Balance: finito si hay carga apreciable (|CL| > 0.05), NaN si no (el panel muestra "—").
        if (s.is_car()) CHECK(std::fabs(r.cl) > 0.05f ? std::isfinite(r.balance) : std::isnan(r.balance), "%s: balance coherente con la carga (%.1f)", c.id, static_cast<double>(r.balance));
        const lbm::FieldView f = s.solver.field();
        bool finite = true;
        for (usize i = 0; i < s.dom.cells(); i += 97) finite &= std::isfinite(f.rho[i]) && std::isfinite(f.ux[i]);
        CHECK(finite, "%s: campo finito", c.id);
    }
}

// ---------------------------------------------------------------------------------------------
static void test_sweep() {
    std::printf("[7] barrido corto sin ventana (ala en efecto suelo, 3 alturas) + CSV\n");
    Sim s;
    s.cfg.model = models::find("f1_wing_ge");
    s.cfg.cells = 400'000;
    s.cfg.ground = lbm::GroundMode::Moving;
    s.init();
    for (int i = 0; i < 20; ++i) s.step(static_cast<int>(0.1f * s.ft_steps()) + 1);   // 2 PF de arranque
    Sweep sw;
    sw.param = SweepParam::Height;
    sw.from = 50; sw.to = 150; sw.n = 3;
    sw.settle_ft = 0.4f; sw.avg_ft = 0.3f;
    const models::Params base = s.params;   // (antes de arrancar: sweep_start aplica ya el primer punto)
    CHECK(sweep_start(sw, s), "el barrido de altura arranca");
    int guard = 0;
    while (sw.running && guard++ < 100000) {
        if (!s.step(25)) break;
        sweep_after_step(sw, s, 25);
    }
    CHECK(sw.done && sw.npts == 3, "3 puntos (%d)", sw.npts);
    bool finite = true;
    for (int i = 0; i < sw.npts; ++i) finite &= std::isfinite(sw.pts[i].cl) && std::isfinite(sw.pts[i].cd) && sw.pts[i].cd > 0.0f;
    CHECK(finite, "coeficientes finitos y CD > 0 en todos los puntos");
    CHECK(sw.npts == 3 && near(sw.pts[0].x, 50, 0.5f) && near(sw.pts[2].x, 150, 0.5f), "alturas de los puntos (%.1f … %.1f)", sw.pts[0].x, sw.pts[2].x);
    CHECK(near(s.params.height_mm, base.height_mm, 1e-3f), "al terminar se restauran los parámetros de partida");
    const std::string path = "build/tests/barrido_test.csv";
    CHECK(sweep_write_csv(sw, s, path), "CSV del barrido");
    FILE* fp = std::fopen(path.c_str(), "r");
    int lines = 0;
    if (fp) { char b[1024]; while (std::fgets(b, sizeof b, fp)) lines += b[0] != '#'; std::fclose(fp); }
    CHECK(lines == 1 + 3, "CSV con cabecera de columnas + 3 puntos (%d líneas sin comentario)", lines);
    for (int i = 0; i < sw.npts; ++i)
        std::printf("    h = %5.1f mm: CL %+.3f CD %.3f\n", static_cast<double>(sw.pts[i].x), static_cast<double>(sw.pts[i].cl), static_cast<double>(sw.pts[i].cd));
}

// ---------------------------------------------------------------------------------------------
static void test_all_models_ui() {
    std::printf("[8] los 18 modelos en la app: cambio de modelo, todas las superficies, cortes y vistas\n");
    Options o;
    o.model = "sphere";
    o.cells = 150'000;
    o.panel = "todo";
    App app;
    CHECK(app.setup(o, true), "setup");
    app.fixed_steps = true;
    int bad = 0;
    for (int m = 0; m < models::count(); ++m) {
        app.sim.cfg.cells = 150'000;
        app.load_model(m);
        app.sim.cfg.cells = 150'000;   // load_model conserva el presupuesto; se fuerza por si acaso
        const models::Info& I = models::info(m);
        CHECK(app.sim.ready && app.sim.dom.cells() > 0, "%s: simulación lista", I.id.c_str());
        CHECK(app.sim.mesh.tri_count() > 0, "%s: malla suave no vacía", I.id.c_str());
        for (int i = 0; i < 3; ++i) app.frame(10, true);
        // Todas las superficies, los 3 cortes, humo y vórtices, 6 vistas.
        for (int sm = 0; sm < static_cast<int>(SurfMode::Count); ++sm) {
            app.view.vs.surf = static_cast<SurfMode>(sm);
            app.view.vs.slice_axis = 1 + sm % 3;
            app.view.vs.smoke = sm == 2;
            app.view.vs.vortex = sm == 3;
            app.view.dirty = true;
            app.cam_view = static_cast<CamView>(sm % static_cast<int>(CamView::Count));
            app.view.frame(app.sim, app.cam_view, app.viewport_rect());
            app.frame(2, true);
        }
        const bool ok = std::isfinite(app.view.cam.distance) && app.view.cam.distance > 0 && std::isfinite(app.perf.last_frame_ms);
        bad += !ok;
        CHECK(ok, "%s: cámara y cuadro válidos", I.id.c_str());
        // El objeto encuadrado queda dentro del visor (centro proyectado dentro del viewport).
        float sx = 0, sy = 0, dz = 0;
        const Vec3 c = app.sim.object_cells().center();
        CHECK(app.view.cam.project(c, sx, sy, dz) && app.view.cam.vp.contains(static_cast<int>(sx), static_cast<int>(sy)),
              "%s: el objeto queda en el visor (%.0f, %.0f)", I.id.c_str(), sx, sy);
    }
    std::printf("    %d modelos recorridos (%d con problemas)\n", models::count(), bad);
}

// ---------------------------------------------------------------------------------------------
static void test_app_frames() {
    std::printf("[9] aplicación sin ventana: cuadros completos, cero asignaciones, captura y atajos\n");
    struct Case { const char* model; const char* vis; usize cells; };
    const Case cases[] = {
        {"f1_2022", "cp,streamlines,smoke,footprint,comp-arrows", 800'000},   // (con < ~0.5 M celdas, dx ≈ 9 cm: demasiado grueso)
        {"f1_wing_ge", "speed,slice-y,q=cp,smoke,vortex", 350'000},
        {"sphere", "voxels,slice-z,q=vort,lic,vortex,cloud", 350'000},
    };
    for (const Case& c : cases) {
        Options o;
        o.model = c.model;
        o.cells = c.cells;
        o.vis = c.vis;
        o.panel = "todo";
        App app;
        CHECK(app.setup(o, true), "%s: setup sin ventana", c.model);
        app.fixed_steps = true;
        app.steps_fixed = 8;
        app.sim.step(static_cast<int>(1.5f * app.sim.ft_steps()));   // flujo ya desarrollado (longitud de líneas estable)
        // Calentamiento: los búferes de la UI y de flowvis/raster crecen (×1.5, amortizado) hasta su
        // tamaño de régimen: el humo tarda en llenarse (vivas = emisión × vida) y las líneas se
        // alargan mientras el flujo se desarrolla. Régimen = el humo dejó de crecer Y 25 cuadros
        // seguidos sin ninguna asignación (máx. 400 cuadros). (Ruido del campo lejano corregido: con el flujo limpio el
        // humo del F1 2022 aún crecía despacio tras 25 cuadros tranquilos y un búfer de draw_points crecía una vez más en
        // la ventana medida; se exigen 60 cuadros seguidos sin asignaciones.)
        app.view.vs.smoke_speed = 8.0f;
        app.view.dirty = true;
        usize alive_prev = 0;
        int stable = 0, quiet = 0;
        int nwarm = 0;
        long warm_allocs = 0;
        const double tw = now_sec();
        for (; nwarm < 400 && (stable < 25 || quiet < 60); ++nwarm) {
            g_allocs.store(0);
            g_count_allocs.store(true);
            app.frame(8, true);
            g_count_allocs.store(false);
            const long a = g_allocs.load();
            warm_allocs += a;
            quiet = a == 0 ? quiet + 1 : 0;
            const usize alive = app.view.vs.smoke ? app.view.smoke.stats().alive : 0;
            stable = (nwarm > 30 && alive <= alive_prev + alive_prev / 200) ? stable + 1 : 0;
            alive_prev = max_(alive_prev, alive);
        }
        std::printf("    %-10s calentamiento: %d cuadros (%.1f s, %ld asignaciones de crecimiento), %zu partículas de humo\n", c.model, nwarm,
                    now_sec() - tw, warm_allocs, alive_prev);
        g_allocs.store(0);
        g_count_allocs.store(true);
        for (int i = 0; i < 30; ++i) app.frame(8, true);
        g_count_allocs.store(false);
        const long na = g_allocs.load();
        CHECK(na == 0, "%s: %ld asignaciones en 30 cuadros en régimen estacionario (deben ser 0)", c.model, na);
        if (na) print_alloc_traces();
        CHECK(app.sim.res.valid && std::isfinite(app.sim.res.cd), "%s: resultados válidos tras los cuadros", c.model);
        CHECK(app.perf.last_frame_ms > 0 && app.perf.steps_per_frame == 8, "%s: estadísticas por cuadro", c.model);
        if (app.view.vs.vortex) {
            // La cuantización del volumen Q no debe saturar por debajo del umbral (si no, isosuperficies en
            // bloques e idénticas para cualquier umbral mayor): full ≥ 4·umbral también con umbrales altos.
            const float q0 = app.view.vs.q_threshold;
            for (float q : {q0, 1000.0f}) {
                app.view.vs.q_threshold = q;
                app.view.dirty = true;
                app.frame(8, true);
                CHECK(app.view.vortex.params.full >= 4.0f * app.view.vortex.params.threshold * 0.999f && app.view.vortex.params.threshold > 0,
                      "%s: volumen Q sin saturar (umbral Q·L²/U² %.0f → %.4g, full %.4g)", c.model, static_cast<double>(q),
                      static_cast<double>(app.view.vortex.params.threshold), static_cast<double>(app.view.vortex.params.full));
            }
            app.view.vs.q_threshold = q0;
            app.view.dirty = true;
        }
        // Captura: la imagen debe tener contenido (no un color uniforme) en visor y panel.
        u64 h_vp = 0, h_pn = 0;
        u32 first_vp = app.fb.color[static_cast<usize>(app.fb.stride) * 600 + 300];
        bool var_vp = false, var_pn = false;
        const render::Rect pr = app.panel_rect();
        for (int y = 0; y < app.fb.h; y += 7)
            for (int x = 0; x < app.fb.w; x += 5) {
                const u32 px = app.fb.color[static_cast<usize>(app.fb.stride) * static_cast<usize>(y) + static_cast<usize>(x)];
                if (x < pr.x) { h_vp += px & 0xFF; var_vp |= px != first_vp; }
                else { h_pn += px & 0xFF; var_pn |= px != app.fb.color[static_cast<usize>(app.fb.stride) * static_cast<usize>(y) + static_cast<usize>(pr.x + 4)]; }
            }
        CHECK(var_vp && var_pn && h_vp > 0 && h_pn > 0, "%s: el framebuffer tiene contenido en visor y panel", c.model);
        // Atajos de teclado simulados (la ventana sin cabeza no produce eventos: se inyectan en Input).
        auto press = [&](int key) {
            app.in.begin_frame();
            app.in.key_pressed[key] = true;
            app.handle_input(app.viewport_rect());
        };
        const bool lines0 = app.view.vs.lines;
        press('l');
        CHECK(app.view.vs.lines != lines0 && app.view.dirty, "%s: L alterna las líneas de corriente", c.model);
        press('x');
        CHECK(app.view.vs.slice_axis == 1, "%s: X activa el corte X", c.model);
        const float p0 = app.view.vs.slice_pos[0];
        press(platform::KeyRight);
        CHECK(app.view.vs.slice_pos[0] > p0, "%s: la flecha mueve el corte", c.model);
        press(' ');
        CHECK(app.paused, "%s: Espacio pausa", c.model);
        const u64 st = app.sim.solver.steps();
        app.frame(8, true);
        CHECK(app.sim.solver.steps() == st, "%s: en pausa no se avanza", c.model);
        press('c');
        press('h');
        CHECK(app.hide_ui && app.panel_rect().w == 0, "%s: H oculta el panel", c.model);
        app.frame(8, true);
        press('h');
        press(platform::KeyF1);
        CHECK(app.show_help, "%s: F1 muestra la ayuda", c.model);
        app.frame(8, true);
        press(platform::KeyEscape);
        CHECK(!app.show_help && !app.quit, "%s: Esc cierra la ayuda sin salir", c.model);
        press('r');
        CHECK(app.sim.solver.steps() == 0, "%s: R reinicia el flujo", c.model);
    }
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (std::getenv("CFD_ALLOC_TRACE")) {
        void* warm[4];
        backtrace(warm, 4);   // la primera llamada carga libgcc (asigna): fuera de la medida
        g_trace_on = true;
    }
    pool().start();
    const double t0 = now_sec();
    test_domain();
    test_balance();
    test_cli();
    test_sanitize();
    test_gauge();
    test_e2e();
    test_sweep();
    test_all_models_ui();
    test_app_frames();
    std::printf("test_app: %d PASS, %d FALLO (%.1f s)\n", g_pass, g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
