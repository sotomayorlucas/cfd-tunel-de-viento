// ============================================================================
//  app/app.hpp — aplicación "túnel de viento": integra todos los módulos.
//
//  Archivos:
//    sim.cpp    dimensionado del dominio, saneado de parámetros, reconstrucción
//               (modelo → vóxeles → solver → malla), fuerzas y coeficientes,
//               balance aerodinámico, filtrado temporal y barridos (polares).
//    view.cpp   escena 3D: cámara, visualización del flujo (flowvis) y dibujo.
//    panel.cpp  panel lateral (ui::Context), HUD, barra de estado, ayuda.
//    cli.cpp    línea de órdenes, modos sin ventana, benchmarks y barridos.
//    app.cpp    estado de la aplicación y bucle principal.
//    main.cpp   punto de entrada.
//
//  Unidades: "modelo" = metros (suelo en z = 0, aire hacia +X); "red"/"celdas"
//  = unidades del LBM (ver geom/lattice_map.hpp). Fuerzas del solver en unidades
//  de red por paso; coeficientes = F / (½ ρ u∞² A) con ρ = 1, A en celdas².
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "../core/mem.hpp"
#include "../core/util.hpp"
#include "../geom/lattice_map.hpp"
#include "../gpu/lbm_gpu.hpp"
#include "../lbm/solver.hpp"
#include "../models/model.hpp"
#include "../platform/platform.hpp"
#include "../render/camera.hpp"
#include "../render/draw2d.hpp"
#include "../render/flowvis.hpp"
#include "../render/mesh.hpp"
#include "../ui/ui.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cfd::app {

inline constexpr float k_nanf = std::numeric_limits<float>::quiet_NaN();
inline constexpr float k_rho_air = 1.225f;      // kg/m³ (aire a nivel del mar, 15 °C)
inline constexpr float k_nu_air = 1.5e-5f;      // m²/s  (viscosidad cinemática del aire)
inline constexpr float k_g = 9.81f;

// ============================================================================
//  Resolución: presupuesto de celdas
// ============================================================================
enum class Preset : u8 { Rapida, Media, Alta, Ultra, Count };
inline constexpr int k_npresets = 4;
const char* preset_name(Preset p);            // "Rápida", ...
const char* preset_key(Preset p);             // "rapida", ...
usize preset_cells(Preset p);                 // 2.5 M, 6 M, 13 M, 28 M
bool parse_preset(const char* s, Preset& out);

// ============================================================================
//  Dominio (túnel) alrededor del objeto
// ============================================================================
struct DomainPlan {
    int nx = 0, ny = 0, nz = 0;
    float dx = 0.0f;               // metros por celda
    LatticeMap map;                // modelo (m) ↔ red (celdas)
    Aabb box_m;                    // caja del dominio en metros (caras exteriores de las celdas)
    Aabb object_m;                 // envolvente del objeto usada para dimensionar (m)
    bool ground = false;           // disposición con suelo: z_modelo = 0 ↔ z_red = 0.5
    float blockage = 0.0f;         // área frontal / sección del túnel
    float design_dx = 0.0f;        // dx máximo para el que está diseñado el modelo (m, 0 = libre)
    usize cells() const { return static_cast<usize>(nx) * static_cast<usize>(ny) * static_cast<usize>(nz); }
};
// Dimensiona el túnel para `budget` celdas. Márgenes por tipo (coche / ala / cuerpo), bloqueo
// frontal ≤ ~10 %, envolvente sobre los rangos de los deslizadores (guiñada, altura, AoA...) para
// que los cambios de parámetros no saquen el objeto del dominio. nx múltiplo de 8 (≥ 16).
// spans_domain: ny·dx = extensión Y exacta de bounds_m.
DomainPlan plan_domain(int model, const models::Params& p, bool ground_layout, usize budget);

// ============================================================================
//  Parámetros: rangos de la UI y saneado
// ============================================================================
struct Range { float lo = 0, hi = 0; };
struct ParamRanges {
    Range ride_front, ride_rear;   // mm
    Range rake;                    // mm (trasera - delantera)
    Range yaw, aoa, flap;          // grados
    Range height, gap;             // mm
};
ParamRanges param_ranges(int model);
// resolve_params + rangos de la UI + límite de rake + holgura con el suelo (la carrocería no puede
// atravesar z = 0: se sube el coche manteniendo el rake). `adjusted` ← true si se corrigió algo.
models::Params sanitize_params(int model, const models::Params& in, bool* adjusted = nullptr);

// ============================================================================
//  Fuerzas y coeficientes
// ============================================================================
// % de la carga vertical (−Fz) sobre el eje delantero a partir de la fuerza total F y del momento
// M respecto al punto `ref` (mismas unidades de longitud para todos los puntos). Equilibrio de
// momentos respecto al eje de cabeceo que pasa por el contacto trasero. NaN si la carga es ~0.
float balance_front_pct(Vec3 F, Vec3 M_ref, Vec3 ref, Vec3 front_contact, Vec3 rear_contact);

struct CompResult {
    sdf::Component comp = sdf::Component::Body;
    const char* name = "";
    float scz = 0, scx = 0;        // m² (carga +, resistencia +)
    Vec3 cop{0, 0, 0};             // punto de aplicación aproximado (celdas)
    Vec3 force{0, 0, 0};           // fuerza filtrada (red)
};
struct AeroResult {
    bool valid = false;
    float cl = 0, cd = 0, cs = 0;  // coeficientes: cl = carga (−Fz) +, cd = +Fx, cs = +Fy
    float scz = 0, scx = 0;        // cl·A, cd·A (m²)
    float ld = 0;                  // cl / cd
    float balance = k_nanf;        // % delantero (coches)
    float down_n = 0, drag_n = 0, side_n = 0, down_kgf = 0, drag_kgf = 0, power_kw = 0;
    Vec3 force{0, 0, 0}, moment{0, 0, 0};   // totales filtrados (red)
    Vec3 cop{0, 0, 0};             // centro de presiones aproximado (celdas)
    int ncomp = 0;
    CompResult comp[static_cast<int>(sdf::Component::Count)];
};

// Media exponencial por id (fuerza y momento) con constante de tiempo en pasos.
struct ForceFilter {
    Vec3 f[256], m[256];
    bool init = false;
    void reset() { init = false; }
    void push(const Vec3* fs, const Vec3* ms, int k, float tau_steps);
};

// Historia muestreada de los coeficientes filtrados (para gráficas y convergencia).
struct History {
    static constexpr int N = 600;
    float scz[N] = {}, scx[N] = {}, cl[N] = {}, cd[N] = {};
    u64 step[N] = {};
    int head = 0, count = 0;       // head = siguiente posición a escribir
    u64 interval = 50, next = 0;
    void reset(u64 every) { head = count = 0; interval = every > 0 ? every : 1; next = 0; }
    void push(u64 st, float cl_, float cd_, float scz_, float scx_);
    int oldest() const { return count < N ? 0 : head; }   // offset del ring buffer para plot_lines
    // Índice de la muestra más reciente con step ≤ s (−1 si no hay).
    int find_at_or_before(u64 s) const;
};

// ============================================================================
//  Simulación
// ============================================================================
struct SimConfig {
    int model = 0;
    models::Params params;         // lo que pide el usuario (se sanea al construir)
    lbm::GroundMode ground = lbm::GroundMode::Moving;
    Preset preset = Preset::Media;
    usize cells = 0;               // presupuesto explícito (0 = el del preset)
    bool fp32 = false;
    float u_lat = 0.09f;           // u∞ en red (Mach ≈ 0.16)
    float nu = 0.0f;               // viscosidad de red (0 = automática)
    float cs = 0.10f;              // Smagorinsky (0.10: valor clásico de flujos de cizalla; con 0.16 la capa límite
                                   // engorda y se despega antes: NACA 0012 6° CL 0.24 → 0.30, ver docs/FISICA.md)
    // ---- Física (fase 2; ver docs/FISICA.md) ----
    lbm::WallModel wall = lbm::WallModel::Slip;   // modelo de pared (ver lbm/solver.hpp y docs/FISICA.md)
    bool interp_bb = true;         // rebote interpolado (Bouzidi) con la superficie real: sin escalones
    // Arranque: 0 = impulsivo (u = u∞ en todo el fluido: el flujo global del túnel ya es el correcto). Con una
    // rampa desde el reposo (> 0, en pasos de flujo) el túnel tarda > 5 pasos de flujo en vaciar la
    // sobrepresión del arranque (medido en el túnel VACÍO: Cp +0.5 y u = 0.76 u∞ a la salida tras 2.3 PF).
    float ramp_ft = 0.0f;
    // ---- Fase 3: backend iGPU (Vulkan de cómputo propio, src/gpu; ver docs/GPU.md) ----
    bool gpu = false;              // el solver avanza en la iGPU; la CPU sólo dibuja (en paralelo en la ventana)
};

struct RebuildTimes { double build = 0, vox = 0, geo = 0, mesh = 0, total = 0; };

class Sim {
public:
    SimConfig cfg;
    models::Built built;
    models::Params params;         // efectivos (saneados): lo que pide el usuario tras el saneado
    // ---- Altura de marcha efectiva (resolución de la red; ver docs/FISICA.md) ----
    // Un hueco suelo-fondo de menos de ~3.5 celdas apenas deja pasar flujo en el LBM (con dx = 36 mm, 30 mm de
    // altura es < 1 celda: el fondo queda "sellado" y da SUSTENTACIÓN). El solver ve el coche subido Δ en
    // los dos ejes (rake conservado): la menor de las dos alturas h pasa a (h⁴ + g⁴)^{1/4}, g = k_gap_cells·dx
    // (= g con h → 0; ≈ h si h ≳ 1.5 g). Monótona y suave: los barridos de altura conservan la tendencia
    // (comprimida por debajo de ~g).
    static constexpr float k_gap_cells = 3.5f;   // 2.5 dejaba el fondo en sustentación (F1 2019 a Media: SCz fondo −0.40 → −0.14 con 3.5)
    models::Params params_eff;     // lo que se construye y ve el solver (= params salvo las alturas)
    float ride_eff_front_mm = 0.0f, ride_eff_rear_mm = 0.0f;   // alturas efectivas (mm)
    float ride_gap_min_mm = 0.0f;  // g = k_gap_cells·dx (mm)
    bool ride_limited = false;     // true si la resolución obliga a subir el coche (Δ > 0.5 mm)
    DomainPlan dom;
    lbm::Solver solver;
    Buffer<u8> solid;
    render::Mesh mesh;             // malla suave (render)
    render::Mesh vox_mesh;         // lo que ve el solver (vista de vóxeles)
    bool vox_dirty = true;
    float mesh_fraction = 0.0f;    // paso de muestreo de la malla actual (× dx)
    float nu = 1e-4f;              // viscosidad de trabajo (puede subir tras una divergencia)
    float wall_nu = 0.0f;          // ν (red) de la ley de pared = aire real a speed_kmh (Reynolds físico)
    float wall_nu_lat() const;     // ν_aire·u_red/(U·dx): la que corresponde a speed_kmh y dx
    // Presión estática de referencia "aguas arriba" (como la toma de un túnel real): ρ medio del plano x a mitad
    // de camino entre la entrada y el objeto (fluido, sin fronteras). Se actualiza en step(). Cp referido a ella:
    // Cp = 2(ρ − rho_ref)/(3u²). Las FUERZAS no dependen de ella (cuerpos cerrados; ver lbm::Config::force_gauge).
    float rho_ref = 1.0f;
    bool ready = false;
    u64 geom_version = 0, domain_version = 0, field_version = 0;
    RebuildTimes times;
    double t_init_ms = 0;
    usize solid_cells = 0;

    // Fuerzas
    ForceFilter filt;
    History hist;
    AeroResult res;
    float speed_kmh = 250.0f;      // reescala N/kgf y fija el Reynolds REAL de la ley de pared (wall_nu); el flujo
                                   // resuelto (u_red, ν_red, LES) no cambia
    bool converged = false;
    float conv_rel = 1.0f;         // cambio relativo máximo (CL, CD) en el último paso de flujo
    int divergences = 0;
    u64 steps_since_geom = 0;      // pasos desde la última reconstrucción o reinicio

    // Reinicio completo (modelo, preset, precisión, disposición del suelo).
    void init();
    // Reconstrucción geométrica con el flujo en marcha. mesh_fraction ≤ 0 → no rehacer la malla.
    // Devuelve true si la geometría cambió.
    bool set_params(const models::Params& p, float mesh_fraction, bool* adjusted = nullptr);
    void remesh(float fraction);
    void ensure_vox_mesh();
    // Suelo: Fijo ↔ Cinta en caliente; Ninguno ↔ con suelo cambia el dominio → init().
    void set_ground(lbm::GroundMode g);
    void reset_flow();
    // Avanza k pasos. false si el flujo divergió (entonces ya se ha recuperado: ν mayor + reinicio).
    bool step(int k);
    void update_results();

    // Fuerzas y momentos por id (red) de una muestra del solver. El solver ya las da MANOMÉTRICAS
    // (p − p∞: lbm::Config::force_gauge) y con el intercambio de momento galileano en paredes móviles
    // (force_galilean, Wen et al. 2014): un id que no es superficie cerrada para el fluido (ruedas
    // apoyadas, grupos que se tocan) no recibe la fuerza espuria p∞·A_contacto (tests/test_aero.cpp).
    void id_forces(const lbm::ForceSample& s, Vec3* f, Vec3* m) const;
    Vec3 mref_cells{0, 0, 0};                  // referencia de momentos del solver (celdas)

    // Utilidades
    float obj_len_m() const;               // longitud para el "paso de flujo"
    float ft_steps() const;                // pasos por paso de flujo (L/U∞)
    float flow_throughs() const;           // pasos de flujo transcurridos desde el reinicio
    float re_lattice() const;              // U·L/ν de la red (sin LES)
    float re_real() const;                 // U·L/ν del aire real a speed_kmh
    Aabb object_cells() const;             // caja actual del objeto (celdas)
    Vec3 to_cells(Vec3 m) const { return dom.map.to_cells(m); }
    float a_ref_cells() const { return built.info.ref_area_m2 / (dom.dx * dom.dx); }
    usize memory_bytes() const;
    bool is_car() const { return built.info.wheelbase_m > 0.0f; }

    // ---- Fase 3: backend iGPU ----
    // Con cfg.gpu el solver de la CPU sigue siendo el dueño del estado; gpu::LbmGpu avanza los pasos. Con
    // gpu_async (bucle interactivo), step(k) publica el lote ANTERIOR y encola k pasos que la GPU hace mientras
    // la CPU dibuja: los resultados (campos, fuerzas) llevan un lote de retraso. Si no, step(k) es síncrono.
    std::unique_ptr<gpu::LbmGpu> gpu;
    bool gpu_async = false;
    std::string gpu_error;         // motivo si no se pudo activar
    int published = 0;             // pasos publicados en el último step() (≠ k con gpu_async)
    bool gpu_on() const { return gpu && gpu->attached(); }
    bool set_gpu(bool on);         // activa/desactiva el backend (false si no hay iGPU utilizable)
    void finish();                 // publica el lote en vuelo (gpu_async)
    double gpu_step_seconds() const;   // tiempo de GPU por paso del último lote (0 si no hay)

private:
    bool post_step(int k);         // tras k pasos publicados: divergencia, ρ de referencia, fuerzas
    bool motion_on_[256] = {};     // ids con movimiento de pared activo en el solver
    float wall_speed_ = -1.0f;     // speed_kmh con la que se fijó wall_nu en el solver
    void sync_wall_law();          // wall_nu ← wall_nu_lat() si cambió speed_kmh
    void update_rho_ref();         // rho_ref ← ρ medio del plano de referencia aguas arriba
    usize fill_contact_pockets();  // huella de contacto de las ruedas (ver sim.cpp)
    void update_effective_params();   // params → params_eff (altura de marcha efectiva)
    void set_moment_ref();
    void apply_wall_motions();
    void rebuild_geometry(float mesh_fraction);
    void reset_forces();
};

// Resultados a partir de fuerzas/momentos por id (red) → coeficientes y desglose por componente.
AeroResult compute_aero(const Sim& s, const Vec3* f, const Vec3* m);

// ============================================================================
//  Barrido (polares): parámetro × N puntos; asentar S pasos de flujo, promediar W.
// ============================================================================
enum class SweepParam : u8 { Aoa, Height, RideHeight, FrontFlap, RearFlap, Yaw, FlapGap, Count };
inline constexpr int k_nsweep = 7;
const char* sweep_param_name(SweepParam p);   // "Ángulo de ataque", ...
const char* sweep_param_key(SweepParam p);    // "aoa", "height", "ride", ...
const char* sweep_param_unit(SweepParam p);   // "°" / "mm"
bool sweep_param_available(int model, SweepParam p);
bool parse_sweep_param(const char* s, SweepParam& out);
float sweep_param_get(const models::Params& p, SweepParam sp);
void sweep_param_set(models::Params& p, SweepParam sp, float v);   // altura de marcha: conserva el rake

struct SweepPoint { float x = 0, cl = 0, cd = 0, cs = 0, scz = 0, scx = 0, ld = 0, bal = k_nanf; };
struct Sweep {
    static constexpr int k_max = 64;
    bool running = false, done = false;
    SweepParam param = SweepParam::Aoa;
    float from = 0, to = 10;
    int n = 6;
    float settle_ft = 1.5f, avg_ft = 1.0f;
    int idx = 0;
    bool averaging = false;
    u64 phase_steps = 0, phase_len = 0;
    double acc_f[256][3] = {}, acc_m[256][3] = {};
    u64 acc_steps = 0;
    models::Params base;
    SweepPoint pts[k_max];
    int npts = 0;
    int model = -1;
    float progress() const;               // 0..1
    float value_at(int i) const { return n > 1 ? from + (to - from) * static_cast<float>(i) / static_cast<float>(n - 1) : from; }
};
// Empieza (aplica el primer punto). false si el parámetro no está disponible.
bool sweep_start(Sweep& sw, Sim& sim);
// Tras cada sim.step(k): acumula / avanza de fase / aplica el siguiente punto. true al terminar.
bool sweep_after_step(Sweep& sw, Sim& sim, int k);
void sweep_stop(Sweep& sw, Sim& sim);   // cancela y restaura los parámetros de partida
bool sweep_write_csv(const Sweep& sw, const Sim& sim, const std::string& path);

// ============================================================================
//  Vista 3D
// ============================================================================
enum class SurfMode : u8 { Cp, Speed, Component, Plain, Voxels, Hidden, Count };
enum class RakeKind : u8 { Vertical, Floor, Tips, Grid, Count };
enum class CamView : u8 { Lateral, Superior, Frontal, Trasera, TresCuartos, Bajo, Count };
const char* surf_mode_name(SurfMode m);
const char* rake_name(RakeKind r);
const char* cam_view_name(CamView v);
const char* cam_view_key(CamView v);
bool parse_cam_view(const char* s, CamView& out);

// Magnitudes disponibles en el plano de corte (orden de la UI).
inline constexpr flowvis::Quantity k_slice_q[7] = {
    flowvis::Quantity::Speed, flowvis::Quantity::Ux, flowvis::Quantity::Uz, flowvis::Quantity::Cp,
    flowvis::Quantity::Cp0, flowvis::Quantity::Vorticity, flowvis::Quantity::QCriterion};

struct VisSettings {
    SurfMode surf = SurfMode::Cp;
    bool surf_auto = false;        // rango de Cp de la superficie automático
    float surf_lo = -2.0f, surf_hi = 1.0f;
    int slice_axis = 0;            // 0 = no, 1 = X, 2 = Y, 3 = Z
    float slice_pos[3] = {0, 0, 0};   // celdas
    int slice_q = 0;               // índice en k_slice_q
    bool slice_auto = true;
    float slice_lo = 0.0f, slice_hi = 1.6f;
    float slice_opacity = 0.85f;
    bool slice_lic = false;
    bool lines = true;
    RakeKind line_rake = RakeKind::Vertical;
    int line_count = 36;
    bool line_cp = false;          // color por Cp (si no, |u|)
    bool smoke = false;
    RakeKind smoke_rake = RakeKind::Vertical;
    float smoke_density = 0.5f;    // 0..1
    float smoke_speed = 3.0f;      // cámara rápida del humo (× tiempo real de la simulación)
    bool vortex = false;
    float q_threshold = 20.0f;     // umbral de Q en unidades del objeto: Q·L²/U∞² (independiente de la resolución)
    bool vortex_cloud = false;
    bool footprint = true;
    bool arrows = true;
    bool comp_arrows = false;
    bool wire = true;
    bool ssao = true;
    bool fxaa = true;
    bool probe = true;
    bool legends = true;
    bool operator==(const VisSettings&) const = default;   // (sin memcmp: el relleno no cuenta)
};

struct View {
    render::Camera cam;
    VisSettings vs;
    flowvis::FlowSampler sampler;
    flowvis::SliceView slice;
    flowvis::Streamlines lines;
    flowvis::Particles smoke;
    flowvis::VortexVolume vortex;
    flowvis::GroundFootprint footprint;
    u32 group_colors[256] = {};
    float belt_offset = 0.0f;
    u64 seen_field = ~0ull, seen_geom = ~0ull, seen_domain = ~0ull;
    bool dirty = true;             // ajustes de visualización cambiados → recalcular
    double field_interval = 0.0;   // s mínimos entre actualizaciones por campo nuevo (0 = cada cuadro)
    double last_field_t = -1e9;
    bool rakes_dirty = true;
    RakeKind lines_rake_built = RakeKind::Count, smoke_rake_built = RakeKind::Count;
    int lines_count_built = -1;
    SurfMode surf_colored = SurfMode::Count;
    // Sonda del ratón
    bool probe_ok = false;
    bool probe_on_slice = false;
    float probe_value = 0;
    flowvis::Probe probe;
    // Tiempos (ms) del último cuadro
    double t_update = 0, t_mesh = 0, t_vis = 0, t_post = 0, t_render = 0;
    double t_upd_sampler = 0, t_upd_lines = 0, t_upd_smoke = 0, t_upd_vortex = 0, t_upd_slice = 0, t_upd_color = 0;

    void defaults_for(const Sim& sim);                       // ajustes de visualización por tipo de modelo
    void frame(const Sim& sim, CamView v, render::Rect vp);  // cámara predefinida encuadrando el objeto
    // Recalcula lo que dependa del campo (tras avanzar `steps` pasos) o de los ajustes.
    void update(Sim& sim, int steps_advanced);
    void render(render::Framebuffer& fb, render::Rect vp, Sim& sim);
    void pick_probe(const render::Framebuffer& fb, Sim& sim, int mx, int my);
    // Punto 3D bajo el píxel usando la profundidad (false si es fondo).
    bool unproject(const render::Framebuffer& fb, int mx, int my, Vec3& out) const;
    void reset_particles() { smoke.reset(); }
};

// ============================================================================
//  Línea de órdenes
// ============================================================================
struct CliParam { std::string key; float value = 0; };
struct Options {
    bool list = false, help = false;
    std::string model = "f1_2022";
    bool model_set = false;
    Preset preset = Preset::Media;
    bool preset_set = false;
    usize cells = 0;               // --cells N (presupuesto explícito)
    int ground = -1;               // -1 defecto del modelo; 0 ninguno, 1 fijo, 2 cinta
    float speed = -1.0f;           // km/h (< 0 → defecto del modelo)
    std::vector<CliParam> params;
    bool fp32 = false;
    bool gpu = false;              // --gpu: solver en la iGPU
    int threads = 0;
    bool headless = false;
    long steps = -1;               // pasos de red (sin ventana: por defecto 2 pasos de flujo)
    float flow_throughs = -1.0f;   // --ft F (alternativa a --steps)
    long frames = -1;              // X11: salir tras N cuadros
    int spf = 0;                   // --spf N: pasos por cuadro fijos en el bucle interactivo (medidas; 0 = adaptativo)
    float quit_after = -1.0f;      // X11: salir tras S segundos
    int view = -1;                 // CamView
    bool cam_set = false;          // --cam x,y,z,guiñada,cabeceo,distancia (m y grados)
    float cam[6] = {0, 0, 0, 0, 0, 0};
    std::string vis;               // lista separada por comas
    std::string panel;             // secciones abiertas del panel
    std::string shot, csv;
    bool sweep = false;
    SweepParam sweep_param = SweepParam::Aoa;
    float sweep_from = 0, sweep_to = 0;
    int sweep_n = 0;
    float settle_ft = 1.5f, avg_ft = 1.0f;
    bool bench = false, bench_pgo = false, stability = false;
    int fb_w = 1920, fb_h = 1200;
    int scale = 0;
    float nu = 0.0f;
    float cs = -1.0f;              // --cs (Smagorinsky; < 0 = defecto de SimConfig)
    int prio = -1;                 // --prio fluidez|equilibrado|max (-1 = equilibrado)
    bool start_paused = false;     // --pause
    int wall = -1;                 // --wall none|log|slip (-1 = defecto de SimConfig)
    int interp_bb = -1;            // --bb interp|implicit (-1 = defecto)
    float ramp_ft = -1.0f;         // --ramp PF (< 0 = defecto)
    std::string error;             // mensaje si el análisis falla
};
// false si hay un error (Options::error lo explica). No termina el proceso.
bool parse_cli(int argc, const char* const* argv, Options& o);
void print_usage(const char* argv0);
int run_cli(const Options& o);     // punto de entrada tras analizar (main)

// ============================================================================
//  Aplicación
// ============================================================================
enum class Priority : u8 { Fluidez, Equilibrado, MaxSim };

struct PerfStats {
    Ema fps, frame_ms, sim_ms, vis_ms, render_ms, ui_ms, present_ms, mlups, step_s;
    int steps_per_frame = 1;
    double last_sim_ms = 0, last_vis_ms = 0, last_render_ms = 0, last_ui_ms = 0, last_present_ms = 0, last_frame_ms = 0;
    // Historia de tiempos por cuadro (anillo) para la gráfica del panel Rendimiento.
    static constexpr int kHist = 240;
    float h_frame[kHist] = {}, h_sim[kHist] = {};
    int h_head = 0, h_count = 0;
    void push_hist(float frame, float sim) {
        h_frame[h_head] = frame; h_sim[h_head] = sim;
        h_head = (h_head + 1) % kHist;
        if (h_count < kHist) ++h_count;
    }
    int h_oldest() const { return h_count < kHist ? 0 : h_head; }
};

// Secciones del panel lateral.
enum PanelSection : int { SecModelo = 0, SecConfig, SecTunel, SecVis, SecResultados, SecBarrido, SecRendimiento, SecCount };

struct App {
    Options opt;
    Sim sim;
    View view;
    ui::Context ui;
    std::unique_ptr<platform::Window> win;
    render::Framebuffer fb;
    platform::Input in;
    render::DrawList overlay;      // HUD, leyendas, barra de estado, ayuda
    Sweep sweep;
    PerfStats perf;

    // Estado de la UI (copias editables)
    models::Params ui_params;      // lo que muestran los deslizadores
    int ui_group = 0;              // 0 = coches F1, 1 = objetos
    int ui_model = 0;
    int ui_preset = 1;
    int ui_ground = 2;
    float rake_mm = 50.0f;
    Priority prio = Priority::Equilibrado;
    bool paused = false, hide_ui = false, show_help = false;
    bool mouse_seen = false;       // el ratón se ha movido sobre la ventana (sin cabeza: nunca → sin sonda)
    bool idle_throttle = false;    // (ventana) en pausa y sin entrada: redibujar a ~10 Hz
    double last_render_t = 0;
    u64 last_render_field = ~0ull, last_render_geom = ~0ull;
    bool sec_open[SecCount] = {true, true, true, true, true, false, false};
    bool fixed_steps = false;      // pasos por cuadro fijos (sin ventana / benchmark)
    int steps_fixed = 20;

    // Cambios pendientes (los fija el panel; se aplican en App::frame)
    bool want_model = false;       // ui_model → load_model
    bool want_ground = false;      // ui_ground → Sim::set_ground
    bool ui_fp32 = false;
    // Física del solver (se aplica con un reinicio completo: "Aplicar")
    lbm::WallModel ui_wall = lbm::WallModel::Slip;   // (se sincronizan con SimConfig en sync_ui_from_sim)
    bool ui_interp_bb = true;
    bool ui_gpu = false;           // conmutador "Solver: CPU / iGPU" del panel
    bool want_gpu = false;         // ui_gpu → Sim::set_gpu
    float ui_ramp_ft = 0.0f;
    bool want_res_apply = false;   // "Aplicar resolución": descarta un --cells explícito
    bool want_csv = false, want_sweep_start = false, want_sweep_csv = false;
    bool geom_final_pending = false;   // tras arrastrar: rehacer la malla fina al soltar
    CpuTopology topo;
    bool want_reinit = false;      // preset/precisión → init()
    bool want_geom = false;        // parámetros geométricos cambiados
    bool geom_dragging = false;
    double last_geom_t = 0;
    bool want_shot = false;
    double esc_t = -10.0;          // confirmación de salida
    bool quit = false;
    bool camera_framed = false;
    CamView cam_view = CamView::TresCuartos;
    long frame_no = 0;
    double t_start = 0;
    std::string last_shot;

    // Listas de nombres para los desplegables (persistentes)
    std::vector<int> models_f1, models_obj;
    std::vector<const char*> names_f1, names_obj;

    // Ciclo de vida
    bool setup(const Options& o, bool headless);   // crea la ventana, el solver y la vista
    void load_model(int model, bool keep_camera = false);   // cambia de modelo (reinicio completo)
    void reinit();                                  // aplica preset/precisión/suelo pendientes
    // Un cuadro completo. steps_override > 0 fija los pasos de red de este cuadro.
    void frame(int steps_override = 0, bool render_frame = true);
    int run_interactive();
    bool save_screenshot(const std::string& path);
    std::string auto_shot_path(const char* prefix, const char* ext) const;

    // Entrada / UI (app.cpp, panel.cpp)
    void handle_input(render::Rect vp);
    void build_panel(render::Rect pr);
    void draw_sweep_plot();
    void sweep_defaults();
    bool write_forces_csv(const std::string& path);
    void draw_overlay(render::Rect vp);
    void apply_ui_params(bool dragging);
    void sync_ui_from_sim();
    void update_title();           // título de la ventana con el modelo actual
    render::Rect panel_rect() const;
    render::Rect viewport_rect() const;
    int adaptive_steps() const;
};

// Utilidades compartidas
struct SiStr { char s[32]; };                    // texto en la pila (sin asignaciones por cuadro)
SiStr fmt_si(double v, int digits = 3);          // 1.23e6 → "1,23·10^6"
bool ensure_dir(const std::string& dir);

} // namespace cfd::app
