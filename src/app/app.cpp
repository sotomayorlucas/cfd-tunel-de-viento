// ============================================================================
//  app/app.cpp — estado de la aplicación y bucle principal.
//
//  Un cuadro: entrada → panel (UI inmediata) → cambios pendientes (reinicio,
//  reconstrucción con antirrebote) → k pasos de red (k adaptativo para el FPS
//  objetivo) → actualización de la visualización → render 3D → HUD → UI →
//  presentación. Sin asignaciones por cuadro en régimen estacionario (búferes
//  persistentes en todos los módulos; textos en char[] de pila).
// ============================================================================
#include "app.hpp"

#include "../core/png.hpp"
#include "../core/threadpool.hpp"
#include "../render/raster.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

namespace cfd::app {

using render::Rect;

bool ensure_dir(const std::string& dir) {
    if (dir.empty()) return true;
    std::string cur;
    for (usize i = 0; i <= dir.size(); ++i) {
        if (i == dir.size() || dir[i] == '/') {
            if (!cur.empty() && cur != ".") {
                if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            }
        }
        if (i < dir.size()) cur += dir[i];
    }
    return true;
}

// ============================================================================
//  Disposición
// ============================================================================
Rect App::panel_rect() const {
    if (hide_ui) return {fb.w, 0, 0, fb.h};
    const int pw = min_(static_cast<int>(440.0f * ui.style().scale), fb.w / 2);
    return {fb.w - pw, 0, pw, fb.h};
}
Rect App::viewport_rect() const {
    const Rect p = panel_rect();
    return {0, 0, p.x, fb.h};
}

// ============================================================================
//  Arranque
// ============================================================================
bool App::setup(const Options& o, bool headless) {
    opt = o;
    topo = detect_topology();
    pool().start(o.threads);
    const int idx = models::find(o.model);
    if (idx < 0) {
        std::fprintf(stderr, "[cfd] modelo desconocido '%s' (usa --list)\n", o.model.c_str());
        return false;
    }
    if (!headless) {
        win = platform::create_x11_window();
        if (!win) std::fprintf(stderr, "[cfd] no hay servidor X (DISPLAY): se usa el modo sin ventana\n");
    }
    if (!win) win = platform::create_headless_window();
    int scale = o.scale > 0 ? o.scale : (win->is_headless() ? 1 : platform::detect_pixel_scale());
    if (!win->open("Túnel de viento CFD — LBM D3Q19", o.fb_w, o.fb_h, scale)) {
        std::fprintf(stderr, "[cfd] no se pudo abrir la ventana\n");
        return false;
    }
    fb.resize(o.fb_w, o.fb_h);
    if (fb.h >= 2000 && scale == 1) ui.set_scale(2.0f);
    // Listas de modelos para el panel (el catálogo es estático e inmutable).
    for (int i = 0; i < models::count(); ++i) {
        const models::Info& I = models::info(i);
        if (I.kind == models::Kind::F1Car) { models_f1.push_back(i); names_f1.push_back(I.name.c_str()); }
        else { models_obj.push_back(i); names_obj.push_back(I.name.c_str()); }
    }
    // Simulación
    const models::Info& I = models::info(idx);
    sim.cfg.model = idx;
    sim.cfg.params = models::Params{};
    for (const CliParam& p : o.params) {
        models::Params& P = sim.cfg.params;
        if (p.key == "ride_front") P.ride_front_mm = p.value;
        else if (p.key == "ride_rear") P.ride_rear_mm = p.value;
        else if (p.key == "ride") { const float rk = P.ride_rear_mm >= 0 && P.ride_front_mm >= 0 ? P.ride_rear_mm - P.ride_front_mm : I.default_ride_rear_mm - I.default_ride_front_mm; P.ride_front_mm = p.value; P.ride_rear_mm = p.value + rk; }
        else if (p.key == "front_flap") P.front_flap_deg = p.value;
        else if (p.key == "rear_flap") P.rear_flap_deg = p.value;
        else if (p.key == "drs") P.drs_open = p.value != 0.0f;
        else if (p.key == "yaw") P.yaw_deg = p.value;
        else if (p.key == "aoa") P.aoa_deg = p.value;
        else if (p.key == "height") P.height_mm = p.value;
        else if (p.key == "gap") P.flap_gap_mm = p.value;
        else if (p.key == "wheels") P.wheels_rotating = p.value != 0.0f;
    }
    sim.cfg.ground = o.ground >= 0 ? static_cast<lbm::GroundMode>(o.ground) : (I.needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None);
    sim.cfg.preset = o.preset;
    sim.cfg.cells = o.cells;
    sim.cfg.fp32 = o.fp32;
    sim.cfg.gpu = o.gpu;
    // La iGPU sólo admite la red uniforme: con --gpu y sin --refine explícito, sin refinamiento.
    sim.cfg.refine = (o.gpu && o.refine < 0 && o.refine_boxes.empty()) ? 0 : o.refine;
    sim.cfg.manual_boxes = o.refine_boxes;
    sim.cfg.nu = o.nu;
    if (o.cs >= 0.0f) sim.cfg.cs = o.cs;
    if (o.wall >= 0) sim.cfg.wall = static_cast<lbm::WallModel>(o.wall);
    if (o.interp_bb >= 0) sim.cfg.interp_bb = o.interp_bb != 0;
    if (o.ramp_ft >= 0.0f) sim.cfg.ramp_ft = o.ramp_ft;
    if (o.prio >= 0) prio = static_cast<Priority>(o.prio);
    paused = o.start_paused;
    sim.speed_kmh = o.speed > 0 ? o.speed : I.default_speed_kmh;
    sim.init();
    view.defaults_for(sim);
    // --vis
    if (!o.vis.empty()) {
        VisSettings& v = view.vs;
        std::string tok;
        auto apply = [&](const std::string& t) {
            if (t.empty()) return;
            const usize eq = t.find('=');
            const std::string k = t.substr(0, eq), val = eq == std::string::npos ? "" : t.substr(eq + 1);
            const float fv = val.empty() ? 0.0f : std::strtof(val.c_str(), nullptr);
            if (k == "cp") v.surf = SurfMode::Cp;
            else if (k == "speed" || k == "velocidad") v.surf = SurfMode::Speed;
            else if (k == "component" || k == "componentes") v.surf = SurfMode::Component;
            else if (k == "plain" || k == "liso") v.surf = SurfMode::Plain;
            else if (k == "voxels" || k == "voxeles") v.surf = SurfMode::Voxels;
            else if (k == "hidden" || k == "oculta") v.surf = SurfMode::Hidden;
            else if (k == "streamlines" || k == "lineas") v.lines = true;
            else if (k == "nostreamlines" || k == "nolineas") v.lines = false;
            else if (k == "smoke" || k == "humo") v.smoke = true;
            else if (k == "vortex" || k == "vortices") v.vortex = true;
            else if (k == "novortex" || k == "novortices") v.vortex = false;
            else if (k == "nosmoke" || k == "nohumo") v.smoke = false;
            else if (k == "noprobe") v.probe = false;
            else if (k == "cloud" || k == "nube") { v.vortex = true; v.vortex_cloud = true; }
            else if (k == "footprint" || k == "huella") v.footprint = true;
            else if (k == "nofootprint" || k == "nohuella") v.footprint = false;
            else if (k == "arrows" || k == "flechas") v.arrows = true;
            else if (k == "boxes" || k == "cajas") v.level_boxes = true;
            else if (k == "noarrows" || k == "noflechas") v.arrows = false;
            else if (k == "comp-arrows") { v.arrows = true; v.comp_arrows = true; }
            else if (k == "nowire") v.wire = false;
            else if (k == "nossao") v.ssao = false;
            else if (k == "nofxaa") v.fxaa = false;
            else if (k == "nolegends") v.legends = false;
            else if (k == "lic") v.slice_lic = true;
            else if (k == "noslice") v.slice_axis = 0;
            else if (k == "slice-x" || k == "slice-y" || k == "slice-z") {
                const int ax = k[6] - 'x';
                v.slice_axis = ax + 1;
                if (!val.empty()) {
                    Vec3 pm = sim.dom.map.to_model(Vec3(v.slice_pos[0], v.slice_pos[1], v.slice_pos[2]));
                    pm[ax] = fv;
                    v.slice_pos[ax] = sim.dom.map.to_cells(pm)[ax];
                }
            } else if (k == "slice-q" || k == "q") {
                static const char* const qk[7] = {"speed", "ux", "uz", "cp", "cp0", "vort", "q"};
                for (int i = 0; i < 7; ++i) if (val == qk[i]) v.slice_q = i;
            } else if (k == "slice-range") {
                float a = 0, b = 0;
                if (std::sscanf(val.c_str(), "%f:%f", &a, &b) == 2) { v.slice_auto = false; v.slice_lo = a; v.slice_hi = b; }
            } else if (k == "opacity") v.slice_opacity = clamp_(fv, 0.05f, 1.0f);
            else if (k == "rake" || k == "smoke-rake") {
                RakeKind rk = RakeKind::Vertical;
                if (val == "floor" || val == "suelo") rk = RakeKind::Floor;
                else if (val == "tips" || val == "puntas") rk = RakeKind::Tips;
                else if (val == "grid" || val == "rejilla") rk = RakeKind::Grid;
                (k == "rake" ? v.line_rake : v.smoke_rake) = rk;
            } else if (k == "lines") { v.lines = true; v.line_count = clamp_(static_cast<int>(fv), 2, 400); }
            else if (k == "line-cp") v.line_cp = true;
            else if (k == "q-threshold") v.q_threshold = fv;
            else if (k == "cp-range") {
                float a = 0, b = 0;
                if (std::sscanf(val.c_str(), "%f:%f", &a, &b) == 2) { v.surf_lo = a; v.surf_hi = b; }
            } else std::fprintf(stderr, "[cfd] --vis: opción desconocida '%s' (se ignora)\n", k.c_str());
        };
        for (char ch : o.vis) { if (ch == ',') { apply(tok); tok.clear(); } else tok += ch; }
        apply(tok);
        view.dirty = true;
    }
    if (!o.panel.empty()) {
        for (bool& b : sec_open) b = false;
        static const char* const keys[SecCount] = {"modelo", "config", "tunel", "vis", "resultados", "barrido", "rendimiento"};
        for (int i = 0; i < SecCount; ++i) if (o.panel.find(keys[i]) != std::string::npos) sec_open[i] = true;
        if (o.panel == "todo" || o.panel == "all") for (bool& b : sec_open) b = true;
    }
    cam_view = o.view >= 0 ? static_cast<CamView>(o.view) : (I.spans_domain ? CamView::Lateral : CamView::TresCuartos);
    view.frame(sim, cam_view, viewport_rect());
    if (o.cam_set) {
        view.cam.target = sim.to_cells(Vec3(o.cam[0], o.cam[1], o.cam[2]));
        view.cam.yaw = o.cam[3] * k_deg2rad;
        view.cam.pitch = o.cam[4] * k_deg2rad;
        view.cam.distance = o.cam[5] / sim.dom.dx;
        view.cam.update(viewport_rect());
    }
    sync_ui_from_sim();
    update_title();
    sweep.param = SweepParam::Count;
    for (int i = 0; i < k_nsweep; ++i)
        if (sweep_param_available(idx, static_cast<SweepParam>(i))) { sweep.param = static_cast<SweepParam>(i); break; }
    sweep_defaults();
    perf.step_s.push(static_cast<double>(sim.dom.cells()) / 8e8);
    t_start = now_sec();
    return true;
}

void App::update_title() {
    if (!win || win->is_headless()) return;
    char t[200];
    std::snprintf(t, sizeof t, "Túnel de viento CFD — %s", sim.built.info.name.c_str());
    win->set_title(t);
}

void App::sync_ui_from_sim() {
    ui_params = sim.params;
    ui_model = sim.cfg.model;
    ui_group = sim.built.info.kind == models::Kind::F1Car ? 0 : 1;
    ui_ground = static_cast<int>(sim.cfg.ground);
    ui_preset = static_cast<int>(sim.cfg.preset);
    ui_fp32 = sim.cfg.fp32;
    ui_gpu = sim.gpu_on();
    ui_refine = sim.cfg.refine;
    ui_wall = sim.cfg.wall;
    ui_interp_bb = sim.cfg.interp_bb;
    ui_ramp_ft = sim.cfg.ramp_ft;
}

void App::sweep_defaults() {
    const int m = sim.cfg.model;
    const models::Info& I = models::info(m);
    switch (sweep.param) {
        case SweepParam::Aoa:
            if (I.needs_ground) { sweep.from = 0; sweep.to = 12; sweep.n = 5; }
            else { sweep.from = -4; sweep.to = 16; sweep.n = 6; }
            break;
        case SweepParam::Height:
            if (I.needs_ground) { sweep.from = 25; sweep.to = 250; sweep.n = 5; }
            else { sweep.from = 100; sweep.to = 1500; sweep.n = 5; }
            break;
        case SweepParam::RideHeight: sweep.from = 15; sweep.to = 75; sweep.n = 5; break;
        case SweepParam::FrontFlap: case SweepParam::RearFlap: sweep.from = -10; sweep.to = 10; sweep.n = 5; break;
        case SweepParam::Yaw: sweep.from = 0; sweep.to = 10; sweep.n = 5; break;
        case SweepParam::FlapGap: sweep.from = 10; sweep.to = 60; sweep.n = 5; break;
        default: break;
    }
    sweep.npts = 0;
    sweep.done = false;
}

void App::load_model(int model, bool keep_camera) {
    if (sweep.running) sweep_stop(sweep, sim);
    const models::Info& I = models::info(model);
    sim.cfg.model = model;
    sim.cfg.params = models::Params{};
    sim.cfg.ground = I.needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None;
    sim.speed_kmh = I.default_speed_kmh;
    sim.init();
    view.defaults_for(sim);
    if (!keep_camera) {
        cam_view = I.spans_domain ? CamView::Lateral : CamView::TresCuartos;
        view.frame(sim, cam_view, viewport_rect());
    }
    sync_ui_from_sim();
    sweep.param = SweepParam::Count;
    for (int i = 0; i < k_nsweep; ++i)
        if (sweep_param_available(model, static_cast<SweepParam>(i))) { sweep.param = static_cast<SweepParam>(i); break; }
    sweep_defaults();
    ui.toast(ui.style().accent, "Modelo: %s · %d×%d×%d, dx %.1f mm", I.name.c_str(), sim.dom.nx, sim.dom.ny, sim.dom.nz,
             static_cast<double>(sim.dom.dx * 1e3f));
    update_title();
}

void App::reinit() {
    if (sweep.running) sweep_stop(sweep, sim);
    const Preset np = static_cast<Preset>(clamp_(ui_preset, 0, k_npresets - 1));
    if (np != sim.cfg.preset || want_res_apply) sim.cfg.cells = 0;   // presupuesto explícito (--cells) sólo hasta cambiar de preset
    want_res_apply = false;
    sim.cfg.preset = np;
    sim.cfg.fp32 = ui_fp32;
    sim.cfg.wall = ui_wall;
    sim.cfg.interp_bb = ui_interp_bb;
    sim.cfg.ramp_ft = ui_ramp_ft;
    sim.cfg.refine = ui_refine;
    sim.init();
    view.dirty = true;
    view.rakes_dirty = true;
    view.frame(sim, cam_view, viewport_rect());
    sync_ui_from_sim();
    if (sim.levels.empty())
        ui.toast(ui.style().accent, "Reiniciado: %s, %s · %d×%d×%d (%.2f M celdas), dx %.1f mm", preset_name(sim.cfg.preset),
                 sim.cfg.fp32 ? "FP32" : "FP16S", sim.dom.nx, sim.dom.ny, sim.dom.nz, static_cast<double>(sim.dom.cells()) * 1e-6,
                 static_cast<double>(sim.dom.dx * 1e3f));
    else
        ui.toast(ui.style().accent, "Reiniciado: %s, %s · %d niveles finos, %.2f M celdas, dx %.1f → %.1f mm", preset_name(sim.cfg.preset),
                 sim.cfg.fp32 ? "FP32" : "FP16S", sim.refine_levels(), static_cast<double>(sim.total_cells) * 1e-6,
                 static_cast<double>(sim.dom.dx * 1e3f), static_cast<double>(sim.levels.back().dx * 1e3f));
}

// ============================================================================
//  Parámetros geométricos con antirrebote
// ============================================================================
void App::apply_ui_params(bool dragging) {
    if (sweep.running) {                  // el barrido manda sobre los parámetros
        want_geom = false;
        ui_params = sim.params;
        return;
    }
    const double now = now_sec();
    if (want_geom) {
        if (dragging && now - last_geom_t < 0.12) return;   // como mucho ~8 reconstrucciones/s al arrastrar
        bool adj = false;
        const bool changed = sim.set_params(ui_params, dragging ? 1.0f : 0.5f, &adj);
        last_geom_t = now;
        want_geom = false;
        view.rakes_dirty = true;
        if (dragging) geom_final_pending = geom_final_pending || changed;
        else {
            geom_final_pending = false;
            if (adj) ui.toast(ui.style().warn, "Alturas corregidas (rake limitado o la carrocería tocaba el suelo)");
            ui_params = sim.params;
        }
        return;
    }
    if (!dragging && geom_final_pending) {
        geom_final_pending = false;
        sim.remesh(0.5f);
        ui_params = sim.params;
    }
}

// ============================================================================
//  Entrada: cámara y atajos
// ============================================================================
void App::handle_input(Rect vp) {
    using namespace platform;
    if (!ui.wants_mouse()) {
        const bool inside = vp.contains(in.mouse_x, in.mouse_y);
        if (in.mouse_down[MouseLeft] && (in.mouse_dx || in.mouse_dy))
            view.cam.orbit(-static_cast<float>(in.mouse_dx) * 0.006f, static_cast<float>(in.mouse_dy) * 0.006f);
        if ((in.mouse_down[MouseRight] || in.mouse_down[MouseMiddle]) && (in.mouse_dx || in.mouse_dy))
            view.cam.pan(static_cast<float>(in.mouse_dx), static_cast<float>(in.mouse_dy));
        if (inside && in.wheel != 0.0f) view.cam.zoom(std::pow(0.88f, in.wheel));
        if (inside && in.double_click) {
            Vec3 p;
            if (view.unproject(fb, in.mouse_x, in.mouse_y, p)) view.cam.target = p;
        }
    }
    if (ui.wants_keyboard()) return;
    const bool* kp = in.key_pressed;
    const ui::Style& st = ui.style();
    if (kp[KeyEscape]) {
        const double now = now_sec();
        if (show_help) show_help = false;
        else if (now - esc_t < 2.0) quit = true;
        else { esc_t = now; ui.toast(st.warn, "Pulsa Esc otra vez para salir"); }
    }
    if (kp[KeyF1]) show_help = !show_help;
    if (kp[KeyF11]) win->set_fullscreen(!win->fullscreen());
    if (kp[KeyF12] || kp['s']) want_shot = true;
    if (kp[' ']) { paused = !paused; ui.toast(st.accent, paused ? "En pausa" : "En marcha"); }
    if (kp['r']) { sim.reset_flow(); view.reset_particles(); ui.toast(st.warn, "Flujo reiniciado"); }
    if (kp['g']) { ui_ground = (static_cast<int>(sim.cfg.ground) + 1) % 3; want_ground = true; }
    if (kp['d'] && (sim.built.info.param_mask & models::P_Drs)) {
        ui_params.drs_open = !ui_params.drs_open;
        want_geom = true;
        ui.toast(st.accent, sim.built.info.id == "f1_2026" ? (ui_params.drs_open ? "Modo X (baja resistencia)" : "Modo Z (carga)")
                                                          : (ui_params.drs_open ? "DRS abierto" : "DRS cerrado"));
    }
    VisSettings& v = view.vs;
    const VisSettings before = v;
    if (kp['c']) v.surf = static_cast<SurfMode>((static_cast<int>(v.surf) + 1) % static_cast<int>(SurfMode::Count));
    for (int a = 0; a < 3; ++a)
        if (kp['x' + a]) v.slice_axis = v.slice_axis == a + 1 ? 0 : a + 1;
    if (v.slice_axis > 0) {
        const int ax = v.slice_axis - 1;
        const float d = in.shift ? 10.0f : 1.0f;
        if (kp[KeyRight] || kp[KeyUp]) v.slice_pos[ax] += d;
        if (kp[KeyLeft] || kp[KeyDown]) v.slice_pos[ax] -= d;
    }
    if (kp['l']) v.lines = !v.lines;
    if (kp['p']) v.smoke = !v.smoke;
    if (kp['v']) v.vortex = !v.vortex;
    if (kp['h']) hide_ui = !hide_ui;
    for (int i = 0; i < 6; ++i)
        if (kp['1' + i]) { cam_view = static_cast<CamView>(i); view.frame(sim, cam_view, vp); }
    if (!(before == v)) view.dirty = true;
}

// ============================================================================
//  Pasos por cuadro adaptativos
// ============================================================================
int App::adaptive_steps() const {
    if (fixed_steps) return steps_fixed;
    // Objetivo de FPS y fracción MÍNIMA del cuadro para el solver según la prioridad: si el resto
    // del cuadro (visualización + render + UI + presentación) ya se come el objetivo, el solver
    // conserva su parte (con una malla grande el FPS baja en vez de congelar la simulación).
    static constexpr double k_target[3] = {1.0 / 45.0, 1.0 / 30.0, 1.0 / 12.0};
    static constexpr double k_share[3] = {0.25, 0.5, 0.8};
    const int pi = clamp_(static_cast<int>(prio), 0, 2);
    const double other = max_(0.0, (perf.frame_ms.v - perf.sim_ms.v) * 1e-3);
    const double step_s = perf.step_s.v > 0 ? perf.step_s.v : static_cast<double>(sim.dom.cells()) / 8e8;
    double budget = max_(k_target[pi] - other, other * k_share[pi] / (1.0 - k_share[pi]));
    if (sim.gpu_on() && sim.gpu_async) {
        // iGPU en paralelo: la GPU puede ocupar el cuadro ENTERO mientras la CPU dibuja (el tiempo de "sim" del
        // cuadro es sólo la espera del lote anterior). Lote ≈ lo que tarda el resto del cuadro (o el objetivo).
        budget = max_(k_target[pi], other);
    }
    int k = static_cast<int>(budget / step_s);
    // Suavizado: como mucho ×1.5 / ÷1.5 respecto al cuadro anterior.
    const int prev = max_(perf.steps_per_frame, 1);
    k = clamp_(k, max_(1, prev * 2 / 3), prev * 3 / 2 + 1);
    return clamp_(k, 1, 2000);
}

// ============================================================================
//  Un cuadro
// ============================================================================
void App::frame(int steps_override, bool render_frame) {
    const double t0 = now_sec();
    in.begin_frame();
    win->poll(in);
    if (in.quit) quit = true;
    if (in.mouse_dx || in.mouse_dy || in.mouse_pressed[0] || in.mouse_pressed[1] || in.mouse_pressed[2]) mouse_seen = true;
    // ¿Hubo entrada este cuadro? (en pausa y sin entrada se redibuja sólo a ~10 Hz: CPU casi a 0)
    bool activity = in.mouse_dx || in.mouse_dy || in.wheel != 0.0f || in.text_len > 0 || in.resized || in.double_click;
    for (int b = 0; b < 3 && !activity; ++b) activity = in.mouse_pressed[b] || in.mouse_released[b] || in.mouse_down[b];
    for (int kc = 0; kc < platform::KeyCount && !activity; ++kc) activity = in.key_pressed[kc];
    if (in.resized && in.fb_w > 0 && in.fb_h > 0) { fb.resize(in.fb_w, in.fb_h); view.cam.update(viewport_rect()); }
    const Rect vp = viewport_rect(), pr = panel_rect();
    // Avisos: sobre las leyendas y la barra de estado (abajo del visor).
    ui.set_toast_area({vp.x, vp.y, vp.w, max_(vp.h - (view.vs.legends ? 104 : 30), 100)});
    ui.begin_frame(in, fb.w, fb.h, t0);
    if (!hide_ui) build_panel(pr);
    ui.end_frame();
    handle_input(vp);
    const double t_ui_build = now_sec() - t0;

    // ---- Cambios pendientes
    if (want_model) { want_model = false; if (ui_model != sim.cfg.model) load_model(ui_model); }
    if (want_reinit) { want_reinit = false; reinit(); }
    if (want_gpu) {
        want_gpu = false;
        if (ui_gpu != sim.gpu_on()) {
            const bool ok = sim.set_gpu(ui_gpu);
            if (ui_gpu && !ok) ui.toast(ui.style().error, "iGPU no disponible: %s", sim.gpu_error.c_str());
            else ui.toast(ui.style().accent, "Solver: %s", sim.gpu_on() ? "iGPU (Vulkan propio; la CPU dibuja en paralelo)" : "CPU");
            perf.step_s = {};
        }
        ui_gpu = sim.gpu_on();
    ui_refine = sim.cfg.refine;
    }
    if (want_ground) {
        want_ground = false;
        const lbm::GroundMode g = static_cast<lbm::GroundMode>(clamp_(ui_ground, 0, 2));
        if (g != sim.cfg.ground) {
            if (sweep.running) sweep_stop(sweep, sim);
            const u64 dv = sim.domain_version;
            sim.set_ground(g);
            if (sim.domain_version != dv) { view.defaults_for(sim); view.frame(sim, cam_view, vp); }
            view.dirty = true;
            ui.toast(ui.style().accent, "Suelo: %s%s", g == lbm::GroundMode::None ? "ninguno" : (g == lbm::GroundMode::Static ? "fijo" : "cinta móvil"),
                     sim.domain_version != dv ? " (dominio redimensionado, flujo reiniciado)" : "");
        }
        sync_ui_from_sim();
    }
    apply_ui_params(geom_dragging);
    if (want_sweep_start) {
        want_sweep_start = false;
        if (sweep_start(sweep, sim)) { paused = false; ui.toast(ui.style().accent, "Barrido de %s: %d puntos", sweep_param_name(sweep.param), sweep.n); }
    }
    if (want_csv) {
        want_csv = false;
        const std::string p = auto_shot_path("fuerzas", "csv");
        if (write_forces_csv(p)) ui.toast(ui.style().success, "CSV guardado: %s", p.c_str());
        else ui.toast(ui.style().error, "No se pudo escribir %s", p.c_str());
    }
    if (want_sweep_csv) {
        want_sweep_csv = false;
        const std::string p = auto_shot_path("barrido", "csv");
        if (ensure_dir("capturas") && sweep_write_csv(sweep, sim, p)) ui.toast(ui.style().success, "Barrido guardado: %s", p.c_str());
        else ui.toast(ui.style().error, "No se pudo escribir %s", p.c_str());
    }

    // ---- Simulación
    const double t1 = now_sec();
    int k = 0;
    if (!paused && sim.ready) {
        k = steps_override > 0 ? steps_override : adaptive_steps();
        const bool ok = sim.step(k);
        const double ts = now_sec() - t1;
        // Con la iGPU en paralelo el tiempo de este cuadro es sólo la espera: se usa el tiempo de GPU por paso.
        const double gs = sim.gpu_async ? sim.gpu_step_seconds() : 0.0;
        perf.step_s.push(gs > 0.0 ? gs : ts / k, 0.2);
        if (!ok) {
            view.reset_particles();
            if (sweep.running) sweep_stop(sweep, sim);
            ui.toast(ui.style().error, "El flujo divergió: viscosidad aumentada a ν = %.1e y flujo reiniciado", static_cast<double>(sim.nu));
            std::fprintf(stderr, "[cfd] divergencia: nu -> %.2e, flujo reiniciado\n", static_cast<double>(sim.nu));
        } else if (sweep.running && sim.published > 0 && sweep_after_step(sweep, sim, sim.published)) {
            ui.toast(ui.style().success, "Barrido terminado (%d puntos)", sweep.npts);
            sync_ui_from_sim();
        }
        perf.mlups.push(sim.solver.last_mlups(), 0.1);
    }
    if (k > 0) perf.steps_per_frame = k;
    const double t2 = now_sec();

    // ---- Visualización (sin ventana / pasos fijos: cada cuadro, para capturas deterministas)
    static constexpr double k_vis_interval[3] = {1.0 / 20.0, 1.0 / 12.0, 1.0 / 6.0};
    view.field_interval = fixed_steps ? 0.0 : k_vis_interval[clamp_(static_cast<int>(prio), 0, 2)];
    view.update(sim, k);
    const double t3 = now_sec();
    double t_render = 0, t_ui_render = 0;
    const bool idle = idle_throttle && paused && !activity && !view.dirty && !want_shot && sim.field_version == last_render_field &&
                      sim.geom_version == last_render_geom && t0 - last_render_t < 0.1;
    if (idle) render_frame = false;
    if (render_frame) {
        last_render_t = t0;
        last_render_field = sim.field_version;
        last_render_geom = sim.geom_version;
        view.render(fb, vp, sim);
        if (mouse_seen && !ui.wants_mouse()) view.pick_probe(fb, sim, in.mouse_x, in.mouse_y);
        else view.probe_ok = false;
        draw_overlay(vp);
        overlay.render(fb);
        const double t4 = now_sec();
        ui.render(fb);
        t_ui_render = now_sec() - t4;
        t_render = t4 - t3;
    }
    if (want_shot) {
        want_shot = false;
        const std::string p = auto_shot_path("captura", "png");
        if (save_screenshot(p)) { last_shot = p; ui.toast(ui.style().success, "Captura guardada: %s", p.c_str()); }
        else ui.toast(ui.style().error, "No se pudo guardar la captura");
    }
    const double t5 = now_sec();
    if (render_frame) win->present(fb);
    const double t6 = now_sec();
    if (idle) {   // cuadro omitido: no cuenta en las estadísticas; cede la CPU
        ++frame_no;
        ::usleep(8000);
        return;
    }
    // ---- Estadísticas
    const double frame_ms = (t6 - t0) * 1e3;
    perf.last_sim_ms = (t2 - t1) * 1e3;
    perf.last_vis_ms = (t3 - t2) * 1e3;
    perf.last_render_ms = t_render * 1e3;
    perf.last_ui_ms = (t_ui_build + t_ui_render) * 1e3;
    perf.last_present_ms = (t6 - t5) * 1e3;
    perf.last_frame_ms = frame_ms;
    perf.sim_ms.push(perf.last_sim_ms, 0.15);
    perf.vis_ms.push(perf.last_vis_ms, 0.15);
    perf.render_ms.push(perf.last_render_ms, 0.15);
    perf.ui_ms.push(perf.last_ui_ms, 0.15);
    perf.present_ms.push(perf.last_present_ms, 0.15);
    perf.frame_ms.push(frame_ms, 0.15);
    if (frame_ms > 0) perf.fps.push(1000.0 / frame_ms, 0.08);
    perf.push_hist(static_cast<float>(frame_ms), static_cast<float>(perf.last_sim_ms));
    ++frame_no;
}

// ============================================================================
//  Capturas y CSV
// ============================================================================
std::string App::auto_shot_path(const char* prefix, const char* ext) const {
    char ts[32];
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    std::strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", &tmv);
    char b[256];
    std::snprintf(b, sizeof b, "capturas/%s_%s_%s_%03ld.%s", prefix, sim.built.info.id.c_str(), ts, frame_no % 1000, ext);
    return b;
}

bool App::save_screenshot(const std::string& path) {
    const usize slash = path.find_last_of('/');
    if (slash != std::string::npos && !ensure_dir(path.substr(0, slash))) return false;
    return png::write_argb(path, fb.color.data(), fb.w, fb.h, fb.stride, false);
}

bool App::write_forces_csv(const std::string& path) {
    const usize slash = path.find_last_of('/');
    if (slash != std::string::npos && !ensure_dir(path.substr(0, slash))) return false;
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    const models::Info& I = sim.built.info;
    const AeroResult& r = sim.res;
    std::fprintf(fp, "# %s (%s) — red %dx%dx%d, dx = %.2f mm, u_red = %.3f, nu = %.2e, suelo = %d, A_ref = %.4f m2, %.0f km/h\n", I.id.c_str(),
                 I.name.c_str(), sim.dom.nx, sim.dom.ny, sim.dom.nz, static_cast<double>(sim.dom.dx * 1e3f), static_cast<double>(sim.cfg.u_lat),
                 static_cast<double>(sim.nu), static_cast<int>(sim.cfg.ground), static_cast<double>(I.ref_area_m2), static_cast<double>(sim.speed_kmh));
    std::fprintf(fp, "# final: CL=%.5f CD=%.5f CS=%.5f SCz=%.5f SCx=%.5f L/D=%.4f balance_del=%.2f%% carga=%.1fN resistencia=%.1fN convergido=%d\n",
                 static_cast<double>(r.cl), static_cast<double>(r.cd), static_cast<double>(r.cs), static_cast<double>(r.scz), static_cast<double>(r.scx),
                 static_cast<double>(r.ld), static_cast<double>(r.balance), static_cast<double>(r.down_n), static_cast<double>(r.drag_n), sim.converged ? 1 : 0);
    if (sim.is_car())
        std::fprintf(fp, "# altura de marcha pedida %.1f/%.1f mm, efectiva (resolución) %.1f/%.1f mm, hueco mínimo %.1f mm\n",
                     static_cast<double>(sim.params.ride_front_mm), static_cast<double>(sim.params.ride_rear_mm),
                     static_cast<double>(sim.ride_eff_front_mm), static_cast<double>(sim.ride_eff_rear_mm), static_cast<double>(sim.ride_gap_min_mm));
    std::fprintf(fp, "# componentes (SCz_m2, SCx_m2):");
    for (int i = 0; i < r.ncomp; ++i) std::fprintf(fp, " %s=(%.4f,%.4f)", r.comp[i].name, static_cast<double>(r.comp[i].scz), static_cast<double>(r.comp[i].scx));
    std::fprintf(fp, "\npaso,pasos_de_flujo,CL,CD,SCz_m2,SCx_m2\n");
    const History& h = sim.hist;
    const float ft = max_(sim.ft_steps(), 1.0f);
    for (int i = 0; i < h.count; ++i) {
        const int k = (h.oldest() + i) % History::N;
        std::fprintf(fp, "%llu,%.3f,%.5f,%.5f,%.5f,%.5f\n", static_cast<unsigned long long>(h.step[k]), static_cast<double>(h.step[k]) / ft,
                     static_cast<double>(h.cl[k]), static_cast<double>(h.cd[k]), static_cast<double>(h.scz[k]), static_cast<double>(h.scx[k]));
    }
    std::fclose(fp);
    return true;
}

// ============================================================================
//  Bucle interactivo (X11)
// ============================================================================
int App::run_interactive() {
    std::vector<double> ft, st, vt, rt, ut, pt;
    const usize cap = opt.frames > 0 ? static_cast<usize>(opt.frames) + 8 : 20000;
    ft.reserve(cap); st.reserve(cap); vt.reserve(cap); rt.reserve(cap); ut.reserve(cap); pt.reserve(cap);
    std::vector<int> ks;
    ks.reserve(cap);
    std::vector<double> ml;
    ml.reserve(cap);
    t_start = now_sec();
    idle_throttle = true;
    sim.gpu_async = true;   // iGPU: la simulación del lote siguiente se solapa con el dibujo del anterior
    long n = 0;
    while (!quit) {
        frame(opt.spf);
        ++n;
        if (n > 10 && ft.size() < cap) {
            ft.push_back(perf.last_frame_ms); st.push_back(perf.last_sim_ms); vt.push_back(perf.last_vis_ms);
            rt.push_back(perf.last_render_ms); ut.push_back(perf.last_ui_ms); pt.push_back(perf.last_present_ms);
            ks.push_back(perf.steps_per_frame);
            ml.push_back(sim.solver.last_mlups());
        }
        if (opt.frames > 0 && n >= opt.frames) break;
        if (opt.quit_after > 0 && now_sec() - t_start > opt.quit_after) break;
    }
    if (!opt.shot.empty()) {
        if (save_screenshot(opt.shot)) std::printf("[cfd] captura → %s\n", opt.shot.c_str());
        else std::fprintf(stderr, "[cfd] no se pudo guardar %s\n", opt.shot.c_str());
    }
    if (!ft.empty()) {
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::vector<double> kd(ks.begin(), ks.end());
        const platform::PresentStats ps = win->present_stats();
        std::printf("[cfd] %ld cuadros en %.1f s · ventana %dx%d (escala %d, %s) · modelo %s · %d×%d×%d (%.2f M celdas)\n", n, now_sec() - t_start,
                    ps.win_w, ps.win_h, win->pixel_scale(), ps.shm ? "MIT-SHM" : "XPutImage", sim.built.info.id.c_str(), sim.dom.nx, sim.dom.ny,
                    sim.dom.nz, static_cast<double>(sim.dom.cells()) * 1e-6);
        const double mf = med(ft);
        std::printf("[cfd] medianas: cuadro %.2f ms (%.1f FPS) | sim %.2f | vis %.2f | render %.2f | UI %.2f | present %.2f ms | %0.f pasos/cuadro | %.0f MLUPS\n",
                    mf, 1000.0 / mf, med(st), med(vt), med(rt), med(ut), med(pt), med(kd), med(ml));
    }
    return 0;
}

} // namespace cfd::app
