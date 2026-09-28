// ============================================================================
//  app/panel.cpp — panel lateral (ui::Context), HUD del visor, leyendas de
//  color, barra de estado, sonda y ayuda (F1). Todo el texto en español.
// ============================================================================
#include "app.hpp"

#include "../render/colormap.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

namespace cfd::app {

using render::Rect;

namespace {

const char* const k_ground_names[3] = {"Ninguno", "Fijo", "Cinta móvil"};
const char* const k_prec_names[2] = {"FP16S", "FP32"};
const char* const k_prio_names[3] = {"Fluidez", "Equilibrado", "Máx. simulación"};
const char* const k_axis_names[4] = {"No", "X", "Y", "Z"};
const char* const k_slice_q_names[7] = {"Velocidad |u|/U∞", "u_x/U∞ (axial)", "u_z/U∞ (vertical)", "Cp (presión)",
                                        "Cp0 (presión total)", "Vorticidad |ω|", "Criterio Q"};
const char* const k_surf_names[6] = {"Cp (presión)", "Velocidad junto a la pared", "Por componente", "Liso", "Vóxeles (lo que ve el solver)",
                                     "Oculta (sólo el flujo)"};
const char* const k_rake_names[4] = {"Vertical (plano central)", "Horizontal (altura del fondo)", "Puntas / laterales", "Rejilla aguas arriba"};
const char* const k_preset_items[4] = {"Rápida · 2.5 M celdas", "Media · 6 M celdas", "Alta · 13 M celdas", "Ultra · 28 M celdas"};
const char* const k_plot_names[4] = {"CL / CD", "SCz / SCx", "L/D", "Balance"};

int g_sweep_plot = 0;   // qué se dibuja en la gráfica del barrido


// El balance (% de la carga en el eje delantero) sólo tiene sentido con carga apreciable hacia
// abajo; con sustentación neta o carga ~0 el cociente se dispara (se muestra "—").
// Convención de presentación según el tipo de objeto: la fuerza "útil" de un coche o de un ala en
// efecto suelo es la carga (hacia abajo); la de un ala en aire libre, la sustentación (hacia arriba);
// en un cuerpo romo en aire libre sólo interesa la resistencia. (AeroResult siempre usa carga +.)
enum class Conv : u8 { Carga, Sustentacion, Resistencia };
Conv conv_of(const models::Info& I) {
    if (I.kind == models::Kind::Wing && !I.needs_ground) return Conv::Sustentacion;
    if (I.kind == models::Kind::Body && !I.needs_ground) return Conv::Resistencia;
    return Conv::Carga;
}

bool balance_ok(const AeroResult& r) { return std::isfinite(r.balance) && r.cl > 0.05f && r.balance > -50.0f && r.balance < 150.0f; }
// Fuera de 0-100 % el centro de presiones cae fuera de la batalla (un eje recibe sustentación): posible, pero
// en la práctica indica un flujo sin converger o una carga mal repartida → se resalta en ámbar.
bool balance_odd(const AeroResult& r) { return r.balance < 0.0f || r.balance > 100.0f; }

} // namespace

SiStr fmt_si(double v, int digits) {
    // La fuente sólo tiene ¹²³ como superíndices: se escribe "1,4·10^6".
    SiStr r{};
    if (!std::isfinite(v)) { std::snprintf(r.s, sizeof r.s, "—"); return r; }
    if (v == 0) { r.s[0] = '0'; return r; }
    const int e = static_cast<int>(std::floor(std::log10(std::fabs(v))));
    const double m = v / std::pow(10.0, e);
    if (e >= -2 && e <= 3) std::snprintf(r.s, sizeof r.s, "%.*f", max_(digits - 1 - e, 0), v);
    else std::snprintf(r.s, sizeof r.s, "%.*f·10^%d", max_(digits - 1, 0), m, e);
    for (char& c : r.s) if (c == '.') c = ',';
    return r;
}

// ============================================================================
//  Panel lateral
// ============================================================================
void App::build_panel(Rect pr) {
    ui::Context& U = ui;
    const ui::Style& st = U.style();
    char sub[160];
    std::snprintf(sub, sizeof sub, "LBM D3Q19 · %d×%d×%d · %.0f FPS", sim.dom.nx, sim.dom.ny, sim.dom.nz, perf.fps.v);
    if (!U.begin_panel("panel_principal", pr)) return;
    U.title("Túnel de viento CFD", sub);
    U.spacing(2);
    const models::Info& I = sim.built.info;
    const u32 mask = I.param_mask;

    // ---------------------------------------------------------------- Modelo
    if (U.header("Modelo", sec_open[SecModelo])) {
        const char* const grp[2] = {"Coches F1", "Objetos"};
        U.segmented("##grupo", grp, 2, &ui_group);
        const std::vector<int>& list = ui_group == 0 ? models_f1 : models_obj;
        const std::vector<const char*>& names = ui_group == 0 ? names_f1 : names_obj;
        int sel = -1;
        for (usize i = 0; i < list.size(); ++i) if (list[i] == sim.cfg.model) sel = static_cast<int>(i);
        if (U.combo("Modelo", &sel, names.data(), static_cast<int>(names.size())) && sel >= 0) {
            ui_model = list[static_cast<usize>(sel)];
            want_model = true;
        }
        U.tooltip("Coches de F1 de 9 épocas reglamentarias y cuerpos/alas de referencia.");
        U.text_wrapped("%s", I.description.c_str());
        U.value("Época", "%s", I.era.c_str());
        if (I.kind == models::Kind::F1Car) {
            U.value("Batalla", "%.2f m", static_cast<double>(I.wheelbase_m));
            if (I.reg_width_m > 0) U.value("Ancho reglamentario", "%.2f m", static_cast<double>(I.reg_width_m));
            U.value("Flaps de la época (del./tras.)", "%.0f° / %.0f°", static_cast<double>(I.default_front_flap_deg), static_cast<double>(I.default_rear_flap_deg));
        }
        U.value("Área de referencia", "%.3f m²", static_cast<double>(I.ref_area_m2));
        if (I.ref_ClA != 0.0f || I.ref_CdA != 0.0f)
            U.value("Valores reales aprox.", "SCz %.2f · SCx %.2f m²", static_cast<double>(I.ref_ClA), static_cast<double>(I.ref_CdA));
        if (I.ref_source && I.ref_source[0]) U.text_wrapped_colored(st.text_disabled, "Fuente: %s", I.ref_source);
    }

    // ---------------------------------------------------------------- Configuración
    if (U.header("Configuración", sec_open[SecConfig])) {
        const ParamRanges R = param_ranges(sim.cfg.model);
        bool ch = false, act = false;
        auto slider = [&](const char* label, float* v, Range r, const char* f, const char* unit, const char* tip) {
            const bool c = U.slider_float(label, v, r.lo, r.hi, f, unit);
            act |= U.item_active();
            if (tip) U.tooltip("%s", tip);
            ch |= c;
        };
        if (mask & models::P_RideHeight) {
            slider("Altura delantera", &ui_params.ride_front_mm, R.ride_front, "%.0f", "mm", "Altura del plano de referencia (plank) en el eje delantero.");
            slider("Altura trasera", &ui_params.ride_rear_mm, R.ride_rear, "%.0f", "mm", "Altura del plank en el eje trasero. El rake se limita a un rango razonable y el coche se sube si la carrocería tocara el suelo.");
            const float rake = ui_params.ride_rear_mm - ui_params.ride_front_mm;
            U.text_colored(st.text_dim, "Rake %+.0f mm (%.2f°) · límite %+.0f…%+.0f mm", static_cast<double>(rake),
                           static_cast<double>(std::atan2(rake * 1e-3f, max_(I.wheelbase_m, 0.5f)) * k_rad2deg),
                           static_cast<double>(R.rake.lo), static_cast<double>(R.rake.hi));
            // Altura efectiva: la red no resuelve huecos de menos de ~Sim::k_gap_cells celdas bajo el fondo → el solver
            // ve el coche más alto (Sim::params_eff). Se muestra siempre; en ámbar si difiere de la pedida.
            if (sim.ride_limited)
                U.text_wrapped_colored(st.warn, "Altura efectiva (resolución): %.0f / %.0f mm (pedida %.0f / %.0f; hueco mínimo %.0f mm = %.1f celdas)",
                                       static_cast<double>(sim.ride_eff_front_mm), static_cast<double>(sim.ride_eff_rear_mm),
                                       static_cast<double>(sim.params.ride_front_mm), static_cast<double>(sim.params.ride_rear_mm),
                                       static_cast<double>(sim.ride_gap_min_mm), static_cast<double>(Sim::k_gap_cells));
            else
                U.text_colored(st.text_dim, "Altura efectiva = pedida (hueco mínimo %.0f mm)", static_cast<double>(sim.ride_gap_min_mm));
            U.tooltip("Bajo el fondo hacen falta ~%.1f celdas de hueco para que pase aire: con alturas menores la red "
                      "sube el coche (la menor altura h pasa a (h^4 + g^4)^(1/4), rake conservado; ≈ h si h ≥ 1.5·g). Sube la resolución para "
                      "estudiar alturas reales. Ver docs/FISICA.md.", static_cast<double>(Sim::k_gap_cells));
        }
        if (mask & models::P_FrontFlap)
            slider(I.kind == models::Kind::Wing ? "Flap (incremento)" : "Flap delantero", &ui_params.front_flap_deg, R.flap, "%+.1f", "°",
                   "Incremento sobre el ángulo de flap de la época (+ = más carga y más resistencia).");
        if (mask & models::P_RearFlap)
            slider("Flap trasero", &ui_params.rear_flap_deg, R.flap, "%+.1f", "°", "Incremento sobre el ángulo de flap de la época (+ = más carga).");
        if (mask & models::P_Drs) {
            const bool x_mode = I.id == "f1_2026";
            const char* lbl = x_mode ? (ui_params.drs_open ? "Modo X: baja resistencia (pulsa → Z)###drs" : "Modo Z: carga (pulsa → X)###drs")
                                     : (ui_params.drs_open ? "DRS ABIERTO (pulsa para cerrar)###drs" : "DRS cerrado (pulsa para abrir)###drs");
            if (U.toggle_button(lbl, &ui_params.drs_open)) { ch = true; }
            U.tooltip(x_mode ? "2026: en modo X se abren los flaps delanteros y el trasero para reducir la resistencia en recta."
                             : "DRS: el flap superior del alerón trasero se abre (-6°) para reducir la resistencia.");
        }
        if (mask & models::P_Yaw) slider("Guiñada", &ui_params.yaw_deg, R.yaw, "%+.1f", "°", "Ángulo del flujo respecto al eje del vehículo (viento cruzado).");
        if (mask & models::P_Aoa)
            slider("Ángulo de ataque", &ui_params.aoa_deg, R.aoa, "%+.1f", "°",
                   I.needs_ground ? "Convención F1: + = borde de ataque hacia abajo (más carga)." : "+ = borde de ataque hacia arriba (sustentación positiva).");
        if ((mask & models::P_Height) && sim.cfg.ground != lbm::GroundMode::None)
            slider("Altura sobre el suelo", &ui_params.height_mm, R.height, "%.0f", "mm", "Altura del punto más bajo del objeto sobre el suelo.");
        else if (mask & models::P_Height)
            U.text_wrapped_colored(st.text_disabled, "Altura sobre el suelo: sólo con suelo (Túnel → Suelo).");
        if (mask & models::P_FlapGap) slider("Hueco del flap", &ui_params.flap_gap_mm, R.gap, "%.0f", "mm", "Ranura entre el plano principal y el flap.");
        if (mask & models::P_Wheels) {
            if (U.toggle("Ruedas girando", &ui_params.wheels_rotating)) ch = true;
            U.tooltip("Sólo tiene efecto con la cinta móvil (ruedas rodando a la velocidad del suelo).");
        }
        if (U.button("Valores por defecto", ui::ButtonKind::Subtle)) {
            ui_params = models::resolve_params(sim.cfg.model, models::Params{});
            ch = true;
        }
        if (ch) { want_geom = true; }
        geom_dragging = act;
        if (!mask) U.text_colored(st.text_dim, "Este modelo no tiene parámetros geométricos.");
    }

    // ---------------------------------------------------------------- Túnel
    if (U.header("Túnel", sec_open[SecTunel])) {
        if (U.slider_float("Velocidad", &sim.speed_kmh, 30.0f, 400.0f, "%.0f", "km/h")) sim.update_results();
        U.tooltip("Reescala las fuerzas en newtons/kgf (F = C·½ρU²A) y fija el Reynolds REAL de la ley de pared "
                  "(fricción en las superficies): cambia poco los coeficientes. El Reynolds de la red (viscosidad) no cambia.");
        U.label("Suelo", st.text_dim);
        int g = ui_ground;
        if (U.segmented("##suelo", k_ground_names, 3, &g)) { ui_ground = g; want_ground = true; }
        U.tooltip("Ninguno: aire libre. Fijo: suelo sin deslizamiento (túnel antiguo, crece capa límite). Cinta móvil: el suelo se mueve a U∞ como la carretera vista desde el coche.");
        U.combo("Resolución", &ui_preset, k_preset_items, 4);
        U.tooltip("Presupuesto de celdas del dominio. dx = raíz cúbica(volumen / celdas).");
        if (ui_preset != static_cast<int>(sim.cfg.preset) || sim.cfg.cells) {
            if (U.button("Aplicar resolución (reinicia el flujo)", ui::ButtonKind::Primary)) { want_reinit = true; want_res_apply = true; }
        }
        U.value("Malla", "%d × %d × %d", sim.dom.nx, sim.dom.ny, sim.dom.nz);
        U.value("Celdas", "%.2f M", static_cast<double>(sim.dom.cells()) * 1e-6);
        U.value("dx", "%.1f mm", static_cast<double>(sim.dom.dx * 1e3f));
        U.value("Memoria del solver", "%.0f MB", static_cast<double>(sim.solver.memory_bytes()) / 1048576.0);
        U.value("Bloqueo frontal", "%.1f %%", static_cast<double>(sim.dom.blockage * 100.0f));
        if (sim.dom.design_dx > 0.0f && sim.dom.dx > sim.dom.design_dx * 1.02f && (mask & (models::P_FrontFlap | models::P_RearFlap)))
            U.text_wrapped_colored(st.warn, "dx > %.0f mm de diseño: ranuras de alerón pueden cerrarse o fugar. Usa una resolución mayor.",
                           static_cast<double>(sim.dom.design_dx * 1e3f));
        int prec = sim.cfg.fp32 ? 1 : 0;
        U.label("Precisión de las poblaciones", st.text_dim);
        if (U.segmented("##precision", k_prec_names, 2, &prec)) { ui_fp32 = prec == 1; want_reinit = true; }
        U.tooltip("FP16S: 16 bits desplazados (mitad de memoria, ~1.8× más rápido). FP32: referencia.");
        {
            // Fase 3: backend iGPU (Vulkan de cómputo propio, sin bibliotecas: src/gpu, docs/GPU.md).
            static const char* const k_backend_names[2] = {"CPU", "iGPU"};
            int be = sim.gpu_on() ? 1 : 0;
            U.label("Solver", st.text_dim);
            if (U.segmented("##backend", k_backend_names, 2, &be)) { ui_gpu = be == 1; want_gpu = true; }
            U.tooltip("CPU: kernel AVX2 en los núcleos P+E. iGPU: el mismo algoritmo en la gráfica integrada (Vulkan de cómputo "
                      "con SPIR-V generado por el propio programa); la CPU queda libre para dibujar el cuadro anterior mientras la "
                      "GPU calcula el siguiente. Mismos resultados (FP32 ~1e-6; FP16S dentro de su redondeo).");
            if (!sim.gpu_error.empty()) U.text_wrapped_colored(st.warn, "iGPU no disponible: %s", sim.gpu_error.c_str());
        }
        float cs = sim.cfg.cs;
        if (U.slider_float("LES Cs", &cs, 0.0f, 0.3f, "%.2f")) { sim.cfg.cs = cs; sim.solver.set_smagorinsky(cs); }
        U.tooltip("Constante de Smagorinsky (modelo de turbulencia de subred). 0 = sin LES (menos estable).");
        // Física del solver: cambia la configuración del solver → se aplica con un reinicio completo.
        U.label("Física del solver (se aplica reiniciando)", st.text_dim);
        {
            static const char* const wall_names[3] = {"Sin ley", "Ley log.", "Deslizam."};
            int wm = clamp_(static_cast<int>(ui_wall), 0, 2);
            if (U.segmented("##pared", wall_names, 3, &wm)) ui_wall = static_cast<lbm::WallModel>(wm);
            U.tooltip("Modelo de pared junto a los sólidos. Sin ley: rebote no deslizante (capa límite sobre-espesa a esta "
                      "resolución). Ley log.: tensión de pared de una capa límite turbulenta al Reynolds real de la velocidad "
                      "elegida. Deslizam. (defecto): pared deslizante parcial + tensión de pared. Ver docs/FISICA.md.");
            U.toggle("Rebote interpolado (Bouzidi)", &ui_interp_bb);
            U.tooltip("La pared se sitúa en la superficie real (no en la escalera de vóxeles): menos resistencia espuria.");
            U.slider_float("Arranque (rampa)", &ui_ramp_ft, 0.0f, 2.0f, "%.2f", "PF");
            U.tooltip("u∞ sube desde el reposo en este número de pasos de flujo al reiniciar. 0 = arranque impulsivo "
                      "(defecto y recomendado: con rampa el túnel tarda más de 5 pasos de flujo en vaciar la sobrepresión "
                      "del arranque y todo sale rojo en Cp mientras tanto).");
            if (ui_wall != sim.cfg.wall || ui_interp_bb != sim.cfg.interp_bb || std::fabs(ui_ramp_ft - sim.cfg.ramp_ft) > 1e-4f) {
                if (U.button("Aplicar física (reinicia el flujo)", ui::ButtonKind::Primary)) want_reinit = true;
            }
        }
        U.value("u∞ de red / Mach", "%.3f / %.2f", static_cast<double>(sim.cfg.u_lat), static_cast<double>(sim.cfg.u_lat * std::sqrt(3.0f)));
        U.value("Viscosidad ν (red)", "%.1e", static_cast<double>(sim.nu));
        U.value("Re de la red (U·L/ν)", "%s", fmt_si(sim.re_lattice(), 2).s);
        U.value("Re real a esta velocidad", "%s", fmt_si(sim.re_real(), 2).s);
        U.tooltip("El Reynolds de la red es muy inferior al real (y el LES añade viscosidad turbulenta): los coeficientes son orientativos.");
        U.row(2);
        if (U.button(paused ? "Reanudar" : "Pausa", ui::ButtonKind::Primary)) paused = !paused;
        if (U.button("Reiniciar flujo", ui::ButtonKind::Danger)) { sim.reset_flow(); view.reset_particles(); U.toast(st.warn, "Flujo reiniciado"); }
        U.label("Prioridad", st.text_dim);
        int pz = static_cast<int>(prio);
        if (U.segmented("##prioridad", k_prio_names, 3, &pz)) prio = static_cast<Priority>(pz);
        U.tooltip("Fluidez ≈ 45 FPS · Equilibrado ≈ 30 FPS · Máx. simulación ≈ 12 FPS (más pasos de red por cuadro).");
    }

    // ---------------------------------------------------------------- Visualización
    if (U.header("Visualización", sec_open[SecVis])) {
        VisSettings& v = view.vs;
        VisSettings before = v;
        int sm = static_cast<int>(v.surf);
        if (U.combo("Superficie", &sm, k_surf_names, 6)) v.surf = static_cast<SurfMode>(sm);
        if (v.surf == SurfMode::Cp || v.surf == SurfMode::Voxels) {
            U.row(2);
            U.slider_float("Cp mín", &v.surf_lo, -6.0f, 0.0f, "%.1f");
            U.slider_float("Cp máx", &v.surf_hi, 0.2f, 1.5f, "%.1f");
        }
        {
            const float u = max_(sim.cfg.u_lat, 1e-4f);
            U.text_colored(st.text_dim, "Cp referido a la toma estática (Δ %+.3f frente a ρ = 1)",
                           static_cast<double>(2.0f * (sim.rho_ref - 1.0f) / (3.0f * u * u)));
            U.tooltip("Como en un túnel real, Cp y Cp0 (superficie, suelo, líneas, corte y sonda) se refieren a la presión "
                      "estática medida aguas arriba del objeto (ρ medio de un plano entre la entrada y el objeto), no a la "
                      "densidad inicial ρ = 1. El bloqueo del túnel sube algo la presión delante del objeto. Las fuerzas no "
                      "dependen de esta referencia.");
        }
        U.label("Plano de corte (X/Y/Z, flechas lo mueven)", st.text_dim);
        U.segmented("##corte", k_axis_names, 4, &v.slice_axis);
        if (v.slice_axis > 0) {
            const int ax = v.slice_axis - 1;
            const int n = ax == 0 ? sim.dom.nx : (ax == 1 ? sim.dom.ny : sim.dom.nz);
            char unit[48];
            const float pm = sim.dom.map.to_model(Vec3(v.slice_pos[0], v.slice_pos[1], v.slice_pos[2]))[ax];
            std::snprintf(unit, sizeof unit, "(%.2f m)", static_cast<double>(pm));
            U.slider_float("Posición", &v.slice_pos[ax], 0.0f, static_cast<float>(n - 1), "%.0f", unit);
            U.combo("Magnitud", &v.slice_q, k_slice_q_names, 7);
            U.tooltip("%s", flowvis::quantity_info(k_slice_q[clamp_(v.slice_q, 0, 6)]).description);
            U.toggle("Rango automático (1-99 %)", &v.slice_auto);
            if (!v.slice_auto) {
                const flowvis::ColorScale ds = flowvis::default_scale(k_slice_q[clamp_(v.slice_q, 0, 6)]);
                const float span = max_(std::fabs(ds.lo), std::fabs(ds.hi)) * 2.0f + 0.01f;
                U.row(2);
                U.slider_float("Mín##corte", &v.slice_lo, -span, span, "%.3g");
                U.slider_float("Máx##corte", &v.slice_hi, -span, span, "%.3g");
            }
            U.slider_float("Opacidad", &v.slice_opacity, 0.1f, 1.0f, "%.2f");
            U.checkbox("Vetas LIC (dirección del flujo)", &v.slice_lic);
        }
        U.separator();
        U.checkbox("Líneas de corriente  [L]", &v.lines);
        if (v.lines) {
            int rk = static_cast<int>(v.line_rake);
            if (U.combo("Rastrillo##lineas", &rk, k_rake_names, 4)) v.line_rake = static_cast<RakeKind>(rk);
            U.slider_int("Número", &v.line_count, 6, 160);
            U.checkbox("Color por Cp (si no, |u|)", &v.line_cp);
        }
        U.checkbox("Humo (partículas)  [P]", &v.smoke);
        if (v.smoke) {
            int rk = static_cast<int>(v.smoke_rake);
            if (U.combo("Emisor##humo", &rk, k_rake_names, 4)) v.smoke_rake = static_cast<RakeKind>(rk);
            U.slider_float("Densidad", &v.smoke_density, 0.0f, 1.0f, "%.2f");
            U.slider_float("Cámara rápida", &v.smoke_speed, 1.0f, 8.0f, "×%.1f");
            U.tooltip("El humo avanza más rápido que la simulación (sólo tiene sentido con flujo casi estacionario).");
        }
        U.checkbox("Vórtices (criterio Q)  [V]", &v.vortex);
        if (v.vortex) {
            float lq = std::log10(max_(v.q_threshold, 1e-3f));
            if (U.slider_float("Umbral Q·L²/U²", &lq, -0.5f, 3.0f, "%.2f", "log10")) v.q_threshold = std::pow(10.0f, lq);
            U.tooltip("Criterio Q (rotación − deformación) adimensionalizado con la longitud del objeto L y U∞: el mismo "
                      "umbral muestra los mismos vórtices en cualquier resolución. Más alto = sólo los núcleos más intensos.");
            U.checkbox("Nube volumétrica (si no, isosuperficies)", &v.vortex_cloud);
        }
        if (sim.cfg.ground != lbm::GroundMode::None) U.checkbox("Huella de Cp en el suelo", &v.footprint);
        U.row(2);
        U.checkbox("Flechas de fuerza", &v.arrows);
        U.checkbox("Por componente", &v.comp_arrows);
        U.row(2);
        U.checkbox("Túnel (alambre)", &v.wire);
        U.checkbox("Leyendas", &v.legends);
        U.row(2);
        U.checkbox("Oclusión (SSAO)", &v.ssao);
        U.checkbox("Antialias (FXAA)", &v.fxaa);
        U.checkbox("Sonda bajo el ratón", &v.probe);
        U.label("Cámara (teclas 1-6)", st.text_dim);
        for (int r = 0; r < 2; ++r) {
            U.row(3);
            for (int c = 0; c < 3; ++c) {
                const CamView cv = static_cast<CamView>(r * 3 + c);
                if (U.button(cam_view_name(cv))) { cam_view = cv; view.frame(sim, cv, viewport_rect()); }
            }
        }
        if (!(before == v)) view.dirty = true;
    }

    // ---------------------------------------------------------------- Resultados
    if (U.header("Resultados", sec_open[SecResultados])) {
        const AeroResult& r = sim.res;
        const Conv cv = conv_of(I);
        char b1[48], b2[48];
        // Sin resultados todavía (transitorio de arranque): "—" en vez de ceros engañosos.
        auto fmt_num = [&](char* b, usize n, float v, const char* f) {
            if (r.valid) std::snprintf(b, n, f, static_cast<double>(v));
            else std::snprintf(b, n, "—");
        };
        if (cv == Conv::Resistencia) {
            // Cuerpo en aire libre: la magnitud de interés es la resistencia.
            U.row(2);
            fmt_num(b1, sizeof b1, r.drag_kgf, "%.1f");
            U.metric("RESISTENCIA", b1, "kgf", st.series[1]);
            fmt_num(b2, sizeof b2, r.cd, "%.3f");
            U.metric("CD", b2, nullptr, st.series[1]);
            U.row(2);
            fmt_num(b1, sizeof b1, r.scx, "%.3f");
            U.metric("SCx (CD·A)", b1, "m²", st.series[1]);
            fmt_num(b2, sizeof b2, r.power_kw, "%.1f");
            U.metric("POTENCIA", b2, "kW", st.warn);
        } else {
            // Coches y alas: fuerza vertical "útil" (carga hacia abajo; sustentación en alas libres).
            const bool lift_conv = cv == Conv::Sustentacion;
            const float useful = lift_conv ? -r.down_kgf : r.down_kgf;   // > 0 = en el sentido deseado
            const char* vlabel = (r.down_kgf >= 0.0f) ? "CARGA" : "SUSTENTACIÓN";
            U.row(2);
            fmt_num(b1, sizeof b1, std::fabs(r.down_kgf), "%.0f");
            U.metric(vlabel, b1, "kgf", useful >= 0.0f ? st.series[0] : st.warn);
            fmt_num(b2, sizeof b2, r.drag_kgf, "%.0f");
            U.metric("RESISTENCIA", b2, "kgf", st.series[1]);
            U.row(2);
            if (lift_conv) {
                fmt_num(b1, sizeof b1, -r.cl, "%+.3f");
                U.metric("CL (sustentación +)", b1, nullptr, st.series[0]);
                fmt_num(b2, sizeof b2, r.cd, "%.3f");
                U.metric("CD", b2, nullptr, st.series[1]);
            } else {
                fmt_num(b1, sizeof b1, r.scz, "%+.3f");
                U.metric("SCz (CL·A, carga +)", b1, "m²", st.series[0]);
                fmt_num(b2, sizeof b2, r.scx, "%.3f");
                U.metric("SCx (CD·A)", b2, "m²", st.series[1]);
            }
            U.row(2);
            fmt_num(b1, sizeof b1, lift_conv ? -r.ld : r.ld, "%+.2f");
            U.metric("EFICIENCIA L/D", b1, nullptr, st.success);
            if (sim.is_car()) {
                if (balance_ok(r)) fmt_num(b2, sizeof b2, r.balance, "%.1f");
                else std::snprintf(b2, sizeof b2, "—");
                U.metric("BALANCE DEL.", b2, "%", balance_ok(r) && balance_odd(r) ? st.error : st.warn);
                U.tooltip("%% de la carga aerodinámica que va al eje delantero (equilibrio de momentos de cabeceo). "
                          "Sólo tiene sentido con carga neta hacia abajo. Fuera de 0-100 %% (en rojo) un eje recibe "
                          "sustentación: mira el desglose por componente y la convergencia.");
            } else if (lift_conv) {
                fmt_num(b2, sizeof b2, -r.scz, "%+.3f");
                U.metric("CL·A (sust. +)", b2, "m²", st.warn);
            } else {
                fmt_num(b2, sizeof b2, r.cl, "%+.3f");
                U.metric("CL (carga +)", b2, nullptr, st.warn);
            }
        }
        U.value("CL (carga + = hacia abajo)", "%+.4f", static_cast<double>(r.cl));
        U.value("CD", "%.4f", static_cast<double>(r.cd));
        U.value("CS (lateral, + = derecha)", "%+.4f", static_cast<double>(r.cs));
        U.value("Área de referencia", "%.3f m²", static_cast<double>(I.ref_area_m2));
        U.value("Fuerzas a", "%.0f km/h (½ρU² = %.0f Pa)", static_cast<double>(sim.speed_kmh),
                0.5 * static_cast<double>(k_rho_air) * static_cast<double>(sim.speed_kmh / 3.6f) * static_cast<double>(sim.speed_kmh / 3.6f));
        U.value("Vertical / resistencia", "%+.0f N / %.0f N", static_cast<double>(r.down_n), static_cast<double>(r.drag_n));
        U.value("Potencia absorbida", "%.1f kW", static_cast<double>(r.power_kw));
        // Convergencia
        char cb[128];
        const float ftn = sim.flow_throughs();
        if (sim.converged) std::snprintf(cb, sizeof cb, "Convergido · Δ %.1f %% · %.1f pasos de flujo", static_cast<double>(sim.conv_rel * 100), static_cast<double>(ftn));
        else std::snprintf(cb, sizeof cb, "Convergiendo… Δ %.1f %% · %.1f pasos de flujo", static_cast<double>(min_(sim.conv_rel, 9.99f) * 100), static_cast<double>(ftn));
        const float conv = sim.converged ? 1.0f : clamp_(1.0f - std::log10(max_(sim.conv_rel / 0.015f, 1.0f)) / 1.5f, 0.02f, 0.97f);
        U.progress(conv, cb, sim.converged ? st.success : st.accent);
        U.tooltip("Media exponencial con constante de ½ paso de flujo (L/U∞). Convergido: cambio de CL y CD < 1.5 %% en el último paso de flujo.");
        const History& h = sim.hist;
        const ui::PlotSeries ser[2] = {{h.scz, h.count, h.oldest(), st.series[0], "SCz"}, {h.scx, h.count, h.oldest(), st.series[1], "SCx"}};
        U.plot_lines("Historia (filtrada)", ser, 2, static_cast<int>(130 * st.scale), "m²", "%.3f");
        // Desglose por componente
        ui::Bar bars[static_cast<int>(sdf::Component::Count)];
        const int nb = min_(r.ncomp, static_cast<int>(sdf::Component::Count));
        if (nb > 1 || cv != Conv::Resistencia) {
            for (int i = 0; i < nb; ++i) bars[i] = {r.comp[i].name, r.comp[i].scz, render::component_color(r.comp[i].comp)};
            if (nb > 0) U.bar_chart("Carga por componente (+ = hacia abajo)", bars, nb, "SCz [m²]", "%+.3f");
            for (int i = 0; i < nb; ++i) bars[i] = {r.comp[i].name, r.comp[i].scx, render::component_color(r.comp[i].comp)};
            if (nb > 0) U.bar_chart("Resistencia por componente", bars, nb, "SCx [m²]", "%+.3f");
        }
        // Comparación con los valores reales aproximados del catálogo (misma convención: carga +).
        if (I.ref_ClA != 0.0f || I.ref_CdA != 0.0f) {
            U.separator();
            U.value_colored("Real aprox. SCz / SCx", st.text_dim, "%+.2f / %.2f m²", static_cast<double>(I.ref_ClA), static_cast<double>(I.ref_CdA));
            U.tooltip("%s. Referencia de orden de magnitud: a estas resoluciones la carga de un F1 sale muy por debajo de la real y la resistencia por encima (tabla de calibración y límites en docs/FISICA.md, secciones 5-6).", I.ref_source);
            char rz[24] = "—", rx[24] = "—";
            if (std::fabs(I.ref_ClA) > 0.05f) std::snprintf(rz, sizeof rz, "%.0f %%", 100.0 * static_cast<double>(r.scz / I.ref_ClA));
            if (I.ref_CdA > 0.005f) std::snprintf(rx, sizeof rx, "%.0f %%", 100.0 * static_cast<double>(r.scx / I.ref_CdA));
            U.value_colored("Simulado / real (SCz · SCx)", st.text_dim, "%s · %s", rz, rx);
        }
        U.text_wrapped("Re de la red %s frente a %s real: los números son coeficientes (F / ½ρU²A) orientativos; compara tendencias.",
                       fmt_si(sim.re_lattice(), 2).s, fmt_si(sim.re_real(), 2).s);
        if (U.button("Exportar historia y componentes (CSV)", ui::ButtonKind::Subtle)) want_csv = true;
    }

    // ---------------------------------------------------------------- Barrido
    if (U.header("Barrido (polares)", sec_open[SecBarrido])) {
        int avail[k_nsweep], na = 0, cur = -1;
        const char* anames[k_nsweep];
        for (int i = 0; i < k_nsweep; ++i)
            if (sweep_param_available(sim.cfg.model, static_cast<SweepParam>(i))) {
                if (static_cast<int>(sweep.param) == i) cur = na;
                anames[na] = sweep_param_name(static_cast<SweepParam>(i));
                avail[na++] = i;
            }
        if (na == 0) {
            U.text_colored(st.text_dim, "Este modelo no tiene parámetros que barrer.");
        } else {
            if (cur < 0) { cur = 0; sweep.param = static_cast<SweepParam>(avail[0]); sweep_defaults(); }
            const bool busy = sweep.running;
            if (!busy && U.combo("Parámetro", &cur, anames, na)) { sweep.param = static_cast<SweepParam>(avail[cur]); sweep_defaults(); }
            if (busy) U.value("Parámetro", "%s", sweep_param_name(sweep.param));
            const ParamRanges R = param_ranges(sim.cfg.model);
            Range rr{};
            switch (sweep.param) {
                case SweepParam::Aoa: rr = R.aoa; break;
                case SweepParam::Height: rr = R.height; break;
                case SweepParam::RideHeight: rr = R.ride_front; break;
                case SweepParam::FrontFlap: case SweepParam::RearFlap: rr = R.flap; break;
                case SweepParam::Yaw: rr = R.yaw; break;
                case SweepParam::FlapGap: rr = R.gap; break;
                default: break;
            }
            const char* un = sweep_param_unit(sweep.param);
            if (!busy) {
                U.row(2);
                U.slider_float("Desde", &sweep.from, rr.lo, rr.hi, "%.1f", un);
                U.slider_float("Hasta", &sweep.to, rr.lo, rr.hi, "%.1f", un);
                U.slider_int("Puntos", &sweep.n, 2, 24);
                U.row(2);
                U.slider_float("Asentar", &sweep.settle_ft, 0.3f, 4.0f, "%.1f", "PF");
                U.slider_float("Promediar", &sweep.avg_ft, 0.3f, 4.0f, "%.1f", "PF");
                U.tooltip("PF = pasos de flujo (L/U∞) por punto: primero se deja asentar el flujo y luego se promedian las fuerzas.");
            }
            // Barrido de altura de marcha por debajo del hueco mínimo resoluble: todos los puntos se simulan casi a la
            // misma altura efectiva ((h⁴+g⁴)^¼ ≈ g), así que las diferencias entre ellos son ruido (revisión).
            if (sweep.param == SweepParam::RideHeight && sim.ride_gap_min_mm > 0.0f &&
                max_(sweep.from, sweep.to) < 1.2f * sim.ride_gap_min_mm)
                U.text_wrapped_colored(st.warn, "Todo el barrido queda por debajo del hueco mínimo de esta resolución (%.0f mm): "
                                       "los puntos se simulan a casi la misma altura efectiva y sus diferencias son ruido. "
                                       "Sube la resolución o usa el ala en efecto suelo.", static_cast<double>(sim.ride_gap_min_mm));
            const double sps = perf.mlups.v > 1 ? perf.mlups.v * 1e6 / static_cast<double>(max_(sim.dom.cells(), usize(1))) : 100.0;
            const double eta = static_cast<double>(sweep.n) * (sweep.settle_ft + sweep.avg_ft) * sim.ft_steps() / sps *
                               (busy ? 1.0 - static_cast<double>(sweep.progress()) : 1.0);
            if (!busy) {
                U.text_colored(st.text_dim, "Duración estimada ≈ %.0f s (%.0f pasos de red por punto)", eta,
                               static_cast<double>((sweep.settle_ft + sweep.avg_ft) * sim.ft_steps()));
                if (U.button("Iniciar barrido", ui::ButtonKind::Primary)) want_sweep_start = true;
            } else {
                char pb[96];
                std::snprintf(pb, sizeof pb, "Punto %d/%d · %s · quedan ≈ %.0f s", sweep.idx + 1, sweep.n, sweep.averaging ? "promediando" : "asentando", eta);
                U.progress(sweep.progress(), pb);
                if (U.button("Detener barrido", ui::ButtonKind::Danger)) { sweep_stop(sweep, sim); sync_ui_from_sim(); }
            }
            if (sweep.npts > 0) {
                const int nplots = sim.is_car() ? 4 : 3;   // el balance sólo en coches
                if (g_sweep_plot >= nplots) g_sweep_plot = 0;
                U.segmented("##grafica_barrido", k_plot_names, nplots, &g_sweep_plot);
                draw_sweep_plot();
                if (U.button("Exportar barrido (CSV)", ui::ButtonKind::Subtle)) want_sweep_csv = true;
            }
        }
    }

    // ---------------------------------------------------------------- Rendimiento
    if (U.header("Rendimiento", sec_open[SecRendimiento])) {
        U.value("FPS", "%.1f", perf.fps.v);
        U.value("MLUPS (solver)", "%.0f%s", perf.mlups.v, sim.gpu_on() ? " (iGPU)" : "");
        if (sim.gpu_on()) {
            const gpu::LbmGpuStats& gs = sim.gpu->stats();
            U.value("iGPU: lote", "%d pasos · %.1f ms de GPU · %.1f GB/s", gs.steps, gs.gpu_ms, gs.gbs);
            U.value("iGPU: publicar", "%.2f ms (fuerzas + campos)", gs.publish_ms);
            U.tooltip("La CPU dibuja el campo del lote anterior mientras la GPU calcula el siguiente; \"Simulación\" es sólo la "
                      "espera al final del cuadro (0 si la GPU terminó antes que el dibujo).");
        }
        U.value("Pasos de red por cuadro", "%d", perf.steps_per_frame);
        U.value("Pasos por segundo", "%.0f", perf.fps.v * perf.steps_per_frame);
        U.value("Simulación", "%.2f ms", perf.sim_ms.v);
        U.value("Visualización (cálculo)", "%.2f ms", perf.vis_ms.v);
        U.value("Render 3D", "%.2f ms", perf.render_ms.v);
        U.text_colored(st.text_dim, "  malla %.2f · flujo %.2f · post %.2f ms", view.t_mesh, view.t_vis, view.t_post);
        U.value("UI", "%.2f ms", perf.ui_ms.v);
        U.value("Presentación", "%.2f ms", perf.present_ms.v);
        U.value("Cuadro", "%.2f ms", perf.frame_ms.v);
        {
            const ui::PlotSeries ps[2] = {{perf.h_frame, perf.h_count, perf.h_oldest(), st.series[2], "cuadro"},
                                          {perf.h_sim, perf.h_count, perf.h_oldest(), st.series[0], "simulación"}};
            U.plot_lines("Tiempo por cuadro", ps, 2, static_cast<int>(90 * st.scale), "ms", "%.1f");
        }
        U.value("Visualización con campo nuevo", "cada %.0f ms (prioridad)", view.field_interval * 1e3);
        U.text_wrapped_colored(st.text_dim, "Actualización: muestreo %.2f · líneas %.2f · humo %.2f · vórtices %.2f · corte %.2f · color %.2f ms",
                       view.t_upd_sampler, view.t_upd_lines, view.t_upd_smoke, view.t_upd_vortex, view.t_upd_slice, view.t_upd_color);
        U.value("Celdas", "%.2f M (%.1f %% sólidas)", static_cast<double>(sim.dom.cells()) * 1e-6,
                100.0 * static_cast<double>(sim.solid_cells) / static_cast<double>(max_(sim.dom.cells(), usize(1))));
        U.value("Memoria (solver + malla)", "%.0f MB", static_cast<double>(sim.memory_bytes()) / 1048576.0);
        U.value("Triángulos (malla suave)", "%zu", sim.mesh.tri_count());
        if (view.vs.smoke) U.value("Partículas vivas", "%zu", view.smoke.stats().alive);
        const CpuTopology& tp = topo;
        U.value("Hilos del pool", "%d (P %d · E %d · LP-E %d)", pool().size(), tp.n_pcores, tp.n_ecores, tp.n_lpe);
        {
            // Carga del sistema (/proc/loadavg, 1 vez por segundo con pread sobre un descriptor abierto una
            // vez: sin asignaciones): con otros procesos pesados las cifras de arriba se degradan.
            static int fd = -2;
            static double t_last = -10.0;
            static float load1 = -1.0f;
            if (fd == -2) fd = ::open("/proc/loadavg", O_RDONLY | O_CLOEXEC);
            const double t = now_sec();
            if (fd >= 0 && t - t_last > 1.0) {
                char lb[64] = {};
                if (::pread(fd, lb, sizeof lb - 1, 0) > 0) load1 = std::strtof(lb, nullptr);
                t_last = t;
            }
            if (load1 >= 0.0f)
                U.value_colored("Carga del sistema (1 min)", load1 > 0.5f * static_cast<float>(tp.n_primary) ? st.warn : st.text_dim, "%.1f",
                                static_cast<double>(load1));
            U.tooltip("Media de procesos ejecutables del sistema. Con otros programas pesados en marcha (compilaciones, "
                      "máquinas virtuales) el solver pierde ancho de banda de memoria y los tiempos suben.");
        }
        U.value("Reconstrucción", "vóx %.1f · solver %.1f · malla %.1f ms", sim.times.vox, sim.times.geo, sim.times.mesh);
        U.value("Reinicio completo", "%.0f ms", sim.t_init_ms);
        if (!win->is_headless()) {
            const platform::PresentStats ps = win->present_stats();
            U.value("Ventana", "%d×%d (escala %d, %s)", ps.win_w, ps.win_h, win->pixel_scale(), ps.shm ? "MIT-SHM" : "XPutImage");
        }
    }
    U.end_panel();
}

// Gráfica X-Y del barrido (dibujo propio en la capa del panel).
void App::draw_sweep_plot() {
    ui::Context& U = ui;
    const ui::Style& st = U.style();
    const int H = static_cast<int>(170 * st.scale);
    const Rect r = U.next_rect(H);
    render::DrawList& dl = U.draw();
    dl.fill_round_rect(r, static_cast<float>(st.radius), st.plot_bg);
    const int n = sweep.npts;
    if (n <= 0) return;
    float xs[Sweep::k_max], y1[Sweep::k_max], y2[Sweep::k_max];
    bool two = true;
    const char* n1 = "CL";
    const char* n2 = "CD";
    // Alas en aire libre: sustentación positiva hacia arriba (el resto: carga positiva hacia abajo).
    const bool lift = conv_of(sim.built.info) == Conv::Sustentacion;
    const float sg = lift ? -1.0f : 1.0f;
    for (int i = 0; i < n; ++i) {
        const SweepPoint& p = sweep.pts[i];
        xs[i] = p.x;
        switch (g_sweep_plot) {
            case 0: y1[i] = sg * p.cl; y2[i] = p.cd; if (lift) n1 = "CL (sust.)"; break;
            case 1: y1[i] = sg * p.scz; y2[i] = p.scx; n1 = lift ? "CL·A" : "SCz"; n2 = "SCx"; break;
            case 2: y1[i] = sg * p.ld; y2[i] = 0; two = false; n1 = "L/D"; break;
            default: y1[i] = std::isfinite(p.bal) ? p.bal : 0.0f; y2[i] = 0; two = false; n1 = "Balance %"; break;
        }
    }
    float x0 = min_(sweep.from, sweep.to), x1 = max_(sweep.from, sweep.to);
    if (x1 - x0 < 1e-6f) x1 = x0 + 1.0f;
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < n; ++i) {
        lo = min_(lo, y1[i]); hi = max_(hi, y1[i]);
        if (two) { lo = min_(lo, y2[i]); hi = max_(hi, y2[i]); }
    }
    if (g_sweep_plot < 2) lo = min_(lo, 0.0f);
    if (hi - lo < 1e-6f) { hi += 0.5f; lo -= 0.5f; }
    const float pad = 0.08f * (hi - lo);
    lo -= pad; hi += pad;
    const int fw = st.fw();
    const Rect a{r.x + fw * 7, r.y + 8, r.w - fw * 7 - 10, r.h - 8 - st.text_h - 6};
    // Rejilla y etiquetas (5 divisiones)
    char b[32];
    for (int k = 0; k <= 4; ++k) {
        const float t = static_cast<float>(k) / 4.0f;
        const int y = a.y + a.h - static_cast<int>(t * static_cast<float>(a.h));
        dl.fill_rect({a.x, y, a.w, 1}, st.grid);
        std::snprintf(b, sizeof b, "%.3g", static_cast<double>(lo + t * (hi - lo)));
        dl.text(r.x + 4, y - render::font_center(st.font), b, st.text_dim, st.font);
        const int x = a.x + static_cast<int>(t * static_cast<float>(a.w));
        dl.fill_rect({x, a.y, 1, a.h}, st.grid);
        std::snprintf(b, sizeof b, "%.3g", static_cast<double>(x0 + t * (x1 - x0)));
        const int tw = render::text_width(b, st.font);
        dl.text(clamp_(x - tw / 2, a.x, a.x + a.w - tw), a.y + a.h + 4, b, st.text_dim, st.font);
    }
    if (lo < 0 && hi > 0) {
        const int y0 = a.y + a.h - static_cast<int>((0 - lo) / (hi - lo) * static_cast<float>(a.h));
        dl.fill_rect({a.x, y0, a.w, 1}, 0xFF4A5261u);
    }
    auto plot = [&](const float* ys, u32 col) {
        Vec2 pts[Sweep::k_max];
        for (int i = 0; i < n; ++i)
            pts[i] = {static_cast<float>(a.x) + (xs[i] - x0) / (x1 - x0) * static_cast<float>(a.w),
                      static_cast<float>(a.y + a.h) - (ys[i] - lo) / (hi - lo) * static_cast<float>(a.h)};
        if (n > 1) dl.polyline(pts, n, col, 2.0f);
        for (int i = 0; i < n; ++i) dl.fill_circle(pts[i].x, pts[i].y, 3.5f, col);
    };
    plot(y1, st.series[0]);
    if (two) plot(y2, st.series[1]);
    // Leyenda
    int lx = a.x + 8;
    dl.fill_rect({lx, a.y + 8, 10, 3}, st.series[0]);
    lx = dl.text(lx + 14, a.y + 2, n1, st.text, st.font) + 12;
    if (two) { dl.fill_rect({lx, a.y + 8, 10, 3}, st.series[1]); dl.text(lx + 14, a.y + 2, n2, st.text, st.font); }
    char xl[64];
    std::snprintf(xl, sizeof xl, "%s [%s]", sweep_param_name(sweep.param), sweep_param_unit(sweep.param));
    const int tw = render::text_width(xl, st.font);
    dl.text(a.x + a.w - tw, a.y + 2, xl, st.text_dim, st.font);
}

// ============================================================================
//  Superposiciones del visor: HUD, leyendas, sonda, barra de estado, ayuda
// ============================================================================
namespace {

void panel_box(render::DrawList& dl, Rect r, u32 bg = 0xB8101318u) {
    dl.fill_round_rect(r, 6.0f, bg);
    dl.round_rect(r, 6.0f, 0x40FFFFFFu, 1);
}

// Barra de colores horizontal con marcas lo / 0 (o centro) / hi.
void legend(render::DrawList& dl, int x, int y, int w, const char* title, const flowvis::ColorScale& s, const char* unit) {
    const int h = 12;
    panel_box(dl, {x - 10, y - 26, w + 20, h + 52});
    dl.text_shadow(x, y - 20, title, 0xFFE6EAF0u);
    if (unit) { const int uw = render::text_width(unit); dl.text_shadow(x + w - uw, y - 20, unit, 0xFF9AA3B2u); }
    dl.blit_scaled(render::colormap_lut(s.map), 256, 1, 256, {x, y, w, h}, false);
    dl.rect({x, y, w, h}, 0x80000000u, 1);
    char b0[24], b1[24], b2[24];
    std::snprintf(b0, sizeof b0, "%.3g", static_cast<double>(s.lo));
    std::snprintf(b2, sizeof b2, "%.3g", static_cast<double>(s.hi));
    const float mid = s.pivoted() ? 0.0f : 0.5f * (s.lo + s.hi);
    std::snprintf(b1, sizeof b1, "%.3g", static_cast<double>(mid));
    for (int k = 0; k <= 2; ++k) dl.fill_rect({x + (w - 1) * k / 2, y + h, 1, 4}, 0xFFB0B8C4u);
    dl.text_shadow(x, y + h + 5, b0, 0xFFD0D6DEu);
    dl.text_shadow(x + (w - render::text_width(b1)) / 2, y + h + 5, b1, 0xFFD0D6DEu);
    dl.text_shadow(x + w - render::text_width(b2), y + h + 5, b2, 0xFFD0D6DEu);
}

} // namespace

void App::draw_overlay(Rect vp) {
    render::DrawList& dl = overlay;
    dl.clear();
    dl.set_base_clip({0, 0, fb.w, fb.h});
    dl.push_clip(vp);
    const models::Info& I = sim.built.info;
    const AeroResult& r = sim.res;
    char b[256];
    // ---- HUD (arriba a la izquierda): líneas en búferes de pila, caja ajustada al texto.
    {
        constexpr int kMax = 9;
        char L[kMax][200];
        u32 C[kMax];
        int n = 0;
        auto line = [&](u32 col, const char* f, auto... a) {
            if (n < kMax) { std::snprintf(L[n], sizeof L[n], f, a...); C[n++] = col; }
        };
        line(0xFFF4F6FAu, "%s", I.name.c_str());
        const Conv cv = conv_of(I);
        if (r.valid) {
            const double dk = std::fabs(static_cast<double>(r.down_kgf));
            if (cv == Conv::Resistencia) {
                line(0xFFE0E6EEu, "Resistencia %.1f kgf · CD %.3f · SCx %.3f m²  @ %.0f km/h", static_cast<double>(r.drag_kgf),
                     static_cast<double>(r.cd), static_cast<double>(r.scx), static_cast<double>(sim.speed_kmh));
                line(0xFF8FC8FFu, "CL %+.3f (carga +) · CS %+.3f · potencia %.1f kW", static_cast<double>(r.cl), static_cast<double>(r.cs),
                     static_cast<double>(r.power_kw));
            } else if (cv == Conv::Sustentacion) {
                line(r.down_kgf <= 0 ? 0xFFE0E6EEu : 0xFFFFB547u, "%s %.0f kgf · Resistencia %.0f kgf · L/D %.2f  @ %.0f km/h",
                     r.down_kgf <= 0 ? "Sustentación" : "Carga (hacia abajo)", dk, static_cast<double>(r.drag_kgf), -static_cast<double>(r.ld),
                     static_cast<double>(sim.speed_kmh));
                line(0xFF8FC8FFu, "CL %+.3f (sustentación +) · CD %.4f · CL·A %+.3f m²", -static_cast<double>(r.cl), static_cast<double>(r.cd),
                     -static_cast<double>(r.scz));
            } else {
                line(r.down_kgf >= 0 ? 0xFFE0E6EEu : 0xFFFFB547u, "%s %.0f kgf · Resistencia %.0f kgf · L/D %.2f  @ %.0f km/h",
                     r.down_kgf >= 0 ? "Carga" : "Sustentación", dk, static_cast<double>(r.drag_kgf), static_cast<double>(r.ld),
                     static_cast<double>(sim.speed_kmh));
                if (sim.is_car() && balance_ok(r))
                    line(balance_odd(r) ? 0xFFFFB547u : 0xFF8FC8FFu, "SCz %+.3f m² · SCx %.3f m² · balance %.1f %% delante%s", static_cast<double>(r.scz),
                         static_cast<double>(r.scx), static_cast<double>(r.balance), balance_odd(r) ? " (fuera de la batalla)" : "");
                else
                    line(0xFF8FC8FFu, "SCz %+.3f m² · SCx %.3f m² · CL %+.3f · CD %.3f", static_cast<double>(r.scz), static_cast<double>(r.scx),
                         static_cast<double>(r.cl), static_cast<double>(r.cd));
            }
        } else {
            line(0xFFE0E6EEu, "%s", sim.solver.steps() > 0 ? "Arrancando el flujo: ondas de presión del arranque (~1 paso de flujo; no se promedian)…" : "Flujo en reposo");
        }
        const int conv_line = n;
        line(0xFFD0D6DEu, "%s (Δ %.1f %%) · %.2f pasos de flujo · paso %llu", sim.converged ? "Convergido" : "Convergiendo",
             static_cast<double>(min_(sim.conv_rel, 9.99f) * 100.0f), static_cast<double>(sim.flow_throughs()),
             static_cast<unsigned long long>(sim.solver.steps()));
        line(0xFF9AA3B2u, "Re red %s (real %s) · dx %.1f mm · suelo: %s", fmt_si(sim.re_lattice(), 2).s, fmt_si(sim.re_real(), 2).s,
             static_cast<double>(sim.dom.dx * 1e3f), k_ground_names[static_cast<int>(sim.cfg.ground)]);
        if (sim.is_car() && sim.ride_limited)
            line(0xFFFFB547u, "Altura efectiva (resolución): %.0f / %.0f mm (pedida %.0f / %.0f)", static_cast<double>(sim.ride_eff_front_mm),
                 static_cast<double>(sim.ride_eff_rear_mm), static_cast<double>(sim.params.ride_front_mm), static_cast<double>(sim.params.ride_rear_mm));
        if (sim.divergences > 0)
            line(0xFFFF7A59u, "%d divergencia%s: ν subida a %.1e (se restablece al cambiar de modelo o de resolución)",
                 sim.divergences, sim.divergences > 1 ? "s" : "", static_cast<double>(sim.nu));
        if (sweep.running)
            line(0xFF8FC8FFu, "Barrido: punto %d/%d (%s = %.1f %s)", sweep.idx + 1, sweep.n, sweep_param_key(sweep.param),
                 static_cast<double>(sweep.value_at(sweep.idx)), sweep_param_unit(sweep.param));
        else if (paused) line(0xFFFFB547u, "%s", "EN PAUSA (Espacio para reanudar) · F1 ayuda");
        else line(0xFF7F8898u, "%s", "F1 ayuda · 1-6 vistas · H oculta la interfaz");
        const int x = vp.x + 14, y0 = vp.y + 12, lh = 18;
        int w = 0;
        for (int i = 0; i < n; ++i) w = max_(w, render::text_width(L[i]) + (i == conv_line ? 16 : 0));
        panel_box(dl, {x - 8, y0 - 6, min_(w + 18, vp.w - 12), n * lh + 12});
        int y = y0;
        for (int i = 0; i < n; ++i, y += lh) {
            if (i == conv_line) {
                dl.fill_circle(static_cast<float>(x + 5), static_cast<float>(y + 8), 4.5f, sim.converged ? 0xFF3DDC97u : 0xFFFFB547u);
                dl.text_shadow(x + 16, y, L[i], C[i]);
            } else {
                dl.text_shadow(x, y, L[i], C[i]);
            }
        }
    }
    // ---- Leyendas (abajo a la izquierda)
    if (view.vs.legends) {
        const int w = 240;
        int x = vp.x + 22;
        const int y = vp.y + vp.h - 40 - 30;
        auto next = [&] { x += w + 34; };
        const VisSettings& v = view.vs;
        if (v.surf == SurfMode::Cp || v.surf == SurfMode::Voxels) {
            legend(dl, x, y, w, "Superficie: Cp", {render::Colormap::CoolWarm, v.surf_lo, v.surf_hi, true}, nullptr);
            next();
        } else if (v.surf == SurfMode::Speed) {
            legend(dl, x, y, w, "Superficie: |u|/U∞", {render::Colormap::Turbo, 0.0f, 1.5f, false}, nullptr);
            next();
        } else if (v.surf == SurfMode::Component) {
            // Muestras de color de los componentes presentes en la escena (crece hacia arriba desde la
            // fila de leyendas). No depende de que haya resultados todavía.
            constexpr int kNC = static_cast<int>(sdf::Component::Count);
            bool present[kNC] = {};
            for (const auto& g : sim.built.scene.groups()) present[clamp_(static_cast<int>(g.component), 0, kNC - 1)] = true;
            int rows = 0;
            for (int c = 0; c < kNC; ++c) rows += present[c];
            const int rh = 18, bottom = y + 12 + 26, top = bottom - (rows * rh + 34);
            panel_box(dl, {x - 10, top, w + 20, bottom - top});
            dl.text_shadow(x, top + 6, "Superficie: componente", 0xFFE6EAF0u);
            int i = 0;
            for (int c = 0; c < kNC; ++c) {
                if (!present[c]) continue;
                const int yy = top + 28 + (i++) * rh;
                dl.fill_round_rect({x, yy + 2, 14, 12}, 3.0f, render::component_color(static_cast<sdf::Component>(c)));
                dl.text_shadow(x + 22, yy, sdf::component_name(static_cast<sdf::Component>(c)), 0xFFD0D6DEu);
            }
            next();
        }
        if (v.slice_axis > 0 && view.slice.tex_w() > 0) {
            const flowvis::QuantityInfo& qi = flowvis::quantity_info(k_slice_q[clamp_(v.slice_q, 0, 6)]);
            std::snprintf(b, sizeof b, "Corte %s: %s", k_axis_names[v.slice_axis], qi.short_name);
            legend(dl, x, y, w, b, view.slice.effective_scale(), nullptr);
            next();
        }
        if (v.lines && x + w < vp.x + vp.w - 20) {
            legend(dl, x, y, w, v.line_cp ? "Líneas: Cp" : "Líneas: |u|/U∞",
                   v.line_cp ? flowvis::ColorScale{render::Colormap::CoolWarm, -2.0f, 1.0f, true} : flowvis::ColorScale{render::Colormap::Turbo, 0.0f, 1.5f, false}, nullptr);
            next();
        }
        if (v.smoke && x + w < vp.x + vp.w - 20) {
            legend(dl, x, y, w, "Humo: |u|/U∞", {render::Colormap::Turbo, 0.0f, 1.5f, false}, nullptr);
            next();
        }
        if (sim.cfg.ground != lbm::GroundMode::None && v.footprint && x + w < vp.x + vp.w - 20) {
            legend(dl, x, y, w, "Suelo: Cp", {render::Colormap::CoolWarm, -2.0f, 1.0f, true}, nullptr);
            next();
        }
        if (v.vortex && x + w < vp.x + vp.w - 20) {
            legend(dl, x, y, w, "Vórtices Q: color u_x/U∞", view.vortex.params.color_scale, nullptr);
            next();
        }
    }
    // ---- Sonda
    if (view.probe_ok && !ui.wants_mouse() && vp.contains(in.mouse_x, in.mouse_y)) {
        const flowvis::Probe& p = view.probe;
        const int x = in.mouse_x + 18, y = in.mouse_y + 16;
        const Vec3 pm = sim.dom.map.to_model(p.pos);
        char l1[128], l2[128], l3[128];
        if (view.probe_on_slice) {
            const flowvis::QuantityInfo& qi = flowvis::quantity_info(k_slice_q[clamp_(view.vs.slice_q, 0, 6)]);
            std::snprintf(l1, sizeof l1, "%s = %.3f", qi.short_name, static_cast<double>(view.probe_value));
        } else {
            std::snprintf(l1, sizeof l1, "Cp = %.3f", static_cast<double>(p.cp));
        }
        std::snprintf(l2, sizeof l2, "|u|/U∞ %.3f · Cp %.3f · Cp0 %.3f", static_cast<double>(p.speed), static_cast<double>(p.cp), static_cast<double>(p.cp0));
        std::snprintf(l3, sizeof l3, "x %.2f  y %.2f  z %.2f m%s", static_cast<double>(pm.x), static_cast<double>(pm.y), static_cast<double>(pm.z),
                      p.solid ? " (sólido)" : "");
        const int w = 8 * static_cast<int>(max_(render::utf8_count(l2), max_(render::utf8_count(l1), render::utf8_count(l3)))) + 16;
        const Rect box{min_(x, vp.x + vp.w - w - 4), min_(y, vp.y + vp.h - 70), w, 62};
        panel_box(dl, box, 0xE0181B22u);
        dl.text(box.x + 8, box.y + 6, l1, 0xFFF4F6FAu);
        dl.text(box.x + 8, box.y + 24, l2, 0xFFC8D0DAu);
        dl.text(box.x + 8, box.y + 42, l3, 0xFF8B93A2u);
    }
    // ---- Barra de estado
    {
        const int h = 22;
        const Rect sb{vp.x, vp.y + vp.h - h, vp.w, h};
        dl.fill_rect(sb, 0xD00C0E12u);
        dl.fill_rect({sb.x, sb.y, sb.w, 1}, 0xFF262A32u);
        // Texto largo, o compacto si no cabe junto al de la derecha (visor estrecho).
        char rb[160];
        std::snprintf(rb, sizeof rb, "%.0f FPS · %d pasos/cuadro · %.0f MLUPS · sim %.1f · render %.1f ms", perf.fps.v, perf.steps_per_frame,
                      perf.mlups.v, perf.sim_ms.v, perf.render_ms.v);
        int rw = render::text_width(rb);
        std::snprintf(b, sizeof b, "%s  %s · %.2f M celdas (%d×%d×%d) · dx %.1f mm · %s · ν %.1e · Cs %.2f · %s",
                      paused ? "PAUSA" : "▶", preset_name(sim.cfg.preset), static_cast<double>(sim.dom.cells()) * 1e-6, sim.dom.nx, sim.dom.ny,
                      sim.dom.nz, static_cast<double>(sim.dom.dx * 1e3f), sim.cfg.fp32 ? "FP32" : "FP16S", static_cast<double>(sim.nu),
                      static_cast<double>(sim.cfg.cs), cam_view_name(cam_view));
        if (render::text_width(b) + rw + 40 > sb.w) {
            std::snprintf(b, sizeof b, "%s  %s · %.2f M · dx %.1f mm · %s", paused ? "PAUSA" : "▶", preset_name(sim.cfg.preset),
                          static_cast<double>(sim.dom.cells()) * 1e-6, static_cast<double>(sim.dom.dx * 1e3f), sim.cfg.fp32 ? "FP32" : "FP16S");
            std::snprintf(rb, sizeof rb, "%.0f FPS · %d pasos · %.0f MLUPS", perf.fps.v, perf.steps_per_frame, perf.mlups.v);
            rw = render::text_width(rb);
        }
        dl.text(sb.x + 10, sb.y + 3, b, paused ? 0xFFFFB547u : 0xFFB8C0CCu);
        if (render::text_width(b) + rw + 40 <= sb.w) dl.text(sb.x + sb.w - rw - 10, sb.y + 3, rb, 0xFF8FC8FFu);
    }
    // ---- Ayuda (F1)
    if (show_help) {
        static const char* const keys[][2] = {
            {"Ratón izquierdo", "orbitar la cámara"},
            {"Ratón derecho / central", "desplazar la cámara"},
            {"Rueda", "zoom"},
            {"Doble clic", "centrar la cámara en el punto"},
            {"1 … 6", "vistas: lateral, superior, frontal, trasera, 3/4, bajo el coche"},
            {"Espacio", "pausa / reanudar"},
            {"R", "reiniciar el flujo"},
            {"G", "ciclar suelo: ninguno → fijo → cinta"},
            {"D", "DRS / modo X"},
            {"C", "ciclar color de la superficie"},
            {"X / Y / Z", "plano de corte (repetir = quitar)"},
            {"Flechas", "mover el corte (Mayús = ×10)"},
            {"L / P / V", "líneas de corriente / humo / vórtices"},
            {"H", "ocultar la interfaz"},
            {"F12 o S", "captura PNG en ./capturas/"},
            {"F11", "pantalla completa"},
            {"Tab", "recorrer los controles del panel"},
            {"Esc", "cerrar ayuda / desplegable; dos veces = salir"},
        };
        const int n = static_cast<int>(sizeof keys / sizeof keys[0]);
        static const char* const notes[] = {
            "Coeficientes: F / (½ ρ U² A_ref). Carga = −Fz (+ hacia abajo). SCz = CL·A, SCx = CD·A.",
            "El Reynolds de la red es mucho menor que el real: valores orientativos, compara tendencias.",
            "Prioridad (panel Túnel): reparte el cuadro entre simulación y visualización.",
            "La velocidad (km/h) reescala N/kgf y fija el Reynolds real de la ley de pared.",
            "Pulsa F1 o Esc para cerrar.",
        };
        const int nn = static_cast<int>(sizeof notes / sizeof notes[0]);
        const int lh = 20, col = 250;
        int w = 0;
        for (int i = 0; i < n; ++i) w = max_(w, col + render::text_width(keys[i][1]));
        for (int i = 0; i < nn; ++i) w = max_(w, render::text_width(notes[i]));
        w = min_(w + 48, vp.w - 20);
        const int h = min_(n * lh + 84 + nn * lh, vp.h - 20);
        const Rect box{vp.x + (vp.w - w) / 2, vp.y + (vp.h - h) / 2, w, h};
        dl.shadow(box, 10.0f, 18.0f, 0xA0000000u);
        panel_box(dl, box, 0xF01A1D24u);
        dl.push_clip(box);
        dl.text(box.x + 20, box.y + 14, "Controles", 0xFFF4F6FAu, render::Font::Large);
        for (int i = 0; i < n; ++i) {
            dl.text(box.x + 24, box.y + 60 + i * lh, keys[i][0], 0xFF8FC8FFu);
            dl.text(box.x + 24 + col - 24, box.y + 60 + i * lh, keys[i][1], 0xFFD9DDE5u);
        }
        const int yb = box.y + 68 + n * lh;
        for (int i = 0; i < nn; ++i) dl.text(box.x + 24, yb + i * lh, notes[i], i + 1 == nn ? 0xFF7F8898u : 0xFF9AA3B2u);
        dl.pop_clip();
    }
    dl.pop_clip();
}

} // namespace cfd::app
