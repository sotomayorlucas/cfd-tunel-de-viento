// ============================================================================
//  ui/demo.cpp — panel de demostración (todos los widgets) y visor ficticio.
// ============================================================================
#include "demo.hpp"
#include "../core/util.hpp"

#include <cmath>
#include <cstdio>

namespace cfd::ui {

namespace {
const char* const k_models[] = {
    "F1 1998 — surcos y estrecho", "F1 2005 — aletas y deflectores", "F1 2009 — alerones simplificados",
    "F1 2014 — morro bajo híbrido", "F1 2017 — ancho y agresivo",   "F1 2022 — efecto suelo (Venturi)",
    "F1 2026 — aerodinámica activa", "Ala NACA 4412",               "Ala multielemento (flap)",
    "Esfera (referencia)",          "Cilindro (referencia)",         "Cuerpo de Ahmed 25°",
    "Placa plana",                  "Perfil NACA 0012",
};
const char* const k_desc[] = {
    "Coche estrecho (1.8 m) con neumáticos ranurados. Carga dominada por los alerones.",
    "Máxima complejidad: aletas, deflectores y difusor de doble piso.",
    "Reglamento que simplificó la carrocería: alerón delantero ancho y trasero alto y estrecho.",
    "Morro bajo, motor híbrido y menos carga del alerón delantero.",
    "Coche más ancho y alerones más grandes: la época de mayor carga antes del efecto suelo.",
    "Vuelve el efecto suelo: túneles Venturi bajo el suelo generan la mayor parte de la carga, "
    "con menos estela sucia para el coche que sigue.",
    "Aerodinámica activa: alerones móviles (modo X de baja resistencia en recta).",
    "Perfil con combadura 4 %: sustentación a ángulo nulo.",
    "Elemento principal + flap con ranura: más carga sin desprendimiento.",
    "Cuerpo romo clásico: crisis de resistencia según el número de Reynolds.",
    "Calle de vórtices de von Kármán detrás del cilindro.",
    "Cuerpo de referencia de automoción con luneta a 25°.",
    "Placa normal al flujo: Cd ≈ 1.2, estela muy ancha.",
    "Perfil simétrico: Cl = 0 a ángulo nulo.",
};
const char* const k_kinds[] = {"Coche F1", "Ala", "Objeto"};
const char* const k_fields[] = {"Cp", "Velocidad", "Vorticidad", "Q"};
const char* const k_cmaps[] = {"Turbo", "Viridis", "Inferno", "Frío-cálido", "Grises"};

float noise(double t, double f, double ph) { return static_cast<float>(std::sin(t * f + ph) * 0.5 + std::sin(t * f * 2.37 + ph * 1.7) * 0.25); }
} // namespace

void demo_init(DemoState& s) {
    s = DemoState{};
    for (int i = 0; i < 400; ++i) demo_tick(s, 1.0 / 30.0);
}

void demo_tick(DemoState& s, double dt) {
    if (s.paused) return;
    s.t += dt;
    s.step += static_cast<u64>(s.steps);
    const double k = 1.0 - std::exp(-s.t * 0.35);                  // convergencia del flujo
    const float v = s.speed_kmh / 250.0f;
    const float cl = static_cast<float>(3.45 * k) * (s.drs ? 0.82f : 1.0f) + 0.06f * noise(s.t, 3.1, 0.3) * static_cast<float>(k);
    const float cd = static_cast<float>(1.09 * k) * (s.drs ? 0.86f : 1.0f) + 0.03f * noise(s.t, 2.3, 1.1) * static_cast<float>(k);
    s.scz[s.head] = cl * (0.97f + 0.03f * v);
    s.scx[s.head] = cd;
    s.res[s.head] = static_cast<float>(std::exp(-s.t * 0.25) * 1e-2 * (1.0 + 0.3 * noise(s.t, 7.0, 0.0)));
    s.head = (s.head + 1) % DemoState::N;
    if (s.count < DemoState::N) ++s.count;
}

void demo_panel(Context& ui, DemoState& s, Rect pr, double fps) {
    const Style& st = ui.style();
    char sub[96];
    std::snprintf(sub, sizeof sub, "LBM D3Q19 · 256×128×96 · %.0f FPS", fps);
    if (!ui.begin_panel("panel_principal", pr)) return;
    ui.title("Túnel de viento", sub);
    ui.spacing(2);

    if (ui.header("Modelo")) {
        ui.combo("Modelo", &s.model, k_models, static_cast<int>(sizeof k_models / sizeof k_models[0]));
        ui.tooltip("Coches de F1 de distintas épocas reglamentarias y cuerpos de referencia.");
        ui.segmented("##tipo", k_kinds, 3, &s.kind);
        ui.text_wrapped("%s", k_desc[s.model]);
    }
    if (ui.header("Túnel de viento")) {
        ui.slider_float("Velocidad", &s.speed_kmh, 50.0f, 350.0f, "%.0f", "km/h");
        ui.tooltip("Velocidad del aire. Mayús = ajuste fino; doble clic o Ctrl+clic para escribir el valor.");
        ui.slider_float("Guiñada", &s.yaw_deg, -10.0f, 10.0f, "%+.1f", "°");
        ui.label("Suelo", st.text_dim);
        ui.row(3);
        ui.radio("Sin suelo", &s.ground, 0);
        ui.radio("Fijo", &s.ground, 1);
        ui.radio("Cinta móvil", &s.ground, 2);
        ui.toggle("Ruedas girando", &s.wheels);
    }
    if (ui.header("Configuración aerodinámica")) {
        ui.slider_float("Altura delantera", &s.ride_f, 10.0f, 80.0f, "%.0f", "mm");
        ui.slider_float("Altura trasera", &s.ride_r, 20.0f, 120.0f, "%.0f", "mm");
        ui.slider_float("Flap delantero", &s.flap_f, -5.0f, 10.0f, "%+.1f", "°");
        ui.slider_float("Flap trasero", &s.flap_r, -5.0f, 10.0f, "%+.1f", "°");
        ui.row(2);
        ui.toggle_button("DRS abierto", &s.drs);
        if (ui.button("Valores por defecto")) { s.ride_f = 30; s.ride_r = 75; s.flap_f = s.flap_r = 0; s.drs = false; ui.toast(st.accent, "Configuración restablecida"); }
    }
    if (ui.header("Visualización")) {
        ui.segmented("##campo", k_fields, 4, &s.field);
        ui.combo("Mapa de color", &s.cmap, k_cmaps, 5);
        ui.colorbar("Coeficiente de presión", static_cast<render::Colormap>(s.cmap), -3.0f, 1.0f, "Cp");
        ui.row(2);
        ui.checkbox("Líneas de corriente", &s.streamlines);
        ui.checkbox("Humo", &s.smoke);
        ui.row(2);
        ui.checkbox("Plano de corte", &s.slice);
        ui.checkbox("Vórtices (Q)", &s.vortices);
        ui.checkbox("Oclusión ambiental", &s.ssao);
        ui.slider_float("Opacidad del plano", &s.slice_alpha, 0.0f, 1.0f, "%.2f");
    }
    const int last = (s.head + DemoState::N - 1) % DemoState::N;
    const float scz = s.count ? s.scz[last] : 0.0f, scx = s.count ? s.scx[last] : 0.0f;
    if (ui.header("Resultados")) {
        char b[32];
        ui.row(2);
        std::snprintf(b, sizeof b, "%.3f", static_cast<double>(scz));
        ui.metric("CARGA · SCz", b, "m²", st.series[0]);
        std::snprintf(b, sizeof b, "%.3f", static_cast<double>(scx));
        ui.metric("RESISTENCIA · SCx", b, "m²", st.series[1]);
        ui.row(2);
        std::snprintf(b, sizeof b, "%.2f", static_cast<double>(scx > 1e-6f ? scz / scx : 0.0f));
        ui.metric("EFICIENCIA L/D", b, nullptr, st.success);
        ui.metric("BALANCE DEL.", "41.8", "%", st.warn);
        const float q = 0.5f * 1.225f * sq(s.speed_kmh / 3.6f);
        ui.value("Cl (sustentación)", "%+.3f", static_cast<double>(-scz / 1.45f));
        ui.value("Cd (resistencia)", "%.3f", static_cast<double>(scx / 1.45f));
        ui.value("Carga", "%.0f kg", static_cast<double>(q * scz / 9.81f));
        ui.value("Resistencia", "%.0f N", static_cast<double>(q * scx));
        ui.value_colored("Potencia absorbida", st.warn, "%.0f kW", static_cast<double>(q * scx * s.speed_kmh / 3.6f / 1000.0f));
        const PlotSeries ser[2] = {{s.scz, s.count, s.count < DemoState::N ? 0 : s.head, st.series[0], "SCz"},
                                   {s.scx, s.count, s.count < DemoState::N ? 0 : s.head, st.series[1], "SCx"}};
        ui.plot_lines("Historia", ser, 2, static_cast<int>(150 * st.scale), "m²", "%.3f");
        const Bar bars[] = {
            {"Alerón delantero", 0.92f * scz / 3.45f, 0},  {"Suelo y difusor", 1.71f * scz / 3.45f, 0},
            {"Alerón trasero", 0.98f * scz / 3.45f, 0},    {"Beam wing", 0.21f * scz / 3.45f, 0},
            {"Carrocería", -0.18f, 0},                      {"Ruedas", -0.29f, 0},
            {"Suspensión", 0.05f, 0},
        };
        ui.bar_chart("Carga por componente", bars, 7, "SCz [m²]", "%+.2f");
    }
    if (ui.header("Solver")) {
        ui.value("Malla", "256 × 128 × 96");
        ui.value("Paso", "%llu", static_cast<unsigned long long>(s.step));
        ui.value("Rendimiento", "%.0f MLUPS", 41.7 + 2.0 * std::sin(s.t));
        ui.value("Reynolds", "%.2e", 1.2e7 * static_cast<double>(s.speed_kmh) / 250.0);
        const float conv = static_cast<float>(1.0 - std::exp(-s.t * 0.35));
        char pb[48];
        std::snprintf(pb, sizeof pb, "Convergencia %.0f %%", static_cast<double>(conv * 100.0f));
        ui.progress(conv, pb);
        const PlotSeries rs[1] = {{s.res, s.count, s.count < DemoState::N ? 0 : s.head, st.series[2], "resid."}};
        ui.plot_lines("Residuo", rs, 1, static_cast<int>(96 * st.scale), nullptr, "%.1e");
        ui.slider_int("Pasos por cuadro", &s.steps, 1, 32);
        ui.row(2);
        if (ui.button(s.paused ? "Reanudar" : "Pausa", ButtonKind::Primary)) s.paused = !s.paused;
        if (ui.button("Reiniciar flujo", ButtonKind::Danger)) { ++s.resets; s.t = 0; s.count = 0; s.head = 0; ui.toast(st.warn, "Flujo reiniciado (ρ = 1, u = u∞)"); }
        if (ui.button("Guardar captura PNG", ButtonKind::Subtle)) { ++s.shots; ui.toast(st.success, "Captura guardada en build/captura_%03d.png", s.shots); }
    }
    ui.end_panel();
}

void demo_viewport(render::Framebuffer& fb, Rect vp, double fps, double ui_ms) {
    // Se graba en una DrawList y se rasteriza en paralelo por franjas (como hará la app).
    static render::DrawList dl;
    dl.clear();
    dl.push_clip(vp);
    dl.gradient_v(vp, 0xFF2A2F38u, 0xFF0D0F13u);
    // Suelo en perspectiva (rejilla) — líneas antialias.
    const float hx = static_cast<float>(vp.x) + vp.w * 0.5f, hy = static_cast<float>(vp.y) + vp.h * 0.42f;
    for (int i = -14; i <= 14; ++i) {
        const float x = hx + static_cast<float>(i) * vp.w * 0.09f;
        dl.line(hx + (x - hx) * 0.08f, hy + 40.0f, x, static_cast<float>(vp.y + vp.h), 0x28FFFFFFu, 1.0f);
    }
    for (int k = 1; k < 14; ++k) {
        const float t = static_cast<float>(k) / 14.0f, y = hy + 40.0f + (vp.h * 0.58f - 40.0f) * t * t;
        dl.line(static_cast<float>(vp.x), y, static_cast<float>(vp.x + vp.w), y, 0x20FFFFFFu, 1.0f);
    }
    // "Líneas de corriente" decorativas coloreadas con Turbo.
    Vec2 pts[160];
    for (int sidx = 0; sidx < 14; ++sidx) {
        const float y0 = hy - 120.0f + static_cast<float>(sidx) * 22.0f;
        for (int i = 0; i < 160; ++i) {
            const float x = static_cast<float>(vp.x) + 40.0f + static_cast<float>(i) * (vp.w - 80.0f) / 159.0f;
            const float u = (x - hx) / (vp.w * 0.18f);
            const float bump = 70.0f * std::exp(-u * u) * (1.0f - static_cast<float>(sidx) / 16.0f);
            pts[i] = {x, y0 - bump + 6.0f * std::sin(u * 2.0f + static_cast<float>(sidx))};
        }
        dl.polyline(pts, 160, render::colormap(render::Colormap::Turbo, static_cast<float>(sidx) / 13.0f), 1.6f);
    }
    // Cuerpo (marcador) + sombra.
    dl.shadow({static_cast<int>(hx) - 170, static_cast<int>(hy) + 20, 340, 40}, 20.0f, 18.0f, 0x80000000u, false);
    dl.fill_round_rect({static_cast<int>(hx) - 160, static_cast<int>(hy) - 10, 320, 48}, 22.0f, 0xFFB8BCC4u);
    dl.fill_round_rect({static_cast<int>(hx) - 40, static_cast<int>(hy) - 34, 110, 40}, 16.0f, 0xFF9AA0AAu);
    dl.fill_circle(hx - 105.0f, hy + 42.0f, 26.0f, 0xFF2E2E32u);
    dl.fill_circle(hx + 110.0f, hy + 42.0f, 28.0f, 0xFF2E2E32u);
    char hud[160];
    std::snprintf(hud, sizeof hud, "Vista 3D (marcador) · %.0f FPS · UI %.2f ms", fps, ui_ms);
    dl.text_shadow(vp.x + 14, vp.y + 12, hud, 0xFFE8ECF2u, 0xC0000000u);
    dl.text_shadow(vp.x + 14, vp.y + 32, "Arrastrar: orbitar · Rueda: zoom · Clic central: desplazar", 0xFFA8B0BCu, 0xC0000000u);
    dl.pop_clip();
    dl.render(fb);
}

} // namespace cfd::ui
