// ============================================================================
//  ui/demo.hpp — panel de demostración con TODOS los widgets (lo usan
//  tests/test_ui.cpp para la captura PNG y el benchmark, y tools/ui_demo.cpp
//  para la demo interactiva X11). Sirve también de referencia a la app para el
//  panel real del túnel de viento.
// ============================================================================
#pragma once

#include "ui.hpp"

namespace cfd::ui {

struct DemoState {
    // Modelo
    int model = 5, kind = 0;
    // Túnel
    float speed_kmh = 250.0f, yaw_deg = 0.0f;
    int ground = 2;
    bool wheels = true;
    // Configuración aerodinámica
    float ride_f = 30.0f, ride_r = 75.0f, flap_f = 0.0f, flap_r = 0.0f;
    bool drs = false;
    // Visualización
    int field = 0, cmap = 3;
    float slice_alpha = 0.75f;
    bool streamlines = true, smoke = false, slice = true, vortices = false, ssao = true;
    // Solver
    int steps = 8;
    bool paused = false;
    // Historia simulada (búferes circulares)
    static constexpr int N = 600;
    float scz[N] = {}, scx[N] = {}, res[N] = {};
    int head = 0, count = 0;
    double t = 0;
    u64 step = 0;
    int resets = 0, shots = 0;
};

void demo_init(DemoState& s);
void demo_tick(DemoState& s, double dt);              // avanza los datos simulados
void demo_panel(Context& ui, DemoState& s, Rect panel, double fps = 60.0);
// Marcador de posición del visor 3D: degradado de túnel, suelo en perspectiva y HUD.
void demo_viewport(render::Framebuffer& fb, Rect vp, double fps, double ui_ms);

} // namespace cfd::ui
