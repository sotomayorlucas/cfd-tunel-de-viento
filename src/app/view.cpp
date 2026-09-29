// ============================================================================
//  app/view.cpp — escena 3D: cámara, visualización del flujo y composición.
//
//  Orden de dibujo (flowvis.hpp): fondo → suelo/huella → malla → SSAO →
//  corte → líneas → humo → volumen de vórtices (último: lee la profundidad) →
//  flechas de fuerza y túnel → FXAA. Todo desde el hilo principal: las
//  funciones de raster/flowvis usan el pool por dentro y no son reentrantes.
// ============================================================================
#include "app.hpp"

#include "../render/colormap.hpp"
#include "../render/raster.hpp"

#include <cmath>
#include <cstring>

namespace cfd::app {

using flowvis::Quantity;
using render::Rect;

namespace {
// Cp referido a la presión estática de referencia del túnel (Sim::rho_ref, la "toma estática" aguas arriba)
// en vez de a ρ = 1: Cp_túnel = Cp_bruto − 2(ρ_ref − 1)/(3u∞²). Se aplica como ColorScale::offset a todo lo
// que se colorea por Cp o Cp0 (superficie, huella, líneas, corte) y a la sonda. Las fuerzas no dependen de él.
float cp_offset(const Sim& s) {
    const float u = max_(s.cfg.u_lat, 1e-4f);
    const float d = 2.0f * (s.rho_ref - 1.0f) / (3.0f * u * u);
    return std::isfinite(d) ? d : 0.0f;
}
bool is_cp_quantity(Quantity q) { return q == Quantity::Cp || q == Quantity::Cp0; }
} // namespace

const char* surf_mode_name(SurfMode m) {
    switch (m) {
        case SurfMode::Cp: return "Cp (presión)";
        case SurfMode::Speed: return "Velocidad junto a la pared";
        case SurfMode::Component: return "Por componente";
        case SurfMode::Plain: return "Liso";
        case SurfMode::Voxels: return "Vóxeles (lo que ve el solver)";
        case SurfMode::Hidden: return "Oculta (sólo el flujo)";
        default: return "?";
    }
}
const char* rake_name(RakeKind r) {
    switch (r) {
        case RakeKind::Vertical: return "Vertical (plano central)";
        case RakeKind::Floor: return "Horizontal (altura del fondo)";
        case RakeKind::Tips: return "Puntas de ala / laterales";
        case RakeKind::Grid: return "Rejilla aguas arriba";
        default: return "?";
    }
}
const char* cam_view_name(CamView v) {
    switch (v) {
        case CamView::Lateral: return "Lateral";
        case CamView::Superior: return "Superior";
        case CamView::Frontal: return "Frontal";
        case CamView::Trasera: return "Trasera";
        case CamView::TresCuartos: return "3/4 delantera";
        case CamView::Bajo: return "Bajo el coche";
        default: return "?";
    }
}
const char* cam_view_key(CamView v) {
    switch (v) {
        case CamView::Lateral: return "lateral";
        case CamView::Superior: return "superior";
        case CamView::Frontal: return "frontal";
        case CamView::Trasera: return "trasera";
        case CamView::TresCuartos: return "34";
        case CamView::Bajo: return "bajo";
        default: return "?";
    }
}
bool parse_cam_view(const char* s, CamView& out) {
    for (int i = 0; i < static_cast<int>(CamView::Count); ++i)
        if (!std::strcmp(s, cam_view_key(static_cast<CamView>(i)))) { out = static_cast<CamView>(i); return true; }
    if (!std::strcmp(s, "3/4") || !std::strcmp(s, "tres_cuartos")) { out = CamView::TresCuartos; return true; }
    return false;
}

// ============================================================================
//  Ajustes por defecto según el tipo de modelo
// ============================================================================
void View::defaults_for(const Sim& sim) {
    const models::Info& I = sim.built.info;
    const VisSettings keep = vs;
    vs = VisSettings{};
    vs.ssao = keep.ssao; vs.fxaa = keep.fxaa; vs.legends = keep.legends; vs.probe = keep.probe; vs.wire = keep.wire;
    const Aabb ob = sim.object_cells();
    const Vec3 c = ob.center();
    vs.slice_pos[0] = min_(ob.hi.x + 0.35f * ob.size().x, static_cast<float>(sim.dom.nx) - 2.0f);
    vs.slice_pos[1] = std::round(c.y);
    vs.slice_pos[2] = c.z;
    const bool ground = sim.cfg.ground != lbm::GroundMode::None;
    vs.footprint = ground;
    if (I.kind == models::Kind::F1Car || (I.kind == models::Kind::Body && I.needs_ground)) {
        vs.lines = true; vs.line_rake = RakeKind::Vertical; vs.line_count = 36;
        vs.slice_axis = 0;
        // Corte horizontal por defecto a la altura del fondo (si se activa con Z).
        vs.slice_pos[2] = 0.5f + max_(sim.params.ride_front_mm * 1e-3f * 0.5f, 0.012f) / sim.dom.dx;
    } else if (I.kind == models::Kind::Wing && I.needs_ground) {
        // Ala en efecto suelo: la huella de Cp en el suelo enseña la succión; líneas por el plano
        // central. El corte Y (Cp) a un cuarto de envergadura queda preparado para la tecla Y.
        vs.lines = true; vs.line_rake = RakeKind::Vertical; vs.line_count = 30;
        vs.slice_axis = 0; vs.slice_q = 3;   // (Y, Cp al activarlo)
        vs.slice_opacity = 0.8f;
        vs.slice_pos[1] = std::round(c.y - 0.25f * ob.size().y);   // a un cuarto de envergadura (lado de la cámara)
    } else if (I.kind == models::Kind::Wing) {
        vs.lines = true;
        vs.line_rake = I.spans_domain ? RakeKind::Vertical : RakeKind::Tips;
        vs.line_count = I.spans_domain ? 30 : 36;
        vs.slice_axis = I.spans_domain ? 2 : 0;
        vs.slice_q = 0;
        vs.slice_opacity = 0.9f;
        vs.vortex = !I.spans_domain;   // ala finita: torbellinos de punta (criterio Q)
    } else {
        vs.lines = true; vs.line_rake = RakeKind::Vertical; vs.line_count = 30;
        vs.slice_axis = 2; vs.slice_q = 0;
        vs.slice_opacity = 0.75f;
    }
    // Umbral de vórtices (Q·L²/U²) por tipo: en un coche hay mucha vorticidad de capa límite y de
    // estela junto a la carrocería; en un ala libre interesan los torbellinos de punta.
    // (Valores elegidos mirando capturas a resolución rápida, 2.5 PF: por debajo de ~400 el volumen del coche
    // se llena de capas de cortadura y ruido de escala de red; con 600 quedan los torbellinos de ruedas,
    // alerones y difusor. En el ala libre los torbellinos de punta están en ~20-50.)
    // Cuerpos (revisión): con 60 el ruido de escala de red del campo lejano (ν = 1e-4) llenaba TODO el dominio
    // de manchas (Ahmed, rápida, 3 PF); con 800 quedan motas sueltas y el cuerpo visible.
    // (Ruido del campo lejano corregido, docs/FISICA.md §1.5) El dominio ya no se llena de motas: 300 en coches y 400 en
    // cuerpos dejan ver los torbellinos de ruedas, alerones y estela (capturas en build/app/shots/noise/). Queda algo de
    // ruido de escala de red dentro de la capa de ~8 celdas junto a los cuerpos (colisión de 2º orden, ver
    // lbm::Config::rr_wall_layer): con 150-200 en coches, 10-20 en alas o 150 en el Ahmed aparece como motas pegadas a
    // la superficie; en alas libres se mantiene 30 por eso.
    vs.q_threshold = I.kind == models::Kind::F1Car ? 300.0f : (I.kind == models::Kind::Wing ? 30.0f : 400.0f);
    dirty = true;
    rakes_dirty = true;
    surf_colored = SurfMode::Count;
}

// ============================================================================
//  Cámara
// ============================================================================
void View::frame(const Sim& sim, CamView v, Rect vp) {
    const Aabb ob = sim.object_cells();
    cam.fov_y = 38.0f * k_deg2rad;
    cam.ortho = false;
    float fill = 0.86f;   // fracción del visor que ocupa el objeto
    switch (v) {
        case CamView::Lateral: cam.yaw = 0.0f; cam.pitch = 0.10f; fill = 0.9f; break;
        case CamView::Superior: cam.yaw = 0.0f; cam.pitch = 1.45f; fill = 0.9f; break;
        case CamView::Frontal: cam.yaw = -0.5f * k_pi; cam.pitch = 0.12f; fill = 0.8f; break;
        case CamView::Trasera: cam.yaw = 0.5f * k_pi; cam.pitch = 0.2f; fill = 0.8f; break;
        case CamView::TresCuartos: cam.yaw = -0.72f; cam.pitch = 0.3f; fill = 0.84f; break;
        case CamView::Bajo: cam.yaw = -0.45f; cam.pitch = -0.75f; fill = 0.84f; break;
        default: break;
    }
    // Los cuerpos y alas se miran con contexto (estela, efecto suelo): el objeto ocupa menos.
    // Los coches llenan el visor (son el sujeto; el panel de resultados da los números).
    const models::Info& I = sim.built.info;
    // Cuerpos en aire libre: 3/4 más de lado (el corte Y central con la estela se ve casi de frente).
    if (v == CamView::TresCuartos && I.kind == models::Kind::Body && !I.needs_ground) { cam.yaw = -0.42f; cam.pitch = 0.22f; }
    if (I.kind == models::Kind::Body && !I.needs_ground) fill *= 0.55f;
    else if (I.kind == models::Kind::Body) fill *= 0.8f;
    else if (I.kind == models::Kind::Wing) fill *= I.spans_domain ? 0.8f : 0.75f;
    const float big = static_cast<float>(max_(sim.dom.nx, max_(sim.dom.ny, sim.dom.nz)));
    cam.target = ob.center();
    // Encuadre: las 8 esquinas de la caja proyectadas deben ocupar `fill` del visor (en su eje más
    // exigente). Búsqueda de la distancia + recentrado del objetivo sobre la caja proyectada.
    const float r = 0.5f * length(ob.size());
    auto extent = [&](float d, float& cx, float& cy) {
        cam.distance = d;
        cam.znear = max_(0.25f, 0.002f * d);
        cam.zfar = 20.0f * big + 4.0f * d;
        cam.update(vp);
        float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
        for (int i = 0; i < 8; ++i) {
            const Vec3 p{(i & 1) ? ob.hi.x : ob.lo.x, (i & 2) ? ob.hi.y : ob.lo.y, (i & 4) ? ob.hi.z : ob.lo.z};
            float sx, sy, dz;
            if (!cam.project(p, sx, sy, dz)) return 1e30f;
            x0 = min_(x0, sx); x1 = max_(x1, sx); y0 = min_(y0, sy); y1 = max_(y1, sy);
        }
        cx = 0.5f * (x0 + x1); cy = 0.5f * (y0 + y1);
        return max_((x1 - x0) / static_cast<float>(max_(vp.w, 1)), (y1 - y0) / static_cast<float>(max_(vp.h, 1)));
    };
    for (int pass = 0; pass < 3; ++pass) {
        float lo = 0.3f * r, hi = 40.0f * r + 10.0f, cx = 0, cy = 0;
        for (int it = 0; it < 40; ++it) {
            const float mid = std::sqrt(lo * hi);
            if (extent(mid, cx, cy) > fill) lo = mid; else hi = mid;
        }
        extent(hi, cx, cy);
        // Recentrar: desplazar el objetivo en el plano de la cámara hacia el centro de la caja proyectada.
        const float ccx = static_cast<float>(vp.x) + 0.5f * static_cast<float>(vp.w), ccy = static_cast<float>(vp.y) + 0.5f * static_cast<float>(vp.h);
        cam.pan(-(cx - ccx), -(cy - ccy));
    }
    cam.update(vp);
}

// ============================================================================
//  Rastrillos
// ============================================================================
namespace {

struct RakeSet { flowvis::Rake r[4]; int n = 0; };

// Altura "del fondo" (celdas) para el rastrillo horizontal.
float floor_z_cells(const Sim& sim) {
    const models::Info& I = sim.built.info;
    const Aabb ob = sim.object_cells();
    if (sim.cfg.ground == lbm::GroundMode::None && !(I.param_mask & models::P_RideHeight)) return ob.center().z;
    if (I.param_mask & models::P_RideHeight) return 0.5f + max_(0.5f * sim.params.ride_front_mm * 1e-3f, 0.8f * sim.dom.dx) / sim.dom.dx;
    if (I.needs_ground && (I.param_mask & models::P_Height)) return 0.5f + max_(0.5f * sim.params.height_mm * 1e-3f, 0.8f * sim.dom.dx) / sim.dom.dx;
    return max_(ob.lo.z - 0.02f * ob.size().z, 1.5f);
}

RakeSet make_rakes(const Sim& sim, RakeKind kind, int count, bool smoke) {
    RakeSet s;
    const Aabb ob = sim.object_cells();
    const Vec3 sz = ob.size();
    const float gap = 0.12f * sz.x + 4.0f;
    const float yc = ob.center().y;
    const float ymin = 1.5f, ymax = static_cast<float>(sim.dom.ny) - 2.5f;
    const float zmin = sim.cfg.ground != lbm::GroundMode::None ? 1.2f : 1.5f, zmax = static_cast<float>(sim.dom.nz) - 2.5f;
    count = max_(count, 2);
    switch (kind) {
        case RakeKind::Vertical: {
            flowvis::Rake r = flowvis::rake_vertical(ob, gap, yc + 0.37f, count, 0.12f);
            r.origin.z = clamp_(r.origin.z, zmin, zmax);
            s.r[s.n++] = r;
            break;
        }
        case RakeKind::Floor: {
            flowvis::Rake r = flowvis::rake_floor(ob, gap, clamp_(floor_z_cells(sim), zmin, zmax), count, 0.08f);
            r.origin.y = max_(r.origin.y, ymin);
            if (r.origin.y + r.du.y > ymax) r.du.y = ymax - r.origin.y;
            s.r[s.n++] = r;
            break;
        }
        case RakeKind::Tips: {
            // Dos rejillas pequeñas a los lados (puntas de ala / ruedas y endplates del coche).
            const float z0 = max_(ob.lo.z - 0.05f * sz.z, zmin);
            const float z1 = sim.built.info.kind == models::Kind::Wing ? ob.hi.z + 0.1f * sz.z : ob.lo.z + 0.55f * sz.z;
            const int per = max_(1, count / 2);
            const int nv = max_(2, per / 3), nu = max_(1, per / nv);
            const float wy = max_(0.07f * sz.y, 2.0f);
            const float x = ob.lo.x - gap;
            const float ys[2] = {ob.lo.y + 0.03f * sz.y, ob.hi.y - 0.03f * sz.y};
            for (float y : ys)
                s.r[s.n++] = flowvis::Rake::grid({x, clamp_(y - 0.5f * wy, ymin, ymax), z0}, {0, wy, 0}, {0, 0, min_(z1, zmax) - z0}, nu, nv);
            break;
        }
        case RakeKind::Grid: {
            const float aspect = max_(sz.y, 1.0f) / max_(sz.z, 1.0f);
            int nu = max_(2, static_cast<int>(std::lround(std::sqrt(static_cast<float>(count) * aspect))));
            int nv = max_(2, count / nu);
            if (smoke) { nu = min_(nu, 12); nv = min_(nv, 8); }
            flowvis::Rake r = flowvis::rake_upstream(ob, gap, nu, nv, 0.1f);
            r.origin.y = max_(r.origin.y, ymin);
            if (r.origin.y + r.du.y > ymax) r.du.y = ymax - r.origin.y;
            r.origin.z = max_(r.origin.z, zmin);
            if (r.origin.z + r.dv.z > zmax) r.dv.z = zmax - r.origin.z;
            s.r[s.n++] = r;
            break;
        }
        default: break;
    }
    return s;
}

} // namespace

// ============================================================================
//  Actualización de la visualización
// ============================================================================
void View::update(Sim& sim, int steps_advanced) {
    if (!sim.ready) return;
    const double t0 = now_sec();
    t_upd_sampler = t_upd_lines = t_upd_smoke = t_upd_vortex = t_upd_slice = t_upd_color = 0;
    // Campo nuevo: se refleja en la visualización como mucho cada `field_interval` s (el flujo
    // cambia despacio frente a los cuadros; así el tiempo va al solver). El humo avanza cada cuadro.
    const double now = now_sec();
    const bool field_changed = sim.field_version != seen_field;
    const bool field_new = field_changed && (now - last_field_t >= field_interval || dirty || sim.geom_version != seen_geom ||
                                             sim.domain_version != seen_domain || !sampler.ready());
    if (field_new) last_field_t = now;
    const bool geom_new = sim.geom_version != seen_geom;
    const bool domain_new = sim.domain_version != seen_domain;
    if (domain_new) { smoke.reset(); rakes_dirty = true; belt_offset = 0.0f; }
    if (geom_new) {
        rakes_dirty = true;
        surf_colored = SurfMode::Count;
        const auto& gs = sim.built.scene.groups();
        for (int i = 0; i < 256; ++i) group_colors[i] = 0xFFB8BCC4u;
        for (usize g = 0; g < gs.size() && g < 254; ++g) group_colors[g + 1] = render::component_color(gs[g].component);
    }
    const lbm::FieldView f = sim.solver.field();
    const bool recompute = field_new || dirty || geom_new || domain_new;
    const float cp_off = cp_offset(sim);   // Cp respecto a la presión de referencia del túnel
    const bool ground = sim.cfg.ground != lbm::GroundMode::None;
    if (sim.cfg.ground == lbm::GroundMode::Moving && steps_advanced > 0) {
        const float spacing = max_(0.5f / sim.dom.dx, 2.0f) * 5.0f;
        belt_offset = std::fmod(belt_offset + sim.cfg.u_lat * static_cast<float>(steps_advanced), spacing);
    }
    // Muestreador empaquetado (líneas y humo).
    const bool need_sampler = vs.lines || vs.smoke;
    if (need_sampler && (field_new || domain_new || !sampler.ready())) {
        const double a = now_sec();
        sampler.update(f);
        t_upd_sampler = (now_sec() - a) * 1e3;
    }
    // Plano de corte.
    if (vs.slice_axis > 0 && recompute) {
        const double a = now_sec();
        const int ax = vs.slice_axis - 1;
        const Quantity q = k_slice_q[clamp_(vs.slice_q, 0, 6)];
        if (slice.params.quantity != q) slice.set_quantity(q);
        slice.params.axis = static_cast<flowvis::Axis>(ax);
        const int n_ax = ax == 0 ? sim.dom.nx : (ax == 1 ? sim.dom.ny : sim.dom.nz);
        vs.slice_pos[ax] = clamp_(vs.slice_pos[ax], 0.0f, static_cast<float>(n_ax - 1));
        slice.params.pos = vs.slice_pos[ax];
        slice.params.auto_range = vs.slice_auto;
        if (!vs.slice_auto) {
            flowvis::ColorScale sc = flowvis::default_scale(q);
            sc.lo = vs.slice_lo;
            sc.hi = max_(vs.slice_hi, vs.slice_lo + 1e-4f);
            slice.params.scale = sc;
        } else {
            slice.params.scale = flowvis::default_scale(q);
        }
        slice.params.scale.offset = is_cp_quantity(q) ? cp_off : 0.0f;
        slice.params.opacity = vs.slice_opacity;
        slice.params.lic = vs.slice_lic ? 2 : 0;
        // Celdas sólidas del corte: gris "carrocería" si se ve la superficie (la sección de vóxeles
        // se lee como parte del cuerpo; transparente dejaba ver el fondo por la escalera), gris
        // oscuro si la superficie está oculta (silueta que resuelve el solver).
        slice.params.solid_color = vs.surf == SurfMode::Hidden ? 0xFF2E2F33u : 0xFF7C828Cu;
        slice.update(f);
        t_upd_slice = (now_sec() - a) * 1e3;
    }
    // Huella en el suelo.
    if (ground && vs.footprint && recompute) {
        footprint.params.quantity = Quantity::Cp;
        footprint.params.scale = {render::Colormap::CoolWarm, -2.0f, 1.0f, true, cp_off};
        footprint.params.auto_range = false;
        footprint.update(f);
    }
    // Líneas de corriente.
    if (vs.lines) {
        if (rakes_dirty || lines_rake_built != vs.line_rake || lines_count_built != vs.line_count) {
            const RakeSet rs = make_rakes(sim, vs.line_rake, vs.line_count, false);
            lines.set_rakes(std::span<const flowvis::Rake>(rs.r, static_cast<usize>(rs.n)));
            lines_rake_built = vs.line_rake;
            lines_count_built = vs.line_count;
        }
        if (recompute || lines.line_count() == 0) {
            const double a = now_sec();
            lines.params.max_steps = clamp_(static_cast<int>(1.4f * static_cast<float>(sim.dom.nx)), 300, 1500);
            lines.params.step = 0.6f;
            lines.params.max_step = 1.2f;
            lines.params.min_step = 0.15f;
            lines.params.width = 1.6f;
            lines.params.color_by = vs.line_cp ? Quantity::Cp : Quantity::Speed;
            lines.params.scale = vs.line_cp ? flowvis::ColorScale{render::Colormap::CoolWarm, -2.0f, 1.0f, true, cp_off}
                                            : flowvis::ColorScale{render::Colormap::Turbo, 0.0f, 1.5f, false};
            lines.compute(sampler);
            t_upd_lines = (now_sec() - a) * 1e3;
        }
    }
    // Humo.
    if (vs.smoke) {
        if (rakes_dirty || smoke_rake_built != vs.smoke_rake) {
            const int n = vs.smoke_rake == RakeKind::Vertical || vs.smoke_rake == RakeKind::Floor ? 28 : 40;
            const RakeSet rs = make_rakes(sim, vs.smoke_rake, n, true);
            smoke.set_emitters(std::span<const flowvis::Rake>(rs.r, static_cast<usize>(rs.n)));
            smoke_rake_built = vs.smoke_rake;
            smoke.reset();
        }
        const int cap = 60000 + static_cast<int>(vs.smoke_density * 240000.0f);
        if (smoke.params.capacity != cap) smoke.params.capacity = cap;
        // Aditivo (brillo) sobre fondo oscuro; con la huella de Cp (suelo claro) el aditivo satura a
        // blanco y el humo no se ve → mezcla alfa con más opacidad.
        const bool light_floor = ground && vs.footprint;
        smoke.params.intensity = light_floor ? 0.45f + 0.4f * vs.smoke_density : 0.18f + 0.3f * vs.smoke_density;
        smoke.params.point_size = 2.0f;
        smoke.params.additive = !light_floor;
        smoke.params.time_scale = vs.smoke_speed;
        smoke.params.jitter = 0.4f;
        smoke.params.scale = {render::Colormap::Turbo, 0.0f, 1.5f, false};
        if (steps_advanced > 0 && sampler.ready()) {
            const double a = now_sec();
            smoke.step(sampler, static_cast<float>(steps_advanced));
            t_upd_smoke = (now_sec() - a) * 1e3;
        }
    }
    // Umbral en unidades del objeto (Q·L²/U²) → normalizado por celda (Q·dx²/U², lo que usa flowvis):
    // el mismo vórtice físico da un Q por celda (L/dx)² veces menor en una red más fina.
    // La densidad del volumen se cuantiza en 8 bits hasta `full` (satura): con un umbral cercano o superior
    // a `full` todas las celdas por encima quedan "planas" (isosuperficies en bloques e idénticas para
    // cualquier umbral mayor). `full` ≥ 4·umbral, en escalones ×2 para no recalcular el volumen por cada
    // movimiento del deslizador.
    bool vortex_full_changed = false;
    {
        const float l_cells = max_(sim.obj_len_m() / max_(sim.dom.dx, 1e-6f), 1.0f);
        vortex.params.threshold = vs.q_threshold / (l_cells * l_cells);
        float full = 0.015f;
        while (full < 4.0f * vortex.params.threshold && full < 1e3f) full *= 2.0f;
        vortex_full_changed = full != vortex.params.full;
        vortex.params.full = full;
    }
    // Volumen de vórtices.
    if (vs.vortex && (recompute || vortex_full_changed)) {
        const double a = now_sec();
        vortex.params.field = flowvis::VolumeField::QCriterion;
        vortex.params.color_by = flowvis::VolumeColor::Streamwise;
        vortex.params.downsample = sim.dom.cells() > 9'000'000 ? 2 : (sim.dom.cells() > 3'500'000 ? 2 : 1);
        vortex.update(f);
        t_upd_vortex = (now_sec() - a) * 1e3;
    }
    vortex.params.style = vs.vortex_cloud ? flowvis::VolumeStyle::Cloud : flowvis::VolumeStyle::Surface;
    // Colores de la superficie.
    {
        const double a = now_sec();
        render::Mesh& m = vs.surf == SurfMode::Voxels ? (sim.ensure_vox_mesh(), sim.vox_mesh) : sim.mesh;
        const SurfMode sm = vs.surf;
        const bool field_dep = sm == SurfMode::Cp || sm == SurfMode::Speed || sm == SurfMode::Voxels;
        if (sm != SurfMode::Hidden && ((field_dep && recompute) || surf_colored != sm || geom_new)) {
            flowvis::SurfaceParams sp;
            sp.group_colors = std::span<const u32>(group_colors, 256);
            switch (sm) {
                case SurfMode::Cp: case SurfMode::Voxels:
                    sp.mode = flowvis::SurfaceMode::Cp;
                    sp.scale = {render::Colormap::CoolWarm, vs.surf_lo, max_(vs.surf_hi, vs.surf_lo + 0.01f), true, cp_off};
                    sp.offset = sm == SurfMode::Voxels ? 1.0f : 0.9f;
                    break;
                case SurfMode::Speed:
                    sp.mode = flowvis::SurfaceMode::Speed;
                    sp.scale = {render::Colormap::Turbo, 0.0f, 1.5f, false};
                    sp.offset = 1.2f;
                    break;
                case SurfMode::Component: sp.mode = flowvis::SurfaceMode::Component; break;
                default: sp.mode = flowvis::SurfaceMode::Solid; sp.solid_color = 0xFFB8BCC4u; break;
            }
            flowvis::color_mesh(m, f, sp);
            surf_colored = sm;
        }
        t_upd_color = (now_sec() - a) * 1e3;
    }
    rakes_dirty = false;
    dirty = false;
    if (field_new || !field_changed) seen_field = sim.field_version;
    seen_geom = sim.geom_version;
    seen_domain = sim.domain_version;
    t_update = (now_sec() - t0) * 1e3;
}

// ============================================================================
//  Render 3D
// ============================================================================
void View::render(render::Framebuffer& fb, Rect vp, Sim& sim) {
    const double t0 = now_sec();
    cam.update(vp);
    fb.gradient(vp, 0xFF2C323Du, 0xFF0A0C10u);
    fb.clear_depth();
    if (!sim.ready) return;
    const bool ground = sim.cfg.ground != lbm::GroundMode::None;
    const float spacing = max_(0.5f / sim.dom.dx, 2.0f);   // rejilla cada 0.5 m (líneas mayores cada 2.5 m)
    const Aabb dom_cells{{-0.5f, -0.5f, -0.5f},
                         {static_cast<float>(sim.dom.nx) - 0.5f, static_cast<float>(sim.dom.ny) - 0.5f, static_cast<float>(sim.dom.nz) - 0.5f}};
    const bool below = cam.eye.z < 0.5f;
    if (ground) {
        const u32 base = below ? 0x50303640u : 0xFF2A2F37u;
        const u32 line = sim.cfg.ground == lbm::GroundMode::Moving ? 0x60D0D8E0u : 0x40B0B8C0u;
        const float off = sim.cfg.ground == lbm::GroundMode::Moving ? belt_offset : 0.0f;
        // Suelo "del pabellón" alrededor del túnel: plano infinito oscuro y sin rejilla, 2 cm por
        // debajo (para que el suelo del túnel gane el test de profundidad). Sin él, el borde del
        // dominio corta la escena en las vistas lateral/frontal (se veía el fondo bajo el suelo).
        if (!below) render::draw_ground(fb, cam, 0.5f - 0.02f / sim.dom.dx, Aabb{}, spacing, 0.0f, 0xFF1C2027u, 0x00000000u);
        if (vs.footprint && footprint.texture()) footprint.draw(fb, cam, spacing, off, base, line);
        else {
            const Aabb ext{{dom_cells.lo.x, dom_cells.lo.y, 0.5f}, {dom_cells.hi.x, dom_cells.hi.y, 0.5f}};
            render::draw_ground(fb, cam, 0.5f, ext, spacing, off, base, line);
        }
    }
    const double t1 = now_sec();
    // Malla del objeto.
    const render::Mesh& m = vs.surf == SurfMode::Voxels ? sim.vox_mesh : sim.mesh;
    render::MeshStyle st;
    st.two_sided = true;
    st.use_vertex_color = true;
    render::Light light;
    if (vs.surf != SurfMode::Hidden) render::draw_mesh(fb, cam, m, light, st);
    const double t2 = now_sec();
    if (vs.ssao) render::screen_space_edges(fb, vp, 0.55f);
    // Corte, líneas, humo, vórtices.
    flowvis::TranslucentPlane tp;
    if (vs.slice_axis > 0 && slice.tex_w() > 0) {
        slice.params.opacity = vs.slice_opacity;
        slice.draw(fb, cam);
        tp = slice.translucent_plane();
    }
    const std::span<const flowvis::TranslucentPlane> behind(&tp, 1);
    if (vs.lines) lines.draw(fb, cam, true, behind);
    if (vs.smoke) smoke.draw(fb, cam, behind);
    if (vs.vortex) vortex.render(fb, cam, behind);
    const double t3 = now_sec();
    // Túnel y flechas de fuerza.
    if (vs.wire) render::draw_box_wire(fb, cam, dom_cells, 0x46A0B4D0u, 1.0f);
    if (vs.arrows && sim.res.valid) {
        const Aabb ob = sim.object_cells();
        const Vec3 sz = ob.size();
        // Longitud de referencia de las flechas: ~¼ de la longitud del objeto para |C| = máx(|CL|,|CD|)
        // (con la mayor de las dos en 0.4 como mínimo, para que un coeficiente pequeño se vea pequeño).
        const float len_ref = 0.22f * max_(sz.x, 0.6f * sz.y);
        const float cmax = max_(max_(std::fabs(sim.res.cl), std::fabs(sim.res.cd)), 0.4f);
        Vec3 cp = sim.res.cop;
        // Coches con carga: la flecha vertical se sitúa donde una carga pura daría el MISMO balance que
        // el panel (x = eje del. + (1 − balance)·batalla). El "centro de presiones" geométrico mezcla el
        // momento de la resistencia y, con poca carga, se va fuera del coche.
        if (sim.is_car() && std::isfinite(sim.res.balance) && sim.res.cl > 0.05f) {
            const Vec3 fa = sim.to_cells(sim.built.front_axle_m), ra = sim.to_cells(sim.built.rear_axle_m);
            const float t = clamp_(1.0f - sim.res.balance * 0.01f, -0.25f, 1.25f);
            const Vec3 p = fa + (ra - fa) * t;
            cp.x = p.x; cp.y = p.y;
        }
        const float lz = len_ref * sim.res.cl / cmax;
        const float top = ob.hi.z + 0.06f * len_ref;
        // Una flecha casi paralela a la dirección de vista se proyecta como un disco (la punta cónica
        // de frente): en las vistas superior/frontal/trasera se omiten las que apuntan a la cámara.
        const bool vert_ok = std::fabs(cam.fwd.z) < 0.93f, horiz_ok = std::fabs(cam.fwd.x) < 0.93f;
        if (std::fabs(lz) > 0.5f && vert_ok) {
            if (lz > 0) render::draw_arrow(fb, cam, {cp.x, cp.y, top + lz}, {cp.x, cp.y, top}, 0xFF4AA8FFu, 3.0f);   // carga
            else render::draw_arrow(fb, cam, {cp.x, cp.y, top}, {cp.x, cp.y, top - lz}, 0xFFFFC04Au, 3.0f);          // sustentación
        }
        const float lx = len_ref * sim.res.cd / cmax;
        if (std::fabs(lx) > 0.5f && horiz_ok) {
            const float x0 = ob.hi.x + 0.04f * len_ref;
            render::draw_arrow(fb, cam, {x0, cp.y, cp.z}, {x0 + lx, cp.y, cp.z}, 0xFFFF7A45u, 3.0f);
        }
        if (vs.comp_arrows && vert_ok) {
            for (int i = 0; i < sim.res.ncomp; ++i) {
                const CompResult& c = sim.res.comp[i];
                const float A = max_(sim.built.info.ref_area_m2, 1e-6f);
                const float l = len_ref * (c.scz / A) / cmax;
                if (std::fabs(l) < 0.8f) continue;
                const u32 col = render::component_color(c.comp);
                const Vec3 p = c.cop;
                if (l > 0) render::draw_arrow(fb, cam, {p.x, p.y, p.z + l}, p, col, 2.2f);
                else render::draw_arrow(fb, cam, p, {p.x, p.y, p.z - l}, col, 2.2f);
            }
        }
    }
    if (vs.fxaa) render::fxaa(fb, vp);
    const double t4 = now_sec();
    t_mesh = (t2 - t1) * 1e3;
    t_vis = (t3 - t2) * 1e3;
    t_post = (t4 - t3) * 1e3;
    t_render = (t4 - t0) * 1e3;
}

// ============================================================================
//  Sonda del ratón
// ============================================================================
bool View::unproject(const render::Framebuffer& fb, int mx, int my, Vec3& out) const {
    if (mx < 0 || my < 0 || mx >= fb.w || my >= fb.h || !cam.vp.contains(mx, my)) return false;
    const float d = fb.drow(my)[mx];
    if (!std::isfinite(d) || d <= 0.0f) return false;
    Vec3 o, dir;
    cam.ray(static_cast<float>(mx) + 0.5f, static_cast<float>(my) + 0.5f, o, dir);
    const float c = dot(dir, cam.fwd);
    if (c <= 1e-6f) return false;
    out = o + dir * (d / c);
    return true;
}

void View::pick_probe(const render::Framebuffer& fb, Sim& sim, int mx, int my) {
    probe_ok = false;
    probe_on_slice = false;
    if (!vs.probe || !sim.ready || !cam.vp.contains(mx, my)) return;
    const lbm::FieldView f = sim.solver.field();
    Vec3 hit;
    float val = 0;
    const float sx = static_cast<float>(mx) + 0.5f, sy = static_cast<float>(my) + 0.5f;
    if (vs.slice_axis > 0 && slice.tex_w() > 0 && slice.pick(cam, sx, sy, hit, val)) {
        // Sólo si el corte está delante de lo opaco (si no, la sonda cae en la superficie).
        const float dz = dot(hit - cam.eye, cam.fwd);
        const float zb = fb.drow(my)[mx];
        if (!(dz > zb * 1.001f) && std::isfinite(val)) {
            probe = flowvis::probe(f, hit);
            const float off = cp_offset(sim);
            probe.cp -= off; probe.cp0 -= off;   // Cp referido a la presión de referencia del túnel
            probe_value = is_cp_quantity(k_slice_q[clamp_(vs.slice_q, 0, 6)]) ? val - off : val;
            probe_on_slice = true;
            probe_ok = probe.valid;
            return;
        }
    }
    Vec3 p;
    if (!unproject(fb, mx, my, p)) return;
    Vec3 o, dir;
    cam.ray(sx, sy, o, dir);
    p = p - dir * 0.8f;                     // un poco hacia la cámara: primera celda de fluido
    probe = flowvis::probe(f, p);
    const float off = cp_offset(sim);
    probe.cp -= off; probe.cp0 -= off;
    probe_ok = probe.valid;
}

} // namespace cfd::app
