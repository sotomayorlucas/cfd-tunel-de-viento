// ============================================================================
//  app/sim.cpp — simulación: dimensionado del túnel, saneado de parámetros,
//  tubería de reconstrucción (modelo → vóxeles → solver → malla), fuerzas,
//  coeficientes, balance aerodinámico, filtrado temporal y barridos.
// ============================================================================
#include "app.hpp"

#include "../geom/voxelizer.hpp"
#include "../lbm/lattice.hpp"
#include "../render/colormap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace cfd::app {

using models::Info;
using models::Params;

// ============================================================================
//  Presets de resolución
// ============================================================================
const char* preset_name(Preset p) {
    switch (p) {
        case Preset::Rapida: return "Rápida";
        case Preset::Media: return "Media";
        case Preset::Alta: return "Alta";
        case Preset::Ultra: return "Ultra";
        default: return "?";
    }
}
const char* preset_key(Preset p) {
    switch (p) {
        case Preset::Rapida: return "rapida";
        case Preset::Media: return "media";
        case Preset::Alta: return "alta";
        case Preset::Ultra: return "ultra";
        default: return "?";
    }
}
usize preset_cells(Preset p) {
    switch (p) {
        case Preset::Rapida: return 2'500'000;
        case Preset::Media: return 6'000'000;
        case Preset::Alta: return 13'000'000;
        case Preset::Ultra: return 28'000'000;
        default: return 6'000'000;
    }
}
bool parse_preset(const char* s, Preset& out) {
    for (int i = 0; i < k_npresets; ++i)
        if (!std::strcmp(s, preset_key(static_cast<Preset>(i))) || !std::strcmp(s, preset_name(static_cast<Preset>(i)))) {
            out = static_cast<Preset>(i);
            return true;
        }
    if (!std::strcmp(s, "rápida")) { out = Preset::Rapida; return true; }
    return false;
}

// ============================================================================
//  Rangos de los parámetros (UI, CLI, envolvente del dominio)
// ============================================================================
namespace {

enum class Cat : u8 { Car, Wing, WingGround, Body };

Cat category(const Info& I) {
    if (I.kind == models::Kind::F1Car) return Cat::Car;
    if (I.kind == models::Kind::Wing) return I.needs_ground ? Cat::WingGround : Cat::Wing;
    return I.needs_ground ? Cat::Car : Cat::Body;   // road_car, ahmed_25 → como coche
}

} // namespace

ParamRanges param_ranges(int model) {
    const Info& I = models::info(model);
    ParamRanges r;
    const bool f1 = I.kind == models::Kind::F1Car;
    r.ride_front = f1 ? Range{10, 120} : Range{60, 250};
    r.ride_rear = f1 ? Range{10, 200} : Range{60, 250};
    // Rake (trasera - delantera): cabeceo ≤ ~2.2° sobre la batalla, y -20 mm de "rake negativo".
    const float wb_mm = max_(I.wheelbase_m, 1.0f) * 1000.0f;
    r.rake = {-20.0f, min_(150.0f, wb_mm * std::tan(2.2f * k_deg2rad))};
    r.yaw = (I.kind == models::Kind::Body && !I.needs_ground) ? Range{-30, 30} : Range{-10, 10};
    r.aoa = I.needs_ground ? Range{-5, 15} : Range{-10, 20};
    r.flap = {-20, 20};
    r.height = I.needs_ground ? Range{10, 300} : Range{20, 2000};
    r.gap = {5, 80};
    return r;
}

namespace {

// ¿Algún grupo de la carrocería (marco Body) atraviesa el plano del suelo? Muestreo del SDF de cada
// grupo cuya caja baja de 2 mm, en una rejilla de 2 cm a z = 1 mm (sólo en casos extremos).
bool body_touches_ground(const models::Built& b) {
    const auto& gs = b.scene.groups();
    for (usize g = 0; g < gs.size(); ++g) {
        if (gs[g].frame != sdf::Frame::Body) continue;
        const Aabb& bx = gs[g].box_model;
        if (bx.empty() || bx.lo.z > 0.002f) continue;
        constexpr float h = 0.02f;
        for (float x = bx.lo.x; x <= bx.hi.x + 1e-4f; x += h)
            for (float y = bx.lo.y; y <= bx.hi.y + 1e-4f; y += h)
                if (b.scene.eval_group(static_cast<int>(g), Vec3(x, y, 0.001f)) < 0.0f) return true;
    }
    return false;
}

} // namespace

Params sanitize_params(int model, const Params& in, bool* adjusted) {
    const Info& I = models::info(model);
    const ParamRanges R = param_ranges(model);
    Params p = models::resolve_params(model, in);
    const Params before = p;
    const u32 m = I.param_mask;
    if (m & models::P_RideHeight) {
        p.ride_front_mm = clamp_(p.ride_front_mm, R.ride_front.lo, R.ride_front.hi);
        const float rake = clamp_(p.ride_rear_mm - p.ride_front_mm, R.rake.lo, R.rake.hi);
        p.ride_rear_mm = clamp_(p.ride_front_mm + rake, R.ride_rear.lo, R.ride_rear.hi);
    }
    if (m & models::P_Yaw) p.yaw_deg = clamp_(p.yaw_deg, R.yaw.lo, R.yaw.hi);
    if (m & models::P_Aoa) p.aoa_deg = clamp_(p.aoa_deg, R.aoa.lo, R.aoa.hi);
    if (m & models::P_Height) p.height_mm = clamp_(p.height_mm, R.height.lo, R.height.hi);
    if (m & models::P_FlapGap) p.flap_gap_mm = clamp_(p.flap_gap_mm, R.gap.lo, R.gap.hi);
    if (m & models::P_FrontFlap) p.front_flap_deg = clamp_(p.front_flap_deg, R.flap.lo, R.flap.hi);
    if (m & models::P_RearFlap) p.rear_flap_deg = clamp_(p.rear_flap_deg, R.flap.lo, R.flap.hi);
    // Holgura con el suelo: si alguna pieza de la carrocería lo atraviesa, se sube el coche
    // (las dos alturas, conservando el rake) de 5 en 5 mm.
    if (m & models::P_RideHeight) {
        for (int it = 0; it < 40; ++it) {
            const models::Built b = models::build(model, p);
            if (!body_touches_ground(b)) break;
            if (p.ride_front_mm + 5.0f > R.ride_front.hi || p.ride_rear_mm + 5.0f > 400.0f) break;
            p.ride_front_mm += 5.0f;
            p.ride_rear_mm += 5.0f;
        }
    }
    if (adjusted)
        *adjusted = std::fabs(p.ride_rear_mm - before.ride_rear_mm) > 0.01f || std::fabs(p.ride_front_mm - before.ride_front_mm) > 0.01f;
    return p;
}

// ============================================================================
//  Dimensionado del dominio
// ============================================================================
namespace {

// Envolvente del objeto (m) sobre los extremos de los rangos de la UI: al mover un deslizador el
// objeto no debe salirse del túnel (el dominio sólo cambia con un reinicio completo).
Aabb envelope_bounds(int model, const Params& p0, bool ground) {
    const Info& I = models::info(model);
    const ParamRanges R = param_ranges(model);
    const Params base = sanitize_params(model, p0);
    const models::Built b0 = models::build(model, base);
    Aabb box = b0.bounds_m;
    auto add = [&](Params q) { box.grow(models::build(model, sanitize_params(model, q)).bounds_m); };
    const u32 m = I.param_mask;
    if (m & models::P_Yaw) { Params q = base; q.yaw_deg = R.yaw.hi; add(q); q.yaw_deg = R.yaw.lo; add(q); }
    if (m & models::P_Aoa) { Params q = base; q.aoa_deg = R.aoa.hi; add(q); q.aoa_deg = R.aoa.lo; add(q); }
    if ((m & models::P_Height) && ground) { Params q = base; q.height_mm = R.height.hi; add(q); }
    if (m & models::P_RideHeight) {
        Params q = base;
        q.ride_front_mm = R.ride_front.hi;
        q.ride_rear_mm = R.ride_front.hi + R.rake.hi;
        add(q);
    }
    if (m & models::P_RearFlap) { Params q = base; q.rear_flap_deg = R.flap.hi; add(q); }
    if (m & models::P_FrontFlap) { Params q = base; q.front_flap_deg = R.flap.hi; add(q); }
    if (m & models::P_FlapGap) { Params q = base; q.flap_gap_mm = R.gap.hi; add(q); }
    if (I.spans_domain) { box.lo.y = b0.bounds_m.lo.y; box.hi.y = b0.bounds_m.hi.y; }
    return box;
}

// Margen m (lateral a cada lado = a·m/2... ver abajo) que da una sección (w + a·m)(h + b·m) = T.
float solve_margin(float w, float h, float a, float b, float T) {
    const float A = a * b, B = a * h + b * w, C = w * h - T;
    if (C >= 0.0f) return 0.0f;
    return (-B + std::sqrt(B * B - 4.0f * A * C)) / (2.0f * A);
}

} // namespace

DomainPlan plan_domain(int model, const Params& p, bool ground_layout, usize budget) {
    const Info& I = models::info(model);
    const Cat cat = category(I);
    DomainPlan d;
    d.ground = ground_layout;
    const Aabb ob = envelope_bounds(model, p, ground_layout);
    d.object_m = ob;
    const Vec3 s = ob.size();
    const float w = s.y;
    const float h_obj = ground_layout ? max_(ob.hi.z, 0.0f) : s.z;   // con suelo, el objeto "empieza" en z = 0
    float up = 0, down = 0, side = 0, top = 0, bottom = 0, a_front = 0;
    switch (cat) {
        case Cat::Car: {
            // Coche: 0.5 L aguas arriba, 1.6 L aguas abajo (la esponja de salida ocupa el 12 % final),
            // sección con bloqueo ≤ 10 % del área frontal: margen lateral m, techo 2m (el suelo hace
            // de espejo: altura efectiva doble), y sin suelo también 2m por debajo.
            const float L = s.x;
            up = 0.5f * L;
            down = 1.6f * L;
            a_front = I.ref_area_m2;
            const float T = a_front / 0.10f;
            float mg = solve_margin(w, h_obj, 2.0f, ground_layout ? 2.0f : 4.0f, T);
            mg = max_(mg, 0.3f * max_(w, s.z));
            side = mg; top = 2.0f * mg; bottom = ground_layout ? 0.0f : 2.0f * mg;
            d.design_dx = I.kind == models::Kind::F1Car ? 0.035f : 0.0f;
            break;
        }
        case Cat::WingGround: {
            // Ala en efecto suelo (dx de diseño 4-12 mm): dominio compacto en cuerdas.
            const float c = s.x;
            up = 1.0f * c;
            down = 2.5f * c;
            side = 0.4f * c;
            top = 1.0f * c;
            bottom = ground_layout ? 0.0f : 1.0f * c;
            a_front = w * s.z * 0.6f;
            d.design_dx = 0.012f;
            break;
        }
        case Cat::Wing: {
            const float c = s.x;
            up = 1.2f * c;
            down = 3.0f * c;
            if (I.spans_domain) side = 0.0f;
            else side = max_(0.8f * c, 0.15f * w);
            top = 1.2f * c;
            bottom = ground_layout ? 0.0f : (I.spans_domain ? 1.5f * c : 1.2f * c);
            if (I.spans_domain) top = 1.5f * c;
            a_front = w * s.z * 0.6f;
            break;
        }
        case Cat::Body: {
            // Cuerpo romo en aire libre: 2 d aguas arriba, 5 d aguas abajo (estela), bloqueo ≤ 5 %.
            const float dd = max_(s.x, s.z);
            up = 2.0f * dd;
            down = 5.0f * dd;
            a_front = I.ref_area_m2;
            const float T = a_front / 0.05f;
            float mg = solve_margin(w, ground_layout ? h_obj : s.z, 2.0f, ground_layout ? 1.0f : 2.0f, T);
            mg = max_(mg, 1.0f * max_(w, s.z) * 0.5f);
            side = mg; top = mg; bottom = ground_layout ? 0.0f : mg;
            break;
        }
    }
    const float Lx = up + s.x + down;
    const float W = I.spans_domain ? w : w + 2.0f * side;
    const float H = h_obj + top + bottom;
    d.blockage = a_front / max_(W * H, 1e-6f);
    const double vol = static_cast<double>(Lx) * W * H;
    float dx = static_cast<float>(std::cbrt(vol / static_cast<double>(max_(budget, usize(4096)))));
    int ny;
    if (I.spans_domain) {
        ny = max_(8, static_cast<int>(std::lround(W / dx)));
        dx = W / static_cast<float>(ny);             // ny·dx = ancho exacto del modelo
    } else {
        ny = max_(8, static_cast<int>(std::lround(W / dx)));
    }
    int nx = static_cast<int>(std::ceil(Lx / dx));
    nx = max_(16, (nx + 7) & ~7);
    // Con suelo: la capa z = 0 es el suelo (celda centrada en -dx/2) → +1 capa.
    int nz = max_(8, static_cast<int>(std::lround(H / dx)) + (ground_layout ? 1 : 0));
    d.nx = nx; d.ny = ny; d.nz = nz; d.dx = dx;
    // Caja del dominio: x desde (inicio del objeto - up); el sobrante de redondear va aguas abajo.
    const float x0 = ob.lo.x - up;
    const float yc = 0.5f * (ob.lo.y + ob.hi.y);
    const float y0 = I.spans_domain ? ob.lo.y : yc - 0.5f * static_cast<float>(ny) * dx;
    float z0;
    if (ground_layout) z0 = -dx;                                  // celda 0 = [-dx, 0] (suelo)
    else z0 = 0.5f * (ob.lo.z + ob.hi.z) + 0.5f * (bottom - top) - 0.5f * static_cast<float>(nz) * dx;
    d.box_m = {{x0, y0, z0}, {x0 + static_cast<float>(nx) * dx, y0 + static_cast<float>(ny) * dx, z0 + static_cast<float>(nz) * dx}};
    d.map.dx = dx;
    d.map.origin = Vec3(x0 + 0.5f * dx, y0 + 0.5f * dx, z0 + 0.5f * dx);   // centro de la celda (0,0,0)
    return d;
}

// ============================================================================
//  Balance aerodinámico
// ============================================================================
float balance_front_pct(Vec3 F, Vec3 M_ref, Vec3 ref, Vec3 front, Vec3 rear) {
    Vec3 e = front - rear;
    e.z = 0.0f;
    const float wb = length(e);
    if (!(wb > 1e-6f)) return k_nanf;
    e = e * (1.0f / wb);
    const Vec3 axis{e.y, -e.x, 0.0f};                  // eje de cabeceo (horizontal, ⟂ a la batalla)
    const Vec3 M_rear = M_ref + cross(ref - rear, F);  // momento de las fuerzas aero respecto al contacto trasero
    const float D = -F.z;                              // carga total (+ hacia abajo)
    // Equilibrio: la reacción extra del eje delantero Nf (en `front`) anula el momento: Nf·wb = −M_rear·axis.
    const float Nf = -dot(M_rear, axis) / wb;
    if (!(std::fabs(D) > 1e-12f)) return k_nanf;
    return 100.0f * Nf / D;
}

// ============================================================================
//  Filtro, historia y resultados
// ============================================================================
void ForceFilter::push(const Vec3* fs, const Vec3* ms, int k, float tau) {
    if (!init) {
        for (int i = 0; i < 256; ++i) { f[i] = fs[i]; m[i] = ms[i]; }
        init = true;
        return;
    }
    const float a = 1.0f - std::exp(-static_cast<float>(k) / max_(tau, 1.0f));
    for (int i = 1; i < 255; ++i) {
        f[i] += (fs[i] - f[i]) * a;
        m[i] += (ms[i] - m[i]) * a;
    }
}

void History::push(u64 st, float cl_, float cd_, float scz_, float scx_) {
    step[head] = st; cl[head] = cl_; cd[head] = cd_; scz[head] = scz_; scx[head] = scx_;
    head = (head + 1) % N;
    if (count < N) ++count;
}
int History::find_at_or_before(u64 s) const {
    for (int i = 0; i < count; ++i) {
        const int k = (head - 1 - i + N) % N;
        if (step[k] <= s) return k;
    }
    return -1;
}

AeroResult compute_aero(const Sim& s, const Vec3* f, const Vec3* m) {
    AeroResult r;
    if (!s.ready) return r;
    const float u = s.cfg.u_lat;
    const float A = s.built.info.ref_area_m2;
    const float qA = 0.5f * u * u * s.a_ref_cells();            // ½ρu²A en unidades de red (ρ = 1)
    const float inv = qA > 0.0f ? 1.0f / qA : 0.0f;
    Vec3 F{0, 0, 0}, M{0, 0, 0};
    for (int id = 1; id < 255; ++id) { F += f[id]; M += m[id]; }
    r.force = F; r.moment = M;
    r.cl = -F.z * inv;
    r.cd = F.x * inv;
    r.cs = F.y * inv;
    r.scz = r.cl * A;
    r.scx = r.cd * A;
    r.ld = std::fabs(r.cd) > 1e-6f ? r.cl / r.cd : 0.0f;
    const float v = s.speed_kmh / 3.6f;
    const float q = 0.5f * k_rho_air * v * v;
    r.down_n = r.scz * q;
    r.drag_n = r.scx * q;
    r.side_n = r.cs * A * q;
    r.down_kgf = r.down_n / k_g;
    r.drag_kgf = r.drag_n / k_g;
    r.power_kw = r.drag_n * v * 1e-3f;
    const Vec3 P = s.to_cells(s.built.moment_ref_m);
    // Punto de aplicación aproximado de una fuerza con momento M respecto a P: sobre la línea de
    // acción, a la altura z_c del centro de la pieza (x de My), sujeto a la caja de la pieza.
    auto cop = [&](Vec3 Fc, Vec3 Mc, const Aabb& box) {
        Vec3 c = box.center();
        if (std::fabs(Fc.z) > 1e-9f) {
            const float x = P.x + ((c.z - P.z) * Fc.x - Mc.y) / Fc.z;
            const float ext = 0.1f * max_(box.size().x, 1.0f);
            if (std::isfinite(x)) c.x = clamp_(x, box.lo.x - ext, box.hi.x + ext);
        }
        return c;
    };
    // Desglose por componente aerodinámico (varios grupos pueden compartir componente).
    constexpr int NC = static_cast<int>(sdf::Component::Count);
    Vec3 cf[NC], cm[NC];
    Aabb cb[NC];
    bool used[NC] = {};
    const auto& gs = s.built.scene.groups();
    Aabb all;
    for (usize g = 0; g < gs.size() && g < 254; ++g) {
        const int c = static_cast<int>(gs[g].component);
        if (c < 0 || c >= NC) continue;
        const int id = static_cast<int>(g) + 1;
        if (!used[c]) { cf[c] = {0, 0, 0}; cm[c] = {0, 0, 0}; cb[c] = Aabb{}; used[c] = true; }
        cf[c] += f[id];
        cm[c] += m[id];
        const Aabb bm = gs[g].box_model;
        if (!bm.empty()) {
            const Aabb bc{s.to_cells(bm.lo), s.to_cells(bm.hi)};
            cb[c].grow(bc);
            all.grow(bc);
        }
    }
    for (int c = 0; c < NC; ++c) {
        if (!used[c]) continue;
        CompResult& o = r.comp[r.ncomp++];
        o.comp = static_cast<sdf::Component>(c);
        o.name = sdf::component_name(o.comp);
        o.scz = -cf[c].z * inv * A;
        o.scx = cf[c].x * inv * A;
        o.force = cf[c];
        o.cop = cb[c].empty() ? P : cop(cf[c], cm[c], cb[c]);
    }
    r.cop = all.empty() ? P : cop(F, M, all);
    if (s.is_car()) {
        const Vec3 fc = s.to_cells(s.built.front_axle_m), rc = s.to_cells(s.built.rear_axle_m);
        r.balance = std::fabs(r.cl) > 0.05f ? balance_front_pct(F, M, P, fc, rc) : k_nanf;
    }
    r.valid = std::isfinite(r.cl) && std::isfinite(r.cd);
    return r;
}

// ============================================================================
//  Sim
// ============================================================================
namespace {
// Viscosidad de red por defecto (suelo molecular; la turbulenta la pone Smagorinsky y la de pared la ley de
// pared). 1e-4 (τ = 0.5003): mismas fuerzas que 1e-5 (NACA 0012 a 6°: CL 0.274 vs 0.277, CD igual; Ahmed igual)
// con 10× más margen de estabilidad y algo menos de ruido de escala de red (ondas acústicas poco amortiguadas).
// Con 3e-4 la resistencia ya sube ~3 %. Si algo diverge, Sim::step la sube ×3 y reinicia el flujo.
constexpr float k_default_nu = 1.0e-4f;
constexpr float k_max_nu = 0.02f;
}

float Sim::obj_len_m() const {
    const float lx = built.bounds_m.size().x;
    return max_(lx, 0.5f * built.info.ref_length_m);
}
float Sim::ft_steps() const { return dom.dx > 0 ? obj_len_m() / dom.dx / cfg.u_lat : 1000.0f; }
float Sim::flow_throughs() const { return static_cast<float>(solver.steps()) / max_(ft_steps(), 1.0f); }
float Sim::re_lattice() const { return cfg.u_lat * (obj_len_m() / dom.dx) / nu; }
float Sim::re_real() const { return (speed_kmh / 3.6f) * obj_len_m() / k_nu_air; }
Aabb Sim::object_cells() const { return {to_cells(built.bounds_m.lo), to_cells(built.bounds_m.hi)}; }
usize Sim::memory_bytes() const {
    return solver.memory_bytes() + solid.size() + (mesh.pos.size() * 28 + mesh.tri.size() * 4) +
           (vox_mesh.pos.size() * 28 + vox_mesh.tri.size() * 4);
}

void Sim::set_moment_ref() {
    mref_cells = to_cells(built.moment_ref_m);
    solver.set_moment_reference(mref_cells);
}

void Sim::id_forces(const lbm::ForceSample& s, Vec3* f, Vec3* m) const {
    // El solver ya entrega fuerzas manométricas y galileanas (lbm::Config::force_gauge/force_galilean).
    for (int id = 0; id < 256; ++id) { f[id] = s.force[id]; m[id] = s.moment[id]; }
}

float Sim::wall_nu_lat() const {
    const float U = max_(speed_kmh, 1.0f) / 3.6f;
    return dom.dx > 0.0f ? k_nu_air * cfg.u_lat / (U * dom.dx) : 0.0f;
}

void Sim::sync_wall_law() {
    if (!ready || speed_kmh == wall_speed_) return;
    wall_speed_ = speed_kmh;
    wall_nu = wall_nu_lat();
    solver.set_wall_model(cfg.wall, wall_nu);
}

void Sim::reset_forces() {
    filt.reset();
    hist.reset(static_cast<u64>(max_(1.0f, ft_steps() / 40.0f)));
    res = AeroResult{};
    converged = false;
    conv_rel = 1.0f;
    steps_since_geom = 0;
}

void Sim::update_effective_params() {
    params_eff = params;
    ride_limited = false;
    ride_gap_min_mm = k_gap_cells * dom.dx * 1000.0f;
    const Info& I = models::info(cfg.model);
    if ((I.param_mask & models::P_RideHeight) && dom.ground) {
        // h_eff = (h⁴ + g⁴)^{1/4}: = g con h → 0, ≈ h en cuanto h ≳ 1.5 g (1.19 g con h = g). Suave y monótona.
        const float h = max_(min_(params.ride_front_mm, params.ride_rear_mm), 0.0f);
        const float g = ride_gap_min_mm;
        const float h2 = h * h, g2 = g * g;
        const float d = std::sqrt(std::sqrt(h2 * h2 + g2 * g2)) - h;   // ≥ 0
        params_eff.ride_front_mm += d;
        params_eff.ride_rear_mm += d;
        ride_limited = d > 0.5f;
    }
    ride_eff_front_mm = params_eff.ride_front_mm;
    ride_eff_rear_mm = params_eff.ride_rear_mm;
}

void Sim::init() {
    const double t0 = now_sec();
    params = sanitize_params(cfg.model, cfg.params);
    cfg.params = params;
    const bool ground = cfg.ground != lbm::GroundMode::None;
    const usize budget = cfg.cells ? cfg.cells : preset_cells(cfg.preset);
    dom = plan_domain(cfg.model, params, ground, budget);
    update_effective_params();
    built = models::build(cfg.model, params_eff);
    if (cfg.nu > 0.0f) nu = cfg.nu;
    else nu = k_default_nu;
    wall_speed_ = speed_kmh;
    wall_nu = wall_nu_lat();
    lbm::Config c;
    c.nx = dom.nx; c.ny = dom.ny; c.nz = dom.nz;
    c.u_inf = cfg.u_lat;
    c.nu = nu;
    c.cs_smag = cfg.cs;
    c.collision = cfg.collision;
    c.precision = cfg.fp32 ? lbm::Precision::FP32 : lbm::Precision::FP16S;
    c.ground = cfg.ground;
    c.sponge_frac = 0.12f;
    c.bulk_omega = cfg.bulk_omega;
    c.rr_wall_layer = cfg.rr_wall_layer;
    // Arranque impulsivo por defecto (ver SimConfig::ramp_ft). Estable con la ley de pared y el rebote
    // interpolado (verificado en los 18 modelos, ver docs/FISICA.md): ya no hace falta el programa de
    // viscosidad del arranque de la primera integración.
    c.ramp_steps = static_cast<int>(max_(0.0f, cfg.ramp_ft) * ft_steps());
    c.wall_model = cfg.wall;
    c.wall_nu = wall_nu;
    c.force_gauge = true;
    c.force_galilean = true;
    c.bounce = cfg.interp_bb ? lbm::BounceBack::Interpolated : lbm::BounceBack::Implicit;
    solver.init(c);
    const usize n = dom.cells();
    if (solid.size() != n) solid.resize(n);
    solid.zero();
    std::memset(motion_on_, 0, sizeof motion_on_);
    ready = true;
    rebuild_geometry(0.5f);
    set_moment_ref();
    reset_forces();
    divergences = 0;
    if (cfg.gpu && !gpu_on()) set_gpu(true);   // (si ya estaba enganchado sigue: el gancho del solver resincroniza)
    ++domain_version;
    field_version += 1;
    t_init_ms = (now_sec() - t0) * 1e3;
}

void Sim::apply_wall_motions() {
    // Sólo con cinta móvil: ruedas girando sobre un suelo fijo no tienen sentido físico (túnel
    // antiguo con el coche quieto). El movimiento sale de Group::motion (faldones de 1979 y
    // deflectores de 2022 son marco Wheels pero sin movimiento).
    const bool belt = cfg.ground == lbm::GroundMode::Moving;
    const auto& gs = built.scene.groups();
    for (int id = 1; id < 255; ++id) {
        const usize g = static_cast<usize>(id - 1);
        const bool want = belt && g < gs.size() && gs[g].motion.moving();
        if (want) {
            const LatticeMap::LatticeMotion lm = dom.map.lattice_motion(gs[g].motion, cfg.u_lat);
            // Huella de contacto (ruedas): las capas z ≤ max(2 celdas, 7 cm) se mueven con la cinta (ver
            // lbm::WallMotion::contact_z): evita la cuña rueda-suelo de 1-2 celdas que "bombea".
            const float cz = gs[g].frame == sdf::Frame::Wheels ? max_(2.0f, 0.07f / dom.dx) : -1.0f;
            solver.set_wall_motion(static_cast<u8>(id), lbm::WallMotion{lm.v, lm.omega, lm.center, cz, true});
            motion_on_[id] = true;
        } else if (motion_on_[id]) {
            solver.set_wall_motion(static_cast<u8>(id), lbm::WallMotion{});
            motion_on_[id] = false;
        }
    }
}

namespace {
// Voxelización coherente con el rebote interpolado: sólido ⇔ d(centro) < k_vox_thicken·dx, SIN voto de
// mayoría 2×2×2 (con él el 30 % de los enlaces tenía el centro sólido FUERA de la superficie → q recortado a 1
// y cuerpo más gordo) y la pared del rebote interpolado en la superficie engrosada d = k_vox_thicken·dx: así
// todo centro sólido está dentro y todo centro fluido fuera (q ∈ [0, 1] exacto). El engrosamiento de 0.12·dx
// (el del voxelizador por defecto) mantiene estancas las piezas finas (diseño de models/ para dx ≤ 3 cm).
constexpr float k_vox_thicken = 0.12f;
// Distancia con signo a la superficie (engrosada) en celdas para el rebote interpolado del solver.
float sim_wall_sdf(const void* ctx, Vec3 pc) {
    const Sim* s = static_cast<const Sim*>(ctx);
    return s->built.scene.eval(s->dom.map.to_model(pc)) / s->dom.dx - k_vox_thicken;
}
} // namespace

// Huella de contacto de las ruedas: las celdas de fluido de las capas z = 1..2 que quedan ENCAJONADAS entre
// una rueda (encima) y el suelo (debajo, o una celda ya rellenada) pasan a ser de la rueda. Una cuña de 1-2
// celdas de alto entre el neumático y la cinta no se puede resolver: la cinta y la rueda "bombean" fluido
// dentro/fuera (medido: Cp +10 / -17 en esas celdas, ρ → 0.13 y divergencia con la ley de pared). Un
// neumático real se aplana en una huella de ~15-20 cm; con dx de 3-5 cm esto rellena 1-3 celdas por
// delante y por detrás del contacto. Esas celdas (z ≤ contact_z) se mueven con la cinta.
usize Sim::fill_contact_pockets() {
    if (!dom.ground) return 0;
    const auto& gs = built.scene.groups();
    bool wheel_id[256] = {};
    bool any = false;
    for (usize g = 0; g < gs.size() && g < 254; ++g)
        if (gs[g].component == sdf::Component::FrontWheels || gs[g].component == sdf::Component::RearWheels) { wheel_id[g + 1] = true; any = true; }
    if (!any) return 0;
    const usize nx = static_cast<usize>(dom.nx), nxny = nx * static_cast<usize>(dom.ny);
    usize filled = 0;
    for (int z = 1; z <= 2 && z + 1 < dom.nz; ++z)
        for (int y = 1; y < dom.ny - 1; ++y)
            for (int x = 1; x < dom.nx - 1; ++x) {
                const usize n = static_cast<usize>(x) + nx * static_cast<usize>(y) + nxny * static_cast<usize>(z);
                if (solid[n]) continue;
                const u8 above = solid[n + nxny];
                const bool below = z == 1 || solid[n - nxny] != 0;
                if (above && wheel_id[above] && below) { solid[n] = above; ++filled; }
            }
    return filled;
}

void Sim::rebuild_geometry(float fraction) {
    const double t0 = now_sec();
    geom::VoxelOptions vo;
    vo.thicken = k_vox_thicken;
    vo.supersample = false;
    const geom::VoxelStats vs = geom::voxelize(built.scene, dom.map, dom.nx, dom.ny, dom.nz, solid.data(), vo);
    solid_cells = vs.solid_cells + fill_contact_pockets();
    const double t1 = now_sec();
    solver.set_geometry(solid.data(), lbm::WallSdf{&sim_wall_sdf, this});
    apply_wall_motions();
    const double t2 = now_sec();
    times.vox = (t1 - t0) * 1e3;
    times.geo = (t2 - t1) * 1e3;
    times.mesh = 0;
    if (fraction > 0.0f) remesh(fraction);
    vox_dirty = true;
    ++geom_version;
}

void Sim::remesh(float fraction) {
    const double t0 = now_sec();
    geom::MeshOptions mo;
    mo.fill_color = true;
    geom::mesh_scene(built.scene, dom.map, fraction, mesh, mo);
    mesh_fraction = fraction;
    times.mesh = (now_sec() - t0) * 1e3;
    ++geom_version;
}

void Sim::ensure_vox_mesh() {
    if (!vox_dirty) return;
    geom::mesh_voxels(solid.data(), dom.nx, dom.ny, dom.nz, vox_mesh, true);
    vox_dirty = false;
}

bool Sim::set_params(const Params& p, float fraction, bool* adjusted) {
    const double t0 = now_sec();
    const Params np = sanitize_params(cfg.model, p, adjusted);
    const Params old_eff = params_eff;
    params = np;
    cfg.params = np;
    update_effective_params();
    const bool same = models::same_geometry(cfg.model, params_eff, old_eff);
    if (same && params_eff.wheels_rotating == old_eff.wheels_rotating) return false;
    const double tb = now_sec();
    built = models::build(cfg.model, params_eff);
    times.build = (now_sec() - tb) * 1e3;
    if (!same) rebuild_geometry(fraction);
    else apply_wall_motions();
    set_moment_ref();
    steps_since_geom = 0;
    converged = false;
    times.total = (now_sec() - t0) * 1e3;
    return !same;
}

void Sim::set_ground(lbm::GroundMode g) {
    if (g == cfg.ground) return;
    const bool layout = (g == lbm::GroundMode::None) != (cfg.ground == lbm::GroundMode::None);
    cfg.ground = g;
    if (layout) { init(); return; }
    solver.set_ground(g);
    apply_wall_motions();
    steps_since_geom = 0;
    converged = false;
}

void Sim::reset_flow() {
    solver.reset_flow();
    reset_forces();
    ++field_version;
}

bool Sim::step(int k) {
    if (!ready || k <= 0) return true;
    sync_wall_law();
    if (gpu_on()) {
        // iGPU. Síncrono: encola, espera y publica. Asíncrono (bucle interactivo): publica el lote anterior (si lo
        // hay) y encola este, que la GPU hace mientras la CPU dibuja el campo publicado (doble búfer: la GPU nunca
        // escribe el que se lee).
        if (!gpu_async) {
            if (!gpu->submit(k)) { std::fprintf(stderr, "[cfd] GPU: %s → vuelta a la CPU\n", gpu->error().c_str()); set_gpu(false); solver.step(k, true); return post_step(k); }
            return post_step(gpu->wait());
        }
        const int done = gpu->cycle(k);   // espera el anterior, encola éste enseguida y publica el anterior
        if (done < 0) { std::fprintf(stderr, "[cfd] GPU: %s → vuelta a la CPU\n", gpu->error().c_str()); set_gpu(false); return true; }
        return post_step(done);
    }
    solver.step(k, true);
    return post_step(k);
}

bool Sim::post_step(int k) {
    published = k;
    if (k <= 0) return true;
    ++field_version;
    steps_since_geom += static_cast<u64>(k);
    if (solver.diverged()) {
        // Recuperación: más viscosidad y flujo reiniciado desde el reposo (la app avisa con un mensaje).
        ++divergences;
        nu = min_(nu * 3.0f, k_max_nu);
        solver.set_viscosity(nu);
        solver.reset_flow();
        reset_forces();
        return false;
    }
    update_rho_ref();
    // La rampa (si la hay) y el primer ½ paso de flujo (transitorio del arranque) no entran en la media.
    if (static_cast<float>(solver.steps()) < static_cast<float>(solver.config().ramp_steps) + 0.5f * ft_steps()) return true;
    Vec3 f[256], m[256];
    id_forces(solver.forces_mean(), f, m);
    filt.push(f, m, k, 0.5f * ft_steps());
    update_results();
    return true;
}

bool Sim::set_gpu(bool on) {
    gpu_error.clear();
    if (!on) {
        if (gpu) gpu->detach();
        cfg.gpu = false;
        return true;
    }
    if (!gpu) gpu = std::make_unique<gpu::LbmGpu>();
    std::string err;
    if (!gpu->init(&err) || !gpu->attach(solver, &err)) {
        gpu_error = err.empty() ? "sin dispositivo Vulkan" : err;
        std::fprintf(stderr, "[cfd] no se pudo activar la iGPU: %s (el solver sigue en la CPU)\n", gpu_error.c_str());
        cfg.gpu = false;
        return false;
    }
    cfg.gpu = true;
    ++field_version;
    return true;
}

void Sim::finish() {
    if (!gpu_on() || !gpu->busy()) return;
    post_step(gpu->wait());
}

double Sim::gpu_step_seconds() const {
    if (!gpu_on()) return 0.0;
    const auto& st = gpu->stats();
    return st.steps > 0 ? st.gpu_ms * 1e-3 / st.steps : 0.0;
}

void Sim::update_rho_ref() {
    // Plano x a mitad de camino entre la entrada y el objeto (≥ 2 celdas de la entrada): ρ medio del fluido.
    const lbm::FieldView fv = solver.field();
    if (!fv.rho || fv.nx < 8) return;
    const int xo = static_cast<int>(object_cells().lo.x);
    const int xr = clamp_(xo / 2, 2, max_(2, fv.nx - 3));
    double acc = 0.0;
    long cnt = 0;
    for (int z = 1; z < fv.nz - 1; ++z)
        for (int y = 1; y < fv.ny - 1; ++y) {
            const usize n = fv.index(xr, y, z);
            if (fv.flags[n] & (lbm::kSolid | lbm::kInlet | lbm::kOutlet)) continue;
            acc += fv.rho[n];
            ++cnt;
        }
    if (cnt > 0) {
        const float r = static_cast<float>(acc / static_cast<double>(cnt));
        if (std::isfinite(r)) rho_ref = r;
    }
}

void Sim::update_results() {
    sync_wall_law();
    if (!filt.init) return;
    res = compute_aero(*this, filt.f, filt.m);
    const u64 st = solver.steps();
    if (st >= hist.next) {
        hist.push(st, res.cl, res.cd, res.scz, res.scx);
        hist.next = st + hist.interval;
    }
    const float ft = ft_steps();
    const u64 ramp = static_cast<u64>(solver.config().ramp_steps);
    const u64 need = static_cast<u64>(ft);
    if (st < ramp + need || steps_since_geom < need || hist.count < 2) {
        converged = false;
        conv_rel = 1.0f;
        return;
    }
    const int k_old = hist.find_at_or_before(st - need);
    if (k_old < 0) { converged = false; conv_rel = 1.0f; return; }
    const float dcl = std::fabs(res.cl - hist.cl[k_old]) / max_(std::fabs(res.cl), 0.05f);
    const float dcd = std::fabs(res.cd - hist.cd[k_old]) / max_(std::fabs(res.cd), 0.05f);
    conv_rel = max_(dcl, dcd);
    converged = conv_rel < 0.015f;
}

// ============================================================================
//  Barridos
// ============================================================================
const char* sweep_param_name(SweepParam p) {
    switch (p) {
        case SweepParam::Aoa: return "Ángulo de ataque";
        case SweepParam::Height: return "Altura sobre el suelo";
        case SweepParam::RideHeight: return "Altura de marcha (rake fijo)";
        case SweepParam::FrontFlap: return "Flap delantero";
        case SweepParam::RearFlap: return "Flap trasero";
        case SweepParam::Yaw: return "Guiñada";
        case SweepParam::FlapGap: return "Hueco del flap";
        default: return "?";
    }
}
const char* sweep_param_key(SweepParam p) {
    switch (p) {
        case SweepParam::Aoa: return "aoa";
        case SweepParam::Height: return "height";
        case SweepParam::RideHeight: return "ride";
        case SweepParam::FrontFlap: return "front_flap";
        case SweepParam::RearFlap: return "rear_flap";
        case SweepParam::Yaw: return "yaw";
        case SweepParam::FlapGap: return "gap";
        default: return "?";
    }
}
const char* sweep_param_unit(SweepParam p) {
    switch (p) {
        case SweepParam::Height: case SweepParam::RideHeight: case SweepParam::FlapGap: return "mm";
        default: return "°";
    }
}
bool parse_sweep_param(const char* s, SweepParam& out) {
    for (int i = 0; i < k_nsweep; ++i)
        if (!std::strcmp(s, sweep_param_key(static_cast<SweepParam>(i)))) { out = static_cast<SweepParam>(i); return true; }
    return false;
}
bool sweep_param_available(int model, SweepParam p) {
    const u32 m = models::info(model).param_mask;
    switch (p) {
        case SweepParam::Aoa: return m & models::P_Aoa;
        case SweepParam::Height: return (m & models::P_Height) && models::info(model).needs_ground;
        case SweepParam::RideHeight: return m & models::P_RideHeight;
        case SweepParam::FrontFlap: return m & models::P_FrontFlap;
        case SweepParam::RearFlap: return m & models::P_RearFlap;
        case SweepParam::Yaw: return m & models::P_Yaw;
        case SweepParam::FlapGap: return m & models::P_FlapGap;
        default: return false;
    }
}
float sweep_param_get(const Params& p, SweepParam sp) {
    switch (sp) {
        case SweepParam::Aoa: return p.aoa_deg;
        case SweepParam::Height: return p.height_mm;
        case SweepParam::RideHeight: return p.ride_front_mm;
        case SweepParam::FrontFlap: return p.front_flap_deg;
        case SweepParam::RearFlap: return p.rear_flap_deg;
        case SweepParam::Yaw: return p.yaw_deg;
        case SweepParam::FlapGap: return p.flap_gap_mm;
        default: return 0.0f;
    }
}
void sweep_param_set(Params& p, SweepParam sp, float v) {
    switch (sp) {
        case SweepParam::Aoa: p.aoa_deg = v; break;
        case SweepParam::Height: p.height_mm = v; break;
        case SweepParam::RideHeight: {
            const float rake = p.ride_rear_mm - p.ride_front_mm;
            p.ride_front_mm = v;
            p.ride_rear_mm = v + rake;
            break;
        }
        case SweepParam::FrontFlap: p.front_flap_deg = v; break;
        case SweepParam::RearFlap: p.rear_flap_deg = v; break;
        case SweepParam::Yaw: p.yaw_deg = v; break;
        case SweepParam::FlapGap: p.flap_gap_mm = v; break;
        default: break;
    }
}

float Sweep::progress() const {
    if (done) return 1.0f;
    if (!running || n <= 0) return 0.0f;
    const float per = settle_ft + avg_ft;
    const float in_pt = (averaging ? settle_ft : 0.0f) +
                        (phase_len ? static_cast<float>(phase_steps) / static_cast<float>(phase_len) : 0.0f) *
                            (averaging ? avg_ft : settle_ft);
    return clamp_((static_cast<float>(idx) * per + in_pt) / (static_cast<float>(n) * per), 0.0f, 1.0f);
}

namespace {
void sweep_apply_point(Sweep& sw, Sim& sim) {
    Params p = sw.base;
    sweep_param_set(p, sw.param, sw.value_at(sw.idx));
    sim.set_params(p, 0.5f);
    sw.averaging = false;
    sw.phase_steps = 0;
    sw.phase_len = static_cast<u64>(max_(1.0f, sw.settle_ft * sim.ft_steps()));
    sw.acc_steps = 0;
}
}

bool sweep_start(Sweep& sw, Sim& sim) {
    if (!sim.ready || !sweep_param_available(sim.cfg.model, sw.param)) return false;
    sw.n = clamp_(sw.n, 2, Sweep::k_max);
    sw.model = sim.cfg.model;
    sw.base = sim.params;
    sw.idx = 0;
    sw.npts = 0;
    sw.running = true;
    sw.done = false;
    sweep_apply_point(sw, sim);
    return true;
}

bool sweep_after_step(Sweep& sw, Sim& sim, int k) {
    if (!sw.running || k <= 0) return false;
    if (sw.averaging) {
        Vec3 gf[256], gm[256];
        sim.id_forces(sim.solver.forces_mean(), gf, gm);
        for (int id = 1; id < 255; ++id) {
            sw.acc_f[id][0] += static_cast<double>(gf[id].x) * k;
            sw.acc_f[id][1] += static_cast<double>(gf[id].y) * k;
            sw.acc_f[id][2] += static_cast<double>(gf[id].z) * k;
            sw.acc_m[id][0] += static_cast<double>(gm[id].x) * k;
            sw.acc_m[id][1] += static_cast<double>(gm[id].y) * k;
            sw.acc_m[id][2] += static_cast<double>(gm[id].z) * k;
        }
        sw.acc_steps += static_cast<u64>(k);
    }
    sw.phase_steps += static_cast<u64>(k);
    if (sw.phase_steps < sw.phase_len) return false;
    if (!sw.averaging) {
        sw.averaging = true;
        sw.phase_steps = 0;
        sw.phase_len = static_cast<u64>(max_(1.0f, sw.avg_ft * sim.ft_steps()));
        std::memset(sw.acc_f, 0, sizeof sw.acc_f);
        std::memset(sw.acc_m, 0, sizeof sw.acc_m);
        sw.acc_steps = 0;
        return false;
    }
    // Punto terminado: coeficientes de la media.
    Vec3 f[256], m[256];
    const double inv = sw.acc_steps ? 1.0 / static_cast<double>(sw.acc_steps) : 0.0;
    for (int id = 0; id < 256; ++id) {
        f[id] = Vec3(static_cast<float>(sw.acc_f[id][0] * inv), static_cast<float>(sw.acc_f[id][1] * inv), static_cast<float>(sw.acc_f[id][2] * inv));
        m[id] = Vec3(static_cast<float>(sw.acc_m[id][0] * inv), static_cast<float>(sw.acc_m[id][1] * inv), static_cast<float>(sw.acc_m[id][2] * inv));
    }
    const AeroResult r = compute_aero(sim, f, m);
    if (sw.npts < Sweep::k_max) {
        SweepPoint& P = sw.pts[sw.npts++];
        P.x = sweep_param_get(sim.params, sw.param);
        P.cl = r.cl; P.cd = r.cd; P.cs = r.cs; P.scz = r.scz; P.scx = r.scx; P.ld = r.ld; P.bal = r.balance;
    }
    ++sw.idx;
    if (sw.idx >= sw.n) {
        sw.running = false;
        sw.done = true;
        sim.set_params(sw.base, 0.5f);
        return true;
    }
    sweep_apply_point(sw, sim);
    return false;
}

void sweep_stop(Sweep& sw, Sim& sim) {
    if (!sw.running) return;
    sw.running = false;
    sw.done = sw.npts > 0;
    sim.set_params(sw.base, 0.5f);
}

bool sweep_write_csv(const Sweep& sw, const Sim& sim, const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    const Info& I = sim.built.info;
    std::fprintf(fp, "# Barrido de %s — modelo %s (%s)\n", sweep_param_name(sw.param), I.id.c_str(), I.name.c_str());
    std::fprintf(fp, "# red %dx%dx%d, dx = %.2f mm, u_red = %.3f, nu_red = %.2e, suelo = %d, A_ref = %.4f m2\n",
                 sim.dom.nx, sim.dom.ny, sim.dom.nz, static_cast<double>(sim.dom.dx * 1e3f), static_cast<double>(sim.cfg.u_lat),
                 static_cast<double>(sim.nu), static_cast<int>(sim.cfg.ground), static_cast<double>(I.ref_area_m2));
    std::fprintf(fp, "# asentamiento %.2f + promedio %.2f pasos de flujo por punto; fuerzas a %.0f km/h, rho = 1.225 kg/m3\n",
                 static_cast<double>(sw.settle_ft), static_cast<double>(sw.avg_ft), static_cast<double>(sim.speed_kmh));
    std::fprintf(fp, "%s_%s,CL_carga,CD,CS,SCz_m2,SCx_m2,L_D,balance_del_pct,carga_N,resistencia_N\n", sweep_param_key(sw.param),
                 std::strcmp(sweep_param_unit(sw.param), "mm") == 0 ? "mm" : "grados");
    const float v = sim.speed_kmh / 3.6f, q = 0.5f * k_rho_air * v * v;
    for (int i = 0; i < sw.npts; ++i) {
        const SweepPoint& P = sw.pts[i];
        std::fprintf(fp, "%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.2f,%.1f,%.1f\n", static_cast<double>(P.x), static_cast<double>(P.cl),
                     static_cast<double>(P.cd), static_cast<double>(P.cs), static_cast<double>(P.scz), static_cast<double>(P.scx),
                     static_cast<double>(P.ld), static_cast<double>(P.bal), static_cast<double>(P.scz * q), static_cast<double>(P.scx * q));
    }
    std::fclose(fp);
    return true;
}

} // namespace cfd::app
