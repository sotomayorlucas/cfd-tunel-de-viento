// ============================================================================
//  tests/test_models.cpp — catálogo de modelos (PASS/FAIL, código ≠ 0 si falla).
//
//  Comprueba para cada modelo: construcción, cotas plausibles de la época,
//  ruedas apoyadas, nada bajo el suelo, simetría, nº de grupos, efecto de los
//  parámetros (altura, DRS, flaps), orientación de los alerones (carga, no
//  sustentación), supervivencia de cada grupo a dx = 3 cm, continuidad de cada
//  elemento de ala en la red (sin fugas diagonales), ausencia de piezas flotantes
//  y coste medio de evaluación del SDF (objetivo < 2 µs por punto).
// ============================================================================
#include "core/mem.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "models/f1_common.hpp"   // utilidades internas (colocación de elementos con ranura)
#include "models/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;
using sdf::Component;

namespace {

int g_fail = 0, g_pass = 0;
void check(bool ok, const char* what, const std::string& detail = "") {
    (ok ? g_pass : g_fail)++;
    if (!ok) std::printf("  FAIL  %s  %s\n", what, detail.c_str());
}
std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, f, a, b, c, d);
    return buf;
}

bool is_wheel(const sdf::Group& g) {
    return g.frame == sdf::Frame::Wheels && (g.component == Component::FrontWheels || g.component == Component::RearWheels);
}

// Punto más bajo de un grupo: columnas cada `step` sobre su AABB; en cada columna se sube con
// sphere tracing sobre el SDF del grupo (no se saltan piezas finas: el SDF es una cota).
float group_min_z(const sdf::Scene& sc, int gi, float step = 0.01f) {
    const Aabb b = sc.groups()[static_cast<usize>(gi)].box_model;
    const int nx = static_cast<int>((b.hi.x - b.lo.x) / step) + 2, ny = static_cast<int>((b.hi.y - b.lo.y) / step) + 2;
    std::vector<Padded<float>> best(static_cast<usize>(pool().size()) + 1);
    for (auto& v : best) v.value = 1e30f;
    parallel_for(0, nx, 4, [&](i64 lo, i64 hi) {
        const int w = max_(ThreadPool::worker_index(), 0);
        float m = best[static_cast<usize>(w)].value;
        for (i64 i = lo; i < hi; ++i)
            for (int j = 0; j < ny; ++j) {
                const float x = b.lo.x + static_cast<float>(i) * step, y = b.lo.y + static_cast<float>(j) * step;
                float z = b.lo.z - 0.002f;
                for (int it = 0; it < 200 && z < min_(b.hi.z, m); ++it) {
                    const float d = sc.eval_group(gi, Vec3(x, y, z));
                    if (d <= 1e-4f) { m = min_(m, z); break; }
                    z += max_(d, 2e-4f);
                }
            }
        best[static_cast<usize>(w)].value = m;
    });
    float m = 1e30f;
    for (auto& v : best) m = min_(m, v.value);
    return m;
}

// ---- Voxelización de prueba (misma regla base que geom::voxelize: sdf(centro) < 0.12·dx) ----
struct Grid {
    int nx = 0, ny = 0, nz = 0;
    float dx = 0.03f;
    Vec3 origin;                  // centro de la celda (0,0,0)
    std::vector<u8> id;           // 0 fluido, 1..254 grupo, 255 suelo
    CFD_INLINE usize at(int i, int j, int k) const { return static_cast<usize>(i) + static_cast<usize>(nx) * (static_cast<usize>(j) + static_cast<usize>(ny) * static_cast<usize>(k)); }
};

Grid voxelize(const models::Built& b, float dx, float thicken, Vec3 offset = Vec3(0.0f)) {
    Grid g;
    g.dx = dx;
    const Aabb bb = b.bounds_m.expanded(2.0f * dx);
    const bool ground = b.info.needs_ground;
    g.origin = Vec3(bb.lo.x, bb.lo.y, ground ? -0.5f * dx : bb.lo.z) + offset;   // con suelo: capa k=0 = suelo
    g.nx = static_cast<int>((bb.hi.x - g.origin.x) / dx) + 2;
    g.ny = static_cast<int>((bb.hi.y - g.origin.y) / dx) + 2;
    g.nz = static_cast<int>((bb.hi.z - g.origin.z) / dx) + 2;
    g.id.assign(static_cast<usize>(g.nx) * static_cast<usize>(g.ny) * static_cast<usize>(g.nz), 0);
    const sdf::Scene& sc = b.scene;
    parallel_for(0, static_cast<i64>(g.nz) * g.ny, 8, [&](i64 lo, i64 hi) {
        for (i64 r = lo; r < hi; ++r) {
            const int k = static_cast<int>(r / g.ny), j = static_cast<int>(r % g.ny);
            for (int i = 0; i < g.nx; ++i) {
                u8 v = 0;
                if (ground && k == 0) v = 255;
                else {
                    const Vec3 p = g.origin + Vec3(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)) * dx;
                    int gid = 0;
                    if (sc.eval(p, &gid) < thicken * dx) v = static_cast<u8>(gid);
                }
                g.id[g.at(i, j, k)] = v;
            }
        }
    });
    return g;
}

// Componentes 6-conexas de celdas sólidas; el suelo (255) actúa de conector (una pieza apoyada
// no flota). Sólo cuentan las componentes con alguna celda del modelo.
int components(const Grid& g, usize& largest) {
    std::vector<u8> seen(g.id.size(), 0);
    std::vector<u32> stack;
    int n = 0;
    largest = 0;
    for (usize s = 0; s < g.id.size(); ++s) {
        if (!g.id[s] || seen[s]) continue;
        usize cnt = 0, model_cells = 0;
        stack.clear();
        stack.push_back(static_cast<u32>(s));
        seen[s] = 1;
        while (!stack.empty()) {
            const u32 c = stack.back();
            stack.pop_back();
            ++cnt;
            model_cells += g.id[c] != 255;
            const int i = static_cast<int>(c % static_cast<u32>(g.nx));
            const int j = static_cast<int>((c / static_cast<u32>(g.nx)) % static_cast<u32>(g.ny));
            const int k = static_cast<int>(c / (static_cast<u32>(g.nx) * static_cast<u32>(g.ny)));
            const int nb[6][3] = {{i - 1, j, k}, {i + 1, j, k}, {i, j - 1, k}, {i, j + 1, k}, {i, j, k - 1}, {i, j, k + 1}};
            for (const auto& q : nb) {
                if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= g.nx || q[1] >= g.ny || q[2] >= g.nz) continue;
                const usize t = g.at(q[0], q[1], q[2]);
                if (g.id[t] && !seen[t]) { seen[t] = 1; stack.push_back(static_cast<u32>(t)); }
            }
        }
        if (model_cells) { ++n; largest = max_(largest, cnt); }
    }
    return n;
}

// Línea media (camber) de una primitiva Wing a media envergadura, en espacio modelo.
std::vector<Vec3> camber_line(const sdf::Scene& sc, const sdf::Group& grp, const sdf::Prim& pr, float s_frac) {
    const float* E = sc.pool.data() + pr.data_off;
    const u32 n = pr.data_n;
    std::vector<Vec3> out;
    const float s = s_frac, c = lerp(pr.p[0], pr.p[1], s);
    const float xle = pr.p[3] * s + pr.p[7] * s * s, zle = pr.p[4] * s + pr.p[6] * s * s;
    const Xform F = grp.frame == sdf::Frame::Body ? sc.body_frame() : (grp.frame == sdf::Frame::Wheels ? sc.wheel_frame() : Xform{});
    for (int k = 1; k < 200; ++k) {
        const float u = static_cast<float>(k) / 200.0f;
        float vmin = 1e30f, vmax = -1e30f;
        for (u32 e = 0; e < n; ++e) {
            const float ax = E[5 * e], ay = E[5 * e + 1], ex = E[5 * e + 2], ey = E[5 * e + 3];
            const float bx = ax + ex;
            if ((u - ax) * (u - bx) > 0.0f || std::fabs(ex) < 1e-9f) continue;
            const float t = (u - ax) / ex, v = ay + t * ey;
            vmin = min_(vmin, v); vmax = max_(vmax, v);
        }
        if (vmin > vmax) continue;
        const Vec3 q(xle + c * u, s * pr.p[2], zle + c * 0.5f * (vmin + vmax));
        out.push_back(F.apply(pr.xf.apply(q)));
    }
    return out;
}

// ¿Es el elemento 4-conexo en la red desde el BA hasta el BS? (plano XZ en su y).
// Una barrera 4-conexa de celdas sólidas separa el fluido 8-conexo: no hay fuga posible por
// los enlaces diagonales de D3Q19 a través del elemento. Se recorre la línea media para hallar
// la primera y la última celda sólida y se hace un BFS 4-conexo por celdas sólidas en una
// ventana alrededor del elemento.
bool element_connected(const sdf::Scene& sc, const std::vector<Vec3>& line, float dx, Vec3 origin, int& path_cells) {
    const float y = line.front().y;
    auto solid = [&](int i, int k) {
        const Vec3 p = origin + Vec3(static_cast<float>(i) * dx, 0.0f, static_cast<float>(k) * dx);
        return sc.eval(Vec3(p.x, y, p.z)) < 0.12f * dx;
    };
    std::vector<std::pair<int, int>> path;
    for (usize a = 0; a + 1 < line.size(); ++a)
        for (int sub = 0; sub < 8; ++sub) {
            const Vec3 p = lerp(line[a], line[a + 1], static_cast<float>(sub) / 8.0f);
            const int i = static_cast<int>(std::lround((p.x - origin.x) / dx)), k = static_cast<int>(std::lround((p.z - origin.z) / dx));
            if (path.empty() || path.back() != std::make_pair(i, k)) path.emplace_back(i, k);
        }
    path_cells = static_cast<int>(path.size());
    int a = 0, e = static_cast<int>(path.size()) - 1;
    while (a <= e && !solid(path[static_cast<usize>(a)].first, path[static_cast<usize>(a)].second)) ++a;
    while (e >= a && !solid(path[static_cast<usize>(e)].first, path[static_cast<usize>(e)].second)) --e;
    if (a > e) return false;                                   // el elemento desaparece
    int i0 = INT32_MAX, i1 = INT32_MIN, k0 = INT32_MAX, k1 = INT32_MIN;
    for (auto [i, k] : path) { i0 = min_(i0, i); i1 = max_(i1, i); k0 = min_(k0, k); k1 = max_(k1, k); }
    i0 -= 2; i1 += 2; k0 -= 2; k1 += 2;
    const int W = i1 - i0 + 1, H = k1 - k0 + 1;
    std::vector<u8> m(static_cast<usize>(W) * static_cast<usize>(H), 0);   // 0 = sin visitar, 1 = sólido, 2 = fluido
    std::vector<std::pair<int, int>> st{path[static_cast<usize>(a)]};
    auto idx = [&](int i, int k) { return static_cast<usize>(i - i0) + static_cast<usize>(W) * static_cast<usize>(k - k0); };
    m[idx(st[0].first, st[0].second)] = 1;
    const auto goal = path[static_cast<usize>(e)];
    while (!st.empty()) {
        const auto [i, k] = st.back();
        st.pop_back();
        if (std::make_pair(i, k) == goal) return true;
        const int nb[4][2] = {{i - 1, k}, {i + 1, k}, {i, k - 1}, {i, k + 1}};
        for (const auto& q : nb) {
            if (q[0] < i0 || q[0] > i1 || q[1] < k0 || q[1] > k1) continue;
            u8& v = m[idx(q[0], q[1])];
            if (v) continue;
            v = solid(q[0], q[1]) ? 1 : 2;
            if (v == 1) st.emplace_back(q[0], q[1]);
        }
    }
    return false;
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// Pose 2D (sección raíz) de una primitiva Wing: inversa de add_elements (rot_y(aoa), BA = xf.t).
models::detail::ElemPose pose_of(const sdf::Prim& pr) {
    models::detail::ElemPose e;
    e.x = pr.xf.t.x; e.z = pr.xf.t.z;
    e.aoa = std::atan2(pr.xf.R.m[0][2], pr.xf.R.m[0][0]);
    e.chord = pr.p[0];
    e.prof = {pr.data_off, pr.data_n};
    return e;
}
// Elementos (primitivas Wing, en orden de construcción) de los alerones delantero y trasero.
std::vector<std::vector<const sdf::Prim*>> wing_chains(const sdf::Scene& sc) {
    std::vector<std::vector<const sdf::Prim*>> out;
    for (const auto& g : sc.groups()) {
        if (g.component != Component::FrontWing && g.component != Component::RearWing) continue;
        out.emplace_back();
        for (u32 i = g.first; i < g.first + g.count; ++i)
            if (sc.prims()[i].type == sdf::PrimType::Wing) out.back().push_back(&sc.prims()[i]);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    pool().start();
    const double t_start = now_sec();
    std::printf("test_models: %d modelos, %d hilos\n", models::count(), pool().size());
    check(models::count() >= 18, "catálogo con >= 18 modelos");
    check(models::find("f1_2022") >= 0 && models::find("no_existe") == -1, "find()");

    // --- Marcos de coche ---------------------------------------------------------------------
    {
        const Xform B = models::car_body_frame(0.03f, 0.08f, 3.6f, 0.0f);
        const Vec3 f = B.apply(Vec3(0, 0, 0)), r = B.apply(Vec3(3.6f, 0, 0));
        check(std::fabs(f.z - 0.03f) < 1e-6f && std::fabs(r.z - 0.08f) < 1e-5f && std::fabs(r.x - 3.6f) < 1e-3f,
              "car_body_frame: plank a ride_front / ride_rear en los ejes", fmt("f.z=%.5f r.z=%.5f r.x=%.5f", f.z, r.z, r.x));
        const Xform Y = models::car_body_frame(0.03f, 0.03f, 3.6f, 10.0f * k_deg2rad);
        const Vec3 m = Y.apply(Vec3(1.8f, 0, 0));
        check(length(m - Vec3(1.8f, 0, 0.03f)) < 1e-5f, "guiñada alrededor del centro entre ejes");
        const Xform W = models::car_wheel_frame(3.6f, 0.0f);
        check(length(W.apply(Vec3(1, 2, 3)) - Vec3(1, 2, 3)) < 1e-6f, "car_wheel_frame(yaw=0) = identidad");
    }

    // --- Colocación de elementos con ranura exacta (regula falsi) y giro del DRS ----------------
    {
        sdf::Scene s;
        models::detail::ProfileCache pc;
        const auto hm = pc.get(s, 0.08f, 0.40f, 0.14f, true, 0.12f), hf = pc.get(s, 0.05f, 0.45f, 0.17f, true, 0.17f);
        check(pc.get(s, 0.08f, 0.40f, 0.14f, true, 0.12f).off == hm.off, "caché de perfiles reutiliza el handle");
        float worst = 0.0f;
        for (float gap : {0.01f, 0.025f, 0.045f, 0.085f})
            for (float aoa : {-10.0f, -25.0f, -40.0f}) {
                const models::detail::ElemPose m{0.0f, 0.0f, -8.0f * k_deg2rad, 0.32f, hm};
                const auto f = models::detail::place_next(s, m, hf, 0.24f, aoa * k_deg2rad, gap, 0.06f);
                worst = max_(worst, std::fabs(models::detail::elem_elem_dist(s, m, f) - gap));
            }
        check(worst < 5e-5f, "place_next: ranura mínima = gap (±0.05 mm)", fmt("error %.2e m", worst));
        const models::detail::ElemPose e{1.0f, 0.5f, -30.0f * k_deg2rad, 0.25f, hf};
        const auto r = models::detail::rotate_about_te(e, -6.0f * k_deg2rad);
        check(length(models::detail::elem_te(r) - models::detail::elem_te(e)) < 1e-5f && r.z > e.z + 0.05f,
              "DRS: el flap gira alrededor del BS y el BA sube");
    }

    std::printf("\n%-15s %6s %6s %6s %4s %7s %9s %9s %7s %8s %6s\n", "modelo", "largo", "ancho", "alto", "grp", "build",
                "eval(ns)", "evalP(ns)", "rueda", "vox3cm", "comp");
    WyRand rng(1234);
    for (int idx = 0; idx < models::count(); ++idx) {
        const models::Info& I = models::info(idx);
        // Tiempo de construcción (mediana de 7)
        std::vector<double> tb;
        models::Built b;
        for (int r = 0; r < 7; ++r) { const double t0 = now_sec(); b = models::build(idx, models::Params{}); tb.push_back(now_sec() - t0); }
        const sdf::Scene& sc = b.scene;
        const Aabb bb = b.bounds_m;
        const Vec3 sz = bb.size();
        const std::string id = I.id;
        const bool car = I.kind == models::Kind::F1Car;
        auto T = [&](const char* w) { static std::string s; s = id + ": " + w; return s.c_str(); };

        check(!bb.empty() && std::isfinite(sz.x) && sz.x > 0.1f, T("se construye con cotas finitas"));
        check(sc.group_count() >= 1 && sc.group_count() <= 254, T("nº de grupos 1..254"));
        check(sc.group_count() <= 60, T("nº de grupos <= 60 (coste)"), fmt("%.0f", static_cast<double>(sc.group_count())));
        check(b.params.ride_front_mm == I.default_ride_front_mm && b.params.ride_rear_mm == I.default_ride_rear_mm, T("defectos resueltos"));

        // --- Cotas plausibles --------------------------------------------------------------
        if (car) {
            check(sz.x > 3.8f && sz.x < 5.9f, T("largo plausible"), fmt("%.2f m", sz.x));
            check(sz.y > 1.6f && sz.y < 2.2f, T("ancho plausible"), fmt("%.2f m", sz.y));
            check(sz.z > 0.75f && bb.hi.z < 1.35f, T("alto plausible"), fmt("%.2f m", bb.hi.z));
            if (I.reg_width_m > 0.0f) check(sz.y <= I.reg_width_m * 1.02f, T("ancho <= reglamento + 2%"), fmt("%.3f > %.3f", sz.y, I.reg_width_m));
            check(std::fabs(b.rear_axle_m.x - b.front_axle_m.x - I.wheelbase_m) < 1e-3f, T("ejes separados por la batalla"));
        }

        // --- Ruedas apoyadas y nada por debajo del suelo ----------------------------------------
        float wheel_min = 1e30f, wheel_max = -1e30f, body_min = 1e30f;
        std::string low_group;
        for (usize gi = 0; gi < sc.group_count(); ++gi) {
            const auto& g = sc.groups()[gi];
            if (!I.needs_ground && !car) continue;
            const float mz = group_min_z(sc, static_cast<int>(gi));
            if (is_wheel(g)) { wheel_min = min_(wheel_min, mz); wheel_max = max_(wheel_max, mz); }
            else if (!(id == "ahmed_25" && g.component == Component::Suspension)) {
                if (mz < body_min) { body_min = mz; low_group = g.name; }
            } else {   // patas del Ahmed: deben apoyar
                check(mz >= -0.010f && mz <= 0.002f, T("patas apoyadas en el suelo"), fmt("%.4f", mz));
            }
        }
        if (wheel_min < 1e29f)
            check(wheel_min >= -0.010f && wheel_max <= 0.002f, T("ruedas apoyadas (min z en [-10, +2] mm)"),
                  fmt("min %.4f max %.4f", wheel_min, wheel_max));
        if (body_min < 1e29f) check(body_min >= -0.0005f, T("ninguna pieza bajo el suelo"), low_group + fmt(" z=%.4f", body_min));
        if (!car && I.kind == models::Kind::Wing) {
            float mz = 1e30f;
            for (usize gi = 0; gi < sc.group_count(); ++gi)
                if (sc.groups()[gi].component != Component::Endplate) mz = min_(mz, group_min_z(sc, static_cast<int>(gi), 0.005f));
            check(std::fabs(mz - b.params.height_mm * 1e-3f) < 0.002f, T("punto más bajo del ala = parámetro de altura"), fmt("%.4f", mz));
        }

        // Aire libre: el objeto flota a la altura por defecto (lejos del suelo potencial z = 0).
        if (!I.needs_ground) {
            check(I.default_height_mm >= 500.0f, T("aire libre: altura por defecto >= 0.5 m"), fmt("%.0f mm", I.default_height_mm));
            check(bb.lo.z >= 0.45f, T("aire libre: el objeto no toca el suelo potencial"), fmt("lo.z = %.3f", bb.lo.z));
        }
        if ((I.param_mask & models::P_Height) && I.kind == models::Kind::Body)
            check(std::fabs(bb.lo.z - b.params.height_mm * 1e-3f) < 0.01f, T("punto más bajo = parámetro de altura"), fmt("%.3f", bb.lo.z));

        // --- Simetría ----------------------------------------------------------------------
        {
            float worst = 0.0f;
            for (int k = 0; k < 20000; ++k) {
                const Vec3 p(rng.uniform(bb.lo.x, bb.hi.x), rng.uniform(0.0f, bb.hi.y), rng.uniform(bb.lo.z, bb.hi.z));
                worst = max_(worst, std::fabs(sc.eval(p) - sc.eval(Vec3(p.x, -p.y, p.z))));
            }
            check(worst < 1e-5f, T("simétrico en Y (guiñada 0)"), fmt("%.2e", worst));
        }

        // --- Plano de simetría: el interior es NEGATIVO también en y = 0 ------------------------
        // (dos mitades espejadas de una extrusión que acaba justo en y = 0 dan sdf = 0 en todo su
        //  interior sobre el plano: una lámina "fuera" para cualquier prueba d < 0 — mallador,
        //  voxelización sin engrosamiento, siembra de partículas).
        {
            int inside = 0, seam = 0;
            float worst = -1e30f;
            WyRand r3(7);
            for (int k = 0; k < 40000; ++k) {
                const Vec3 p(r3.uniform(bb.lo.x, bb.hi.x), 0.0f, r3.uniform(bb.lo.z, bb.hi.z));
                // Hondo dentro a ±2 cm del plano (simetría) → también debe estarlo en y = 0.
                // Junto a la costura el sdf vale max(d2, -|y|): a 2 mm sería ≥ -2 mm (no sirve), a 2 cm
                // llega a -2 cm. El umbral -1.5 cm excluye muescas reales pequeñas en el plano (vértice
                // del halo: dos cápsulas espejadas en V, +1 mm en y = 0 y sólo -5 mm a ±2 cm).
                if (sc.eval(Vec3(p.x, 0.02f, p.z)) > -0.015f) continue;
                ++inside;
                const float d0 = sc.eval(p);
                worst = max_(worst, d0);
                seam += d0 >= 0.0f;
            }
            check(seam == 0, T("sin costura en y = 0 (sdf < 0 en el interior del plano de simetría)"),
                  fmt("%.0f de %.0f puntos interiores con sdf >= 0 (máx %.4f)", seam, inside, worst));
        }

        // --- Efecto de los parámetros -----------------------------------------------------------
        auto differs = [&](const models::Params& pa, const models::Params& pb, Component comp) {
            const models::Built A = models::build(idx, pa), B = models::build(idx, pb);
            Aabb box;
            for (const auto& g : A.scene.groups()) if (g.component == comp) box.grow(g.box_model);
            if (box.empty()) return false;
            box = box.expanded(0.05f);
            WyRand r2(99);
            int diff = 0;
            for (int k = 0; k < 20000; ++k) {
                const Vec3 p(r2.uniform(box.lo.x, box.hi.x), r2.uniform(box.lo.y, box.hi.y), r2.uniform(box.lo.z, box.hi.z));
                diff += (A.scene.eval(p) < 0.0f) != (B.scene.eval(p) < 0.0f);
            }
            return diff > 20;
        };
        if (I.param_mask & models::P_RideHeight) {
            int gf = -1;   // fondo si existe; si no, la carrocería
            for (usize gi = 0; gi < sc.group_count(); ++gi)
                if (sc.groups()[gi].component == Component::Floor) { gf = static_cast<int>(gi); break; }
            for (usize gi = 0; gi < sc.group_count() && gf < 0; ++gi)
                if (sc.groups()[gi].component == Component::Body) gf = static_cast<int>(gi);
            models::Params p2;
            p2.ride_front_mm = I.default_ride_front_mm + 30.0f;
            p2.ride_rear_mm = I.default_ride_rear_mm + 30.0f;
            const models::Built b2 = models::build(idx, p2);
            const float z1 = group_min_z(sc, gf), z2 = group_min_z(b2.scene, gf);
            check(std::fabs((z2 - z1) - 0.030f) < 0.004f, T("+30 mm de altura sube el fondo 30 mm"), fmt("dz=%.4f", z2 - z1));
            float wz = 1e30f;
            for (usize gi = 0; gi < b2.scene.group_count(); ++gi)
                if (is_wheel(b2.scene.groups()[gi])) wz = min_(wz, group_min_z(b2.scene, static_cast<int>(gi)));
            if (wz < 1e29f) check(wz >= -0.010f && wz <= 0.002f, T("las ruedas siguen apoyadas al cambiar la altura"));
        }
        if (I.param_mask & models::P_Drs) {
            models::Params p2; p2.drs_open = true;
            check(differs(models::Params{}, p2, Component::RearWing), T("DRS / modo X cambia el alerón trasero"));
            if (id == "f1_2026") check(differs(models::Params{}, p2, Component::FrontWing), T("modo X abre el alerón delantero"));
        }
        if (I.param_mask & models::P_FrontFlap) {
            models::Params p2; p2.front_flap_deg = 6.0f;
            const Component c = car ? Component::FrontWing : Component::WingFlap;
            check(differs(models::Params{}, p2, c), T("flap delantero cambia la geometría"));
        }
        if (I.param_mask & models::P_RearFlap) {
            models::Params p2; p2.rear_flap_deg = 6.0f;
            check(differs(models::Params{}, p2, Component::RearWing), T("flap trasero cambia la geometría"));
        }
        if (I.param_mask & models::P_Aoa) {
            models::Params p2; p2.aoa_deg = I.default_aoa_deg + 5.0f;
            const Component c = sc.groups()[0].component;
            check(differs(models::Params{}, p2, c), T("AoA cambia la geometría"));
        }
        if (I.param_mask & models::P_Yaw) {
            models::Params p2; p2.yaw_deg = 10.0f;
            const models::Built b2 = models::build(idx, p2);
            check(b2.bounds_m.size().y > sz.y + 0.02f, T("la guiñada gira el modelo"));
        }
        check(models::same_geometry(idx, models::Params{}, models::resolve_params(idx, models::Params{})), T("same_geometry(defecto, resuelto)"));

        // --- Movimiento de ruedas ---------------------------------------------------------------
        for (const auto& g : sc.groups()) {
            if (!is_wheel(g)) continue;
            const Vec3 c = g.motion.center;
            const float R = 1.0f / std::fabs(g.motion.omega_hat.y);
            const Vec3 v = g.motion.velocity_at(c - Vec3(0, 0, R));
            check(length(v - Vec3(1, 0, 0)) < 1e-4f, T("banda de rodadura a +U (igual que la cinta)"));
        }

        // --- Alerones de F1: generan CARGA (ángulo negativo y combadura invertida) -------------
        if (car) {
            int bad = 0, nw = 0;
            for (const auto& g : sc.groups()) {
                if (g.component != Component::FrontWing && g.component != Component::RearWing && g.component != Component::BeamWing) continue;
                for (u32 i = g.first; i < g.first + g.count; ++i) {
                    const sdf::Prim& pr = sc.prims()[i];
                    if (pr.type != sdf::PrimType::Wing) continue;
                    ++nw;
                    const float sa = pr.xf.R.m[0][2];              // rot_y(aoa): m[0][2] = sin(aoa)
                    double area = 0, cy = 0;                       // centroide del perfil (v < 0 = combado hacia abajo)
                    const float* E = sc.pool.data() + pr.data_off;
                    for (u32 e = 0; e < pr.data_n; ++e) {
                        const float ax = E[5 * e], ay = E[5 * e + 1], bx = ax + E[5 * e + 2], by = ay + E[5 * e + 3];
                        const double cr = static_cast<double>(ax) * by - static_cast<double>(bx) * ay;
                        area += cr; cy += (ay + by) * cr;
                    }
                    const double centroid = cy / (3.0 * area);
                    bad += !(sa < 0.0f && centroid < 0.0);
                }
            }
            if (nw > 0 || id != "f1_1967")
                check(nw > 0 && bad == 0, T("todos los elementos de alerón generan carga"), fmt("%.0f de %.0f mal", bad, nw));
        }

        // --- Coste de evaluación: puntos uniformes en la AABB, 1 hilo, mediana de 5 -------------
        double ns1 = 0, nsp = 0;
        {
            constexpr int N = 60000;
            std::vector<Vec3> pts(N);
            for (auto& p : pts) p = Vec3(rng.uniform(bb.lo.x, bb.hi.x), rng.uniform(bb.lo.y, bb.hi.y), rng.uniform(bb.lo.z, bb.hi.z));
            std::vector<double> t1;
            volatile float sink = 0;
            for (int r = 0; r < (quick ? 3 : 5); ++r) {
                const double t0 = now_sec();
                float acc = 0;
                for (const Vec3& p : pts) acc += sc.eval(p);
                t1.push_back(now_sec() - t0);
                sink = sink + acc;
            }
            ns1 = median(t1) * 1e9 / N;
            // Paralelo (throughput por punto con todo el pool).
            std::vector<double> tp;
            std::vector<Padded<float>> part(static_cast<usize>(pool().size()) + 1);
            for (int r = 0; r < 5; ++r) {
                const double t0 = now_sec();
                parallel_for(0, N, 1024, [&](i64 lo, i64 hi) {
                    float acc = 0;
                    for (i64 i = lo; i < hi; ++i) acc += sc.eval(pts[static_cast<usize>(i)]);
                    part[static_cast<usize>(max_(ThreadPool::worker_index(), 0))].value += acc;
                });
                tp.push_back(now_sec() - t0);
            }
            nsp = median(tp) * 1e9 / N;
            check(ns1 < 2000.0, T("coste medio de eval < 2 µs"), fmt("%.0f ns", ns1));
        }

        // --- Supervivencia a dx = 3 cm: cada grupo deja celdas sólidas --------------------------
        usize vox_cells = 0;
        {
            const Grid g = voxelize(b, 0.03f, 0.12f);
            std::vector<usize> per(256, 0);
            for (u8 v : g.id) per[v]++;
            int missing = 0;
            std::string names;
            for (usize gi = 0; gi < sc.group_count(); ++gi)
                if (per[gi + 1] < 3) { ++missing; names += sc.groups()[gi].name + "; "; }
            for (usize k = 1; k < 255; ++k) vox_cells += per[k];
            check(missing == 0, T("todos los grupos sobreviven a dx = 3 cm"), names);
        }

        // --- Continuidad de los elementos de ala a dx = 3 cm (sin fugas diagonales) ---------------
        if (car) {
            int leaks = 0, tests = 0, elems = 0;
            const Vec3 offs[4] = {{0, 0, 0}, {0.011f, 0, 0.007f}, {0.019f, 0, 0.023f}, {0.005f, 0, 0.016f}};
            for (const auto& g : sc.groups()) {
                if (g.component != Component::FrontWing && g.component != Component::RearWing && g.component != Component::BeamWing) continue;
                for (u32 i = g.first; i < g.first + g.count; ++i) {
                    const sdf::Prim& pr = sc.prims()[i];
                    if (pr.type != sdf::PrimType::Wing) continue;
                    ++elems;
                    for (float sfrac : {0.2f, 0.5f, 0.8f}) {
                        const auto line = camber_line(sc, g, pr, sfrac);
                        for (const Vec3& o : offs) {
                            int c = 0;
                            ++tests;
                            if (!element_connected(sc, line, 0.03f, Vec3(bb.lo.x, 0, -0.015f) + o, c)) {
                                ++leaks;
                                if (std::getenv("CFD_VERBOSE"))
                                    std::printf("  nota  %s / %s: elemento BA=(%.2f, %.2f) s=%.1f no es 4-conexo a dx=3 cm\n", id.c_str(),
                                                g.name.c_str(), pr.xf.t.x, pr.xf.t.z, sfrac);
                            }
                        }
                    }
                }
            }
            check(leaks == 0, T("elementos de ala 4-conexos a dx = 3 cm (sin fugas diagonales)"),
                  fmt("%.0f de %.0f cortes con fuga (%.0f elementos)", leaks, tests, elems));
        }

        // --- Piezas flotantes: todo conectado (con el suelo) a resolución fina ----------------------
        int ncomp = 0;
        if (!quick) {
            const float dxf = max_comp(sz) / 320.0f;
            const Grid g = voxelize(b, dxf, 0.0f);
            usize largest = 0;
            ncomp = components(g, largest);
            check(ncomp == 1, T("sin piezas flotantes (1 componente conexa)"), fmt("%.0f componentes, dx=%.1f mm", ncomp, dxf * 1000));
        }

        std::printf("%-15s %6.2f %6.2f %6.2f %4zu %5.2fms %9.0f %9.1f %7.1f %8zu %6d\n", id.c_str(), sz.x, sz.y, bb.hi.z,
                    sc.group_count(), median(tb) * 1e3, ns1, nsp, (wheel_min < 1e29f ? wheel_min * 1000.0f : 0.0f), vox_cells, ncomp);
    }

    // --- Parámetros fuera de rango se sujetan ---------------------------------------------------
    {
        const int i = models::find("f1_2022");
        models::Params p; p.ride_front_mm = 9999; p.yaw_deg = 90; p.front_flap_deg = -99;
        const models::Params r = models::resolve_params(i, p);
        check(r.ride_front_mm <= 400.0f && r.yaw_deg <= 30.0f && r.front_flap_deg >= -20.0f, "resolve_params sujeta rangos");
        const models::Built b = models::build(i, p);
        check(!b.bounds_m.empty(), "construye con parámetros extremos");
        const int j = models::find("sphere");
        models::Params q; q.drs_open = true; q.yaw_deg = 20;
        check(!models::resolve_params(j, q).drs_open && models::resolve_params(j, q).yaw_deg == 0.0f, "parámetros fuera de param_mask se ignoran");
        // No finitos → defecto (clamp_ deja pasar NaN: acabaría en float→int indefinido aguas abajo).
        models::Params nf; nf.yaw_deg = NAN; nf.ride_front_mm = INFINITY; nf.front_flap_deg = NAN; nf.rear_flap_deg = -INFINITY;
        const models::Params rn = models::resolve_params(i, nf), rd = models::resolve_params(i, models::Params{});
        check(rn.yaw_deg == 0.0f && rn.ride_front_mm == rd.ride_front_mm && rn.front_flap_deg == 0.0f && rn.rear_flap_deg == 0.0f &&
                  models::same_geometry(i, nf, models::Params{}),
              "resolve_params: NaN/inf → valores por defecto");
        const int w = models::find("f1_wing_ge");
        models::Params nw; nw.aoa_deg = NAN; nw.height_mm = NAN; nw.flap_gap_mm = INFINITY;
        const models::Built bw = models::build(w, nw);
        check(std::isfinite(bw.bounds_m.size().x) && bw.params.aoa_deg == models::info(w).default_aoa_deg, "ala con NaN construye con defectos");
    }

    // --- Flaps y DRS en TODO su rango: ranura exacta y el DRS sólo abre -----------------------------
    // (antes el DRS/modo X giraba el flap "hasta" un ángulo fijo: con flap -20° le AÑADÍA incidencia
    //  y en el modo X de 2026 la ranura delantera caía de 45 a 12.6 mm → se cerraba a dx = 3 cm).
    {
        int pairs = 0, bad_gap = 0, bad_drs = 0;
        float worst = 1e9f;
        std::string where;
        for (int idx = 0; idx < models::count(); ++idx) {
            const models::Info& I = models::info(idx);
            if (I.kind != models::Kind::F1Car || !(I.param_mask & (models::P_FrontFlap | models::P_RearFlap))) continue;
            for (float ff : {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f})
                for (float rf : {-20.0f, 0.0f, 20.0f}) {
                    models::Params pc; pc.front_flap_deg = ff; pc.rear_flap_deg = rf;
                    models::Params po = pc; po.drs_open = true;
                    const models::Built bc = models::build(idx, pc);
                    const models::Built bo = models::build(idx, po);
                    const auto cc = wing_chains(bc.scene), co = wing_chains(bo.scene);
                    for (usize c = 0; c < cc.size() && c < co.size(); ++c)
                        for (usize k = 0; k < cc[c].size() && k < co[c].size(); ++k) {
                            // DRS / modo X: nunca aumenta la incidencia (aoa sdf menos negativo o igual).
                            if (pose_of(*co[c][k]).aoa < pose_of(*cc[c][k]).aoa - 1e-5f) {
                                ++bad_drs;
                                where = I.id + fmt(" ff=%.0f rf=%.0f elem %.0f", ff, rf, static_cast<double>(k));
                            }
                            if (k == 0) continue;
                            for (const models::Built* bb : {&bc, &bo}) {
                                const auto& ch = bb == &bc ? cc[c] : co[c];
                                const float d = models::detail::elem_elem_dist(bb->scene, pose_of(*ch[k - 1]), pose_of(*ch[k]));
                                ++pairs;
                                worst = min_(worst, d);
                                const bool open = bb == &bo && bb->params.drs_open;
                                // Cerrado: ranura EXACTA; abierto: al menos la de diseño.
                                if (open ? d < models::detail::k_min_gap - 1e-4f : std::fabs(d - models::detail::k_min_gap) > 1e-4f) {
                                    ++bad_gap;
                                    if (where.empty()) where = I.id + fmt(" ff=%.0f rf=%.0f gap=%.1f mm", ff, rf, d * 1000.0f);
                                }
                            }
                        }
                }
        }
        check(bad_drs == 0, "DRS / modo X nunca añade incidencia a un flap (flaps -20..+20)", fmt("%.0f casos", bad_drs) + " " + where);
        check(bad_gap == 0 && pairs > 300, "ranura = 45 mm (cerrado) / >= 45 mm (DRS) con flaps -20..+20",
              fmt("%.0f de %.0f pares, mínima %.1f mm ", bad_gap, pairs, worst * 1000.0f) + where);
    }

    // --- Perfil pseudo-2D: la sección llega a las paredes del dominio (bounds_m.y exacto) ------------
    for (int idx = 0; idx < models::count(); ++idx) {
        const models::Info& I = models::info(idx);
        if (!I.spans_domain) continue;
        const models::Built b = models::build(idx, models::Params{});
        const Vec3 q = b.moment_ref_m;   // 25% de cuerda, dentro del perfil
        float worst = -1e30f;
        for (float t : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f})
            worst = max_(worst, b.scene.eval(Vec3(q.x, lerp(b.bounds_m.lo.y, b.bounds_m.hi.y, t), q.z)));
        check(worst < -0.005f, "spans_domain: sección sólida hasta las paredes (y = bounds_m.lo/hi)", I.id + fmt(" sdf máx %.4f", worst));
        // Área de referencia = cuerda (1 m) × ancho del dominio → el ancho nominal es ref_area_m2.
        check(std::fabs(b.bounds_m.size().y - I.ref_area_m2) < 1e-4f, "spans_domain: ancho de bounds_m = envergadura nominal",
              fmt("%.4f m", b.bounds_m.size().y));
    }
    // Referencia de momentos de las alas: sobre el ala (25% de cuerda del principal), no en el suelo.
    for (int idx = 0; idx < models::count(); ++idx) {
        const models::Info& I = models::info(idx);
        if (I.kind != models::Kind::Wing) continue;
        const models::Built b = models::build(idx, models::Params{});
        const float d = b.scene.eval(b.moment_ref_m);
        check(std::fabs(d) < 0.02f, "alas: moment_ref_m en el 25% de la cuerda (sobre el perfil)", I.id + fmt(" sdf=%.4f", d));
    }

    std::printf("\n%d PASS, %d FAIL  (%.1f s)\n", g_pass, g_fail, now_sec() - t_start);
    pool().stop();
    return g_fail ? 1 : 0;
}
