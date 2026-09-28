// ============================================================================
//  tests/test_geom.cpp — voxelizador y malladores (módulo geom).
//
//  Uso:  test_geom            → tests de corrección + tiempos (PASS/FAIL, código ≠ 0 si falla)
//        test_geom --bench    → además, comparativas antes/después de cada optimización
//        test_geom --png      → vuelca imágenes de control en build/geom/*.png
// ============================================================================
#include "../src/core/png.hpp"
#include "../src/core/threadpool.hpp"
#include "../src/core/util.hpp"
#include "../src/geom/voxelizer.hpp"
#include "../src/lbm/field.hpp"
#include "../src/render/colormap.hpp"
#include "../src/render/framebuffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;
using sdf::Component;
using sdf::Frame;
using sdf::Op;

static int g_fail = 0;
static bool g_bench = false, g_png = false;

static void check(bool ok, const char* name, const char* fmt = "", ...) __attribute__((format(printf, 3, 4)));
static void check(bool ok, const char* name, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("[%s] %-50s %s\n", ok ? "PASS" : "FAIL", name, buf);
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
static void info(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void info(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("       ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    std::fflush(stdout);
}

template <class F>
static double median_time(int reps, F&& f) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { const double t0 = now_sec(); f(); t.push_back(now_sec() - t0); }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ---------------------------------------------------------------------------------------
//  Escena de prueba "tipo F1" (~5.6 m) construida con primitivas SDF.
//  with_subtract=false quita las restas que sobresalen de la pieza base (cockpit, canal del
//  difusor, llantas): ver la nota sobre el bug de sdf.cpp en test_sdf_subtract_bug().
// ---------------------------------------------------------------------------------------
static sdf::Scene make_f1(bool with_subtract = true, float rake_deg = 0.8f, float ride_m = 0.03f, float yaw_deg = 0.0f) {
    sdf::Scene s;
    const auto prof_main = s.naca4(0.08f, 0.4f, 0.12f, true, 40);
    const auto prof_flap = s.naca4(0.06f, 0.4f, 0.10f, true, 32);

    s.begin_group("Carrocería", Component::Body, Frame::Body);
    s.taper_box({0.225f, 0, 0}, 0.775f, {0.07f, 0.06f}, {0.22f, 0.20f}, 0.20f, 0.36f, 0.04f);          // morro
    s.round_box({1.7f, 0, 0.42f}, {0.72f, 0.32f, 0.22f}, 0.08f, {}, Op::SmoothUnion, 0.08f);         // monocasco
    s.taper_box({3.25f, 0, 0}, 0.85f, {0.30f, 0.28f}, {0.10f, 0.10f}, 0.45f, 0.30f, 0.05f, {}, Op::SmoothUnion, 0.10f);
    s.round_cone({2.1f, 0, 0.75f}, {3.2f, 0, 0.55f}, 0.12f, 0.06f, Op::SmoothUnion, 0.08f);           // toma de aire
    if (with_subtract) s.round_box({1.55f, 0, 0.66f}, {0.35f, 0.18f, 0.12f}, 0.06f, {}, Op::SmoothSubtract, 0.03f);   // cockpit
    s.end_group();

    s.begin_group("Pontones", Component::Sidepods, Frame::Body);
    s.taper_box({2.35f, 0.5f, 0}, 0.75f, {0.22f, 0.22f}, {0.12f, 0.12f}, 0.35f, 0.25f, 0.06f, {}, Op::Union, 0, true);
    s.end_group();

    s.begin_group("Fondo plano", Component::Floor, Frame::Body);
    s.round_box({2.2f, 0, 0.035f}, {1.6f, 0.75f, 0.012f}, 0.005f);
    s.end_group();

    s.begin_group("Difusor", Component::Diffuser, Frame::Body);
    {
        const Vec2 ramp[] = {{3.3f, 0.02f}, {4.3f, 0.02f}, {4.3f, 0.30f}, {4.2f, 0.30f}, {3.3f, 0.05f}};
        const auto h = s.polygon(ramp);
        s.extrude_xz(h, 0.0f, 0.5f);
        if (with_subtract) s.round_box({3.8f, 0, 0.12f}, {0.45f, 0.44f, 0.10f}, 0.01f, {}, Op::Subtract);   // canal
    }
    s.end_group();

    s.begin_group("Alerón delantero", Component::FrontWing, Frame::Body);
    s.wing(prof_main, {-0.85f, 0.10f, 0.07f}, 0.32f, 0.26f, 0.80f, -0.04f, 0.03f, 0.01f, 0, 0, 0, Op::Union, 0, true);
    s.wing(prof_flap, {-0.58f, 0.16f, 0.13f}, 0.20f, 0.15f, 0.72f, -0.40f, 0.04f, 0.02f, 0.05f, 0, 0, Op::Union, 0, true);
    s.round_box({-0.62f, 0.92f, 0.16f}, {0.30f, 0.01f, 0.13f}, 0.004f, {}, Op::Union, 0, true);     // endplates
    s.end_group();

    s.begin_group("Alerón trasero", Component::RearWing, Frame::Body);
    s.wing(prof_main, {4.10f, 0.0f, 0.82f}, 0.34f, 0.34f, 0.46f, -0.10f, 0, 0, 0, 0, 0, Op::Union, 0, true);
    s.wing(prof_flap, {4.40f, 0.0f, 0.93f}, 0.22f, 0.22f, 0.46f, -0.55f, 0, 0, 0, 0, 0, Op::Union, 0, true);
    s.round_box({4.35f, 0.47f, 0.78f}, {0.36f, 0.01f, 0.24f}, 0.004f, {}, Op::Union, 0, true);
    s.round_cone({4.0f, 0.0f, 0.35f}, {4.25f, 0.0f, 0.80f}, 0.03f, 0.025f);                          // pilón
    s.end_group();

    s.begin_group("Ruedas del.", Component::FrontWheels, Frame::Wheels, {{0, 0, 0}, {0, -1.0f / 0.33f, 0}, {0, 0, 0.33f}});
    s.cylinder_y({0.0f, 0.80f, 0.33f}, 0.33f, 0.18f, 0.05f, Op::Union, 0, true);
    if (with_subtract) s.cylinder_y({0.0f, 0.99f, 0.33f}, 0.20f, 0.05f, 0.01f, Op::SmoothSubtract, 0.02f, true);
    s.end_group();

    s.begin_group("Ruedas tras.", Component::RearWheels, Frame::Wheels, {{0, 0, 0}, {0, -1.0f / 0.36f, 0}, {3.6f, 0, 0.36f}});
    s.cylinder_y({3.6f, 0.78f, 0.36f}, 0.36f, 0.20f, 0.05f, Op::Union, 0, true);
    if (with_subtract) s.cylinder_y({3.6f, 0.99f, 0.36f}, 0.22f, 0.05f, 0.01f, Op::SmoothSubtract, 0.02f, true);
    s.end_group();

    s.begin_group("Halo", Component::Halo, Frame::Body);
    s.round_cone({1.15f, 0.0f, 0.62f}, {1.45f, 0.0f, 0.80f}, 0.025f, 0.025f);
    s.round_cone({1.45f, 0.0f, 0.80f}, {1.85f, 0.20f, 0.78f}, 0.025f, 0.025f, Op::Union, 0, true);
    s.round_cone({1.85f, 0.20f, 0.78f}, {1.95f, 0.22f, 0.60f}, 0.025f, 0.025f, Op::Union, 0, true);
    s.end_group();

    s.begin_group("Suspensión", Component::Suspension, Frame::Body);
    s.round_cone({0.10f, 0.12f, 0.30f}, {0.0f, 0.68f, 0.40f}, 0.018f, 0.015f, Op::Union, 0, true);
    s.round_cone({0.25f, 0.12f, 0.22f}, {0.0f, 0.68f, 0.24f}, 0.018f, 0.015f, Op::Union, 0, true);
    s.round_cone({3.30f, 0.15f, 0.32f}, {3.6f, 0.62f, 0.44f}, 0.018f, 0.015f, Op::Union, 0, true);
    s.round_cone({3.20f, 0.15f, 0.22f}, {3.6f, 0.62f, 0.26f}, 0.018f, 0.015f, Op::Union, 0, true);
    s.end_group();

    const float rake = rake_deg * k_deg2rad, yaw = yaw_deg * k_deg2rad;
    const Vec3 mid(1.8f, 0, 0);
    Xform body{Mat3::rot_z(yaw) * Mat3::rot_y(-rake), Vec3(0, 0, ride_m)};
    body.t = body.t + mid - Mat3::rot_z(yaw) * mid;
    const Xform wheels{Mat3::rot_z(yaw), mid - Mat3::rot_z(yaw) * mid};
    s.set_frames(body, wheels);
    return s;
}

// Red de 320×128×96 con dx = 0.03 m (dominio 9.6 × 3.84 × 2.88 m), suelo en z = 0.5 celdas.
static constexpr int F1NX = 320, F1NY = 128, F1NZ = 96;
static LatticeMap f1_map() {
    LatticeMap m;
    m.dx = 0.03f;
    m.origin = Vec3(-2.0f, -63.5f * 0.03f, -0.5f * 0.03f);
    return m;
}
static usize lin(int nx, int ny, int x, int y, int z) {
    return static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * static_cast<usize>(z));
}

// Referencia: Scene::eval en TODAS las celdas (definición literal del contrato).
static void voxelize_brute(const sdf::Scene& s, const LatticeMap& map, int nx, int ny, int nz, u8* out,
                           const geom::VoxelOptions& o) {
    const float thr = o.thicken * map.dx;
    parallel_for(o.z_min, nz, 1, [&](i64 zl, i64 zh) {
        for (int z = static_cast<int>(zl); z < static_cast<int>(zh); ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const Vec3 pc(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                    int id = 0;
                    const float d = s.eval(map.to_model(pc), &id);
                    bool solid = d < thr;
                    if (o.supersample && std::fabs(d) < map.dx) {
                        int cnt = 0;
                        for (int k = 0; k < 8; ++k) {
                            const Vec3 q(pc.x + ((k & 1) ? 0.25f : -0.25f), pc.y + ((k & 2) ? 0.25f : -0.25f), pc.z + ((k & 4) ? 0.25f : -0.25f));
                            cnt += s.eval(map.to_model(q)) < thr;
                        }
                        solid = cnt >= 4;
                    }
                    out[lin(nx, ny, x, y, z)] = solid ? static_cast<u8>(id) : u8{0};
                }
    });
}

static usize count_diff(const u8* a, const u8* b, usize n) {
    usize c = 0;
    for (usize i = 0; i < n; ++i) c += a[i] != b[i];
    return c;
}

// =========================================================================================
//  Utilidades de malla
// =========================================================================================
struct MeshCheck {
    usize tris = 0, verts = 0;
    usize bad_edges = 0;          // aristas no compartidas por exactamente 2 triángulos
    usize dup_directed = 0;       // aristas dirigidas repetidas (no-manifold o incoherentes)
    usize inconsistent = 0;       // aristas con 2 triángulos recorridas en el mismo sentido
    usize degenerate = 0;         // índices repetidos o área < eps
    usize unreferenced = 0;
    usize flipped_vs_normal = 0;  // normal geométrica · normal de vértice media ≤ 0
    double area = 0;              // en celdas²
    double volume = 0;            // divergencia (celdas³)
    Vec3 area_vec{0, 0, 0};
    float min_area = 1e30f;
};
static MeshCheck analyze(const render::Mesh& m, float eps_area) {
    MeshCheck c;
    c.tris = m.tri_count();
    c.verts = m.vertex_count();
    std::vector<u64> und, dir;
    und.reserve(c.tris * 3);
    dir.reserve(c.tris * 3);
    std::vector<u8> used(c.verts, 0);
    double ax = 0, ay = 0, az = 0;
    for (usize t = 0; t < c.tris; ++t) {
        const u32 i[3] = {m.tri[3 * t], m.tri[3 * t + 1], m.tri[3 * t + 2]};
        for (int k = 0; k < 3; ++k) used[i[k]] = 1;
        const Vec3 a = m.pos[i[0]], b = m.pos[i[1]], cc = m.pos[i[2]];
        const Vec3 cr = cross(b - a, cc - a);
        const float ar = 0.5f * length(cr);
        c.area += ar;
        c.min_area = std::min(c.min_area, ar);
        ax += cr.x * 0.5; ay += cr.y * 0.5; az += cr.z * 0.5;
        c.volume += static_cast<double>(dot(a, cross(b, cc))) / 6.0;
        if (i[0] == i[1] || i[1] == i[2] || i[0] == i[2] || ar < eps_area) ++c.degenerate;
        if (!m.nrm.empty()) {
            const Vec3 nv = m.nrm[i[0]] + m.nrm[i[1]] + m.nrm[i[2]];
            if (dot(nv, cr) <= 0) ++c.flipped_vs_normal;
        }
        for (int k = 0; k < 3; ++k) {
            const u64 u = i[k], v = i[(k + 1) % 3];
            und.push_back(u < v ? (u << 32 | v) : (v << 32 | u));
            dir.push_back(u << 32 | v);
        }
    }
    c.area_vec = Vec3(static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(az));
    std::sort(und.begin(), und.end());
    for (usize i = 0; i < und.size();) {
        usize j = i;
        while (j < und.size() && und[j] == und[i]) ++j;
        if (j - i != 2) ++c.bad_edges;
        i = j;
    }
    std::sort(dir.begin(), dir.end());
    for (usize i = 1; i < dir.size(); ++i) c.dup_directed += dir[i] == dir[i - 1];
    // Aristas manifold (2 triángulos) recorridas en el MISMO sentido = orientación incoherente.
    for (usize i = 0; i + 1 < dir.size(); ++i) {
        if (dir[i] != dir[i + 1]) continue;
        const u64 rev = (dir[i] << 32) | (dir[i] >> 32);
        const u64 u = dir[i] >> 32, v = dir[i] & 0xFFFFFFFFu;
        const u64 key = u < v ? (u << 32 | v) : (v << 32 | u);
        const auto r = std::equal_range(und.begin(), und.end(), key);
        if (r.second - r.first == 2 && !std::binary_search(dir.begin(), dir.end(), rev)) ++c.inconsistent;
    }
    for (u8 u : used) c.unreferenced += !u;
    return c;
}

// Mini "splatter" ortográfico con z-buffer para inspección visual (no es el rasterizador real).
static void dump_png(const render::Mesh& m, const char* path, float yaw_deg, float pitch_deg, int W = 1400, int H = 700) {
    if (m.tri.empty()) return;
    const float yw = yaw_deg * k_deg2rad, pt = pitch_deg * k_deg2rad;
    const Vec3 fwd = normalize(Vec3(std::sin(yw) * std::cos(pt), -std::cos(yw) * std::cos(pt), -std::sin(pt)) * -1.0f) * -1.0f;
    const Vec3 right = normalize(cross(fwd, Vec3(0, 0, 1))), up = cross(right, fwd);
    Aabb b;
    for (const Vec3& p : m.pos) b.grow(Vec3(dot(p, right), dot(p, up), dot(p, fwd)));
    const float sc = 0.95f * std::min(W / (b.hi.x - b.lo.x), H / (b.hi.y - b.lo.y));
    std::vector<u32> col(static_cast<usize>(W) * H, 0xFF20242Cu);
    std::vector<float> zb(static_cast<usize>(W) * H, 1e30f);
    const Vec3 light = normalize(Vec3(-0.4f, -0.5f, 0.75f));
    for (usize t = 0; t < m.tri_count(); ++t) {
        float sx[3], sy[3], sz[3];
        Vec3 n(0.0f);
        for (int k = 0; k < 3; ++k) {
            const u32 vi = m.tri[3 * t + k];
            const Vec3 p = m.pos[vi];
            sx[k] = (dot(p, right) - b.lo.x) * sc + 0.025f * W;
            sy[k] = H - ((dot(p, up) - b.lo.y) * sc + 0.025f * H);
            sz[k] = dot(p, fwd);
            n += m.nrm.empty() ? Vec3(0.0f) : m.nrm[vi];
        }
        const u8 g = m.group.empty() ? 0 : m.group[m.tri[3 * t]];
        const u32 base = g == 0 ? 0xFFB8BCC4u : render::component_color(static_cast<sdf::Component>((g - 1) % static_cast<int>(sdf::Component::Count)));
        const Vec3 geo = normalize(cross(m.pos[m.tri[3 * t + 1]] - m.pos[m.tri[3 * t]], m.pos[m.tri[3 * t + 2]] - m.pos[m.tri[3 * t]]));
        const float lam = 0.3f + 0.7f * std::max(0.0f, dot(normalize(n), light));
        const bool back = dot(geo, fwd) > 0;       // cara trasera: rojo (orientación)
        const Vec3 rgb = back ? Vec3(0.9f, 0.1f, 0.1f) : render::unpack_rgb(base) * lam;
        const u32 c = render::rgbf(rgb.x, rgb.y, rgb.z);
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({sx[0], sx[1], sx[2]}))));
        const int x1 = std::min(W - 1, static_cast<int>(std::ceil(std::max({sx[0], sx[1], sx[2]}))));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min({sy[0], sy[1], sy[2]}))));
        const int y1 = std::min(H - 1, static_cast<int>(std::ceil(std::max({sy[0], sy[1], sy[2]}))));
        const float den = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
        if (std::fabs(den) < 1e-12f) continue;
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float px = x + 0.5f, py = y + 0.5f;
                const float w1 = ((px - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (py - sy[0])) / den;
                const float w2 = ((sx[1] - sx[0]) * (py - sy[0]) - (px - sx[0]) * (sy[1] - sy[0])) / den;
                const float w0 = 1 - w1 - w2;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                const float z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
                float& zz = zb[static_cast<usize>(y) * W + x];
                if (z < zz) { zz = z; col[static_cast<usize>(y) * W + x] = c; }
            }
    }
    png::write_argb(path, col.data(), W, H, W);
    info("PNG: %s", path);
}

// =========================================================================================
//  Tests del voxelizador
// =========================================================================================
static void test_sdf_subtract_bug() {
    // Documenta un bug de sdf.cpp (no es de este módulo): Scene::eval_group descarta una
    // resta (Op::Subtract) cuando el punto está FUERA de la pieza base (d > 0) aunque esté
    // dentro de la caja de lo restado: la condición `bd > -d` debería ser `bd > max(-d, 0)`
    // (y `bd > max(k - d, 0)` en SmoothSubtract). El signo sigue siendo correcto, pero el
    // SDF es discontinuo y con thicken > 0 aparecen "tapas" fantasma de grosor thicken·dx.
    sdf::Scene s;
    s.begin_group("a", Component::Object, Frame::Fixed);
    s.sphere({0, 0, 0}, 1.0f);
    s.round_box({0, 0, 1.0f}, {0.3f, 0.3f, 0.5f}, 0.0f, {}, Op::Subtract);
    s.end_group();
    const float d = s.eval({0, 0, 1.02f});
    info("INFO sdf.cpp: eval(0,0,1.02) = %.3f (esperado 0.300): %s", d,
         std::fabs(d - 0.3f) < 1e-3f ? "bug corregido" : "BUG de culling en Op::Subtract sigue presente (ver informe)");
}

// Regresión (revisión): restas que SOBRESALEN de la pieza base + thicken > 0. Con el bug de
// culling de sdf.cpp el SDF es discontinuo (positivo a ambos lados) dentro de la caja restada y
// las cotas de Lipschitz del salto de bloques fallaban: 307 celdas distintas de la definición
// literal en estos 48 casos antes de la ruta literal (modelos reales f1_1998 / f1_2008: 56 y 112).
static void test_voxel_protruding_subtract() {
    WyRand r(5);
    usize tot = 0, lit = 0;
    int cases = 0;
    for (int t = 0; t < 12; ++t) {
        sdf::Scene s;
        s.begin_group("base", Component::Object, Frame::Fixed);
        s.round_box({0.5f, 0.5f, 0.4f}, {0.30f, 0.22f, 0.15f}, 0.02f);
        s.round_box({0.45f + r.uniform(-0.02f, 0.02f), 0.5f, 0.55f}, {0.12f, 0.08f, 0.10f}, 0.01f, {}, Op::Subtract);
        s.round_box({0.75f, 0.5f + r.uniform(-0.02f, 0.02f), 0.33f}, {0.10f, 0.06f, 0.05f}, 0.01f, {}, Op::SmoothSubtract, 0.02f);
        s.end_group();
        LatticeMap map;
        map.dx = 0.02f;
        map.origin = Vec3(r.uniform(0, 0.02f), r.uniform(0, 0.02f), r.uniform(0, 0.02f));
        const int nx = 56, ny = 48, nz = 40;
        const usize N = static_cast<usize>(nx) * ny * nz;
        Buffer<u8> a(N, true), b(N, true);
        for (float th : {0.12f, 0.3f})
            for (int ss = 0; ss < 2; ++ss) {
                geom::VoxelOptions o;
                o.thicken = th;
                o.supersample = ss;
                const auto st = geom::voxelize(s, map, nx, ny, nz, a.data(), o);
                voxelize_brute(s, map, nx, ny, nz, b.data(), o);
                const usize L0 = static_cast<usize>(nx) * ny;
                tot += count_diff(a.data() + L0, b.data() + L0, N - L0);
                lit += st.literal_cells;
                o.block_skip = false;                                   // ruta de referencia: también literal
                geom::voxelize(s, map, nx, ny, nz, a.data(), o);
                tot += count_diff(a.data() + L0, b.data() + L0, N - L0);
                ++cases;
            }
    }
    check(tot == 0, "voxel: restas que sobresalen, thicken>0 == bruta", "%d casos, distintas=%zu, celdas por ruta literal=%zu", cases, tot, lit);
}

static void test_voxel_sphere() {
    const int n = 48;
    LatticeMap map;
    map.dx = 0.03f;
    map.origin = Vec3(-0.7f, -0.71f, -0.69f);
    const Vec3 c(0.013f, -0.007f, 0.021f);      // centro no alineado con la red
    const float r = 10.0f * map.dx;
    sdf::Scene s;
    s.begin_group("esfera", Component::Object, Frame::Fixed);
    s.sphere(c, r);
    s.end_group();
    Buffer<u8> a(static_cast<usize>(n) * n * n);
    const double vex = 4.0 / 3.0 * k_pi * 1000.0;   // celdas³
    for (int ss = 0; ss < 2; ++ss) {
        geom::VoxelOptions o;
        o.thicken = 0.0f;
        o.supersample = ss;
        o.z_min = 0;
        const auto st = geom::voxelize(s, map, n, n, n, a.data(), o);
        const double err = (static_cast<double>(st.solid_cells) - vex) / vex;
        char name[80];
        std::snprintf(name, sizeof name, "voxel esfera r=10 celdas: volumen ±3%% (ss=%d)", ss);
        check(std::fabs(err) < 0.03, name, "%zu celdas vs %.1f (%+.2f%%)", st.solid_cells, vex, err * 100);
    }
    geom::VoxelOptions o;
    o.z_min = 0;
    const auto st = geom::voxelize(s, map, n, n, n, a.data(), o);
    info("con thicken=0.12 por defecto: %zu celdas (%+.2f%%, el engrosamiento añade ≈ 3·0.12/10)", st.solid_cells,
         (static_cast<double>(st.solid_cells) - vex) / vex * 100);
}

static void test_voxel_groups_ground() {
    const int nx = 64, ny = 32, nz = 32;
    const usize N = static_cast<usize>(nx) * ny * nz;
    LatticeMap map;
    map.dx = 0.05f;
    map.origin = Vec3(0, 0, -0.025f);          // suelo del modelo (z=0) en z_celdas = 0.5
    sdf::Scene s;
    s.begin_group("esfera", Component::Object, Frame::Fixed);
    s.sphere({0.8f, 0.8f, 0.1f}, 0.35f);        // atraviesa el suelo
    s.end_group();
    s.begin_group("caja", Component::Object, Frame::Fixed);
    s.round_box({2.2f, 0.8f, 0.6f}, {0.3f, 0.25f, 0.2f}, 0.03f);
    s.end_group();
    Buffer<u8> a(N);
    for (usize i = 0; i < N; ++i) a[i] = i < static_cast<usize>(nx) * ny ? 255 : 0x77;   // suelo + basura
    geom::VoxelOptions o;
    const auto st = geom::voxelize(s, map, nx, ny, nz, a.data(), o);
    usize ground_bad = 0, garbage = 0, n1 = 0, n2 = 0, wrong = 0;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const u8 v = a[lin(nx, ny, x, y, z)];
                if (z == 0) { ground_bad += v != 255; continue; }
                garbage += v != 0 && v != 1 && v != 2;
                if (!v) continue;
                const Vec3 p = map.to_model(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
                n1 += v == 1; n2 += v == 2;
                wrong += (v == 1) != (p.x < 1.5f);
            }
    check(ground_bad == 0, "voxel: capa del suelo (z < z_min) intacta", "%zu celdas tocadas", ground_bad);
    check(garbage == 0, "voxel: escribe toda la red (z ≥ z_min)", "%zu celdas con basura", garbage);
    check(n1 > 0 && n2 > 0 && wrong == 0, "voxel: ids de grupo de 2 objetos separados", "id1=%zu id2=%zu mal=%zu", n1, n2, wrong);
    check(st.solid_cells == n1 + n2 && st.lo[2] >= 1, "voxel: VoxelStats coherente", "solid=%zu z_lo=%d", st.solid_cells, st.lo[2]);
}

static void test_voxel_f1() {
    const int nx = F1NX, ny = F1NY, nz = F1NZ;
    const usize N = static_cast<usize>(nx) * ny * nz, L0 = static_cast<usize>(nx) * ny;
    const LatticeMap map = f1_map();
    Buffer<u8> a(N), b(N);
    const sdf::Scene full = make_f1(true), clean = make_f1(false);

    {   // Coste de referencia de Scene::eval.
        const Aabb bb = full.bounds();
        WyRand r(7);
        const int n = 200000;
        std::vector<Vec3> pts(n);
        for (auto& p : pts) p = Vec3(r.uniform(bb.lo.x, bb.hi.x), r.uniform(bb.lo.y, bb.hi.y), r.uniform(bb.lo.z, bb.hi.z));
        float acc = 0;
        const double t = median_time(3, [&] { for (const Vec3& p : pts) acc += full.eval(p); });
        info("coste Scene::eval (escena F1, 1 hilo): %.3f us/eval (%zu grupos, %zu primitivas)%s", t / n * 1e6,
             full.group_count(), full.prims().size(), acc == 12345.f ? " " : "");
    }

    // Identidad salto de bloques == fuerza bruta.
    struct Case { const sdf::Scene* s; float thicken; bool ss; const char* name; };
    const Case cases[] = {
        {&full, 0.0f, false, "voxel F1 (con restas) thicken=0 ss=0 == bruta"},
        {&full, 0.0f, true, "voxel F1 (con restas) thicken=0 ss=1 == bruta"},
        {&clean, 0.12f, false, "voxel F1 (sin restas) thicken=.12 ss=0 == bruta"},
        {&clean, 0.12f, true, "voxel F1 (sin restas) thicken=.12 ss=1 == bruta"},
    };
    for (const Case& c : cases) {
        geom::VoxelOptions o;
        o.thicken = c.thicken;
        o.supersample = c.ss;
        std::memset(a.data(), 0xAB, N);
        std::memset(b.data(), 0xCD, N);
        const auto st = geom::voxelize(*c.s, map, nx, ny, nz, a.data(), o);
        voxelize_brute(*c.s, map, nx, ny, nz, b.data(), o);
        const usize diff = count_diff(a.data() + L0, b.data() + L0, N - L0);
        usize cnt = 0;
        for (usize i = L0; i < N; ++i) cnt += a[i] != 0;
        check(diff == 0 && cnt == st.solid_cells, c.name, "distintas=%zu sólidas=%zu (stats %zu)", diff, cnt, st.solid_cells);
    }
    {   // Opciones por defecto (thicken 0.12, ss) con restas: idéntico también con el bug de sdf.cpp
        // (las cajas de las restas van por la ruta literal mientras la sonda detecte el bug).
        geom::VoxelOptions o;
        const auto st = geom::voxelize(full, map, nx, ny, nz, a.data(), o);
        voxelize_brute(full, map, nx, ny, nz, b.data(), o);
        const usize diff = count_diff(a.data() + L0, b.data() + L0, N - L0);
        check(diff == 0, "voxel F1 (con restas) por defecto == bruta", "distintas=%zu, celdas por ruta literal=%zu (bug sdf.cpp %s)", diff,
              st.literal_cells, geom::sdf_subtract_culling_bug() ? "presente" : "corregido");
    }

    // Simetría (todas las piezas simétricas en y; y=0 cae en el centro de la red).
    {
        geom::VoxelOptions o;
        geom::voxelize(full, map, nx, ny, nz, a.data(), o);
        usize asym = 0;
        for (int z = 1; z < nz; ++z)
            for (int y = 0; y < ny / 2; ++y)
                for (int x = 0; x < nx; ++x) asym += a[lin(nx, ny, x, y, z)] != a[lin(nx, ny, x, ny - 1 - y, z)];
        check(asym == 0, "voxel F1: simetría especular en y", "%zu celdas asimétricas", asym);
    }

    // Tiempos (mediana).
    geom::VoxelOptions o;
    geom::VoxelStats st;
    const double t1 = median_time(15, [&] { st = geom::voxelize(full, map, nx, ny, nz, a.data(), o); });
    check(t1 < 0.060, "voxel F1 320x128x96 dx=0.03 ss=1: < 60 ms", "mediana %.2f ms, %zu sólidas", t1 * 1e3, st.solid_cells);
    o.supersample = false;
    const double t0 = median_time(15, [&] { geom::voxelize(full, map, nx, ny, nz, a.data(), o); });
    info("voxelize F1 ss=0: mediana %.2f ms", t0 * 1e3);
    if (g_bench) {
        geom::VoxelOptions ob;
        ob.block_skip = false;
        const double r1 = median_time(3, [&] { geom::voxelize(full, map, nx, ny, nz, a.data(), ob); });
        ob.supersample = false;
        const double r0 = median_time(3, [&] { geom::voxelize(full, map, nx, ny, nz, a.data(), ob); });
        info("[bench] voxelize sin salto de bloques (Scene::eval por celda, misma caja): ss=1 %.2f ms, ss=0 %.2f ms", r1 * 1e3, r0 * 1e3);
        const double rb = median_time(3, [&] { geom::VoxelOptions q; voxelize_brute(full, map, nx, ny, nz, b.data(), q); });
        info("[bench] fuerza bruta en TODA la red (ss=1 literal): %.2f ms", rb * 1e3);
        // Sensibilidad a la constante de Lipschitz.
        for (float L : {1.0f, 1.25f, 1.5f, 2.0f, 3.0f}) {
            geom::VoxelOptions q;
            q.lipschitz = L;
            const double t = median_time(9, [&] { geom::voxelize(full, map, nx, ny, nz, a.data(), q); });
            info("[bench] lipschitz=%.2f: %.2f ms", L, t * 1e3);
        }
        // Un solo hilo.
        pool().stop();
        pool().start(1);
        geom::VoxelOptions q;
        const double ts = median_time(5, [&] { geom::voxelize(full, map, nx, ny, nz, a.data(), q); });
        pool().stop();
        pool().start();
        info("[bench] voxelize ss=1 con 1 hilo: %.2f ms (escalado ×%.1f con %d hilos)", ts * 1e3, ts / t1, pool().size());
    }
}

// =========================================================================================
//  Tests de mesh_scene
// =========================================================================================
static void check_smooth_mesh(const char* tag, const sdf::Scene& s, const LatticeMap& map, float frac, double area_exact_m2,
                              auto radial_dir) {
    render::Mesh m;
    geom::MeshStats ms;
    geom::MeshOptions mo;
    geom::mesh_scene(s, map, frac, m, mo, &ms);
    const float h_cells = frac;
    const MeshCheck c = analyze(m, 1e-6f * h_cells * h_cells);
    char name[96];
    std::snprintf(name, sizeof name, "mesh_scene %s: estanca (arista = 2 triángulos)", tag);
    check(c.tris > 0 && c.bad_edges == 0 && c.dup_directed == 0 && c.unreferenced == 0, name,
          "%zu tris, %zu verts, aristas malas=%zu, dirigidas dup=%zu, sin usar=%zu", c.tris, c.verts, c.bad_edges, c.dup_directed, c.unreferenced);
    usize bad_n = 0, bad_t = 0;
    float maxd = 0;
    for (usize i = 0; i < m.vertex_count(); ++i) {
        const Vec3 pm = map.to_model(m.pos[i]);
        if (dot(m.nrm[i], radial_dir(pm)) <= 0) ++bad_n;
        maxd = std::max(maxd, std::fabs(s.eval(pm)));
    }
    for (usize t = 0; t < m.tri_count(); ++t) {
        const Vec3 a = m.pos[m.tri[3 * t]], b = m.pos[m.tri[3 * t + 1]], cc = m.pos[m.tri[3 * t + 2]];
        const Vec3 cen = map.to_model((a + b + cc) * (1.0f / 3.0f));
        if (dot(cross(b - a, cc - a), radial_dir(cen)) <= 0) ++bad_t;
    }
    std::snprintf(name, sizeof name, "mesh_scene %s: normales hacia fuera (CCW)", tag);
    check(bad_n == 0 && bad_t == 0 && c.flipped_vs_normal == 0 && ms.normal_fixes == 0, name,
          "vértices mal=%zu, triángulos mal=%zu, vs normal=%zu, normales sustituidas=%zu (superficie suave: 0)",
          bad_n, bad_t, c.flipped_vs_normal, ms.normal_fixes);
    const double area_m2 = c.area * map.dx * map.dx, err = (area_m2 - area_exact_m2) / area_exact_m2;
    std::snprintf(name, sizeof name, "mesh_scene %s: área ±2%% de la analítica", tag);
    check(std::fabs(err) < 0.02, name, "%.5f vs %.5f m² (%+.3f%%), |d| máx en vértices = %.2e m", area_m2, area_exact_m2, err * 100, maxd);
    std::snprintf(name, sizeof name, "mesh_scene %s: sin triángulos degenerados", tag);
    check(c.degenerate == 0, name, "%zu degenerados, área mínima %.2e celdas²", c.degenerate, c.min_area);
    const double vol_m3 = c.volume * map.dx * map.dx * map.dx;
    info("%s: volumen encerrado %.5f m³, %zu muestras evaluadas, %.2f ms", tag, vol_m3, ms.sampled, ms.seconds * 1e3);
}

static void test_mesh_scene_shapes() {
    LatticeMap map;
    map.dx = 0.03f;
    map.origin = Vec3(-1.0f, -1.0f, -1.0f);
    {
        const Vec3 c(0.011f, -0.023f, 0.017f);
        const float r = 0.3f;
        sdf::Scene s;
        s.begin_group("esfera", Component::Object, Frame::Fixed);
        s.sphere(c, r);
        s.end_group();
        check_smooth_mesh("esfera", s, map, 0.5f, 4.0 * k_pi * r * r, [c](Vec3 p) { return p - c; });
        if (g_png) { render::Mesh m; geom::mesh_scene(s, map, 0.5f, m); dump_png(m, "build/geom/mesh_esfera.png", 30, 25, 700, 700); }
    }
    {
        const Vec3 c(0.007f, 0.013f, -0.019f);
        const float R = 0.4f, r = 0.15f;
        sdf::Scene s;
        s.begin_group("toro", Component::Object, Frame::Fixed);
        s.torus_y(c, R, r);
        s.end_group();
        check_smooth_mesh("toro", s, map, 0.5f, 4.0 * k_pi * k_pi * R * r, [c, R](Vec3 p) {
            const Vec3 q = p - c;
            const float l = std::sqrt(q.x * q.x + q.z * q.z);
            return q - Vec3(q.x / l * R, 0, q.z / l * R);
        });
        if (g_png) { render::Mesh m; geom::mesh_scene(s, map, 0.5f, m); dump_png(m, "build/geom/mesh_toro.png", 20, 35, 900, 700); }
    }
}

static void test_mesh_scene_f1() {
    const LatticeMap map = f1_map();
    const sdf::Scene s = make_f1(true);
    render::Mesh m;
    geom::MeshStats ms;
    geom::MeshOptions mo;
    geom::mesh_scene(s, map, 0.5f, m, mo, &ms);   // calentamiento (reserva del espacio de trabajo)
    std::vector<double> ts;
    for (int i = 0; i < 9; ++i) { geom::mesh_scene(s, map, 0.5f, m, mo, &ms); ts.push_back(ms.seconds); }
    std::sort(ts.begin(), ts.end());
    const MeshCheck c = analyze(m, 1e-6f * 0.25f);
    check(ts[4] < 0.300 && c.tris < 600000, "mesh_scene F1 frac=0.5: < 300 ms y < 600k tris", "mediana %.1f ms, %zu tris, %zu verts",
          ts[4] * 1e3, c.tris, c.verts);
    info("rejilla %dx%dx%d h=%.1f mm, %zu muestras en hojas + %zu esquinas (%.1f%% de la rejilla); fases (última): muestreo %.1f, cubos %.1f, esquinas %.1f, vértices %.1f, quads %.1f, normales %.1f ms",
         ms.n[0], ms.n[1], ms.n[2], ms.h * 1e3, ms.sampled, ms.corner_fill,
         100.0 * (ms.sampled + ms.corner_fill) / (static_cast<double>(ms.n[0]) * ms.n[1] * ms.n[2]),
         ms.t_sample * 1e3, ms.t_cubes * 1e3, ms.t_fill * 1e3, ms.t_verts * 1e3, ms.t_quads * 1e3, ms.t_normals * 1e3);
    info("F1: aristas no-manifold=%zu (%.3f%%), dirigidas dup=%zu, degenerados=%zu, orientación vs normal mal=%zu (%.3f%%)",
         c.bad_edges, 100.0 * c.bad_edges / (1.5 * c.tris), c.dup_directed, c.degenerate, c.flipped_vs_normal, 100.0 * c.flipped_vs_normal / c.tris);
    check(c.inconsistent == 0 && c.unreferenced == 0 && c.bad_edges * 200 < c.tris, "mesh_scene F1: orientación coherente, sin sueltos",
          "incoherentes=%zu sueltos=%zu no-manifold=%zu (< 0.5%% tris)", c.inconsistent, c.unreferenced, c.bad_edges);
    // Normales robustas (revisión): antes 4 332 triángulos (1.30 %) con la normal de vértice
    // opuesta a la geométrica, 2 880 en el difusor (resta que sobresale → gradiente del SDF
    // discontinuo); flowvis muestrea Cp en pos + n·offset → con normal invertida, dentro del sólido.
    // Queda ≈ 0.14 % en piezas más finas que h (bordes de salida, endplates).
    check(c.flipped_vs_normal * 300 < c.tris, "mesh_scene F1: normales de vértice coherentes (< 0.33%)",
          "%zu triángulos con normal de vértice opuesta (%.3f%%), %zu normales sustituidas por la de caras",
          c.flipped_vs_normal, 100.0 * c.flipped_vs_normal / c.tris, ms.normal_fixes);
    usize bad_group = 0;
    for (u8 g : m.group) bad_group += g == 0 || g > s.group_count();
    check(bad_group == 0, "mesh_scene F1: id de grupo por vértice válido", "%zu inválidos", bad_group);

    // Muestreo en banda estrecha == muestreo completo (misma malla bit a bit).
    {
        render::Mesh m2;
        geom::MeshOptions mb;
        mb.block_skip = false;
        geom::MeshStats mbs;
        geom::mesh_scene(s, map, 0.5f, m2, mb, &mbs);
        bool same = m2.tri == m.tri && m2.pos.size() == m.pos.size();
        for (usize i = 0; same && i < m.pos.size(); ++i) same = std::memcmp(&m.pos[i], &m2.pos[i], sizeof(Vec3)) == 0;
        check(same, "mesh_scene F1: banda estrecha == muestreo completo", "idéntica bit a bit; completo: %.1f ms (%zu muestras)",
              mbs.seconds * 1e3, mbs.sampled);
    }
    if (g_bench) {
        // A/B intercalado (la máquina tiene carga de fondo): halo vs sin halo + relleno de esquinas.
        {
            std::vector<double> ta, tb;
            geom::MeshStats sa, sb2;
            render::Mesh ma, mb;
            geom::MeshOptions oa, ob;
            oa.halo = true;
            for (int i = 0; i < 7; ++i) {
                geom::mesh_scene(s, map, 0.5f, ma, oa, &sa); ta.push_back(sa.seconds);
                geom::mesh_scene(s, map, 0.5f, mb, ob, &sb2); tb.push_back(sb2.seconds);
            }
            std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
            info("[bench] muestreo con halo: %.1f ms (%zu evals) | sin halo + esquinas: %.1f ms (%zu + %zu evals) | misma malla: %s",
                 ta[3] * 1e3, sa.sampled + sa.corner_fill, tb[3] * 1e3, sb2.sampled, sb2.corner_fill,
                 (ma.tri == mb.tri && ma.pos.size() == mb.pos.size() && std::memcmp(ma.pos.data(), mb.pos.data(), ma.pos.size() * sizeof(Vec3)) == 0) ? "sí" : "NO");
        }
        for (int st = 0; st <= 3; ++st) {
            geom::MeshOptions q;
            q.projection_steps = st;
            geom::MeshStats qs;
            std::vector<double> tt;
            render::Mesh mm;
            for (int i = 0; i < 5; ++i) { geom::mesh_scene(s, map, 0.5f, mm, q, &qs); tt.push_back(qs.seconds); }
            std::sort(tt.begin(), tt.end());
            // Error medio |d| en vértices.
            double err = 0;
            for (const Vec3& p : mm.pos) err += std::fabs(s.eval(map.to_model(p)));
            info("[bench] projection_steps=%d: %.1f ms, |d| medio en vértices = %.3f mm", st, tt[2] * 1e3, err / mm.pos.size() * 1e3);
        }
        for (float frac : {1.0f, 0.5f, 0.35f, 0.25f}) {
            geom::MeshStats qs;
            render::Mesh mm;
            std::vector<double> tt;
            for (int i = 0; i < 3; ++i) { geom::mesh_scene(s, map, frac, mm, geom::MeshOptions{}, &qs); tt.push_back(qs.seconds); }
            std::sort(tt.begin(), tt.end());
            info("[bench] cell_fraction=%.2f: %.1f ms, %zu tris", frac, tt[1] * 1e3, mm.tri_count());
        }
    }
    if (g_png) {
        dump_png(m, "build/geom/mesh_f1_34.png", -35, 25);
        dump_png(m, "build/geom/mesh_f1_lado.png", 0, 0);
        dump_png(m, "build/geom/mesh_f1_arriba.png", 0, 89);
    }
}

// =========================================================================================
//  Tests de mesh_voxels
// =========================================================================================
// Conjunto de caras unitarias (clave) de la definición ingenua.
static u64 face_key(int axis, int plane, int u, int v, int pos, int id) {
    return (static_cast<u64>(axis) << 62) | (static_cast<u64>(pos) << 61) | (static_cast<u64>(id) << 53) |
           (static_cast<u64>(plane & 0x3FFF) << 36) | (static_cast<u64>(u & 0x3FFFF) << 18) | static_cast<u64>(v & 0x3FFFF);
}
static void naive_faces(const u8* g, int nx, int ny, int nz, bool skip_ground, std::vector<u64>& out) {
    auto solid = [&](int x, int y, int z) -> int {
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
        const u8 v = g[lin(nx, ny, x, y, z)];
        return (v != 0 && !(skip_ground && v == 255)) ? v : 0;
    };
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const int v = solid(x, y, z);
                if (!v) continue;
                if (!solid(x + 1, y, z)) out.push_back(face_key(0, x + 1, y, z, 1, v));
                if (!solid(x - 1, y, z)) out.push_back(face_key(0, x, y, z, 0, v));
                if (!solid(x, y + 1, z)) out.push_back(face_key(1, y + 1, x, z, 1, v));
                if (!solid(x, y - 1, z)) out.push_back(face_key(1, y, x, z, 0, v));
                if (!solid(x, y, z + 1)) out.push_back(face_key(2, z + 1, x, y, 1, v));
                if (!solid(x, y, z - 1)) out.push_back(face_key(2, z, x, y, 0, v));
            }
}
// Descompone la malla greedy (4 vértices por rectángulo) en caras unitarias.
static bool greedy_faces(const render::Mesh& m, std::vector<u64>& out) {
    if (m.pos.size() % 4 || m.tri.size() != m.pos.size() / 4 * 6) return false;
    for (usize r = 0; r < m.pos.size() / 4; ++r) {
        const Vec3* p = &m.pos[4 * r];
        const Vec3 n = m.nrm[4 * r];
        const int axis = std::fabs(n.x) > 0.5f ? 0 : (std::fabs(n.y) > 0.5f ? 1 : 2);
        const int au = axis == 0 ? 1 : 0, av = axis == 2 ? 1 : 2;
        const int pos = n[axis] > 0;
        const int plane = static_cast<int>(std::lround(p[0][axis] + 0.5f));
        float ulo = 1e30f, uhi = -1e30f, vlo = 1e30f, vhi = -1e30f;
        for (int c = 0; c < 4; ++c) {
            if (std::fabs(p[c][axis] - p[0][axis]) > 1e-4f) return false;
            ulo = std::min(ulo, p[c][au]); uhi = std::max(uhi, p[c][au]);
            vlo = std::min(vlo, p[c][av]); vhi = std::max(vhi, p[c][av]);
        }
        // Orientación del triángulo coherente con la normal.
        for (int t = 0; t < 2; ++t) {
            const u32* tr = &m.tri[6 * r + 3 * t];
            if (dot(cross(m.pos[tr[1]] - m.pos[tr[0]], m.pos[tr[2]] - m.pos[tr[0]]), n) <= 0) return false;
        }
        for (int u = static_cast<int>(std::lround(ulo + 0.5f)); u < static_cast<int>(std::lround(uhi + 0.5f)); ++u)
            for (int v = static_cast<int>(std::lround(vlo + 0.5f)); v < static_cast<int>(std::lround(vhi + 0.5f)); ++v)
                out.push_back(face_key(axis, plane, u, v, pos, m.group[4 * r]));
    }
    return true;
}

static void check_voxel_mesh(const char* tag, const u8* g, int nx, int ny, int nz, bool skip_ground, bool timing) {
    render::Mesh m;
    geom::mesh_voxels(g, nx, ny, nz, m, skip_ground);
    std::vector<u64> nf, gf;
    naive_faces(g, nx, ny, nz, skip_ground, nf);
    const bool ok_struct = greedy_faces(m, gf);
    std::sort(nf.begin(), nf.end());
    std::sort(gf.begin(), gf.end());
    usize solid = 0;
    for (usize i = 0; i < static_cast<usize>(nx) * ny * nz; ++i) solid += g[i] != 0 && !(skip_ground && g[i] == 255);
    const MeshCheck c = analyze(m, 1e-6f);
    char name[96];
    std::snprintf(name, sizeof name, "mesh_voxels %s: cubre exactamente las caras", tag);
    check(ok_struct && nf == gf, name, "%zu caras unitarias, %zu rectángulos", nf.size(), m.pos.size() / 4);
    std::snprintf(name, sizeof name, "mesh_voxels %s: cerrada (volumen = nº celdas)", tag);
    check(std::fabs(c.volume - static_cast<double>(solid)) < 1e-3 * std::max<double>(1.0, solid) && length(c.area_vec) < 1e-3f * c.area, name,
          "vol %.2f vs %zu celdas, |Σ n·A| = %.2e", c.volume, solid, length(c.area_vec));
    std::snprintf(name, sizeof name, "mesh_voxels %s: menos triángulos que caras ingenuas", tag);
    check(c.tris < nf.size() * 2, name, "%zu tris vs %zu ingenuos (×%.1f menos)", c.tris, nf.size() * 2,
          static_cast<double>(nf.size() * 2) / std::max<usize>(c.tris, 1));
    if (timing) {
        const double t = median_time(15, [&] { geom::mesh_voxels(g, nx, ny, nz, m, skip_ground); });
        info("mesh_voxels %s: mediana %.2f ms", tag, t * 1e3);
    }
}

static void test_mesh_voxels() {
    {   // Esfera voxelizada.
        const int n = 40;
        LatticeMap map;
        map.dx = 0.03f;
        map.origin = Vec3(-0.6f, -0.6f, -0.6f);
        sdf::Scene s;
        s.begin_group("esfera", Component::Object, Frame::Fixed);
        s.sphere({0.01f, 0.0f, -0.02f}, 0.3f);
        s.end_group();
        Buffer<u8> a(static_cast<usize>(n) * n * n);
        geom::VoxelOptions o;
        o.z_min = 0;
        geom::voxelize(s, map, n, n, n, a.data(), o);
        check_voxel_mesh("esfera", a.data(), n, n, n, true, false);
    }
    {   // F1 con suelo (255) en z=0, que se ignora; y otra vez incluyéndolo.
        const int nx = F1NX, ny = F1NY, nz = F1NZ;
        const usize N = static_cast<usize>(nx) * ny * nz;
        Buffer<u8> a(N);
        std::memset(a.data(), 255, static_cast<usize>(nx) * ny);
        geom::voxelize(make_f1(true), f1_map(), nx, ny, nz, a.data());
        check_voxel_mesh("F1", a.data(), nx, ny, nz, true, true);
        check_voxel_mesh("F1+suelo", a.data(), nx, ny, nz, false, false);
        if (g_png) {
            render::Mesh m;
            geom::mesh_voxels(a.data(), nx, ny, nz, m, true);
            dump_png(m, "build/geom/voxels_f1_34.png", -35, 25);
        }
    }
    {   // Caso límite: sólidos tocando los bordes de la red y ids distintos adyacentes.
        const int nx = 16, ny = 8, nz = 8;
        Buffer<u8> a(static_cast<usize>(nx) * ny * nz, true);
        WyRand r(3);
        for (usize i = 0; i < a.size(); ++i) a[i] = r.below(3) == 0 ? static_cast<u8>(1 + r.below(3)) : 0;
        check_voxel_mesh("aleatoria", a.data(), nx, ny, nz, true, false);
    }
}

// ---------------------------------------------------------------------------------------
//  --benchonly: medidas compactas para comparar variantes compiladas con -D (docs/opt/geom.md).
//  Mediana de N repeticiones con 1 hilo y con el pool completo + nº de llamadas a eval_group
//  (independiente de la carga de la máquina) si se compila con -DCFD_GEOM_COUNTING.
// ---------------------------------------------------------------------------------------
static void bench_line(const char* tag, int reps, auto&& fn) {
#ifdef CFD_GEOM_COUNTING
    geom::g_eval_count[0] = 0; geom::g_eval_count[1] = 0;
    fn();
    const u64 c0 = geom::g_eval_count[0], c1 = geom::g_eval_count[1];
#else
    const u64 c0 = 0, c1 = 0;
#endif
    const int nt = pool().size();
    const double tn = median_time(reps, fn);
    // 1 hilo: tiempo de CPU del hilo (CLOCK_THREAD_CPUTIME_ID) → inmune a desalojos por la carga de fondo.
    pool().stop(); pool().start(1);
    std::vector<double> tc;
    for (int i = 0; i < std::max(5, reps / 3); ++i) {
        timespec a, b;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &a);
        fn();
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
        tc.push_back(static_cast<double>(b.tv_sec - a.tv_sec) + 1e-9 * static_cast<double>(b.tv_nsec - a.tv_nsec));
    }
    std::sort(tc.begin(), tc.end());
    pool().stop(); pool().start(nt);
    std::printf("BENCH %-26s | pared %d hilos: %8.2f ms | CPU 1 hilo: %8.2f ms | evals refine=%llu puntos=%llu\n", tag, nt, tn * 1e3,
                tc[tc.size() / 2] * 1e3, static_cast<unsigned long long>(c0), static_cast<unsigned long long>(c1));
    std::fflush(stdout);
}
static void bench_only() {
    const sdf::Scene s = make_f1(true);
    const LatticeMap map = f1_map();
    Buffer<u8> a(static_cast<usize>(F1NX) * F1NY * F1NZ);
    geom::VoxelOptions o1, o0;
    o0.supersample = false;
    bench_line("voxelize F1 ss=1", 21, [&] { geom::voxelize(s, map, F1NX, F1NY, F1NZ, a.data(), o1); });
    bench_line("voxelize F1 ss=0", 21, [&] { geom::voxelize(s, map, F1NX, F1NY, F1NZ, a.data(), o0); });
    render::Mesh m;
    geom::MeshOptions mo;
    geom::mesh_scene(s, map, 0.5f, m, mo);
    bench_line("mesh_scene F1 frac=0.5", 15, [&] { geom::mesh_scene(s, map, 0.5f, m, mo); });
    geom::voxelize(s, map, F1NX, F1NY, F1NZ, a.data(), o1);
    render::Mesh mv;
    bench_line("mesh_voxels F1", 31, [&] { geom::mesh_voxels(a.data(), F1NX, F1NY, F1NZ, mv, true); });
    std::printf("BENCH tris: mesh_scene=%zu mesh_voxels=%zu\n", m.tri_count(), mv.tri_count());
    if (!g_bench) return;
    // Opciones de ejecución (antes/después de los trucos algorítmicos).
    geom::VoxelOptions nb1, nb0;
    nb1.block_skip = false;
    nb0.block_skip = false;
    nb0.supersample = false;
    bench_line("voxelize sin bloques ss=1", 5, [&] { geom::voxelize(s, map, F1NX, F1NY, F1NZ, a.data(), nb1); });
    bench_line("voxelize sin bloques ss=0", 5, [&] { geom::voxelize(s, map, F1NX, F1NY, F1NZ, a.data(), nb0); });
    bench_line("bruta red completa ss=1", 3, [&] { voxelize_brute(s, map, F1NX, F1NY, F1NZ, a.data(), o1); });
    geom::MeshOptions mfull, mhalo, mst2;
    mfull.block_skip = false;
    mhalo.halo = true;
    mst2.projection_steps = 2;
    bench_line("mesh_scene muestreo completo", 5, [&] { geom::mesh_scene(s, map, 0.5f, m, mfull); });
    bench_line("mesh_scene con halo", 9, [&] { geom::mesh_scene(s, map, 0.5f, m, mhalo); });
    bench_line("mesh_scene 2 proyecciones", 9, [&] { geom::mesh_scene(s, map, 0.5f, m, mst2); });
}

int main(int argc, char** argv) {
    bool bench_only_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) g_bench = true;
        if (!std::strcmp(argv[i], "--png")) g_png = true;
        if (!std::strcmp(argv[i], "--benchonly")) bench_only_mode = true;
        if (!std::strcmp(argv[i], "--benchonly-full")) bench_only_mode = g_bench = true;
    }
    pool().start();
    if (bench_only_mode) { bench_only(); return 0; }
    std::printf("test_geom: %d hilos%s%s\n", pool().size(), g_bench ? " [bench]" : "", g_png ? " [png]" : "");
    const double t0 = now_sec();
    test_sdf_subtract_bug();
    test_voxel_sphere();
    test_voxel_protruding_subtract();
    test_voxel_groups_ground();
    test_voxel_f1();
    test_mesh_scene_shapes();
    test_mesh_scene_f1();
    test_mesh_voxels();
    std::printf("%s (%d fallos, %.1f s)\n", g_fail ? "FALLÓ" : "OK", g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
