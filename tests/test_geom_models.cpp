// ============================================================================
//  tests/test_geom_models.cpp — módulo geom contra el catálogo REAL de models/.
//
//  test_geom.cpp usa una escena F1 de prueba propia (no depende de models/); aquí se
//  comprueba lo mismo con los modelos que usará la app, porque cambian con el tiempo
//  (en la revisión, dos coches nuevos rompieron la identidad con la fuerza bruta: restas
//  que sobresalen + bug de culling de sdf.cpp → ruta literal en voxelizer.cpp).
//    * voxelize == definición literal (Scene::eval en toda la red) con las opciones por
//      defecto (thicken 0.12, submuestreo) y con thicken 0 sin submuestreo;
//    * mesh_scene: banda estrecha == muestreo completo (bit a bit), orientación coherente,
//      sin degenerados, índices válidos, ids de grupo válidos, normales coherentes.
//  Compilar: g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_geom_models.cpp
//            src/geom/{voxelizer,mesher,sdf}.cpp src/models/*.cpp src/core/threadpool.cpp
//  PASS/FAIL por modelo, código ≠ 0 si falla. ≈ 30-60 s con 20 hilos.
// ============================================================================
#include "../src/core/threadpool.hpp"
#include "../src/core/util.hpp"
#include "../src/geom/voxelizer.hpp"
#include "../src/models/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cfd;

static int g_fail = 0;

static usize lin(int nx, int ny, int x, int y, int z) {
    return static_cast<usize>(x) + static_cast<usize>(nx) * (static_cast<usize>(y) + static_cast<usize>(ny) * static_cast<usize>(z));
}

// Definición literal del contrato (igual que en test_geom.cpp).
static void voxelize_brute(const sdf::Scene& s, const LatticeMap& map, int nx, int ny, int nz, u8* out, const geom::VoxelOptions& o) {
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

struct MeshCheck { usize tris = 0, bad_idx = 0, degenerate = 0, inconsistent = 0, flipped = 0, bad_group = 0, nonfinite = 0; };
static MeshCheck analyze(const render::Mesh& m, usize ngroups) {
    MeshCheck c;
    c.tris = m.tri_count();
    std::vector<u64> dir;
    dir.reserve(c.tris * 3);
    for (usize t = 0; t < c.tris; ++t) {
        const u32 i[3] = {m.tri[3 * t], m.tri[3 * t + 1], m.tri[3 * t + 2]};
        if (i[0] >= m.pos.size() || i[1] >= m.pos.size() || i[2] >= m.pos.size()) { ++c.bad_idx; continue; }
        const Vec3 cr = cross(m.pos[i[1]] - m.pos[i[0]], m.pos[i[2]] - m.pos[i[0]]);
        if (i[0] == i[1] || i[1] == i[2] || i[0] == i[2] || length(cr) < 1e-7f) ++c.degenerate;
        if (dot(m.nrm[i[0]] + m.nrm[i[1]] + m.nrm[i[2]], cr) <= 0) ++c.flipped;
        for (int k = 0; k < 3; ++k) dir.push_back(static_cast<u64>(i[k]) << 32 | i[(k + 1) % 3]);
    }
    // Arista dirigida repetida con su inversa ausente = dos triángulos que la recorren igual.
    std::sort(dir.begin(), dir.end());
    for (usize k = 0; k + 1 < dir.size(); ++k)
        if (dir[k] == dir[k + 1] && !std::binary_search(dir.begin(), dir.end(), (dir[k] << 32) | (dir[k] >> 32))) ++c.inconsistent;
    for (u8 g : m.group) c.bad_group += g == 0 || g > ngroups;
    for (usize v = 0; v < m.pos.size(); ++v)
        c.nonfinite += !(std::isfinite(m.pos[v].x) && std::isfinite(m.pos[v].y) && std::isfinite(m.pos[v].z) && std::isfinite(m.nrm[v].x) &&
                         std::isfinite(m.nrm[v].y) && std::isfinite(m.nrm[v].z));
    return c;
}

int main() {
    pool().start();
    std::printf("test_geom_models: %d modelos, %d hilos, bug de restas de sdf.cpp %s\n", models::count(), pool().size(),
                geom::sdf_subtract_culling_bug() ? "PRESENTE (ruta literal activa)" : "corregido");
    const double t0 = now_sec();
    constexpr int NX = 256, NY = 128, NZ = 96;
    const usize N = static_cast<usize>(NX) * NY * NZ, L0 = static_cast<usize>(NX) * NY;
    Buffer<u8> a(N, true), b(N, true);
    for (int mi = 0; mi < models::count(); ++mi) {
        const models::Built bm = models::build(mi, models::Params{});
        const Aabb bb = bm.bounds_m;
        // Red que envuelve el modelo con margen (≈ 200×90×60 celdas de objeto), suelo en z = 0.5.
        const float dx = std::max({bb.size().x / 200.0f, bb.size().y / 90.0f, bb.size().z / 60.0f, 0.01f});
        LatticeMap map;
        map.dx = dx;
        map.origin = Vec3(bb.lo.x - 20 * dx, 0.5f * (bb.lo.y + bb.hi.y) - 63.5f * dx, -0.5f * dx);

        usize vdiff = 0, lit = 0;
        for (int cfg = 0; cfg < 2; ++cfg) {
            geom::VoxelOptions o;                      // cfg 0: por defecto (0.12, ss)
            if (cfg == 1) { o.thicken = 0.0f; o.supersample = false; }
            const auto st = geom::voxelize(bm.scene, map, NX, NY, NZ, a.data(), o);
            voxelize_brute(bm.scene, map, NX, NY, NZ, b.data(), o);
            for (usize i = L0; i < N; ++i) vdiff += a[i] != b[i];
            lit += st.literal_cells;
        }

        render::Mesh m, mf;
        geom::MeshStats ms;
        geom::mesh_scene(bm.scene, map, 0.5f, m, geom::MeshOptions{}, &ms);
        geom::MeshOptions full;
        full.block_skip = false;
        geom::mesh_scene(bm.scene, map, 0.5f, mf, full);
        const bool same = m.tri == mf.tri && m.pos.size() == mf.pos.size() &&
                          std::memcmp(m.pos.data(), mf.pos.data(), m.pos.size() * sizeof(Vec3)) == 0 &&
                          std::memcmp(m.nrm.data(), mf.nrm.data(), m.nrm.size() * sizeof(Vec3)) == 0;
        const MeshCheck c = analyze(m, bm.scene.group_count());
        const bool ok = vdiff == 0 && same && c.tris > 0 && c.bad_idx == 0 && c.degenerate == 0 && c.inconsistent == 0 && c.bad_group == 0 &&
                        c.nonfinite == 0 && c.flipped * 100 < c.tris;   // < 1 % (piezas más finas que h)
        std::printf("[%s] %-15s voxel≠bruta=%zu (literal %zu) | malla %6zu tris, ≡completo=%s, incoh=%zu, degen=%zu, normal opuesta=%zu (%.2f%%), sustituidas=%zu, %.1f ms\n",
                    ok ? "PASS" : "FAIL", bm.info.id.c_str(), vdiff, lit, c.tris, same ? "sí" : "NO", c.inconsistent, c.degenerate, c.flipped,
                    100.0 * c.flipped / std::max<usize>(c.tris, 1), ms.normal_fixes, ms.seconds * 1e3);
        std::fflush(stdout);
        if (!ok) ++g_fail;
    }
    std::printf("%s (%d fallos, %.1f s)\n", g_fail ? "FALLÓ" : "OK", g_fail, now_sec() - t0);
    return g_fail ? 1 : 0;
}
