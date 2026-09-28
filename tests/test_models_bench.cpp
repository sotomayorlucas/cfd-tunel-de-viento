// ============================================================================
//  tests/test_models_bench.cpp — micro-benchmark del coste de evaluación del
//  SDF de cada modelo (ns por punto, 1 hilo, mediana de 9 repeticiones) y de
//  su construcción. Sirve para medir las decisiones de diseño del módulo:
//    * nº de puntos por cara de los perfiles (compilar con -DCFD_MODELS_PROFILE_PTS=40),
//    * orden estático de los grupos (--order none|volume|reverse).
//  Puntos: uniformes en la AABB (como el voxelizador) y "cerca de la superficie"
//  (|sdf| < 5 cm, como el mallador / refinamiento).
// ============================================================================
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "models/model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

int main(int argc, char** argv) {
    std::string order = "none", only;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--order") && i + 1 < argc) order = argv[++i];
        else only = argv[i];
    }
    std::printf("%-15s %5s %6s %10s %10s %10s %10s %9s\n", "modelo", "grp", "prims", "unif(ns)", "min(ns)", "cerca(ns)", "alas(min)", "build(us)");
    double sum_u = 0, sum_n = 0, sum_m = 0, sum_w = 0;
    int n_models = 0;
    for (int idx = 0; idx < models::count(); ++idx) {
        const models::Info& I = models::info(idx);
        if (!only.empty() && I.id != only) continue;
        std::vector<double> tb;
        models::Built b;
        for (int r = 0; r < 9; ++r) { const double t0 = now_sec(); b = models::build(idx, models::Params{}); tb.push_back(now_sec() - t0); }
        auto& G = b.scene.groups();
        auto vol = [](const Aabb& a) { const Vec3 s = a.size(); return s.x * s.y * s.z; };
        if (order == "volume") std::stable_sort(G.begin(), G.end(), [&](const sdf::Group& a, const sdf::Group& c) { return vol(a.box_model) > vol(c.box_model); });
        if (order == "reverse") std::reverse(G.begin(), G.end());
        const Aabb bb = b.bounds_m;
        WyRand rng(42);
        constexpr int N = 100000;
        std::vector<Vec3> uni(N), near;
        for (auto& p : uni) p = Vec3(rng.uniform(bb.lo.x, bb.hi.x), rng.uniform(bb.lo.y, bb.hi.y), rng.uniform(bb.lo.z, bb.hi.z));
        while (near.size() < 50000) {
            const Vec3 p(rng.uniform(bb.lo.x, bb.hi.x), rng.uniform(bb.lo.y, bb.hi.y), rng.uniform(bb.lo.z, bb.hi.z));
            if (std::fabs(b.scene.eval(p)) < 0.05f * max_comp(bb.size()) / 5.0f) near.push_back(p);
        }
        double tmin = 0;
        auto run = [&](const std::vector<Vec3>& pts) {
            std::vector<double> t;
            volatile float sink = 0;
            for (int r = 0; r < 9; ++r) {
                const double t0 = now_sec();
                float acc = 0;
                for (const Vec3& p : pts) acc += b.scene.eval(p);
                t.push_back(now_sec() - t0);
                sink = sink + acc;
            }
            tmin = vmin(t) * 1e9 / static_cast<double>(pts.size());
            return median(t) * 1e9 / static_cast<double>(pts.size());
        };
        // Región de alerones (si existen): puntos uniformes en la AABB de los grupos de alas.
        Aabb wbox;
        for (const auto& g : b.scene.groups())
            if (g.component == sdf::Component::FrontWing || g.component == sdf::Component::RearWing || g.component == sdf::Component::WingMain ||
                g.component == sdf::Component::WingFlap)
                wbox.grow(g.box_model);
        std::vector<Vec3> wing(wbox.empty() ? 0 : 50000);
        for (auto& p : wing) p = Vec3(rng.uniform(wbox.lo.x, wbox.hi.x), rng.uniform(wbox.lo.y, wbox.hi.y), rng.uniform(wbox.lo.z, wbox.hi.z));
        const double nu = run(uni), mu = tmin, nn = run(near);
        double nw = 0;
        if (!wing.empty()) { run(wing); nw = tmin; }
        sum_w += nw;
        sum_u += nu; sum_n += nn; sum_m += mu; ++n_models;
        std::printf("%-15s %5zu %6zu %10.0f %10.0f %10.0f %10.0f %9.1f\n", I.id.c_str(), b.scene.group_count(), b.scene.prims().size(), nu, mu, nn,
                    nw, vmin(tb) * 1e6);
    }
    std::printf("media %-9s %25.0f %10.0f %10.0f %10.0f   (orden=%s, puntos/perfil=%s)\n", "", sum_u / n_models, sum_m / n_models, sum_n / n_models,
                sum_w / n_models, order.c_str(),
#ifdef CFD_MODELS_PROFILE_PTS
                "macro"
#else
                "14"
#endif
    );
    return 0;
}
