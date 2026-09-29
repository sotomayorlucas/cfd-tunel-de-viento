// ============================================================================
//  tools/noise_probe.cpp — métrica reproducible del RUIDO del campo lejano.
//
//  Simula un modelo (o el túnel VACÍO con el dominio de ese modelo) con la misma
//  `app::Sim` que la aplicación y mide, a lo largo del tiempo, en regiones de
//  campo lejano (lejos del objeto, fuera de las fronteras de equilibrio):
//
//    UPin   x ∈ [2, 7)                        junto a la entrada
//    UP40   x ∈ [xo−44, xo−36)                ~40 celdas aguas arriba del morro
//    SIDE   y ∈ [2, 2+ny/10), x < inicio de la esponja
//    TOP    z ∈ [nz−2−nz/10, nz−2), x < inicio de la esponja
//    NEAR   fluido a ≤ 8 celdas de los cuerpos (la capa de Config::rr_wall_layer)
//    ALL    todo el fluido (Cp: el pulso acústico del arranque)
//
//  Por región: RMS(u_x/U − 1) (respecto a 1, como lo pide la revisión), media y
//  desviación espacial de u_x/U, RMS de u_y,u_z/U, Cp medio y su desviación
//  espacial; el cociente de PASO ALTO hp = RMS(φ − media de los 6 vecinos)/σ(φ)
//  (≈ 0 en un campo suave, 1.08 en ruido blanco, 2 en un damero (−1)^(i+j+k)) y
//  la energía del modo de Nyquist (−1)^(i+j+k) relativa a la varianza.
//  Temporal (últimos `--tw` PF): σ temporal por celda de u_x/U y Cp (media en la
//  región) y el indicador par/impar de la ω≈2: RMS(φ_{t+1} − ½(φ_t + φ_{t+2}))
//  (segunda diferencia en el tiempo, grande si hay una oscilación de periodo 2).
//  Espectro de u_x a lo largo de y en el plano UP40 (bandas de k/k_Nyquist) y
//  serie temporal de Cp en una sonda aguas arriba (periodos dominantes).
//
//    noise_probe --model f1_2026 --res rapida --ft 2.5 [--empty] [--gpu] [--fp32]
//                [--every PF] [--tw PF] [--csv serie.csv] [--calib PF_media]
//                [--coll pr|rr|bgk] [--bulk ω_b] [--layer N] [--ramp PF] …  (perillas: ver --help)
//
//  --calib A: además promedia las fuerzas de los A últimos PF (como la tabla de
//  calibración de docs/FISICA.md: 4 PF coches / 5 objetos, media de los 2 últimos).
// ============================================================================
#include "../src/app/app.hpp"
#include "../src/core/threadpool.hpp"
#include "../src/core/util.hpp"
#include "../src/models/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;

namespace {

struct Box { int x0, x1, y0, y1, z0, z1; const char* name; const u8* mask = nullptr; };   // mask: sólo celdas ≠ 0

struct RegionStats {
    double rms_ux1 = 0, mean_ux = 0, sd_ux = 0, rms_uyz = 0, mean_cp = 0, sd_cp = 0;
    double hp_ux = 0, hp_cp = 0, nyq_ux = 0, nyq_cp = 0;
    long n = 0;
};

// Estadísticas espaciales de una región en un instante.
RegionStats region_stats(const lbm::FieldView& f, const Box& b, float U, float rho0) {
    RegionStats s;
    const u8 bad = lbm::kSolid | lbm::kInlet | lbm::kOutlet;
    const float kcp = 2.0f / (3.0f * U * U);
    double su = 0, suu = 0, sc = 0, scc = 0, s1 = 0, syz = 0;
    double hu = 0, hc = 0, nu = 0, nc = 0;
    long nh = 0;
    for (int z = b.z0; z < b.z1; ++z)
        for (int y = b.y0; y < b.y1; ++y)
            for (int x = b.x0; x < b.x1; ++x) {
                const usize n = f.index(x, y, z);
                if ((f.flags[n] & bad) || (b.mask && !b.mask[n])) continue;
                const double u = f.ux[n] / U, c = kcp * (f.rho[n] - rho0);
                su += u; suu += u * u; sc += c; scc += c * c;
                s1 += (u - 1.0) * (u - 1.0);
                syz += (double(f.uy[n]) * f.uy[n] + double(f.uz[n]) * f.uz[n]) / (double(U) * U);
                const double sg = ((x + y + z) & 1) ? -1.0 : 1.0;
                nu += sg * u; nc += sg * c;
                ++s.n;
                // Paso alto: sólo si los 6 vecinos son fluido.
                const usize o[6] = {n - 1, n + 1, n - usize(f.nx), n + usize(f.nx), n - usize(f.nx) * usize(f.ny), n + usize(f.nx) * usize(f.ny)};
                bool ok = x > 0 && x < f.nx - 1 && y > 0 && y < f.ny - 1 && z > 0 && z < f.nz - 1;
                double au = 0, ac = 0;
                if (ok)
                    for (int k = 0; k < 6; ++k) {
                        if (f.flags[o[k]] & bad) { ok = false; break; }
                        au += f.ux[o[k]] / U; ac += kcp * (f.rho[o[k]] - rho0);
                    }
                if (ok) {
                    const double ru = u - au / 6.0, rc = c - ac / 6.0;
                    hu += ru * ru; hc += rc * rc; ++nh;
                }
            }
    if (!s.n) return s;
    const double N = static_cast<double>(s.n);
    s.mean_ux = su / N;
    s.sd_ux = std::sqrt(std::max(0.0, suu / N - s.mean_ux * s.mean_ux));
    s.mean_cp = sc / N;
    s.sd_cp = std::sqrt(std::max(0.0, scc / N - s.mean_cp * s.mean_cp));
    s.rms_ux1 = std::sqrt(s1 / N);
    s.rms_uyz = std::sqrt(syz / N);
    if (nh) {
        s.hp_ux = s.sd_ux > 0 ? std::sqrt(hu / nh) / s.sd_ux : 0;
        s.hp_cp = s.sd_cp > 0 ? std::sqrt(hc / nh) / s.sd_cp : 0;
    }
    // Energía del modo de Nyquist relativa a la varianza (1/N para ruido blanco; 1 para un damero puro).
    s.nyq_ux = s.sd_ux > 0 ? (nu / N) * (nu / N) / (s.sd_ux * s.sd_ux) : 0;
    s.nyq_cp = s.sd_cp > 0 ? (nc / N) * (nc / N) / (s.sd_cp * s.sd_cp) : 0;
    return s;
}

// Acumulador temporal por celda de una región (σ temporal y segunda diferencia par/impar).
struct TempAcc {
    Box b{};
    std::vector<double> su, suu, sc, scc;
    long m = 0;
    double d2u = 0, d2c = 0;   // Σ (φ_{t+1} − ½(φ_t+φ_{t+2}))²
    long nd2 = 0;
    void init(const Box& bb) {
        b = bb;
        const usize n = static_cast<usize>(b.x1 - b.x0) * static_cast<usize>(b.y1 - b.y0) * static_cast<usize>(b.z1 - b.z0);
        su.assign(n, 0); suu.assign(n, 0); sc.assign(n, 0); scc.assign(n, 0);
    }
};

template <class Fn>
void for_region(const lbm::FieldView& f, const Box& b, Fn&& fn) {
    const u8 bad = lbm::kSolid | lbm::kInlet | lbm::kOutlet;
    usize i = 0;
    for (int z = b.z0; z < b.z1; ++z)
        for (int y = b.y0; y < b.y1; ++y)
            for (int x = b.x0; x < b.x1; ++x, ++i) {
                const usize n = f.index(x, y, z);
                if ((f.flags[n] & bad) || (b.mask && !b.mask[n])) continue;
                fn(i, n);
            }
}

struct Series { std::vector<float> cp; int dt = 1; };

// Periodos dominantes (en pasos) de una serie por DFT directa (sin la media).
void dominant_periods(const std::vector<float>& v, int dt, char* out, usize cap) {
    const usize M = v.size();
    out[0] = 0;
    if (M < 16) return;
    double mean = 0;
    for (float x : v) mean += x;
    mean /= static_cast<double>(M);
    std::vector<std::pair<double, int>> pw;
    for (usize k = 1; k <= M / 2; ++k) {
        double re = 0, im = 0;
        for (usize t = 0; t < M; ++t) {
            const double a = 2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(M);
            re += (v[t] - mean) * std::cos(a);
            im -= (v[t] - mean) * std::sin(a);
        }
        pw.push_back({re * re + im * im, static_cast<int>(k)});
    }
    std::sort(pw.begin(), pw.end(), [](auto& a, auto& b) { return a.first > b.first; });
    double tot = 0;
    for (auto& p : pw) tot += p.first;
    usize len = 0;
    for (int i = 0; i < 3 && i < static_cast<int>(pw.size()); ++i) {
        const double per = static_cast<double>(M) * dt / pw[static_cast<usize>(i)].second;
        len += static_cast<usize>(std::snprintf(out + len, cap - len, "%s%.0f pasos (%.0f%%)", i ? ", " : "", per,
                                                100.0 * pw[static_cast<usize>(i)].first / std::max(tot, 1e-300)));
        if (len >= cap) break;
    }
}

// Espectro de u_x a lo largo de y (plano x = xp, varias z): fracción de la energía por cuartos de k/k_Nyquist.
void y_spectrum(const lbm::FieldView& f, int xp, float U, double bands[4]) {
    for (int i = 0; i < 4; ++i) bands[i] = 0;
    const int y0 = 2, y1 = f.ny - 2, L = y1 - y0;
    if (L < 16) return;
    const u8 bad = lbm::kSolid | lbm::kInlet | lbm::kOutlet;
    for (int z = 2; z < f.nz - 2; z += 2) {
        std::vector<double> v(static_cast<usize>(L));
        bool ok = true;
        double mean = 0;
        for (int y = y0; y < y1; ++y) {
            const usize n = f.index(xp, y, z);
            if (f.flags[n] & bad) { ok = false; break; }
            v[static_cast<usize>(y - y0)] = f.ux[n] / U;
            mean += v[static_cast<usize>(y - y0)];
        }
        if (!ok) continue;
        mean /= L;
        for (int k = 1; k <= L / 2; ++k) {
            double re = 0, im = 0;
            for (int t = 0; t < L; ++t) {
                const double a = 2.0 * M_PI * k * t / L;
                re += (v[static_cast<usize>(t)] - mean) * std::cos(a);
                im -= (v[static_cast<usize>(t)] - mean) * std::sin(a);
            }
            const double kn = static_cast<double>(k) / (L / 2.0);   // k/k_Nyquist ∈ (0, 1]
            bands[std::min(3, static_cast<int>(kn * 4.0 - 1e-9))] += re * re + im * im;
        }
    }
    const double tot = bands[0] + bands[1] + bands[2] + bands[3];
    if (tot > 0) for (int i = 0; i < 4; ++i) bands[i] /= tot;
}

// Celdas de fluido a distancia de Chebyshev ≤ R de un sólido que no es el suelo (la "capa" junto a los cuerpos).
std::vector<u8> near_mask(const lbm::FieldView& f, int R) {
    const usize N = static_cast<usize>(f.nx) * f.ny * f.nz;
    std::vector<u8> a(N), b(N);
    for (int z = 0; z < f.nz; ++z)
        for (int y = 0; y < f.ny; ++y)
            for (int x = 0; x < f.nx; ++x) {
                const usize n = f.index(x, y, z);
                a[n] = (f.flags[n] & lbm::kSolid) && !(z == 0 && f.solid_id[n] == lbm::k_ground_id);
            }
    auto pass = [&](std::vector<u8>& src, std::vector<u8>& dst, int axis) {
        const int len = axis == 0 ? f.nx : (axis == 1 ? f.ny : f.nz);
        for (int z = 0; z < (axis == 2 ? 1 : f.nz); ++z)
            for (int y = 0; y < (axis == 1 ? 1 : f.ny); ++y)
                for (int x = 0; x < (axis == 0 ? 1 : f.nx); ++x) {
                    std::vector<int> fw(static_cast<usize>(len));
                    auto at = [&](int i) { return f.index(axis == 0 ? i : x, axis == 1 ? i : y, axis == 2 ? i : z); };
                    int d = 1 << 20;
                    for (int i = 0; i < len; ++i) { d = src[at(i)] ? 0 : d + 1; fw[static_cast<usize>(i)] = d; }
                    d = 1 << 20;
                    for (int i = len - 1; i >= 0; --i) { d = src[at(i)] ? 0 : d + 1; dst[at(i)] = std::min(d, fw[static_cast<usize>(i)]) <= R; }
                }
    };
    pass(a, b, 0); pass(b, a, 1); pass(a, b, 2);
    for (usize n = 0; n < N; ++n) b[n] = b[n] && !(f.flags[n] & lbm::kSolid);
    return b;
}

void usage() {
    std::printf(
        "noise_probe — ruido del campo lejano (ver cabecera de tools/noise_probe.cpp)\n"
        "  --model id  --res rapida|media|alta  --ft PF  --empty  --gpu  --fp32  --ground none|static|moving\n"
        "  --every PF (muestreo, defecto 0.25)  --tw PF (ventana temporal final, defecto 0.5)\n"
        "  --csv f.csv (serie temporal)  --calib PF (fuerzas medias de los últimos PF)  --quiet\n"
        "  --nu v  --cs v  --ramp PF  --param k=v\n"
        "  Perillas del solver (lbm::Config): --coll pr|rr|bgk (colisión; defecto rr)\n"
        "  --bulk ω_b (relajación de la traza de Π^neq; 0 = como la cortante; defecto 1)\n"
        "  --layer N (capa junto a los cuerpos sin el término de 3er orden, celdas; defecto 8)\n");
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    pool().start();
    std::string model = "f1_2026", res = "rapida", csv;
    float ft = 2.5f, every = 0.25f, tw = 0.5f, calib = 0.0f;
    bool empty = false, gpu = false, fp32 = false, quiet = false;
    app::SimConfig cfg;
    bool ground_set = false;
    lbm::GroundMode ground = lbm::GroundMode::Moving;
    std::vector<std::pair<std::string, float>> params;
    float knob_bulk = -1.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = nx();
        else if (a == "--res") res = nx();
        else if (a == "--ft") ft = std::strtof(nx(), nullptr);
        else if (a == "--every") every = std::strtof(nx(), nullptr);
        else if (a == "--tw") tw = std::strtof(nx(), nullptr);
        else if (a == "--calib") calib = std::strtof(nx(), nullptr);
        else if (a == "--empty") empty = true;
        else if (a == "--gpu") gpu = true;
        else if (a == "--fp32") fp32 = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--csv") csv = nx();
        else if (a == "--nu") cfg.nu = std::strtof(nx(), nullptr);
        else if (a == "--cs") cfg.cs = std::strtof(nx(), nullptr);
        else if (a == "--ramp") cfg.ramp_ft = std::strtof(nx(), nullptr);
        else if (a == "--coll") { const std::string c = nx(); cfg.collision = c == "bgk" ? lbm::Collision::BGK : (c == "pr" ? lbm::Collision::Regularized : lbm::Collision::Recursive); }
        else if (a == "--layer") cfg.rr_wall_layer = std::atoi(nx());
        else if (a == "--bulk") knob_bulk = std::strtof(nx(), nullptr);
        else if (a == "--ground") {
            const std::string g = nx();
            ground_set = true;
            ground = g == "none" ? lbm::GroundMode::None : (g == "static" ? lbm::GroundMode::Static : lbm::GroundMode::Moving);
        } else if (a == "--param") {
            const std::string p = nx();
            const auto eq = p.find('=');
            if (eq != std::string::npos) params.push_back({p.substr(0, eq), std::strtof(p.c_str() + eq + 1, nullptr)});
        } else { usage(); return a == "--help" ? 0 : 1; }
    }
    const int mi = models::find(model);
    if (mi < 0) { std::fprintf(stderr, "modelo desconocido: %s\n", model.c_str()); return 1; }
    app::Preset pr;
    if (!app::parse_preset(res.c_str(), pr)) { std::fprintf(stderr, "resolución desconocida: %s\n", res.c_str()); return 1; }

    app::Sim sim;
    cfg.model = mi;
    cfg.preset = pr;
    cfg.fp32 = fp32;
    cfg.params = models::resolve_params(mi, models::Params{});
    const models::Info& I = models::info(mi);
    cfg.ground = ground_set ? ground : (I.needs_ground ? lbm::GroundMode::Moving : lbm::GroundMode::None);
    for (auto& [k, v] : params) {
        if (k == "aoa") cfg.params.aoa_deg = v;
        else if (k == "height") cfg.params.height_mm = v;
        else if (k == "yaw") cfg.params.yaw_deg = v;
        else std::fprintf(stderr, "aviso: --param %s no soportado por la sonda\n", k.c_str());
    }
    if (knob_bulk >= 0.0f) cfg.bulk_omega = knob_bulk;
    sim.cfg = cfg;
    sim.init();
    if (empty) {
        sim.solver.clear_wall_motions();
        sim.solver.set_geometry(nullptr);
        sim.solver.reset_flow();
    }
    if (gpu && !sim.set_gpu(true)) { std::fprintf(stderr, "sin iGPU\n"); return 2; }

    const lbm::FieldView f0 = sim.solver.field();
    const int nx = f0.nx, ny = f0.ny, nz = f0.nz;
    const float U = sim.cfg.u_lat;
    const Aabb ob = sim.object_cells();
    const int xo = std::max(8, static_cast<int>(ob.lo.x));
    const int xs = static_cast<int>(static_cast<float>(nx) * (1.0f - 0.12f));   // inicio de la esponja
    const int zlo = sim.dom.ground ? 2 : 2;
    const int dy = std::max(2, ny / 10), dz = std::max(2, nz / 10);
    // NEAR: celdas a ≤ 8 celdas de los cuerpos (donde la colisión no añade el término de 3er orden: Config::rr_wall_layer).
    const std::vector<u8> near = near_mask(f0, 8);
    const Box boxes[] = {
        {2, std::min(7, xo), 2, ny - 2, zlo, nz - 2, "UPin"},
        {std::max(2, xo - 44), std::max(3, xo - 36), 2, ny - 2, zlo, nz - 2, "UP40"},
        {2, xs, 2, 2 + dy, zlo, nz - 2, "SIDE"},
        {2, xs, 2, ny - 2, nz - 2 - dz, nz - 2, "TOP"},
        {1, nx - 1, 1, ny - 1, 1, nz - 1, "NEAR", near.data()},
        {1, nx - 1, 1, ny - 1, 1, nz - 1, "ALL"},
    };
    constexpr int NB = 6;
    const float fts = sim.ft_steps();
    const long total = static_cast<long>(std::lround(ft * fts));
    const int dstep = std::max(1, static_cast<int>(std::lround(every * fts)));
    const long tw_start = total - static_cast<long>(std::lround(tw * fts));
    const int ser_dt = std::max(1, static_cast<int>(std::lround(fts / 200.0f)));   // serie: ~200 muestras por PF
    std::printf("# %s%s res=%s dominio %dx%dx%d (%.2f M) dx=%.1f mm  1 PF = %.0f pasos  U=%.3f nu=%.2g cs=%.2f %s%s  xo=%d\n",
                model.c_str(), empty ? " (VACÍO)" : "", res.c_str(), nx, ny, nz, nx * double(ny) * nz * 1e-6, sim.dom.dx * 1e3, fts, U,
                sim.nu, sim.cfg.cs, fp32 ? "FP32" : "FP16S", gpu ? " GPU" : " CPU", xo);
    std::printf("# perillas: colisión %s bulk_omega=%.3f capa=%d ramp=%.2f PF\n",
                sim.cfg.collision == lbm::Collision::Recursive ? "RR" : (sim.cfg.collision == lbm::Collision::BGK ? "BGK" : "PR"),
                sim.cfg.bulk_omega, sim.cfg.rr_wall_layer, sim.cfg.ramp_ft);
    if (!quiet)
        std::printf("%6s %-5s %8s %8s %8s %8s %8s %8s %6s %6s %8s %8s\n", "PF", "reg", "rms(u-1)", "<u>", "sd(u)", "rms(uyz)", "<Cp>",
                    "sd(Cp)", "hp_u", "hp_cp", "nyq_u", "nyq_cp");

    // Sonda (serie temporal): centro de UP40.
    const int px = (boxes[1].x0 + boxes[1].x1) / 2, py = ny / 2 + ny / 5, pz = std::min(nz - 3, (nz * 3) / 4);
    Series ser;
    ser.dt = ser_dt;
    TempAcc tacc[NB - 1];
    for (int b = 0; b < NB - 1; ++b) tacc[b].init(boxes[b]);
    std::vector<float> prev_u[NB - 1], prev_c[NB - 1], prev2_u[NB - 1], prev2_c[NB - 1];
    const float kcp = 2.0f / (3.0f * U * U);
    // Fuerzas medias de calibración.
    const long cal_start = calib > 0 ? total - static_cast<long>(std::lround(calib * fts)) : total + 1;
    double cf[256][3] = {}, cm[256][3] = {};
    long cal_n = 0;
    double t_run = 0;
    long t = 0, next_sample = dstep;
    RegionStats last[NB];
    while (t < total) {
        long k = std::min<long>(next_sample, total) - t;
        if (t >= tw_start) k = std::min<long>(k, ser_dt);
        if (t < cal_start && t + k > cal_start) k = cal_start - t;
        k = std::max<long>(1, k);
        const double a = now_sec();
        if (!sim.step(static_cast<int>(k))) { std::printf("DIVERGENCIA en el paso %ld\n", t); return 3; }
        t_run += now_sec() - a;
        t += k;
        if (t > cal_start) {
            const lbm::ForceSample& fm = sim.solver.forces_mean();
            for (int id = 0; id < 256; ++id)
                for (int c = 0; c < 3; ++c) { cf[id][c] += (&fm.force[id].x)[c] * double(k); cm[id][c] += (&fm.moment[id].x)[c] * double(k); }
            cal_n += k;
        }
        const lbm::FieldView f = sim.solver.field();
        if (t >= tw_start) {
            ser.cp.push_back(kcp * (f.rho[f.index(px, py, pz)] - 1.0f));
            // Temporal: σ por celda y par/impar (paso a paso con step(1) al tomar cada muestra de la ventana).
            for (int b = 0; b < NB - 1; ++b) {
                TempAcc& T = tacc[b];
                for_region(f, T.b, [&](usize i, usize n) {
                    const double u = f.ux[n] / U, c = kcp * (f.rho[n] - 1.0f);
                    T.su[i] += u; T.suu[i] += u * u; T.sc[i] += c; T.scc[i] += c * c;
                });
                ++T.m;
            }
        }
        if (t >= next_sample || t >= total) {
            next_sample += dstep;
            const float pf = static_cast<float>(t) / fts;
            for (int b = 0; b < NB; ++b) {
                last[b] = region_stats(f, boxes[b], U, 1.0f);
                const RegionStats& s = last[b];
                if (!quiet)
                    std::printf("%6.2f %-5s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.2f %6.2f %8.2e %8.2e\n", pf, boxes[b].name, s.rms_ux1,
                                s.mean_ux, s.sd_ux, s.rms_uyz, s.mean_cp, s.sd_cp, s.hp_ux, s.hp_cp, s.nyq_ux, s.nyq_cp);
            }
            std::fflush(stdout);
        }
    }
    // Indicador par/impar al final: 3 pasos consecutivos con campo macro.
    std::vector<float> fu[3][NB - 1], fc[3][NB - 1];
    for (int s = 0; s < 3; ++s) {
        sim.step(1);
        const lbm::FieldView f = sim.solver.field();
        for (int b = 0; b < NB - 1; ++b) {
            fu[s][b].clear(); fc[s][b].clear();
            for_region(f, boxes[b], [&](usize, usize n) { fu[s][b].push_back(f.ux[n] / U); fc[s][b].push_back(kcp * (f.rho[n] - 1.0f)); });
        }
    }
    std::printf("\n# resumen final (PF %.2f; σ_t = desviación temporal media por celda en los últimos %.2f PF, %ld muestras cada %d pasos)\n",
                t / fts, tw, tacc[0].m, ser_dt);
    std::printf("%-5s %9s %9s %9s %9s %7s %7s %9s %9s %9s %9s\n", "reg", "rms(u-1)", "sd(u)", "<Cp>", "sd(Cp)", "hp_u", "hp_cp", "σt(u)",
                "σt(Cp)", "d2t(u)", "d2t(Cp)");
    for (int b = 0; b < NB; ++b) {
        const RegionStats& s = last[b];
        double stu = 0, stc = 0, d2u = 0, d2c = 0;
        if (b < NB - 1) {
            const TempAcc& T = tacc[b];
            long nc = 0;
            for (usize i = 0; i < T.su.size(); ++i) {
                if (T.m < 2) break;
                const double mu = T.su[i] / T.m, mc = T.sc[i] / T.m;
                const double vu = T.suu[i] / T.m - mu * mu, vc = T.scc[i] / T.m - mc * mc;
                if (T.su[i] == 0 && T.suu[i] == 0) continue;
                stu += std::max(0.0, vu); stc += std::max(0.0, vc); ++nc;
            }
            if (nc) { stu = std::sqrt(stu / nc); stc = std::sqrt(stc / nc); }
            const usize M = fu[0][b].size();
            for (usize i = 0; i < M; ++i) {
                const double a = fu[1][b][i] - 0.5 * (fu[0][b][i] + fu[2][b][i]);
                const double c = fc[1][b][i] - 0.5 * (fc[0][b][i] + fc[2][b][i]);
                d2u += a * a; d2c += c * c;
            }
            if (M) { d2u = std::sqrt(d2u / M); d2c = std::sqrt(d2c / M); }
        }
        std::printf("%-5s %9.4f %9.4f %9.4f %9.4f %7.2f %7.2f %9.4f %9.4f %9.2e %9.2e\n", boxes[b].name, s.rms_ux1, s.sd_ux, s.mean_cp, s.sd_cp,
                    s.hp_ux, s.hp_cp, stu, stc, d2u, d2c);
    }
    {
        const lbm::FieldView f = sim.solver.field();
        double bands[4];
        y_spectrum(f, px, U, bands);
        std::printf("espectro u_x(y) en x=%d: k/kN [0,.25) %.2f  [.25,.5) %.2f  [.5,.75) %.2f  [.75,1] %.2f\n", px, bands[0], bands[1], bands[2],
                    bands[3]);
        char buf[256];
        dominant_periods(ser.cp, ser.dt, buf, sizeof buf);
        std::printf("sonda Cp (%d,%d,%d): periodos dominantes %s   (tránsito acústico nx/cs = %.0f pasos)\n", px, py, pz, buf,
                    nx / 0.57735);
    }
    if (calib > 0 && cal_n > 0 && !empty) {
        Vec3 fl[256], ml[256], ff[256], mm[256];
        for (int id = 0; id < 256; ++id) {
            fl[id] = Vec3(float(cf[id][0] / cal_n), float(cf[id][1] / cal_n), float(cf[id][2] / cal_n));
            ml[id] = Vec3(float(cm[id][0] / cal_n), float(cm[id][1] / cal_n), float(cm[id][2] / cal_n));
        }
        lbm::ForceSample fs;
        for (int id = 0; id < 256; ++id) { fs.force[id] = fl[id]; fs.moment[id] = ml[id]; }
        sim.id_forces(fs, ff, mm);
        const app::AeroResult r = app::compute_aero(sim, ff, mm);
        std::printf("CALIB %s %s: CL %.4f CD %.4f SCz %.3f SCx %.3f bal %.1f  (media de %ld pasos)\n", model.c_str(), res.c_str(), r.cl, r.cd,
                    r.scz, r.scx, r.balance, cal_n);
        for (int c = 0; c < r.ncomp; ++c) std::printf("   %-22s SCz %+.3f SCx %.3f\n", r.comp[c].name, r.comp[c].scz, r.comp[c].scx);
    }
    std::printf("tiempo de simulación %.1f s (%.0f MLUPS medios)\n", t_run, nx * double(ny) * nz * t / std::max(t_run, 1e-9) * 1e-6);
    if (!csv.empty()) {
        if (FILE* fo = std::fopen(csv.c_str(), "w")) {
            std::fprintf(fo, "paso,cp\n");
            for (usize i = 0; i < ser.cp.size(); ++i) std::fprintf(fo, "%ld,%g\n", tw_start + static_cast<long>(i) * ser.dt, ser.cp[i]);
            std::fclose(fo);
        }
    }
    if (sim.gpu_on()) sim.set_gpu(false);
    return 0;
}
