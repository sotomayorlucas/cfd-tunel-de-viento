// ============================================================================
//  render/flowvis_volume.cpp — volumen de vórtices (criterio Q o |ω|).
//
//  update(): Q por filas con el núcleo AVX2 compartido (detail::quantity_row),
//            promedio 2³ opcional, cuantización a u8 (densidad = 255·Q/full y
//            color) y ladrillos 8³ con el máximo (+1 vóxel de margen).
//            El umbral NO se hornea: es un índice de LUT evaluado en render()
//            → mover el deslizador del umbral no requiere recalcular nada.
//  render(): raymarching por píxel (o por bloque 2×2), salto de ladrillos
//            vacíos (máximo < umbral), prueba SWAR de 8 esquinas ("hay byte
//            mayor que n" en un u64), dos estilos:
//              Surface — cruces del isovalor refinados por interpolación
//                        lineal, normal por gradiente (6 muestras SÓLO en el
//                        cruce), Blinn-Phong, varias capas semitransparentes;
//              Cloud   — absorción-emisión con LUT de opacidad.
//            Composición front-to-back premultiplicada, terminación temprana,
//            parada en la profundidad opaca de la escena, ruido IGN contra el
//            bandeado y reescalado 2× bilineal con SWAR de 16 bits por canal
//            (PDEP/PEXT de BMI2) con rechazo por profundidad a resolución completa.
// ============================================================================
#include "flowvis.hpp"
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace cfd::flowvis {

using lbm::FieldView;

namespace {
constexpr int k_brick = 8;                     // vóxeles por lado de ladrillo
constexpr u64 k_lanes = 0x00FF00FF00FF00FFull; // 4 canales u8 → 4 carriles de 16 bits
constexpr float k_inf = std::numeric_limits<float>::infinity();

// Ruido de gradiente entrelazado (Jimenez 2014): desplaza el inicio de cada rayo
// para convertir el bandeado del muestreo regular en grano fino (lo suaviza el reescalado).
CFD_INLINE float ign(int x, int y) {
    const float f = 0.06711056f * static_cast<float>(x) + 0.00583715f * static_cast<float>(y);
    const float g = 52.9829189f * (f - std::floor(f));
    return g - std::floor(g);
}

// SWAR "¿algún byte > n?" (bit twiddling hacks, hasmore) — válido para 0 ≤ n ≤ 127.
// Si un byte ≥ 128, el acarreo puede ensuciar otros bytes, pero la respuesta global
// ya es "sí" gracias al OR con x.
CFD_INLINE bool any_byte_gt(u64 x, u32 n) {
    constexpr u64 k01 = 0x0101010101010101ull, k80 = 0x8080808080808080ull;
    return (((x + k01 * (127u - n)) | x) & k80) != 0;
}
} // namespace

// Anula (NaN) los valores de la fila (y,z) en celdas sólidas o con algún vecino 6-conexo
// sólido. SWAR: OR de 7 palabras de 8 flags (fila, x±1, y±1, z±1) → salta 8 celdas por
// comparación cuando no hay paredes cerca (el caso común).
// (En detail:: para poder contrastarla en los tests con una versión de fuerza bruta.)
void detail::mask_near_wall(const FieldView& f, int y, int z, float* CFD_RESTRICT out) {
    const int nx = f.nx;
    const usize sy = static_cast<usize>(nx), sz = sy * static_cast<usize>(f.ny);
    const u8* r0 = f.flags + f.index(0, y, z);
    const u8* rym = y > 0 ? r0 - sy : r0;
    const u8* ryp = y < f.ny - 1 ? r0 + sy : r0;
    const u8* rzm = z > 0 ? r0 - sz : r0;
    const u8* rzp = z < f.nz - 1 ? r0 + sz : r0;
    constexpr u64 kS = 0x0101010101010101ull * lbm::kSolid;
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    auto ld = [](const u8* p) { u64 w; std::memcpy(&w, p, 8); return w; };
    int x = 0;
    if (nx >= 10) {
        for (x = 1; x + 9 <= nx; x += 8) {
            const u64 w = ld(r0 + x - 1) | ld(r0 + x) | ld(r0 + x + 1) | ld(rym + x) | ld(ryp + x) | ld(rzm + x) | ld(rzp + x);
            u64 m = w & kS;
            while (m) {                                   // sólo los bytes con pared cerca
                const int b = std::countr_zero(m) >> 3;
                out[x + b] = nan;
                m &= ~(0xFFull << (b * 8));
            }
        }
    }
    for (int xx = 0; xx < nx; ++xx) {                     // extremos (x = 0 y cola) en escalar
        if (xx >= 1 && xx < x) { xx = x - 1; continue; }
        u8 w = r0[xx] | rym[xx] | ryp[xx] | rzm[xx] | rzp[xx];
        if (xx > 0) w |= r0[xx - 1];
        if (xx < nx - 1) w |= r0[xx + 1];
        if (w & lbm::kSolid) out[xx] = nan;
    }
}

// ============================================================================
//  Actualización del volumen
// ============================================================================
void VortexVolume::update(const FieldView& f) {
    CFD_CHECK(f.valid() && f.flags != nullptr, "VortexVolume: campo sin datos o sin flags");
    const int ds = params.downsample >= 2 ? 2 : 1;
    const int vx = max_(f.nx / ds, 2), vy = max_(f.ny / ds, 2), vz = max_(f.nz / ds, 2);
    const usize N = static_cast<usize>(vx) * static_cast<usize>(vy) * static_cast<usize>(vz);
    if (vx != vx_ || vy != vy_ || vz != vz_ || ds != ds_ || dens_.size() != N) {
        vx_ = vx; vy_ = vy; vz_ = vz; ds_ = ds;
        dens_.resize(N); col_.resize(N);
        bx_ = (vx + k_brick - 1) / k_brick; by_ = (vy + k_brick - 1) / k_brick; bz_ = (vz + k_brick - 1) / k_brick;
        brick_.resize(static_cast<usize>(bx_) * static_cast<usize>(by_) * static_cast<usize>(bz_));
    }
    const Quantity q = params.field == VolumeField::Vorticity ? Quantity::Vorticity : Quantity::QCriterion;
    full_ = max_(params.full, 1e-12f);   // escala de cuantización horneada (render la reutiliza para el umbral)
    const float kq = 255.0f / full_;
    const bool by_ux = params.color_by == VolumeColor::Streamwise;
    const ColorScale csc = params.color_scale;
    const float inv_u = 1.0f / f.u_inf;
    const int nthr = max_(pool().size(), 1);
    const usize rowpad = (static_cast<usize>(f.nx) + 15) & ~usize{15};
    const usize per = rowpad * static_cast<usize>(ds * ds);
    if (scratch_.size() < per * static_cast<usize>(nthr)) scratch_.resize(per * static_cast<usize>(nthr));
    const float inv_cells = 1.0f / static_cast<float>(ds * ds * ds);
    const bool near_wall = params.hide_near_wall;
    std::atomic<u32> vmax_bits{0};

    parallel_for(0, static_cast<i64>(vy) * vz, 2, [&](i64 lo, i64 hi) {
        const int wi = max_(ThreadPool::worker_index(), 0);
        float* CFD_RESTRICT sc = scratch_.data() + per * static_cast<usize>(wi);
        float local_max = 0.0f;
        for (i64 r = lo; r < hi; ++r) {
            const int yv = static_cast<int>(r % vy), zv = static_cast<int>(r / vy);
            for (int dz = 0; dz < ds; ++dz)
                for (int dy = 0; dy < ds; ++dy) {
                    const int y = min_(yv * ds + dy, f.ny - 1), z = min_(zv * ds + dz, f.nz - 1);
                    float* row = sc + rowpad * static_cast<usize>(dz * ds + dy);
                    detail::quantity_row(f, q, y, z, row);
                    if (near_wall) detail::mask_near_wall(f, y, z, row);
                }
            const usize vo = static_cast<usize>(vx) * static_cast<usize>(r);
            u8* CFD_RESTRICT dd = dens_.data() + vo;
            u8* CFD_RESTRICT cc = col_.data() + vo;
            for (int xv = 0; xv < vx; ++xv) {
                float s = 0.0f, su = 0.0f;
                for (int k = 0; k < ds * ds; ++k) {
                    const float* rowq = sc + rowpad * static_cast<usize>(k);
                    const int y = min_(yv * ds + (k % ds), f.ny - 1), z = min_(zv * ds + (k / ds), f.nz - 1);
                    const usize nrow = f.index(0, y, z);
                    for (int dx = 0; dx < ds; ++dx) {
                        const int x = min_(xv * ds + dx, f.nx - 1);
                        const float v = rowq[x];
                        s += v == v ? v : 0.0f;                                    // sólido → 0
                        const usize n = nrow + static_cast<usize>(x);
                        su += (f.flags[n] & lbm::kSolid) ? 0.0f : f.ux[n];
                    }
                }
                s *= inv_cells;
                local_max = max_(local_max, s);
                const float dq = s * kq;                                            // Q < 0 → 0
                dd[xv] = static_cast<u8>(dq > 0.0f ? min_(dq, 255.0f) + 0.5f : 0.0f);
                // NaN (campo divergido) → 0: saturate(NaN) = NaN y (u8)NaN es UB.
                const float tc = by_ux ? saturate(csc.normalize(su * inv_cells * inv_u)) : 0.0f;
                cc[xv] = by_ux ? static_cast<u8>((tc == tc ? tc : 0.0f) * 255.0f + 0.5f) : dd[xv];
            }
        }
        // Máximo global sin locks: los floats ≥ 0 se ordenan igual que sus bits como u32.
        u32 cur = vmax_bits.load(std::memory_order_relaxed);
        const u32 mine = std::bit_cast<u32>(max_(local_max, 0.0f));
        while (mine > cur && !vmax_bits.compare_exchange_weak(cur, mine, std::memory_order_relaxed)) {}
    });

    // Ladrillos: máximo de densidad en [b·8-1, b·8+8] (margen para la trilineal).
    const int nb = bx_ * by_ * bz_;
    parallel_for(0, nb, 4, [&](i64 lo, i64 hi) {
        for (i64 b = lo; b < hi; ++b) {
            const int ix = static_cast<int>(b % bx_), iy = static_cast<int>((b / bx_) % by_), iz = static_cast<int>(b / (static_cast<i64>(bx_) * by_));
            const int x0 = max_(ix * k_brick - 1, 0), x1 = min_(ix * k_brick + k_brick, vx - 1);
            const int y0 = max_(iy * k_brick - 1, 0), y1 = min_(iy * k_brick + k_brick, vy - 1);
            const int z0 = max_(iz * k_brick - 1, 0), z1 = min_(iz * k_brick + k_brick, vz - 1);
            u8 m = 0;
            for (int z = z0; z <= z1; ++z)
                for (int y = y0; y <= y1; ++y) {
                    const u8* row = dens_.data() + static_cast<usize>(vx) * (static_cast<usize>(y) + static_cast<usize>(vy) * static_cast<usize>(z));
                    for (int x = x0; x <= x1; ++x) m = max_(m, row[x]);
                }
            brick_[static_cast<usize>(b)] = m;
        }
    });
    col_ux_ = by_ux;                 // col_ guarda u_x normalizada o la densidad: el render usa la LUT
    col_map_ = csc.map;              // correspondiente a lo que se calculó aquí (no a params actuales)
    stats_.vx = vx; stats_.vy = vy; stats_.vz = vz;
    stats_.bricks = static_cast<usize>(nb);
    stats_.max_value = std::bit_cast<float>(vmax_bits.load());
    count_bricks();
}

// Cuenta los ladrillos ocupados para el umbral actual y su caja envolvente (≤ unos miles
// de ladrillos: coste despreciable, se repite en cada render porque el umbral es de render).
void VortexVolume::count_bricks() {
    const u32 iso = iso_index();
    usize ne = 0;
    int lo[3] = {bx_, by_, bz_}, hi[3] = {-1, -1, -1};
    for (int z = 0; z < bz_; ++z)
        for (int y = 0; y < by_; ++y)
            for (int x = 0; x < bx_; ++x) {
                if (brick_[static_cast<usize>(x) + static_cast<usize>(bx_) * (static_cast<usize>(y) + static_cast<usize>(by_) * static_cast<usize>(z))] < iso) continue;
                ++ne;
                lo[0] = min_(lo[0], x); lo[1] = min_(lo[1], y); lo[2] = min_(lo[2], z);
                hi[0] = max_(hi[0], x); hi[1] = max_(hi[1], y); hi[2] = max_(hi[2], z);
            }
    stats_.bricks_nonempty = ne;
    for (int k = 0; k < 3; ++k) { occ_lo_[k] = lo[k]; occ_hi_[k] = hi[k]; }
}

u32 VortexVolume::iso_index() const {
    const float t = params.threshold / full_ * 255.0f;
    return static_cast<u32>(clamp_(std::ceil(t), 1.0f, 255.0f));
}

// ============================================================================
//  Raymarching
// ============================================================================
namespace {

struct MarchCtx {
    const u8* dens;
    const u8* col;
    const u8* brick;
    int vx, vy, vz, bx, by;
    usize sy, sz;
    float hx, hy, hz;          // v-1-ε (sujeción trilineal)
    const float* alpha_lut;    // Cloud
    const float* rgb_lut;      // 4 floats por entrada
    float dt;                  // paso en t (unidades de mundo)
    float iso;                 // isovalor en unidades de densidad (0..255)
    u32 iso_i;                 // ceil(iso): ladrillo/esquinas por debajo → vacío
    Vec3 light_v;              // dirección de la luz (1 vóxel)
    bool shading, surface;
    float surf_alpha;
};

// Densidad trilineal en p (vóxeles). false si las 8 esquinas están por debajo de iso_i (SWAR).
CFD_INLINE bool dens_at(const MarchCtx& m, float px, float py, float pz, float& out) {
    const float fx = clamp_(px, 0.0f, m.hx), fy = clamp_(py, 0.0f, m.hy), fz = clamp_(pz, 0.0f, m.hz);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), z0 = static_cast<int>(fz);
    const u8* b = m.dens + static_cast<usize>(x0) + m.sy * static_cast<usize>(y0) + m.sz * static_cast<usize>(z0);
    u16 a00, a10, a01, a11;                    // pares (x0, x0+1) en 16 bits
    std::memcpy(&a00, b, 2); std::memcpy(&a10, b + m.sy, 2);
    std::memcpy(&a01, b + m.sz, 2); std::memcpy(&a11, b + m.sy + m.sz, 2);
    const u64 all = static_cast<u64>(a00) | (static_cast<u64>(a10) << 16) | (static_cast<u64>(a01) << 32) | (static_cast<u64>(a11) << 48);
    if (m.iso_i <= 128 ? !any_byte_gt(all, m.iso_i - 1) : all == 0) { out = 0.0f; return false; }
    const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
    auto lx = [&](u16 v) { const float l = static_cast<float>(v & 0xFF), h = static_cast<float>(v >> 8); return l + (h - l) * tx; };
    const float l00 = lx(a00), l10 = lx(a10), l01 = lx(a01), l11 = lx(a11);
    const float c0 = l00 + (l10 - l00) * ty;
    const float c1 = l01 + (l11 - l01) * ty;
    out = c0 + (c1 - c0) * tz;
    return true;
}
// Sin la prueba de vacío (para gradientes en el cruce).
CFD_INLINE float dens_raw(const MarchCtx& m, float px, float py, float pz) {
    const float fx = clamp_(px, 0.0f, m.hx), fy = clamp_(py, 0.0f, m.hy), fz = clamp_(pz, 0.0f, m.hz);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), z0 = static_cast<int>(fz);
    const u8* b = m.dens + static_cast<usize>(x0) + m.sy * static_cast<usize>(y0) + m.sz * static_cast<usize>(z0);
    const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
    auto lx = [&](const u8* q) { return static_cast<float>(q[0]) + (static_cast<float>(q[1]) - static_cast<float>(q[0])) * tx; };
    const float c0 = lx(b) + (lx(b + m.sy) - lx(b)) * ty;
    const float c1 = lx(b + m.sz) + (lx(b + m.sy + m.sz) - lx(b + m.sz)) * ty;
    return c0 + (c1 - c0) * tz;
}
CFD_INLINE float col_at(const MarchCtx& m, float px, float py, float pz) {
    const float fx = clamp_(px, 0.0f, m.hx), fy = clamp_(py, 0.0f, m.hy), fz = clamp_(pz, 0.0f, m.hz);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), z0 = static_cast<int>(fz);
    const u8* b = m.col + static_cast<usize>(x0) + m.sy * static_cast<usize>(y0) + m.sz * static_cast<usize>(z0);
    const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
    auto lx = [&](const u8* q) { return static_cast<float>(q[0]) + (static_cast<float>(q[1]) - static_cast<float>(q[0])) * tx; };
    const float c0 = lx(b) + (lx(b + m.sy) - lx(b)) * ty;
    const float c1 = lx(b + m.sz) + (lx(b + m.sy + m.sz) - lx(b + m.sz)) * ty;
    return c0 + (c1 - c0) * tz;
}

struct RayOut {
    float r = 0, g = 0, b = 0, a = 0;
    float tfront = k_inf;
};

// Cruces del rayo con planos translúcidos: a partir de tc[i] los aportes pesan ×kf[i].
struct Crossings {
    float tc[5];
    float kf[5];
    int n = 0;
};

CFD_INLINE RayOut march(const MarchCtx& m, Vec3 o, Vec3 d, float t0, float t1, float jitter, const Crossings& X) {
    RayOut R;
    int xi = 0;
    float kplane = 1.0f;                        // transmisión acumulada de los planos ya cruzados
    // Inversas de la dirección para el salto de ladrillos (slabs). Componente nula → ese eje
    // nunca limita la salida.
    const bool fx0 = std::fabs(d.x) < 1e-12f, fy0 = std::fabs(d.y) < 1e-12f, fz0 = std::fabs(d.z) < 1e-12f;
    const float ix = fx0 ? 0.0f : 1.0f / d.x, iy = fy0 ? 0.0f : 1.0f / d.y, iz = fz0 ? 0.0f : 1.0f / d.z;
    constexpr float far = 1e30f;
    const float dn = 1.0f / length(d);          // |d| = 1/ds en vóxeles → dirección unitaria para la vista
    const Vec3 view{-d.x * dn, -d.y * dn, -d.z * dn};
    float t = t0 + m.dt * jitter;
    float r = 0, g = 0, bl = 0, A = 0;
    float prev = 0.0f;                          // densidad de la muestra anterior (Surface)
    while (t < t1) {
        const float px = o.x + d.x * t, py = o.y + d.y * t, pz = o.z + d.z * t;
        const int bxi = clamp_(static_cast<int>(px) >> 3, 0, m.bx - 1);
        const int byi = clamp_(static_cast<int>(py) >> 3, 0, m.by - 1);
        const int bzi = max_(static_cast<int>(pz) >> 3, 0);
        const usize bi = static_cast<usize>(bxi) + static_cast<usize>(m.bx) * (static_cast<usize>(byi) + static_cast<usize>(m.by) * static_cast<usize>(bzi));
#ifndef FLOWVIS_VOL_BRICKS
#define FLOWVIS_VOL_BRICKS 1   // salto de ladrillos vacíos (0 = desactivado, sólo para medir)
#endif
        if (FLOWVIS_VOL_BRICKS && m.brick[bi] < m.iso_i) {
            // Salto al borde de salida del ladrillo vacío.
            const float ex = fx0 ? far : (static_cast<float>((d.x > 0 ? bxi + 1 : bxi) * k_brick) - o.x) * ix;
            const float ey = fy0 ? far : (static_cast<float>((d.y > 0 ? byi + 1 : byi) * k_brick) - o.y) * iy;
            const float ez = fz0 ? far : (static_cast<float>((d.z > 0 ? bzi + 1 : bzi) * k_brick) - o.z) * iz;
            const float te = min_(ex, min_(ey, ez));
            // Reanuda en la rejilla de pasos (mismo patrón de ruido → sin costuras). Progreso ≥ 1 paso.
            const float tn = t0 + m.dt * (std::floor((te - t0) / m.dt - jitter) + 1.0f + jitter);
            t = tn > t ? tn : t + m.dt;
            prev = 0.0f;
            continue;
        }
        while (xi < X.n && t >= X.tc[xi]) kplane *= X.kf[xi++];
        float dv;
        const bool hit = dens_at(m, px, py, pz, dv);
        if (m.surface) {
            if (hit && dv >= m.iso && prev < m.iso) {
                // Cruce de entrada: refina por interpolación lineal entre la muestra anterior y ésta.
                const float f = (m.iso - prev) / max_(dv - prev, 1e-6f);
                const float th = t - m.dt * (1.0f - f);
                const float hx = o.x + d.x * th, hy = o.y + d.y * th, hz = o.z + d.z * th;
                Vec3 c;
                {
                    const float* lc = m.rgb_lut + 4 * static_cast<usize>(clamp_(static_cast<int>(col_at(m, hx, hy, hz) + 0.5f), 0, 255));
                    c = {lc[0], lc[1], lc[2]};
                }
                float shade = 1.0f;
                if (m.shading) {
                    // Normal = -∇densidad (diferencias centradas: 6 muestras sólo en el cruce).
                    Vec3 gd{dens_raw(m, hx + 1, hy, hz) - dens_raw(m, hx - 1, hy, hz), dens_raw(m, hx, hy + 1, hz) - dens_raw(m, hx, hy - 1, hz),
                            dens_raw(m, hx, hy, hz + 1) - dens_raw(m, hx, hy, hz - 1)};
                    const Vec3 n = normalize(-gd);
                    const float ndl = dot(n, m.light_v), ndv = dot(n, view);
                    const float diff = std::fabs(ndl);                       // dos caras
                    const Vec3 hv = normalize(m.light_v + view);
                    const float sp = std::pow(std::fabs(dot(n, hv)), 40.0f);
                    const float rim = 1.0f - std::fabs(ndv);
                    shade = 0.30f + 0.70f * diff + 0.25f * rim * rim;
                    c = c * shade + Vec3(0.35f * sp);
                }
                const float w = (1.0f - A) * m.surf_alpha * kplane;
                r += w * c.x; g += w * c.y; bl += w * c.z;
                A += w;
                if (R.tfront == k_inf) R.tfront = th;
                if (A > 0.985f) break;
            }
            prev = hit ? dv : 0.0f;
        } else if (hit) {
            const float a = m.alpha_lut[static_cast<int>(dv + 0.5f)];
            if (a > 0.002f) {
                const usize ci = static_cast<usize>(clamp_(static_cast<int>(px + 0.5f), 0, m.vx - 1)) +
                                 m.sy * static_cast<usize>(clamp_(static_cast<int>(py + 0.5f), 0, m.vy - 1)) +
                                 m.sz * static_cast<usize>(clamp_(static_cast<int>(pz + 0.5f), 0, m.vz - 1));
                const float* c = m.rgb_lut + 4 * static_cast<usize>(m.col[ci]);
                float shade = 1.0f;
                if (m.shading) {
                    // Derivada direccional hacia la luz: si la densidad baja hacia la luz, la "superficie" la mira.
                    const float d2 = dens_raw(m, px + m.light_v.x, py + m.light_v.y, pz + m.light_v.z);
                    shade = 0.62f + 0.38f * clamp_((dv - d2) * (1.0f / 40.0f), -1.0f, 1.0f);
                }
                const float w = (1.0f - A) * a * kplane;
                r += w * c[0] * shade; g += w * c[1] * shade; bl += w * c[2] * shade;
                A += w;
                if (A > 0.03f && R.tfront == k_inf) R.tfront = t;
                if (A > 0.985f) break;                      // terminación temprana
            }
        }
        t += m.dt;
    }
    R.r = r; R.g = g; R.b = bl; R.a = A;
    return R;
}

CFD_INLINE u32 pack_premul(const RayOut& R) {
    auto c = [](float v) { return static_cast<u32>(clamp_(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    const u32 a = c(R.a);
    u32 r = c(R.r), g = c(R.g), b = c(R.b);
    r = min_(r, a); g = min_(g, a); b = min_(b, a);    // premultiplicado válido
    return (a << 24) | (r << 16) | (g << 8) | b;
}

// dst = src + dst·(255-A)/255 (src premultiplicado). Suma saturada por canal (PADDUSB).
CFD_INLINE u32 over(u32 dst, u32 src) {
    const u32 ia = 255u - (src >> 24);
    const u32 rb = ((dst & 0x00FF00FFu) * ia + 0x00800080u) >> 8 & 0x00FF00FFu;
    const u32 g = ((dst & 0x0000FF00u) * ia + 0x00008000u) >> 8 & 0x0000FF00u;
    return simd::add_sat_argb(rb | g, src & 0x00FFFFFFu) | 0xFF000000u;
}

} // namespace

void VortexVolume::render(Framebuffer& fb, const Camera& cam, std::span<const TranslucentPlane> behind) {
    if (dens_.empty()) return;
    count_bricks();                                        // el umbral puede haber cambiado
    if (stats_.bricks_nonempty == 0) return;
    // Viewport ∩ framebuffer (antes sólo se recortaba por la derecha/abajo: un vp.x < 0
    // escribía fuera de la fila). Los rayos se generan con cam.ray → coherentes igualmente.
    Rect vp;
    vp.x = max_(cam.vp.x, 0); vp.y = max_(cam.vp.y, 0);
    vp.w = min_(cam.vp.x + cam.vp.w, fb.w) - vp.x; vp.h = min_(cam.vp.y + cam.vp.h, fb.h) - vp.y;
    if (vp.w <= 0 || vp.h <= 0) return;

    // --- LUTs de transferencia (256 entradas; 256 exp por cuadro, despreciable).
    const float ds = static_cast<float>(ds_);
    const float stepv = clamp_(params.step, 0.1f, 2.0f);
    const float dt_world = stepv * ds;
    const float iso = clamp_(params.threshold / full_ * 255.0f, 0.5f, 254.5f);
    const float sigma = -std::log(1.0f - clamp_(params.density, 0.0f, 0.999f));   // por celda a densidad máx.
    for (int i = 0; i < 256; ++i) {
        const float fi = static_cast<float>(i);
        // Cloud: opacidad nula bajo el umbral, rampa suave (8 niveles) y proporcional encima.
        const float dn = fi < iso ? 0.0f : smoothstep(iso, iso + 8.0f, fi) * (fi - iso) / (255.0f - iso + 1e-3f);
        alpha_lut_[i] = 1.0f - std::exp(-sigma * dn * dt_world);
        const u32 c = col_ux_ ? render::colormap_lut(col_map_)[i] : render::colormap_lut(Colormap::Inferno)[64 + (i * 191) / 255];
        const Vec3 rgb = render::unpack_rgb(c);
        rgb_lut_[4 * i + 0] = rgb.x; rgb_lut_[4 * i + 1] = rgb.y; rgb_lut_[4 * i + 2] = rgb.z; rgb_lut_[4 * i + 3] = 1.0f;
    }

    MarchCtx m;
    m.dens = dens_.data(); m.col = col_.data(); m.brick = brick_.data();
    m.vx = vx_; m.vy = vy_; m.vz = vz_; m.bx = bx_; m.by = by_;
    m.sy = static_cast<usize>(vx_); m.sz = static_cast<usize>(vx_) * static_cast<usize>(vy_);
    m.hx = static_cast<float>(vx_ - 1) - 1e-3f; m.hy = static_cast<float>(vy_ - 1) - 1e-3f; m.hz = static_cast<float>(vz_ - 1) - 1e-3f;
    m.alpha_lut = alpha_lut_; m.rgb_lut = rgb_lut_;
    m.dt = dt_world;
    m.iso = iso;
    m.iso_i = iso_index();
    m.light_v = normalize(params.light_dir);
    m.shading = params.shading;
    m.surface = params.style == VolumeStyle::Surface;
    m.surf_alpha = clamp_(params.surface_alpha, 0.02f, 1.0f);
    const float off = 0.5f * (ds - 1.0f);                  // centro del vóxel i en el mundo: ds·i + off
    const float inv_ds = 1.0f / ds;
    // Recorte de los rayos a la caja de los LADRILLOS OCUPADOS (no al dominio entero): con
    // vórtices dispersos casi todos los rayos se descartan con un test de slabs, sin recorrer ladrillos.
    const int vmaxi[3] = {vx_ - 1, vy_ - 1, vz_ - 1};
    Vec3 bmin, bmax;
    for (int k = 0; k < 3; ++k) {
        bmin[k] = static_cast<float>(max_(occ_lo_[k] * k_brick - 1, 0));
        bmax[k] = static_cast<float>(min_((occ_hi_[k] + 1) * k_brick, vmaxi[k]));
    }
    // Rectángulo de pantalla que cubre la caja (proyección de sus 8 esquinas). Si alguna
    // esquina queda detrás de la cámara se usa todo el viewport.
    Rect scr = vp;
    {
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        bool ok = true;
        for (int c = 0; c < 8 && ok; ++c) {
            const Vec3 pv{(c & 1) ? bmax.x : bmin.x, (c & 2) ? bmax.y : bmin.y, (c & 4) ? bmax.z : bmin.z};
            const Vec3 pw = pv * ds + Vec3(off);
            float sx, sy, dep;
            if (!cam.project(pw, sx, sy, dep)) { ok = false; break; }
            x0 = min_(x0, sx); y0 = min_(y0, sy); x1 = max_(x1, sx); y1 = max_(y1, sy);
        }
        if (ok) {
            const int ix0 = max_(static_cast<int>(std::floor(x0)) - 2, vp.x), iy0 = max_(static_cast<int>(std::floor(y0)) - 2, vp.y);
            const int ix1 = min_(static_cast<int>(std::ceil(x1)) + 2, vp.x + vp.w), iy1 = min_(static_cast<int>(std::ceil(y1)) + 2, vp.y + vp.h);
            if (ix1 <= ix0 || iy1 <= iy0) return;
            scr = {ix0, iy0, ix1 - ix0, iy1 - iy0};
        }
    }

    // Planos translúcidos activos (máx. 4).
    TranslucentPlane planes[4];
    int nplanes = 0;
    for (const TranslucentPlane& p : behind)
        if (p.opacity > 0.0f && nplanes < 4) planes[nplanes++] = p;

    // Rayo en coordenadas de vóxel (t conserva unidades de mundo) + recorte a la caja y a la escena.
    auto setup = [&](float sx, float sy, float scene_depth, Vec3& o, Vec3& d, float& t0, float& t1, float& cosf, Crossings& X) -> bool {
        Vec3 ow, dw;
        cam.ray(sx, sy, ow, dw);
        X.n = 0;
        for (int k = 0; k < nplanes; ++k) {   // cruces ordenados por t (inserción: ≤ 4)
            const TranslucentPlane& pl = planes[k];
            const int a = static_cast<int>(pl.axis), ua = a == 0 ? 1 : 0, va = a == 2 ? 1 : 2;
            if (std::fabs(dw[a]) < 1e-12f) continue;
            const float tp = (pl.pos - ow[a]) / dw[a];
            if (!(tp > 0.0f)) continue;
            const Vec3 xp = ow + dw * tp;
            if (xp[ua] < pl.lo.x || xp[ua] > pl.hi.x || xp[va] < pl.lo.y || xp[va] > pl.hi.y) continue;
            int j = X.n++;
            while (j > 0 && X.tc[j - 1] > tp) { X.tc[j] = X.tc[j - 1]; X.kf[j] = X.kf[j - 1]; --j; }
            X.tc[j] = tp; X.kf[j] = 1.0f - pl.opacity;
        }
        cosf = cam.ortho ? 1.0f : dot(dw, cam.fwd);
        o = (ow - Vec3(off)) * inv_ds;
        d = dw * inv_ds;
        float a0 = 0.0f, a1 = 1e30f;
        for (int k = 0; k < 3; ++k) {
            const float dk = d[k], ok = o[k];
            if (std::fabs(dk) < 1e-12f) { if (ok < bmin[k] || ok > bmax[k]) return false; continue; }
            float ta = (bmin[k] - ok) / dk, tb = (bmax[k] - ok) / dk;
            if (ta > tb) { const float tmp = ta; ta = tb; tb = tmp; }
            a0 = max_(a0, ta); a1 = min_(a1, tb);
        }
        const float ts = scene_depth / max_(cosf, 1e-6f);  // profundidad de vista → t a lo largo del rayo
        a1 = min_(a1, ts);
        t0 = a0; t1 = a1;
        return t0 < t1;
    };

    if (!params.half_res) {
        parallel_for(scr.y, scr.y + scr.h, 2, [&](i64 lo, i64 hi) {
            for (i64 yy = lo; yy < hi; ++yy) {
                const int y = static_cast<int>(yy);
                u32* crow = fb.row(y);
                const float* drow = fb.drow(y);
                for (int x = scr.x; x < scr.x + scr.w; ++x) {
                    Vec3 o, d;
                    float t0, t1, cf;
                    Crossings X;
                    if (!setup(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, drow[x], o, d, t0, t1, cf, X)) continue;
                    const RayOut R = march(m, o, d, t0, t1, ign(x, y), X);
                    if (R.a <= 0.0f) continue;
                    crow[x] = over(crow[x], pack_premul(R));
                }
            }
        });
        return;
    }

    // --- Media resolución: 1 rayo por bloque 2×2 (centro del bloque); profundidad = máxima del bloque.
    const int rw = (vp.w + 1) / 2, rh = (vp.h + 1) / 2;
    const usize nlow = static_cast<usize>(rw) * static_cast<usize>(rh);
    if (low_.size() < nlow) { low_.resize(nlow); low_depth_.resize(nlow); }
    u32* CFD_RESTRICT low = low_.data();
    float* CFD_RESTRICT lowd = low_depth_.data();
    // Subrectángulo de media resolución a trazar (+1 de margen: el reescalado lee vecinos).
    const int i0 = max_((scr.x - vp.x) / 2 - 1, 0), i1 = min_((scr.x + scr.w - vp.x + 1) / 2 + 1, rw);
    const int j0 = max_((scr.y - vp.y) / 2 - 1, 0), j1 = min_((scr.y + scr.h - vp.y + 1) / 2 + 1, rh);
    parallel_for(j0, j1, 1, [&](i64 lo, i64 hi) {
        for (i64 jj = lo; jj < hi; ++jj) {
            const int j = static_cast<int>(jj);
            const int y0 = vp.y + 2 * j, y1 = min_(y0 + 1, vp.y + vp.h - 1);
            const float* d0 = fb.drow(y0);
            const float* d1 = fb.drow(y1);
            for (int i = i0; i < i1; ++i) {
                const int x0 = vp.x + 2 * i, x1 = min_(x0 + 1, vp.x + vp.w - 1);
                const float sd = max_(max_(d0[x0], d0[x1]), max_(d1[x0], d1[x1]));
                const usize li = static_cast<usize>(j) * static_cast<usize>(rw) + static_cast<usize>(i);
                Vec3 o, d;
                float t0, t1, cf;
                Crossings X;
                if (!setup(static_cast<float>(x0) + 1.0f, static_cast<float>(y0) + 1.0f, sd, o, d, t0, t1, cf, X)) {
                    low[li] = 0; lowd[li] = k_inf;
                    continue;
                }
                const RayOut R = march(m, o, d, t0, t1, ign(i, j), X);
                low[li] = R.a > 0.0f ? pack_premul(R) : 0u;
                lowd[li] = R.tfront * cf;   // profundidad de vista del primer aporte
            }
        }
    });

    // --- Reescalado 2× bilineal (pesos 3:1 fijos) + composición, con SWAR en u64
    //     (4 canales × 16 bits): expandir con PDEP, 3a+b dos veces, >>4, comprimir con PEXT.
    const int nthr = max_(pool().size(), 1);
    const usize vpad = static_cast<usize>(rw) + 2;
    if (vpad_.size() < vpad * 4 * static_cast<usize>(nthr)) vpad_.resize(vpad * 4 * static_cast<usize>(nthr));
    // Filas/columnas de resolución completa cubiertas por el subrectángulo trazado. Fuera de él
    // la media resolución vale 0 (el margen trazado ya es 0) → relleno con ceros; en el borde del
    // viewport se replica el último valor.
    const int ry0 = 2 * j0, ry1 = min_(2 * j1, vp.h);
    const int rx0 = 2 * i0, rx1 = min_(2 * i1, vp.w);
    parallel_for(vp.y + ry0, vp.y + ry1, 4, [&](i64 lo, i64 hi) {
        const int wi = max_(ThreadPool::worker_index(), 0);
        u64* CFD_RESTRICT V = reinterpret_cast<u64*>(vpad_.data() + vpad * 4 * static_cast<usize>(wi)) + 1;   // V[-1], V[rw] válidos
        for (i64 yy = lo; yy < hi; ++yy) {
            const int y = static_cast<int>(yy);
            const int ry = y - vp.y, j = ry >> 1;
            const int jm = clamp_(j, j0, j1 - 1);
            int jo = (ry & 1) ? j + 1 : j - 1;                             // fila vecina (peso 1/4)
            if (jo < 0 || jo >= rh) jo = jm;                               // borde del viewport: replica
            const bool lo_zero = jo < j0 || jo >= j1;                      // fuera de lo trazado: 0
            const u32* Lm = low + static_cast<usize>(jm) * static_cast<usize>(rw);
            const u32* Lo = low + static_cast<usize>(lo_zero ? jm : jo) * static_cast<usize>(rw);
            bool any = false;
            for (int i = i0; i < i1; ++i) {
                const u32 a = Lm[i], b = lo_zero ? 0u : Lo[i];
                any |= (a | b) != 0;
                V[i] = 3 * _pdep_u64(a, k_lanes) + _pdep_u64(b, k_lanes);   // ≤ 1020 por carril
            }
            if (!any) continue;
            V[i0 - 1] = i0 == 0 ? V[0] : 0; V[i1] = i1 == rw ? V[rw - 1] : 0;
            u32* crow = fb.row(y);
            const float* drow = fb.drow(y);
            const float* LD = lowd + static_cast<usize>(jm) * static_cast<usize>(rw);
            for (int rx = rx0; rx < rx1; ++rx) {
                const int i = rx >> 1;
                const u64 vc = V[i], vn = V[(rx & 1) ? i + 1 : i - 1];
                if ((vc | vn) == 0) continue;
                const int x = vp.x + rx;
                if (drow[x] < LD[i]) continue;                               // la escena tapa al vórtice
                const u64 s = ((3 * vc + vn) >> 4) & k_lanes;                // ≤ 4080 >> 4 = 255
                const u32 src = static_cast<u32>(_pext_u64(s, k_lanes));
                if (src == 0) continue;
                crow[x] = over(crow[x], src);
            }
        }
    });
}

} // namespace cfd::flowvis
