// ============================================================================
//  render/flowvis.cpp — magnitudes derivadas (núcleos AVX2 por filas), copia
//  empaquetada FP16 del campo, planos de corte, huella en el suelo, estela,
//  colores de malla y sondas.  Ver render/flowvis.hpp para el contrato.
// ============================================================================
#include "flowvis.hpp"
#include "../core/util.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cfd::flowvis {

using lbm::FieldView;
using simd::f8;

namespace {

constexpr float k_nan = std::numeric_limits<float>::quiet_NaN();

// Constantes de normalización (una división por fila, no por celda).
struct Norm {
    float inv_u, inv_u2, cp_k;
    explicit Norm(float u) : inv_u(1.0f / u), inv_u2(1.0f / (u * u)), cp_k(2.0f / (3.0f * u * u)) {}
};

constexpr bool is_grad(Quantity q) { return q == Quantity::Vorticity || q == Quantity::QCriterion; }

// Mayor float estrictamente < n-1 (n ≥ 2): sujeción trilineal que garantiza x0 ≤ n-2 para
// CUALQUIER n (decremento de 1 ulp en los bits). Antes n-1-1e-4 redondeaba a n-1 cuando
// n-1 ≥ 2048 (espaciado 2.4e-4) → la carga de la celda x0+1 leía fuera del búfer (ASan).
CFD_INLINE float below_last(int n) { return std::bit_cast<float>(std::bit_cast<u32>(static_cast<float>(n - 1)) - 1u); }

// ---------------------------------------------------------------------------
//  Velocidad en una celda. En celdas sólidas el solver escribe la velocidad de
//  PARED en cada paso macro (0 si es fija; cinta del suelo o ruedas si kMoving):
//  es justo el valor de no deslizamiento que necesitan los gradientes, así que se
//  usa tal cual (con la cinta móvil no aparece una falsa capa de cortadura en el suelo).
// ---------------------------------------------------------------------------
CFD_INLINE Vec3 vel_at(const FieldView& f, usize n) { return {f.ux[n], f.uy[n], f.uz[n]}; }

// g[i][j] = ∂u_i/∂x_j por diferencias centradas (unilaterales en el borde del dominio).
CFD_INLINE void grad_scalar(const FieldView& f, int x, int y, int z, float g[3][3]) {
    const int xm = x > 0 ? x - 1 : x, xp = x < f.nx - 1 ? x + 1 : x;
    const int ym = y > 0 ? y - 1 : y, yp = y < f.ny - 1 ? y + 1 : y;
    const int zm = z > 0 ? z - 1 : z, zp = z < f.nz - 1 ? z + 1 : z;
    const float ix = 1.0f / static_cast<float>(max_(xp - xm, 1));
    const float iy = 1.0f / static_cast<float>(max_(yp - ym, 1));
    const float iz = 1.0f / static_cast<float>(max_(zp - zm, 1));
    const Vec3 dx = (vel_at(f, f.index(xp, y, z)) - vel_at(f, f.index(xm, y, z))) * ix;
    const Vec3 dy = (vel_at(f, f.index(x, yp, z)) - vel_at(f, f.index(x, ym, z))) * iy;
    const Vec3 dz = (vel_at(f, f.index(x, y, zp)) - vel_at(f, f.index(x, y, zm))) * iz;
    g[0][0] = dx.x; g[1][0] = dx.y; g[2][0] = dx.z;
    g[0][1] = dy.x; g[1][1] = dy.y; g[2][1] = dy.z;
    g[0][2] = dz.x; g[1][2] = dz.y; g[2][2] = dz.z;
}

// Q = ½(|Ω|²-|S|²) = -½ Σ_ij g_ij g_ji  (identidad: evita construir S y Ω).
CFD_INLINE float q_from_grad(const float g[3][3]) {
    return -0.5f * (g[0][0] * g[0][0] + g[1][1] * g[1][1] + g[2][2] * g[2][2] +
                    2.0f * (g[0][1] * g[1][0] + g[0][2] * g[2][0] + g[1][2] * g[2][1]));
}
CFD_INLINE float vort_from_grad(const float g[3][3]) {
    const float wx = g[2][1] - g[1][2], wy = g[0][2] - g[2][0], wz = g[1][0] - g[0][1];
    return std::sqrt(wx * wx + wy * wy + wz * wz);
}

template <Quantity Q>
CFD_INLINE float cell_value(const FieldView& f, const Norm& k, int x, int y, int z) {
    const usize n = f.index(x, y, z);
    if (f.flags[n] & lbm::kSolid) return k_nan;
    if constexpr (is_grad(Q)) {
        float g[3][3];
        grad_scalar(f, x, y, z, g);
        if constexpr (Q == Quantity::Vorticity) return vort_from_grad(g) * k.inv_u;
        else return q_from_grad(g) * k.inv_u2;
    } else {
        const float ux = f.ux[n], uy = f.uy[n], uz = f.uz[n];
        if constexpr (Q == Quantity::Speed) return std::sqrt(ux * ux + uy * uy + uz * uz) * k.inv_u;
        else if constexpr (Q == Quantity::Ux) return ux * k.inv_u;
        else if constexpr (Q == Quantity::Uz) return uz * k.inv_u;
        else if constexpr (Q == Quantity::Cp) return (f.rho[n] - 1.0f) * k.cp_k;
        else {   // Cp0
            const float r = f.rho[n];
            return (r - 1.0f) * k.cp_k + r * (ux * ux + uy * uy + uz * uz) * k.inv_u2;
        }
    }
}

// ---------------------------------------------------------------------------
//  AVX2: 8 celdas consecutivas en X a partir del índice lineal n.
//  Para magnitudes con gradiente se exige 1 ≤ x0 y x0+8 ≤ nx-1 (vecinos x±1).
// ---------------------------------------------------------------------------
struct RowGeom {
    usize oyp, oym, ozp, ozm;   // desplazamientos a vecinos y±1 / z±1 (0 en el borde → unilateral)
    float iy, iz;               // 1 / distancia de la diferencia
};

CFD_INLINE void load_vel(const FieldView& f, usize m, f8& a, f8& b, f8& c) {
    a = _mm256_loadu_ps(f.ux + m);
    b = _mm256_loadu_ps(f.uy + m);
    c = _mm256_loadu_ps(f.uz + m);
}

template <Quantity Q>
CFD_INLINE f8 vec_value(const FieldView& f, const Norm& k, usize n, const RowGeom& gm) {
    const f8 solid_c = simd::mask_bits_u8(f.flags + n, lbm::kSolid);
    f8 r;
    if constexpr (is_grad(Q)) {
        f8 pa, pb, pc, ma, mb, mc;
        const f8 half(0.5f);
        load_vel(f, n + 1, pa, pb, pc);
        load_vel(f, n - 1, ma, mb, mc);
        const f8 g00 = (pa - ma) * half, g10 = (pb - mb) * half, g20 = (pc - mc) * half;
        const f8 iy(gm.iy), iz(gm.iz);
        load_vel(f, n + gm.oyp, pa, pb, pc);
        load_vel(f, n - gm.oym, ma, mb, mc);
        const f8 g01 = (pa - ma) * iy, g11 = (pb - mb) * iy, g21 = (pc - mc) * iy;
        load_vel(f, n + gm.ozp, pa, pb, pc);
        load_vel(f, n - gm.ozm, ma, mb, mc);
        const f8 g02 = (pa - ma) * iz, g12 = (pb - mb) * iz, g22 = (pc - mc) * iz;
        if constexpr (Q == Quantity::Vorticity) {
            const f8 wx = g21 - g12, wy = g02 - g20, wz = g10 - g01;
            r = simd::sqrt(simd::fmadd(wx, wx, simd::fmadd(wy, wy, wz * wz))) * f8(k.inv_u);
        } else {
            f8 s = simd::fmadd(g00, g00, simd::fmadd(g11, g11, g22 * g22));
            const f8 c = simd::fmadd(g01, g10, simd::fmadd(g02, g20, g12 * g21));
            s = simd::fmadd(c, f8(2.0f), s);
            r = s * f8(-0.5f * k.inv_u2);
        }
    } else {
        const f8 ux = f8::load(f.ux + n), uy = f8::load(f.uy + n), uz = f8::load(f.uz + n);
        if constexpr (Q == Quantity::Speed) r = simd::sqrt(simd::fmadd(ux, ux, simd::fmadd(uy, uy, uz * uz))) * f8(k.inv_u);
        else if constexpr (Q == Quantity::Ux) r = ux * f8(k.inv_u);
        else if constexpr (Q == Quantity::Uz) r = uz * f8(k.inv_u);
        else if constexpr (Q == Quantity::Cp) r = (f8::load(f.rho + n) - f8(1.0f)) * f8(k.cp_k);
        else {
            const f8 rho = f8::load(f.rho + n);
            const f8 u2 = simd::fmadd(ux, ux, simd::fmadd(uy, uy, uz * uz));
            r = simd::fmadd(rho * u2, f8(k.inv_u2), (rho - f8(1.0f)) * f8(k.cp_k));
        }
    }
    return simd::select(solid_c, f8(k_nan), r);
}

template <Quantity Q>
void row_impl(const FieldView& f, int y, int z, float* CFD_RESTRICT out) {
    const Norm k(f.u_inf);
    const int nx = f.nx;
    const usize n0 = f.index(0, y, z);
    if constexpr (!is_grad(Q)) {
        const RowGeom gm{};
        int x = 0;
        for (; x + 8 <= nx; x += 8) vec_value<Q>(f, k, n0 + static_cast<usize>(x), gm).store(out + x);
        for (; x < nx; ++x) out[x] = cell_value<Q>(f, k, x, y, z);
    } else {
        RowGeom gm;
        const usize sy = static_cast<usize>(nx), sz = static_cast<usize>(nx) * static_cast<usize>(f.ny);
        gm.oyp = y < f.ny - 1 ? sy : 0; gm.oym = y > 0 ? sy : 0;
        gm.ozp = z < f.nz - 1 ? sz : 0; gm.ozm = z > 0 ? sz : 0;
        gm.iy = 1.0f / static_cast<float>(max_((gm.oyp + gm.oym) / sy, usize{1}));
        gm.iz = 1.0f / static_cast<float>(max_((gm.ozp + gm.ozm) / sz, usize{1}));
        const int head = min_(8, nx);
        int x = 0;
        for (; x < head; ++x) out[x] = cell_value<Q>(f, k, x, y, z);          // x = 0 no tiene vecino x-1
        for (; x + 9 <= nx; x += 8) vec_value<Q>(f, k, n0 + static_cast<usize>(x), gm).store(out + x);
        for (; x < nx; ++x) out[x] = cell_value<Q>(f, k, x, y, z);            // cola (vecino x+1 fuera)
    }
}

using RowFn = void (*)(const FieldView&, int, int, float* CFD_RESTRICT);
using CellFn = float (*)(const FieldView&, const Norm&, int, int, int);
// Despacho único por llamada (desconmutación de bucles vía plantillas: el switch
// sobre la magnitud queda FUERA de los bucles por celda).
constexpr RowFn k_row_fn[] = {row_impl<Quantity::Speed>, row_impl<Quantity::Ux>, row_impl<Quantity::Uz>,
                              row_impl<Quantity::Cp>, row_impl<Quantity::Cp0>, row_impl<Quantity::Vorticity>,
                              row_impl<Quantity::QCriterion>};
constexpr CellFn k_cell_fn[] = {cell_value<Quantity::Speed>, cell_value<Quantity::Ux>, cell_value<Quantity::Uz>,
                                cell_value<Quantity::Cp>, cell_value<Quantity::Cp0>, cell_value<Quantity::Vorticity>,
                                cell_value<Quantity::QCriterion>};

// ---------------------------------------------------------------------------
//  Trilineal ponderada por fluido: pesos de las 8 esquinas × (esquina fluida).
// ---------------------------------------------------------------------------
struct Corners {
    usize o[8];
    float w[8];
    int x0, y0, z0;
    float wsum;
};
CFD_INLINE void fluid_corners(const FieldView& f, Vec3 p, Corners& c) {
    const float fx = clamp_(p.x, 0.0f, below_last(f.nx));
    const float fy = clamp_(p.y, 0.0f, below_last(f.ny));
    const float fz = clamp_(p.z, 0.0f, below_last(f.nz));
    c.x0 = static_cast<int>(fx); c.y0 = static_cast<int>(fy); c.z0 = static_cast<int>(fz);
    const float tx = fx - static_cast<float>(c.x0), ty = fy - static_cast<float>(c.y0), tz = fz - static_cast<float>(c.z0);
    const usize n0 = f.index(c.x0, c.y0, c.z0);
    const usize sy = static_cast<usize>(f.nx), sz = sy * static_cast<usize>(f.ny);
    const usize o[8] = {n0, n0 + 1, n0 + sy, n0 + sy + 1, n0 + sz, n0 + sz + 1, n0 + sz + sy, n0 + sz + sy + 1};
    const float w[8] = {(1 - tx) * (1 - ty) * (1 - tz), tx * (1 - ty) * (1 - tz), (1 - tx) * ty * (1 - tz), tx * ty * (1 - tz),
                        (1 - tx) * (1 - ty) * tz,       tx * (1 - ty) * tz,       (1 - tx) * ty * tz,       tx * ty * tz};
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) {
        c.o[i] = o[i];
        c.w[i] = (f.flags[o[i]] & lbm::kSolid) ? 0.0f : w[i];
        s += c.w[i];
    }
    c.wsum = s;
}

// Primitivas (u, ρ) interpoladas con pesos de fluido. false si no hay fluido.
CFD_INLINE bool sample_prims(const FieldView& f, Vec3 p, Vec3& u, float& rho) {
    Corners c;
    fluid_corners(f, p, c);
    if (c.wsum < 1e-6f) return false;
    Vec3 a{0, 0, 0};
    float r = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const usize o = c.o[i];
        a.x += c.w[i] * f.ux[o]; a.y += c.w[i] * f.uy[o]; a.z += c.w[i] * f.uz[o];
        r += c.w[i] * f.rho[o];
    }
    const float inv = 1.0f / c.wsum;
    u = a * inv; rho = r * inv;
    return true;
}

CFD_INLINE float quantity_from_prims(Quantity q, const Norm& k, Vec3 u, float rho) {
    switch (q) {
        case Quantity::Speed: return length(u) * k.inv_u;
        case Quantity::Ux: return u.x * k.inv_u;
        case Quantity::Uz: return u.z * k.inv_u;
        case Quantity::Cp: return (rho - 1.0f) * k.cp_k;
        case Quantity::Cp0: return (rho - 1.0f) * k.cp_k + rho * length2(u) * k.inv_u2;
        default: return k_nan;
    }
}

// Paleta por id de grupo si la app no da colores: tono dorado (golden ratio) por id.
CFD_INLINE u32 hash_color(u32 g) {
    if (g == 0) return 0xFFB8BCC4u;
    const float h = std::fmod(static_cast<float>(g) * 0.61803398875f, 1.0f) * 6.0f;
    const int i = static_cast<int>(h);
    const float fr = h - static_cast<float>(i);
    const float s = 0.55f, v = 0.85f;
    const float p = v * (1 - s), q = v * (1 - s * fr), t = v * (1 - s * (1 - fr));
    float r = v, gg = t, b = p;
    switch (i % 6) {
        case 0: r = v; gg = t; b = p; break;
        case 1: r = q; gg = v; b = p; break;
        case 2: r = p; gg = v; b = t; break;
        case 3: r = p; gg = q; b = v; break;
        case 4: r = t; gg = p; b = v; break;
        default: r = v; gg = p; b = q; break;
    }
    return render::rgbf(r, gg, b);
}

} // namespace

// ============================================================================
//  Tabla de magnitudes
// ============================================================================
const QuantityInfo& quantity_info(Quantity q) {
    static const QuantityInfo k_info[] = {
        {"Velocidad |u|/U∞", "|u|/U∞", "Módulo de la velocidad relativo a la corriente libre.",
         {Colormap::Turbo, 0.0f, 1.6f, false}},
        {"Velocidad axial u_x/U∞", "u_x/U∞", "Componente en la dirección del flujo; negativa = recirculación.",
         {Colormap::CoolWarm, -0.6f, 1.4f, true}},
        {"Velocidad vertical u_z/U∞", "u_z/U∞", "Componente vertical: upwash (+) / downwash (-).",
         {Colormap::CoolWarm, -0.5f, 0.5f, true}},
        {"Presión estática Cp", "Cp", "Coeficiente de presión: rojo = sobrepresión, azul = succión.",
         {Colormap::CoolWarm, -2.5f, 1.0f, true}},
        {"Presión total Cp0", "Cp0", "Presión total: 1 = sin pérdidas; la estela (pérdidas) aparece en azul.",
         {Colormap::Turbo, 0.0f, 1.0f, false}},
        {"Vorticidad |ω|·Δx/U∞", "|ω|", "Intensidad de rotación local (capas límite, vórtices, estela).",
         {Colormap::Inferno, 0.0f, 0.3f, false}},
        {"Criterio Q·Δx²/U∞²", "Q", "Q > 0: domina la rotación (núcleo de vórtice); Q < 0: domina la deformación.",
         {Colormap::CoolWarm, -0.01f, 0.01f, true}},
    };
    const int i = static_cast<int>(q);
    return k_info[i >= 0 && i < static_cast<int>(Quantity::Count) ? i : 0];
}

const char* surface_mode_name(SurfaceMode m) {
    switch (m) {
        case SurfaceMode::Cp: return "Presión Cp";
        case SurfaceMode::Speed: return "Velocidad junto a la pared";
        case SurfaceMode::Component: return "Por componente";
        case SurfaceMode::Solid: return "Color sólido";
        default: return "?";
    }
}

float cell_quantity(const FieldView& f, Quantity q, int x, int y, int z) {
    const Norm k(f.u_inf);
    return k_cell_fn[static_cast<int>(q)](f, k, x, y, z);
}

float sample_quantity(const FieldView& f, Quantity q, Vec3 p) {
    const Norm k(f.u_inf);
    if (!is_grad(q)) {
        Vec3 u;
        float rho;
        if (!sample_prims(f, p, u, rho)) return k_nan;
        return quantity_from_prims(q, k, u, rho);
    }
    Corners c;
    fluid_corners(f, p, c);
    if (c.wsum < 1e-6f) return k_nan;
    const CellFn fn = k_cell_fn[static_cast<int>(q)];
    float acc = 0.0f, ws = 0.0f;
    for (int i = 0; i < 8; ++i) {
        if (c.w[i] <= 0.0f) continue;
        const float v = fn(f, k, c.x0 + (i & 1), c.y0 + ((i >> 1) & 1), c.z0 + ((i >> 2) & 1));
        acc += c.w[i] * v; ws += c.w[i];
    }
    return acc / ws;
}

Probe probe(const FieldView& f, Vec3 p) {
    Probe r;
    r.pos = p;
    if (!f.valid() || !f.inside(p)) return r;
    r.valid = true;
    const int xi = static_cast<int>(p.x + 0.5f), yi = static_cast<int>(p.y + 0.5f), zi = static_cast<int>(p.z + 0.5f);
    r.solid = f.solid(min_(xi, f.nx - 1), min_(yi, f.ny - 1), min_(zi, f.nz - 1));
    const Norm k(f.u_inf);
    Vec3 u;
    float rho;
    if (!sample_prims(f, p, u, rho)) { r.solid = true; return r; }
    r.u = u * k.inv_u;
    r.speed = length(u) * k.inv_u;
    r.cp = (rho - 1.0f) * k.cp_k;
    r.cp0 = r.cp + rho * length2(u) * k.inv_u2;
    r.vorticity = sample_quantity(f, Quantity::Vorticity, p);
    r.q = sample_quantity(f, Quantity::QCriterion, p);
    return r;
}

namespace {

// ---------------------------------------------------------------------------
//  Rangos: sólo cuentan los valores FINITOS. NaN = sólido; ±inf aparece si el solver
//  diverge (FP16 satura a inf). Antes un único inf hacía (x-vmin)·0 = NaN → (int)NaN
//  como índice del histograma → escritura fuera de la pila (fallo reproducido en
//  tests/test_flowvis_review.cpp [R1]/[R9]).
// ---------------------------------------------------------------------------
// Mín/máx de los finitos con AVX2: |x| ≤ FLT_MAX es falso para NaN y ±inf → esos carriles
// se sustituyen por el neutro (+inf para el mín, -inf para el máx) con un blend, sin ramas.
bool minmax_finite(const float* v, usize n, float& vmin, float& vmax) {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 big = _mm256_set1_ps(std::numeric_limits<float>::max());
    const __m256 pinf = _mm256_set1_ps(kInf), ninf = _mm256_set1_ps(-kInf);
    __m256 mn = pinf, mx = ninf;
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(v + i);
        const __m256 ok = _mm256_cmp_ps(_mm256_and_ps(x, absm), big, _CMP_LE_OQ);
        mn = _mm256_min_ps(mn, _mm256_blendv_ps(pinf, x, ok));
        mx = _mm256_max_ps(mx, _mm256_blendv_ps(ninf, x, ok));
    }
    alignas(32) float a[8], b[8];
    _mm256_store_ps(a, mn); _mm256_store_ps(b, mx);
    float lo = a[0], hi = b[0];
    for (int j = 1; j < 8; ++j) { lo = min_(lo, a[j]); hi = max_(hi, b[j]); }
    for (; i < n; ++i)
        if (std::fabs(v[i]) <= std::numeric_limits<float>::max()) { lo = min_(lo, v[i]); hi = max_(hi, v[i]); }
    vmin = lo; vmax = hi;
    return lo <= hi;
}

// Percentiles plo/phi (histograma de 1024 cubetas) de los finitos, con [vmin, vmax] ya conocidos.
// * Índices de cubeta 8 a la vez (AVX2) sobre MITADES: x/2 - vmin/2 nunca desborda aunque
//   vmax - vmin > FLT_MAX (antes: inf·0 = NaN). Medido (24.5 k valores, 1 hilo): 22 → 12 µs.
// * Medido y descartado: 2/4/8 sub-histogramas intercalados (contra el reenvío almacén→carga
//   de ++ repetidos en la misma cubeta): 12.7/14.8/13.6 µs frente a 12.2 µs con uno solo.
void percentiles(const float* v, usize n, float vmin, float vmax, float plo, float phi, float& lo, float& hi) {
    if (vmax - vmin < 1e-12f) { lo = vmin - 1e-3f; hi = vmax + 1e-3f; return; }
    constexpr int kB = 1024;
    alignas(64) u32 hist[kB];
    std::memset(hist, 0, sizeof(hist));
    const float h0 = 0.5f * vmin;
    const float sc = static_cast<float>(kB) / (0.5f * vmax - h0);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 big = _mm256_set1_ps(std::numeric_limits<float>::max());
    const __m256 vh0 = _mm256_set1_ps(h0), vsc = _mm256_set1_ps(sc), top = _mm256_set1_ps(static_cast<float>(kB - 1)), half = _mm256_set1_ps(0.5f);
    usize cnt = 0, i = 0;
    alignas(32) int idx[8];
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(v + i);
        const __m256 ok = _mm256_cmp_ps(_mm256_and_ps(x, absm), big, _CMP_LE_OQ);
        // finito ⇒ x ≥ vmin ⇒ argumento ≥ 0; min_ps(NaN, top) = top y el AND con `ok` lo anula
        const __m256 b = _mm256_and_ps(_mm256_min_ps(_mm256_mul_ps(_mm256_fmsub_ps(x, half, vh0), vsc), top), ok);
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx), _mm256_cvttps_epi32(b));
        const u32 m = static_cast<u32>(_mm256_movemask_ps(ok));
        if (m == 0xFFu) {
#pragma GCC unroll 8
            for (int l = 0; l < 8; ++l) ++hist[idx[l]];
            cnt += 8;
        } else {
            for_each_bit(m, [&](int l) { ++hist[idx[l]]; });
            cnt += static_cast<usize>(std::popcount(m));
        }
    }
    for (; i < n; ++i) {
        const float x = v[i];
        if (!(std::fabs(x) <= std::numeric_limits<float>::max())) continue;
        ++hist[static_cast<int>(min_((x * 0.5f - h0) * sc, static_cast<float>(kB - 1)))];
        ++cnt;
    }
    const usize tlo = static_cast<usize>(plo * static_cast<float>(cnt)), thi = static_cast<usize>(phi * static_cast<float>(cnt));
    usize acc = 0;
    int blo = 0, bhi = kB - 1;
    bool got_lo = false;
    for (int b2 = 0; b2 < kB; ++b2) {
        acc += hist[b2];
        if (!got_lo && acc > tlo) { blo = b2; got_lo = true; }
        if (acc >= thi) { bhi = b2; break; }
    }
    lo = max_(2.0f * (h0 + static_cast<float>(blo) / sc), vmin);
    hi = min_(2.0f * (h0 + static_cast<float>(bhi + 1) / sc), vmax);
    if (hi <= lo) hi = lo + max_(1e-3f, std::fabs(lo) * 1e-3f);
}

} // namespace

namespace detail {

void quantity_row(const FieldView& f, Quantity q, int y, int z, float* CFD_RESTRICT out) {
    k_row_fn[static_cast<int>(q)](f, y, z, out);
}

bool robust_range(const float* v, usize n, float plo, float phi, float& lo, float& hi) {
    float vmin, vmax;
    if (!minmax_finite(v, n, vmin, vmax)) return false;
    if (plo <= 0.0f && phi >= 1.0f) { lo = vmin; hi = vmax; return true; }   // extremos: sin histograma
    percentiles(v, n, vmin, vmax, plo, phi, lo, hi);
    return true;
}

} // namespace detail

// ============================================================================
//  FlowSampler::update — empaquetado AoS FP16 con transposición 4×8 en SSE.
// ============================================================================
void FlowSampler::update(const FieldView& f) {
    CFD_CHECK(f.valid() && f.flags != nullptr, "FlowSampler: campo sin datos o sin flags");
    CFD_CHECK(f.nx >= 2 && f.ny >= 2 && f.nz >= 2, "FlowSampler: dominio demasiado pequeño");
    nx = f.nx; ny = f.ny; nz = f.nz; u_inf = f.u_inf;
    const usize N = static_cast<usize>(nx) * static_cast<usize>(ny) * static_cast<usize>(nz);
    if (cells_.size() != N) { cells_.resize(N); flags_.resize(N); }
    sy_ = static_cast<usize>(nx);
    sz_ = sy_ * static_cast<usize>(ny);
    hi_[0] = below_last(nx); hi_[1] = below_last(ny); hi_[2] = below_last(nz); hi_[3] = 0.0f;
    max_[0] = static_cast<float>(nx - 1); max_[1] = static_cast<float>(ny - 1); max_[2] = static_cast<float>(nz - 1); max_[3] = 1.0f;

    u64* CFD_RESTRICT dst = cells_.data();
    u8* CFD_RESTRICT fdst = flags_.data();
    const int rows = ny * nz;
#ifndef FLOWVIS_NT_PACK
#define FLOWVIS_NT_PACK 1   // stores no temporales en el empaquetado (medido: ver docs/opt/flowvis.md)
#endif
    const bool aligned = FLOWVIS_NT_PACK && (nx % 8) == 0;
    parallel_for(0, rows, 16, [&](i64 lo, i64 hi) {
        for (i64 r = lo; r < hi; ++r) {
            const usize base = static_cast<usize>(r) * sy_;
            std::memcpy(fdst + base, f.flags + base, static_cast<usize>(nx));
            int x = 0;
            for (; x + 8 <= nx; x += 8) {
                const usize n = base + static_cast<usize>(x);
                const __m256 ux = _mm256_loadu_ps(f.ux + n), uy = _mm256_loadu_ps(f.uy + n), uz = _mm256_loadu_ps(f.uz + n);
                const __m256 rr = _mm256_sub_ps(_mm256_loadu_ps(f.rho + n), _mm256_set1_ps(1.0f));
                constexpr int kR = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                const __m128i hx = _mm256_cvtps_ph(ux, kR), hy = _mm256_cvtps_ph(uy, kR);
                const __m128i hz = _mm256_cvtps_ph(uz, kR), hr = _mm256_cvtps_ph(rr, kR);
                // Transposición 4×8 de medias palabras: (x,y,z,r) por celda.
                const __m128i t0 = _mm_unpacklo_epi16(hx, hy), t1 = _mm_unpackhi_epi16(hx, hy);
                const __m128i t2 = _mm_unpacklo_epi16(hz, hr), t3 = _mm_unpackhi_epi16(hz, hr);
                __m128i* o = reinterpret_cast<__m128i*>(dst + n);
                if (aligned) {
                    // Stores no temporales: 64 B completos por iteración, no contaminan la caché.
                    _mm_stream_si128(o + 0, _mm_unpacklo_epi32(t0, t2));
                    _mm_stream_si128(o + 1, _mm_unpackhi_epi32(t0, t2));
                    _mm_stream_si128(o + 2, _mm_unpacklo_epi32(t1, t3));
                    _mm_stream_si128(o + 3, _mm_unpackhi_epi32(t1, t3));
                } else {
                    _mm_storeu_si128(o + 0, _mm_unpacklo_epi32(t0, t2));
                    _mm_storeu_si128(o + 1, _mm_unpackhi_epi32(t0, t2));
                    _mm_storeu_si128(o + 2, _mm_unpacklo_epi32(t1, t3));
                    _mm_storeu_si128(o + 3, _mm_unpackhi_epi32(t1, t3));
                }
            }
            for (; x < nx; ++x) {
                const usize n = base + static_cast<usize>(x);
                const u64 h = static_cast<u64>(f32_to_f16(f.ux[n])) | (static_cast<u64>(f32_to_f16(f.uy[n])) << 16) |
                              (static_cast<u64>(f32_to_f16(f.uz[n])) << 32) | (static_cast<u64>(f32_to_f16(f.rho[n] - 1.0f)) << 48);
                dst[n] = h;
            }
        }
        _mm_sfence();
        // Sólidos: ρ-1 = media de las vecinas fluidas (6-conexas) → Cp limpio junto a paredes
        // (el solver escribe ρ = 1 en sólidos, que sesgaría Cp hacia 0 al interpolar).
        for (i64 r = lo; r < hi; ++r) {
            const usize base = static_cast<usize>(r) * sy_;
            const int y = static_cast<int>(r % ny), z = static_cast<int>(r / ny);
            for (int x = 0; x < nx; x += 8) {
                const int xe = min_(x + 8, nx);
                if (xe - x == 8 && !simd::any_bits_u8x8(f.flags + base + static_cast<usize>(x), lbm::kSolid)) continue;
                for (int xx = x; xx < xe; ++xx) {
                    const usize n = base + static_cast<usize>(xx);
                    if (!(f.flags[n] & lbm::kSolid)) continue;
                    float acc = 0.0f;
                    int cnt = 0;
                    auto add = [&](usize m) { if (!(f.flags[m] & lbm::kSolid)) { acc += f.rho[m] - 1.0f; ++cnt; } };
                    if (xx > 0) add(n - 1);
                    if (xx < nx - 1) add(n + 1);
                    if (y > 0) add(n - sy_);
                    if (y < ny - 1) add(n + sy_);
                    if (z > 0) add(n - sz_);
                    if (z < nz - 1) add(n + sz_);
                    const float rv = cnt ? acc / static_cast<float>(cnt) : 0.0f;
                    dst[n] = (dst[n] & 0x0000FFFFFFFFFFFFull) | (static_cast<u64>(f32_to_f16(rv)) << 48);
                }
            }
        }
    });
}

// ============================================================================
//  SliceView
// ============================================================================
namespace {
// Posición normalizada t·255 de 8 valores (misma fórmula que map_color8).
CFD_INLINE f8 scale_t255(const ColorScale& s, f8 v) {
    f8 t;
    v = v - f8(s.offset);
    if (s.pivoted()) {
        const f8 kneg(127.5f / -s.lo), kpos(127.5f / s.hi);
        t = simd::fmadd(v, simd::select(v < f8::zero(), kneg, kpos), f8(127.5f));
    } else {
        t = (v - f8(s.lo)) * f8(255.0f / (s.hi - s.lo));
    }
    return simd::min(simd::max(t, f8::zero()), f8(255.0f));
}
} // namespace

void SliceView::corners(Vec3 out[4]) const {
    const float p = pos_;
    const float ex = static_cast<float>(nx_) - 0.5f, ey = static_cast<float>(ny_) - 0.5f, ez = static_cast<float>(nz_) - 0.5f;
    switch (axis_) {
        case Axis::X:
            out[0] = {p, -0.5f, -0.5f}; out[1] = {p, ey, -0.5f}; out[2] = {p, ey, ez}; out[3] = {p, -0.5f, ez};
            break;
        case Axis::Y:
            out[0] = {-0.5f, p, -0.5f}; out[1] = {ex, p, -0.5f}; out[2] = {ex, p, ez}; out[3] = {-0.5f, p, ez};
            break;
        default:
            out[0] = {-0.5f, -0.5f, p}; out[1] = {ex, -0.5f, p}; out[2] = {ex, ey, p}; out[3] = {-0.5f, ey, p};
            break;
    }
}

void SliceView::update(const FieldView& f) {
    CFD_CHECK(f.valid() && f.flags != nullptr, "SliceView: campo sin datos o sin flags");
    nx_ = f.nx; ny_ = f.ny; nz_ = f.nz;
    const Axis ax = params.axis;
    axis_ = ax;   // corners/pick/value_at_world/translucent_plane usan el eje CALCULADO, no params.axis
    fade_ = params.fade_below;
    n_axis_ = ax == Axis::X ? nx_ : (ax == Axis::Y ? ny_ : nz_);
    const int tw = ax == Axis::X ? ny_ : nx_;
    const int th = ax == Axis::Z ? ny_ : nz_;
    const int S = params.lic > 0 ? clamp_(params.lic, 1, 6) : 0;
    if (tw != tw_ || th != th_) {
        tw_ = tw; th_ = th;
        vals_.resize(static_cast<usize>(tw) * static_cast<usize>(th));
    }
    lw_ = S ? tw * S : tw; lh_ = S ? th * S : th;
    if (tex_.size() < static_cast<usize>(lw_) * static_cast<usize>(lh_)) tex_.resize(static_cast<usize>(lw_) * static_cast<usize>(lh_));
    pos_ = clamp_(params.pos == params.pos ? params.pos : 0.0f, 0.0f, static_cast<float>(n_axis_ - 1));   // NaN → 0
    res_ = 1.0f;
    int k0 = static_cast<int>(pos_);
    float t = pos_ - static_cast<float>(k0);
    if (k0 >= n_axis_ - 1) { k0 = n_axis_ - 1; t = 0.0f; }
    const int k1 = min_(k0 + 1, n_axis_ - 1);
    plane_values(f, ax, pos_, params.quantity, vals_.data(), tw, th);
    compute_range();
    if (S) { build_lic(f, k0, k1, t); return; }
    colorize();
}

// Valores del plano eje = pos (celdas de f) de la magnitud q en out (tw×th, NaN en sólidos).
void SliceView::plane_values(const FieldView& f, Axis ax, float pos, Quantity q, float* vals, int tw, int th) {
    const int n_axis = ax == Axis::X ? f.nx : (ax == Axis::Y ? f.ny : f.nz);
    pos = clamp_(pos == pos ? pos : 0.0f, 0.0f, static_cast<float>(n_axis - 1));
    int k0 = static_cast<int>(pos);
    float t = pos - static_cast<float>(k0);
    if (k0 >= n_axis - 1) { k0 = n_axis - 1; t = 0.0f; }
    const int k1 = min_(k0 + 1, n_axis - 1);
    const bool two = t > 1e-3f;
    const int ks = t < 0.5f ? k0 : k1;   // capa más cercana: decide sólido

    const int nthr = max_(pool().size(), 1);
    const usize spad = (static_cast<usize>(tw) + 15) & ~usize{15};
    if (scratch_.size() < spad * static_cast<usize>(nthr)) scratch_.resize(spad * static_cast<usize>(nthr));
    const Norm k(f.u_inf);
    const CellFn cfn = k_cell_fn[static_cast<int>(q)];

    parallel_for(0, th, 2, [&](i64 lo, i64 hi) {
        const int wi = max_(ThreadPool::worker_index(), 0);
        float* CFD_RESTRICT tmp = scratch_.data() + spad * static_cast<usize>(wi);
        for (i64 vv = lo; vv < hi; ++vv) {
            const int v = static_cast<int>(vv);
            float* CFD_RESTRICT out = vals + static_cast<usize>(v) * static_cast<usize>(tw);
            if (ax == Axis::X) {
                const int z = v;
                for (int y = 0; y < tw; ++y) {
                    if (f.flags[f.index(ks, y, z)] & lbm::kSolid) { out[y] = k_nan; continue; }
                    const float a = cfn(f, k, k0, y, z);
                    if (!two) { out[y] = a; continue; }
                    const float b = cfn(f, k, k1, y, z);
                    out[y] = a != a ? b : (b != b ? a : a + (b - a) * t);
                }
            } else {
                // Filas contiguas en X: núcleo AVX2.
                const int ya = ax == Axis::Y ? k0 : v, za = ax == Axis::Y ? v : k0;
                const int yb = ax == Axis::Y ? k1 : v, zb = ax == Axis::Y ? v : k1;
                const int yn = ax == Axis::Y ? ks : v, zn = ax == Axis::Y ? v : ks;
                detail::quantity_row(f, q, ya, za, out);
                if (two) {
                    detail::quantity_row(f, q, yb, zb, tmp);
                    const f8 tt(t);
                    int x = 0;
                    for (; x + 8 <= tw; x += 8) {
                        const f8 a = f8::load(out + x), b = f8::load(tmp + x);
                        const f8 an = _mm256_cmp_ps(a, a, _CMP_UNORD_Q), bn = _mm256_cmp_ps(b, b, _CMP_UNORD_Q);
                        f8 r = simd::fmadd(b - a, tt, a);
                        r = simd::select(an, b, simd::select(bn, a, r));
                        r.store(out + x);
                    }
                    for (; x < tw; ++x) {
                        const float a = out[x], b = tmp[x];
                        out[x] = a != a ? b : (b != b ? a : a + (b - a) * t);
                    }
                }
                // Sólido según la capa más cercana (SWAR: salta bloques de 8 sin sólidos).
                const u8* fl = f.flags + f.index(0, yn, zn);
                for (int x = 0; x < tw; x += 8) {
                    const int xe = min_(x + 8, tw);
                    if (xe - x == 8 && !simd::any_bits_u8x8(fl + x, lbm::kSolid)) continue;
                    for (int xx = x; xx < xe; ++xx) if (fl[xx] & lbm::kSolid) out[xx] = k_nan;
                }
            }
        }
    });

}

// Rango (robusto o fijo) sobre vals_ (tw_×th_) → eff_, vmin_, vmax_.
void SliceView::compute_range() {
    const int tw = tw_, th = th_;
    // Rango (robusto o fijo) + extremos reales (sólo valores finitos). Una pasada de mín/máx
    // y el histograma sólo si hay rango automático (antes: 2 pasadas + 2 histogramas).
    const usize n = static_cast<usize>(tw) * static_cast<usize>(th);
    eff_ = params.scale;
    float mn = 0.0f, mx = 0.0f;
    const bool any = minmax_finite(vals_.data(), n, mn, mx);
    vmin_ = any ? mn : 0.0f; vmax_ = any ? mx : 0.0f;
    if (params.auto_range && any) {
        float lo, hi;
        percentiles(vals_.data(), n, mn, mx, 0.01f, 0.99f, lo, hi);
        lo -= eff_.offset; hi -= eff_.offset;   // rango en unidades corregidas (ColorScale::offset)
        if (eff_.diverging && lo < 0.0f && hi > 0.0f) { eff_.lo = lo; eff_.hi = hi; }
        else if (eff_.diverging) {   // todo de un signo: rango simétrico para conservar el centro
            const float m = max_(std::fabs(lo), std::fabs(hi));
            eff_.lo = -m; eff_.hi = m;
        } else { eff_.lo = lo; eff_.hi = hi; }
    }
    if (!(eff_.hi > eff_.lo)) eff_.hi = eff_.lo + max_(1e-3f, std::fabs(eff_.lo) * 1e-3f);

}

// Textura de color de vals_ (sin LIC).
void SliceView::colorize() {
    const int tw = tw_, th = th_;
    // Pasada de color: 8 texeles por iteración (VPGATHERDD sobre la LUT) + sólidos + desvanecido.
    const ColorScale sc = eff_;
    const u32 solid = params.solid_color;
    const float fade = params.fade_below;
    const u32* lut = render::colormap_lut(sc.map);
    parallel_for(0, th, 8, [&](i64 lo2, i64 hi2) {
        for (i64 v = lo2; v < hi2; ++v) {
            const float* in = vals_.data() + static_cast<usize>(v) * static_cast<usize>(tw);
            u32* o = tex_.data() + static_cast<usize>(v) * static_cast<usize>(tw);
            int x = 0;
            for (; x + 8 <= tw; x += 8) {
                const f8 val = f8::load(in + x);
                const f8 nanm = _mm256_cmp_ps(val, val, _CMP_UNORD_Q);
                const f8 t255 = scale_t255(sc, val);
                __m256i c = _mm256_i32gather_epi32(reinterpret_cast<const int*>(lut), _mm256_cvtps_epi32(t255.v), 4);
                if (fade > 0.0f) {
                    // alfa = saturate(dist/fade)·255, dist = t (lineal) o |2t-1| (divergente)
                    f8 d = sc.pivoted() ? simd::abs(t255 * f8(2.0f / 255.0f) - f8(1.0f)) : t255 * f8(1.0f / 255.0f);
                    d = simd::min(d * f8(255.0f / fade), f8(255.0f));
                    const __m256i a = _mm256_slli_epi32(_mm256_cvtps_epi32(d.v), 24);
                    c = _mm256_or_si256(_mm256_and_si256(c, _mm256_set1_epi32(0x00FFFFFF)), a);
                }
                c = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(c), _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(solid))), nanm.v));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + x), c);
            }
            for (; x < tw; ++x) {
                const float val = in[x];
                if (val != val) { o[x] = solid; continue; }
                u32 c = map_color(sc, val);
                if (fade > 0.0f) {
                    const float tn = saturate(sc.normalize(val));
                    const float d = sc.pivoted() ? std::fabs(2.0f * tn - 1.0f) : tn;
                    c = (c & 0x00FFFFFFu) | (static_cast<u32>(min_(d / fade, 1.0f) * 255.0f + 0.5f) << 24);
                }
                o[x] = c;
            }
        }
    });
    (void)lut;
}

// ---------------------------------------------------------------------------
//  LIC (Cabral & Leedom 1993): para cada subtexel se promedia un ruido blanco fijo
//  a lo largo de la línea de corriente 2D del campo proyectado en el plano (RK1 de
//  paso = 1 subtexel, núcleo de Hann, ±lic_length celdas). La intensidad normalizada
//  (z-score) modula el brillo del color de la magnitud → vetas alineadas con el flujo.
// ---------------------------------------------------------------------------
void SliceView::build_lic(const FieldView& f, int k0, int k1, float t) {
    const int tw = tw_, th = th_;
    const Axis ax = axis_;
    const usize ncell = static_cast<usize>(tw) * static_cast<usize>(th);
    if (vel2_.size() < 2 * ncell) vel2_.resize(2 * ncell);
    // 1. Dirección de la velocidad en el plano por celda (interpolada entre capas); sólido → 0.
    float* CFD_RESTRICT V2 = vel2_.data();
    parallel_for(0, th, 8, [&](i64 lo, i64 hi) {
        for (i64 vv = lo; vv < hi; ++vv) {
            const int v = static_cast<int>(vv);
            for (int u = 0; u < tw; ++u) {
                usize a, b;
                if (ax == Axis::X) { a = f.index(k0, u, v); b = f.index(k1, u, v); }
                else if (ax == Axis::Y) { a = f.index(u, k0, v); b = f.index(u, k1, v); }
                else { a = f.index(u, v, k0); b = f.index(u, v, k1); }
                const float* cu = ax == Axis::X ? f.uy : f.ux;
                const float* cv = ax == Axis::Z ? f.uy : f.uz;
                const usize o = 2 * (static_cast<usize>(v) * static_cast<usize>(tw) + static_cast<usize>(u));
                const bool sol = vals_[o / 2] != vals_[o / 2];
                const float va = lerp(cu[a], cu[b], t), vb = lerp(cv[a], cv[b], t);
                // Dirección UNITARIA por celda: el bucle de convolución sólo interpola (sin sqrt/div por paso).
                const float l2 = va * va + vb * vb;
                const float inv = (sol || l2 < 1e-14f) ? 0.0f : 1.0f / std::sqrt(l2);
                V2[o] = va * inv;
                V2[o + 1] = vb * inv;
            }
        }
    });
    lic_convolve();
}

// Convolución LIC + color sobre vel2_ (dirección unitaria por texel) y vals_ (tw_×th_ texeles; lw_×lh_ subtexeles).
void SliceView::lic_convolve() {
    const int S = clamp_(params.lic, 1, 6);
    const int tw = tw_, th = th_, lw = lw_, lh = lh_;
    const usize nsub = static_cast<usize>(lw) * static_cast<usize>(lh);
    if (lic_.size() < nsub) lic_.resize(nsub);
    if (noise_w_ != lw || noise_h_ != lh) {   // ruido fijo (semilla constante): no parpadea entre cuadros
        noise_.resize(nsub + 8);
        WyRand rng(0x11C0FFEEull);
        for (usize i = 0; i < nsub + 8; ++i) noise_[i] = static_cast<u8>(rng.next() >> 56);
        noise_w_ = lw; noise_h_ = lh;
    }
    const float* CFD_RESTRICT V2 = vel2_.data();
    // 2. Convolución a lo largo de las líneas de corriente del plano.
    const float h = 1.0f / static_cast<float>(S);                      // paso = 1 subtexel (en celdas)
    const float fS = static_cast<float>(S);
    const int L = max_(2, static_cast<int>(params.lic_length * static_cast<float>(S)));
    const float fu = static_cast<float>(tw) - 0.5f, fv = static_cast<float>(th) - 0.5f;
    const float* CFD_RESTRICT vals = vals_.data();
    const u8* CFD_RESTRICT nz = noise_.data();
    float* CFD_RESTRICT out = lic_.data();
    float wk[64];                                                      // núcleo de Hann (hasta 63 pasos)
    const int Lc = min_(L, 63);
    for (int i = 0; i <= Lc; ++i) wk[i] = 0.5f + 0.5f * std::cos(k_pi * static_cast<float>(i) / static_cast<float>(Lc + 1));
    // AVX2: 8 subtexeles consecutivos de la fila integran sus líneas A LA VEZ (un carril cada
    // uno). Bilineal de la dirección con 8 VPGATHERDPS, sólido y ruido con 2 gathers más; los
    // carriles que terminan (sólido, borde, punto crítico) quedan enmascarados. Medido: la
    // versión escalar está limitada por nº de instrucciones (~75 por paso), no por latencia.
    using simd::i8x;
    const i8x vtw(tw), vlw(lw);
    const f8 vzero = f8::zero(), vhalf(0.5f), vfS(fS);
    const f8 xmax(static_cast<float>(tw - 1) - 1e-3f), ymax(static_cast<float>(th - 1) - 1e-3f);
    const f8 blo(-0.5f), bhx(fu), bhy(fv);
    const __m256i ilwm1 = _mm256_set1_epi32(lw - 1), ilhm1 = _mm256_set1_epi32(lh - 1);
    const __m256i itwm1 = _mm256_set1_epi32(tw - 1), ithm1 = _mm256_set1_epi32(th - 1);
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int* nzi = reinterpret_cast<const int*>(nz);
    parallel_for(0, lh, 4, [&](i64 lo, i64 hi) {
        for (i64 jj = lo; jj < hi; ++jj) {
            const int j = static_cast<int>(jj);
            const f8 py((static_cast<float>(j) + 0.5f) * h - 0.5f);
            for (int i0 = 0; i0 < lw; i0 += 8) {
                const __m256i ii = _mm256_add_epi32(_mm256_set1_epi32(i0), lane);
                const f8 valid = _mm256_castsi256_ps(_mm256_cmpgt_epi32(vlw.v, ii));
                const f8 px = simd::fmadd(f8(_mm256_cvtepi32_ps(ii)) + vhalf, f8(h), -vhalf);
                // Celda propia: sólido → NaN
                const __m256i c0 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_min_epi32(_mm256_cvttps_epi32(py + vhalf), ithm1), vtw.v),
                                                    _mm256_min_epi32(_mm256_cvttps_epi32(px + vhalf), itwm1));
                const f8 cv0 = _mm256_mask_i32gather_ps(vzero, vals, c0, valid, 4);
                const f8 alive = valid & f8(_mm256_cmp_ps(cv0, cv0, _CMP_ORD_Q));
                const usize so = static_cast<usize>(j) * static_cast<usize>(lw) + static_cast<usize>(i0);
                const f8 n0 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mask_i32gather_epi32(_mm256_setzero_si256(), nzi, _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(so)), lane), _mm256_castps_si256(valid), 1), _mm256_set1_epi32(255)));
                f8 acc = n0 * f8(wk[0]), ws(wk[0]);
                for (int dir = -1; dir <= 1; dir += 2) {
                    const f8 k(static_cast<float>(dir) * h);
                    f8 x = px, y = py, live = alive;
                    for (int st = 1; st <= Lc; ++st) {
                        if (!simd::movemask(live)) break;
                        const f8 cx = simd::min(simd::max(x, vzero), xmax), cy = simd::min(simd::max(y, vzero), ymax);
                        const __m256i ix = _mm256_cvttps_epi32(cx), iy = _mm256_cvttps_epi32(cy);
                        const f8 tx = cx - f8(_mm256_cvtepi32_ps(ix)), ty = cy - f8(_mm256_cvtepi32_ps(iy));
                        const __m256i e00 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vtw.v), ix), 1);   // índice de float en V2
                        const __m256i e01 = _mm256_add_epi32(e00, _mm256_set1_epi32(2 * tw));
                        const f8 a00 = _mm256_i32gather_ps(V2, e00, 4), b00 = _mm256_i32gather_ps(V2 + 1, e00, 4);
                        const f8 a10 = _mm256_i32gather_ps(V2 + 2, e00, 4), b10 = _mm256_i32gather_ps(V2 + 3, e00, 4);
                        const f8 a01 = _mm256_i32gather_ps(V2, e01, 4), b01 = _mm256_i32gather_ps(V2 + 1, e01, 4);
                        const f8 a11 = _mm256_i32gather_ps(V2 + 2, e01, 4), b11 = _mm256_i32gather_ps(V2 + 3, e01, 4);
                        const f8 ta = simd::fmadd(a10 - a00, tx, a00), tb = simd::fmadd(b10 - b00, tx, b00);
                        const f8 ua = simd::fmadd(simd::fmadd(a11 - a01, tx, a01) - ta, ty, ta);
                        const f8 ub = simd::fmadd(simd::fmadd(b11 - b01, tx, b01) - tb, ty, tb);
                        live = live & (simd::fmadd(ua, ua, ub * ub) >= f8(1e-6f));             // punto crítico → fin
                        const f8 xn = simd::fmadd(ua, k, x), yn = simd::fmadd(ub, k, y);
                        live = live & (xn >= blo) & (yn >= blo) & (xn <= bhx) & (yn <= bhy);
                        const __m256i ci = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(yn + vhalf), _mm256_setzero_si256()), ithm1), vtw.v),
                                                            _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(xn + vhalf), _mm256_setzero_si256()), itwm1));
                        const f8 cv = _mm256_i32gather_ps(vals, ci, 4);
                        live = live & f8(_mm256_cmp_ps(cv, cv, _CMP_ORD_Q));                  // sólido → fin
                        const __m256i si = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32((xn + vhalf) * vfS), _mm256_setzero_si256()), ilwm1);
                        const __m256i sj = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32((yn + vhalf) * vfS), _mm256_setzero_si256()), ilhm1);
                        const __m256i ni = _mm256_add_epi32(_mm256_mullo_epi32(sj, vlw.v), si);
                        const f8 nv = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_i32gather_epi32(nzi, ni, 1), _mm256_set1_epi32(255)));
                        const f8 w = live & f8(wk[st]);                                           // 0 en carriles muertos
                        acc = simd::fmadd(w, nv, acc);
                        ws = ws + w;
                        x = simd::select(live, xn, x);
                        y = simd::select(live, yn, y);
                    }
                }
                const f8 r = simd::select(alive, acc / ws, f8(k_nan));
                alignas(32) float tmp[8];
                r.store_a(tmp);
                const int nl = min_(8, lw - i0);
                for (int l = 0; l < nl; ++l) out[so + static_cast<usize>(l)] = tmp[l];
            }
        }
    });
    // 3. Normalización (media y desviación de la intensidad LIC en el fluido).
    double s1 = 0, s2 = 0;
    usize cnt = 0;
    for (usize i = 0; i < nsub; ++i) { const float x = out[i]; if (x == x) { s1 += x; s2 += static_cast<double>(x) * x; ++cnt; } }
    const double mean = cnt ? s1 / static_cast<double>(cnt) : 128.0;
    const double sd = cnt ? std::sqrt(max_(s2 / static_cast<double>(cnt) - mean * mean, 1e-6)) : 1.0;
    const float fm = static_cast<float>(mean), fis = static_cast<float>(1.0 / sd);
    const float gain = 0.45f * clamp_(params.lic_contrast, 0.0f, 1.0f);
    // 4. Color: magnitud bilineal → mapa → brillo × (1 + gain·z), z sujeto a ±2.
    const ColorScale sc = eff_;
    const u32 solid = params.solid_color;
    const float fade = params.fade_below;
    parallel_for(0, lh, 8, [&](i64 lo, i64 hi) {
        for (i64 jj = lo; jj < hi; ++jj) {
            const int j = static_cast<int>(jj);
            u32* o = tex_.data() + static_cast<usize>(j) * static_cast<usize>(lw);
            for (int i = 0; i < lw; ++i) {
                const float lv = out[static_cast<usize>(j) * static_cast<usize>(lw) + static_cast<usize>(i)];
                if (lv != lv) { o[i] = solid; continue; }
                const float val = value_tex((static_cast<float>(i) + 0.5f) * h - 0.5f, (static_cast<float>(j) + 0.5f) * h - 0.5f);
                u32 c = map_color(sc, val);
                const float m = 1.0f + gain * clamp_((lv - fm) * fis, -2.0f, 2.0f);
                const u32 r = min_(static_cast<u32>(static_cast<float>((c >> 16) & 255) * m), 255u);
                const u32 g = min_(static_cast<u32>(static_cast<float>((c >> 8) & 255) * m), 255u);
                const u32 b = min_(static_cast<u32>(static_cast<float>(c & 255) * m), 255u);
                u32 a = 255u;
                if (fade > 0.0f) {
                    const float tn = saturate(sc.normalize(val));
                    const float d = sc.pivoted() ? std::fabs(2.0f * tn - 1.0f) : tn;
                    a = static_cast<u32>(min_(d / fade, 1.0f) * 255.0f + 0.5f);
                }
                o[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }
    });
}

float SliceView::value_at(float u, float v) const {
    // (u, v) en celdas de la base → texeles (con refinamiento hay res_ texeles por celda).
    return value_tex((u + 0.5f) * res_ - 0.5f, (v + 0.5f) * res_ - 0.5f);
}

float SliceView::value_tex(float u, float v) const {
    if (vals_.empty()) return k_nan;
    if (!(u >= -0.5f && v >= -0.5f && u <= static_cast<float>(tw_) - 0.5f && v <= static_cast<float>(th_) - 0.5f)) return k_nan;
    const float fu = clamp_(u, 0.0f, static_cast<float>(tw_ - 1)), fv = clamp_(v, 0.0f, static_cast<float>(th_ - 1));
    const int u0 = min_(static_cast<int>(fu), max_(tw_ - 2, 0)), v0 = min_(static_cast<int>(fv), max_(th_ - 2, 0));
    const int u1 = min_(u0 + 1, tw_ - 1), v1 = min_(v0 + 1, th_ - 1);
    const float a = fu - static_cast<float>(u0), b = fv - static_cast<float>(v0);
    auto at = [&](int x, int y) { return vals_[static_cast<usize>(y) * static_cast<usize>(tw_) + static_cast<usize>(x)]; };
    const float c00 = at(u0, v0), c10 = at(u1, v0), c01 = at(u0, v1), c11 = at(u1, v1);
    if (c00 != c00 || c10 != c10 || c01 != c01 || c11 != c11) {   // junto a sólidos: el más cercano
        return at(a < 0.5f ? u0 : u1, b < 0.5f ? v0 : v1);
    }
    return lerp(lerp(c00, c10, a), lerp(c01, c11, a), b);
}

float SliceView::value_at_world(Vec3 p) const {
    switch (axis_) {
        case Axis::X: return value_at(p.y, p.z);
        case Axis::Y: return value_at(p.x, p.z);
        default: return value_at(p.x, p.y);
    }
}

TranslucentPlane SliceView::translucent_plane() const {
    TranslucentPlane t;
    t.axis = axis_;
    t.pos = pos_;
    const bool opaque = params.opacity >= 0.999f && fade_ <= 0.0f;
    t.opacity = (tex_.empty() || opaque) ? 0.0f : clamp_(params.opacity, 0.0f, 1.0f);
    t.lo = {-0.5f, -0.5f};
    t.hi = {static_cast<float>(tw_) / res_ - 0.5f, static_cast<float>(th_) / res_ - 0.5f};
    return t;
}

float plane_transmission(const Camera& cam, Vec3 p, std::span<const TranslucentPlane> planes) {
    float k = 1.0f;
    for (const TranslucentPlane& pl : planes) {
        if (!(pl.opacity > 0.0f)) continue;
        const int a = static_cast<int>(pl.axis), ua = a == 0 ? 1 : 0, va = a == 2 ? 1 : 2;
        // Segmento p → cámara (perspectiva: hacia el ojo; ortográfica: a lo largo de -fwd).
        const Vec3 d = cam.ortho ? -cam.fwd : cam.eye - p;
        const float dp = pl.pos - p[a];
        if (std::fabs(d[a]) < 1e-12f) continue;
        const float t = dp / d[a];
        if (!(t > 0.0f) || (!cam.ortho && t >= 1.0f)) continue;   // el plano no está entre p y la cámara
        const Vec3 x = p + d * t;
        if (x[ua] < pl.lo.x || x[ua] > pl.hi.x || x[va] < pl.lo.y || x[va] > pl.hi.y) continue;
        k *= 1.0f - pl.opacity;
    }
    return k;
}

bool pick_plane(const Camera& cam, float sx, float sy, Axis axis, float pos, Vec3& hit) {
    Vec3 o, d;
    cam.ray(sx, sy, o, d);
    const int a = static_cast<int>(axis);
    if (std::fabs(d[a]) < 1e-7f) return false;
    const float t = (pos - o[a]) / d[a];
    if (t <= 0.0f) return false;
    hit = o + d * t;
    hit[a] = pos;
    return true;
}

bool SliceView::pick(const Camera& cam, float sx, float sy, Vec3& hit, float& value) const {
    if (tex_.empty()) return false;
    if (!pick_plane(cam, sx, sy, axis_, pos_, hit)) return false;
    const Vec3 lo{-0.5f, -0.5f, -0.5f};
    const Vec3 hi{static_cast<float>(nx_) - 0.5f, static_cast<float>(ny_) - 0.5f, static_cast<float>(nz_) - 0.5f};
    for (int i = 0; i < 3; ++i) if (hit[i] < lo[i] - 1e-3f || hit[i] > hi[i] + 1e-3f) return false;
    value = value_at_world(hit);
    return true;
}

// ---------------------------------------------------------------------------
//  Refinamiento local: plano compuesto. Cada rejilla que corta el plano calcula sus valores con su propia resolución
//  (plane_values, en sus celdas); la textura final (res_ texeles por celda de la base) toma en cada texel la rejilla
//  más fina cuya región propia lo contiene (bilineal con pesos de fluido: NaN = sólido).
// ---------------------------------------------------------------------------
void SliceView::update(const MultiField& m, bool ground_layers) {
    const FieldView& f = m.base();
    if (m.n <= 1) { update(f); return; }
    CFD_CHECK(f.valid() && f.flags != nullptr, "SliceView: campo sin datos o sin flags");
    nx_ = f.nx; ny_ = f.ny; nz_ = f.nz;
    const Axis ax = params.axis;
    axis_ = ax;
    fade_ = params.fade_below;
    n_axis_ = ax == Axis::X ? nx_ : (ax == Axis::Y ? ny_ : nz_);
    const int a = static_cast<int>(ax), ua = a == 0 ? 1 : 0, va = a == 2 ? 1 : 2;   // ejes del plano (u, v)
    pos_ = clamp_(params.pos == params.pos ? params.pos : 0.0f, 0.0f, static_cast<float>(n_axis_ - 1));
    const Quantity q = params.quantity;
    // Rejillas que cortan el plano y sus valores.
    int use[MultiField::k_max];
    int nuse = 0, depth_max = 0;
    int gw[MultiField::k_max] = {}, gh[MultiField::k_max] = {};
    for (int g = 0; g < m.n; ++g) {
        const GridField& G = m.g[g];
        const int gn[3] = {G.f.nx, G.f.ny, G.f.nz};
        float pg;
        if (ground_layers && g > 0) {
            if (!G.ground) continue;
            pg = 1.0f;                                       // primera capa de fluido de esta rejilla
        } else {
            pg = (pos_ - G.org[a]) / G.scale;
            if (g > 0 && (pg < 0.5f || pg > static_cast<float>(gn[a]) - 1.5f)) continue;   // el plano no cruza su región propia
        }
        const int w = gn[ua], h = gn[va];
        if (gbuf_[g].size() < static_cast<usize>(w) * static_cast<usize>(h)) gbuf_[g].resize(static_cast<usize>(w) * static_cast<usize>(h));
        plane_values(G.f, ax, pg, q, gbuf_[g].data(), w, h);
        const float ks = quantity_grid_scale(q, G.scale);
        if (ks != 1.0f) {
            float* b = gbuf_[g].data();
            for (usize i = 0; i < static_cast<usize>(w) * static_cast<usize>(h); ++i) b[i] *= ks;
        }
        gw[g] = w; gh[g] = h;
        use[nuse++] = g;
        depth_max = max_(depth_max, G.depth);
    }
    // Resolución de la textura: la de la rejilla más fina que corta el plano, hasta ~4 M texeles.
    const int tw0 = ax == Axis::X ? ny_ : nx_, th0 = ax == Axis::Z ? ny_ : nz_;
    int r = 1 << depth_max;
    while (r > 1 && static_cast<double>(tw0) * r * th0 * r > 4.0e6) r >>= 1;
    res_ = static_cast<float>(r);
    const int tw = tw0 * r, th = th0 * r;
    const int S = params.lic > 0 ? clamp_(params.lic, 1, 6) : 0;
    if (tw != tw_ || th != th_) { tw_ = tw; th_ = th; vals_.resize(static_cast<usize>(tw) * static_cast<usize>(th)); }
    lw_ = S ? tw * S : tw; lh_ = S ? th * S : th;
    if (tex_.size() < static_cast<usize>(lw_) * static_cast<usize>(lh_)) tex_.resize(static_cast<usize>(lw_) * static_cast<usize>(lh_));
    const float ir = 1.0f / static_cast<float>(r);
    const float zg = ground_layers ? 0.75f : pos_;          // coordenada normal al plano para elegir la rejilla más fina
    if (S && vel2_.size() < 2 * static_cast<usize>(tw) * static_cast<usize>(th)) vel2_.resize(2 * static_cast<usize>(tw) * static_cast<usize>(th));
    float* CFD_RESTRICT V2 = S ? vel2_.data() : nullptr;
    parallel_for(0, th, 4, [&](i64 lo, i64 hi) {
        for (i64 vv = lo; vv < hi; ++vv) {
            const int v = static_cast<int>(vv);
            float* out = vals_.data() + static_cast<usize>(v) * static_cast<usize>(tw);
            const float pv = (static_cast<float>(v) + 0.5f) * ir - 0.5f;
            for (int u = 0; u < tw; ++u) {
                const float pu = (static_cast<float>(u) + 0.5f) * ir - 0.5f;
                Vec3 p;
                p[a] = zg; p[ua] = pu; p[va] = pv;
                // Rejilla más fina (entre las que cortan el plano) que contiene el texel.
                int g = 0, bd = -1;
                for (int k = 0; k < nuse; ++k) {
                    const GridField& G = m.g[use[k]];
                    if (G.depth <= bd) continue;
                    if (use[k] == 0 || (p.x >= G.inner.lo.x && p.x <= G.inner.hi.x && p.y >= G.inner.lo.y && p.y <= G.inner.hi.y &&
                                        (ground_layers ? p.z <= G.inner.hi.z : (p.z >= G.inner.lo.z && p.z <= G.inner.hi.z)))) {
                        g = use[k];
                        bd = G.depth;
                    }
                }
                const GridField& G = m.g[g];
                const float gu = (pu - G.org[ua]) / G.scale, gv = (pv - G.org[va]) / G.scale;
                const int w = gw[g], h = gh[g];
                const float fu = clamp_(gu, 0.0f, below_last(w)), fvv = clamp_(gv, 0.0f, below_last(h));
                const int u0 = static_cast<int>(fu), v0 = static_cast<int>(fvv);
                const float tu = fu - static_cast<float>(u0), tv = fvv - static_cast<float>(v0);
                const float* b = gbuf_[g].data();
                const float c[4] = {b[static_cast<usize>(v0) * w + u0], b[static_cast<usize>(v0) * w + u0 + 1], b[static_cast<usize>(v0 + 1) * w + u0],
                                    b[static_cast<usize>(v0 + 1) * w + u0 + 1]};
                const float wt[4] = {(1 - tu) * (1 - tv), tu * (1 - tv), (1 - tu) * tv, tu * tv};
                float acc = 0.0f, ws = 0.0f;
                for (int k = 0; k < 4; ++k) if (c[k] == c[k]) { acc += wt[k] * c[k]; ws += wt[k]; }
                // Sólido si la celda más cercana lo es (como el plano de una sola rejilla).
                const int kn = (tu < 0.5f ? 0 : 1) + (tv < 0.5f ? 0 : 2);
                const float val = (c[kn] != c[kn] || ws <= 1e-6f) ? k_nan : acc / ws;
                out[u] = val;
                if (V2) {
                    Vec3 vel{0, 0, 0};
                    if (val == val) {
                        Vec3 pp;
                        pp[a] = ground_layers && g > 0 ? 1.0f : (pos_ - G.org[a]) / G.scale;
                        if (ground_layers && g == 0) pp[a] = pos_;
                        pp[ua] = gu; pp[va] = gv;
                        vel = G.f.velocity(pp);
                    }
                    const float cu = vel[ua], cv = vel[va];
                    const float l2 = cu * cu + cv * cv;
                    const float inv = l2 < 1e-14f ? 0.0f : 1.0f / std::sqrt(l2);
                    const usize o = 2 * (static_cast<usize>(v) * static_cast<usize>(tw) + static_cast<usize>(u));
                    V2[o] = cu * inv;
                    V2[o + 1] = cv * inv;
                }
            }
        }
    });
    compute_range();
    if (S) { lic_convolve(); return; }
    colorize();
}

// ============================================================================
//  Huella en el suelo
// ============================================================================
void GroundFootprint::update(const MultiField& m) {
    if (m.n <= 1) { update(m.base()); return; }
    const FieldView& f = m.base();
    nx_ = f.nx; ny_ = f.ny;
    float z = params.z;
    if (z < 0.0f) {
        const usize n = f.index(f.nx / 2, f.ny / 2, 0);
        const bool ground = (f.flags[n] & lbm::kSolid) && (!f.solid_id || f.solid_id[n] == lbm::k_ground_id);
        z = ground ? 1.0f : 0.0f;
    }
    slice_.params.axis = Axis::Z;
    slice_.params.pos = z;
    slice_.params.quantity = params.quantity;
    slice_.params.scale = params.scale;
    slice_.params.auto_range = params.auto_range;
    slice_.params.opacity = 1.0f;
    slice_.params.fade_below = 0.0f;
    slice_.params.lic = 0;
    slice_.update(m, z >= 1.0f);   // con suelo: la primera capa de fluido de cada rejilla apoyada en él
}

void GroundFootprint::update(const FieldView& f) {
    nx_ = f.nx; ny_ = f.ny;
    float z = params.z;
    if (z < 0.0f) {
        // ¿Hay suelo? (capa z = 0 sólida con id 255 en el centro del dominio)
        const usize n = f.index(f.nx / 2, f.ny / 2, 0);
        const bool ground = (f.flags[n] & lbm::kSolid) && (!f.solid_id || f.solid_id[n] == lbm::k_ground_id);
        z = ground ? 1.0f : 0.0f;
    }
    slice_.params.axis = Axis::Z;
    slice_.params.pos = z;
    slice_.params.quantity = params.quantity;
    slice_.params.scale = params.scale;
    slice_.params.auto_range = params.auto_range;
    slice_.params.opacity = 1.0f;
    slice_.params.fade_below = 0.0f;
    slice_.update(f);
}

Aabb GroundFootprint::extent() const {
    const float z0 = slice_.plane_pos() >= 1.0f ? 0.5f : -0.5f;
    return {{-0.5f, -0.5f, z0}, {static_cast<float>(nx_) - 0.5f, static_cast<float>(ny_) - 0.5f, z0}};
}

// ============================================================================
//  Estela
// ============================================================================
SliceParams wake_slice_params(float x_cells) {
    SliceParams p;
    p.axis = Axis::X;
    p.pos = x_cells;
    p.quantity = Quantity::Cp0;
    p.scale = default_scale(Quantity::Cp0);
    p.opacity = 1.0f;
    return p;
}

WakeStats wake_survey(const FieldView& f, float x_cells) {
    WakeStats w;
    const float xc = clamp_(x_cells, 0.0f, static_cast<float>(f.nx - 1));
    w.x = xc;
    int x0 = static_cast<int>(xc);
    float t = xc - static_cast<float>(x0);
    if (x0 >= f.nx - 1) { x0 = f.nx - 1; t = 0.0f; }
    const int x1 = min_(x0 + 1, f.nx - 1);
    const Norm k(f.u_inf);
    double loss = 0, cross = 0, cy = 0, cz = 0;
    float mn = 1e30f;
    int cells = 0;
    for (int z = 0; z < f.nz; ++z)
        for (int y = 0; y < f.ny; ++y) {
            const usize a = f.index(x0, y, z), b = f.index(x1, y, z);
            if ((f.flags[a] | f.flags[b]) & lbm::kSolid) continue;
            const float r = lerp(f.rho[a], f.rho[b], t);
            const Vec3 u{lerp(f.ux[a], f.ux[b], t), lerp(f.uy[a], f.uy[b], t), lerp(f.uz[a], f.uz[b], t)};
            const float cp0 = (r - 1.0f) * k.cp_k + r * length2(u) * k.inv_u2;
            mn = min_(mn, cp0);
            cross += static_cast<double>((u.y * u.y + u.z * u.z) * k.inv_u2);
            if (cp0 < 1.0f) {
                const double l = 1.0 - static_cast<double>(cp0);
                loss += l; cy += l * y; cz += l * z;
                if (cp0 < 0.98f) ++cells;
            }
        }
    w.loss_area = static_cast<float>(loss);
    w.crossflow_area = static_cast<float>(cross);
    w.min_cp0 = mn < 1e29f ? mn : 1.0f;
    if (loss > 0) w.centroid = {static_cast<float>(cy / loss), static_cast<float>(cz / loss)};
    w.cells = cells;
    return w;
}

// ============================================================================
//  Refinamiento local: utilidades del campo compuesto
// ============================================================================
MultiField single_field(const FieldView& f) {
    MultiField m;
    m.n = 1;
    m.g[0].f = f;
    m.g[0].inner = {{-0.5f, -0.5f, -0.5f}, {static_cast<float>(f.nx) - 0.5f, static_cast<float>(f.ny) - 0.5f, static_cast<float>(f.nz) - 0.5f}};
    return m;
}

float quantity_grid_scale(Quantity q, float scale) {
    if (q == Quantity::Vorticity) return 1.0f / scale;
    if (q == Quantity::QCriterion) return 1.0f / (scale * scale);
    return 1.0f;
}

float sample_quantity(const MultiField& m, Quantity q, Vec3 p) {
    const int g = m.finest(p);
    const GridField& G = m.g[g];
    return g ? sample_quantity(G.f, q, G.to_grid(p)) * quantity_grid_scale(q, G.scale) : sample_quantity(G.f, q, p);
}

Probe probe(const MultiField& m, Vec3 p) {
    const int g = m.finest(p);
    if (!g) return probe(m.base(), p);
    const GridField& G = m.g[g];
    Probe r = probe(G.f, G.to_grid(p));
    r.pos = p;
    r.vorticity *= quantity_grid_scale(Quantity::Vorticity, G.scale);
    r.q *= quantity_grid_scale(Quantity::QCriterion, G.scale);
    return r;
}

void MultiSampler::update(const MultiField& m) {
    n_ = max_(min_(m.n, MultiField::k_max), 0);
    for (int i = 0; i < n_; ++i) {
        s_[i].update(m.g[i].f);
        g_[i] = m.g[i];
        org_[i][0] = m.g[i].org.x; org_[i][1] = m.g[i].org.y; org_[i][2] = m.g[i].org.z; org_[i][3] = 0.0f;
        inv_[i] = 1.0f / m.g[i].scale;
    }
    nx = s_[0].nx; ny = s_[0].ny; nz = s_[0].nz; u_inf = s_[0].u_inf;
}

// ============================================================================
//  Colores de malla
// ============================================================================
void color_mesh(Mesh& mesh, const FieldView& f, const SurfaceParams& p) {
    const usize nv = mesh.pos.size();
    if (mesh.color.size() != nv) mesh.color.resize(nv);
    if (nv == 0) return;
    u32* CFD_RESTRICT col = mesh.color.data();
    const Vec3* pos = mesh.pos.data();
    const Vec3* nrm = mesh.nrm.size() == nv ? mesh.nrm.data() : nullptr;
    switch (p.mode) {
        case SurfaceMode::Solid: {
            const u32 c = p.solid_color;
            parallel_for(0, static_cast<i64>(nv), 1 << 14, [&](i64 lo, i64 hi) { for (i64 i = lo; i < hi; ++i) col[i] = c; });
            return;
        }
        case SurfaceMode::Component: {
            const u8* g = mesh.group.size() == nv ? mesh.group.data() : nullptr;
            u32 pal[256];
            for (u32 i = 0; i < 256; ++i) pal[i] = p.group_colors.size() >= 256 ? p.group_colors[i] : hash_color(i);
            parallel_for(0, static_cast<i64>(nv), 1 << 14, [&](i64 lo, i64 hi) {
                for (i64 i = lo; i < hi; ++i) col[i] = g ? pal[g[i]] : p.solid_color;
            });
            return;
        }
        default: break;
    }
    if (!f.valid() || !f.flags) {
        parallel_for(0, static_cast<i64>(nv), 1 << 14, [&](i64 lo, i64 hi) { for (i64 i = lo; i < hi; ++i) col[i] = p.solid_color; });
        return;
    }
    const Norm k(f.u_inf);
    const bool cp = p.mode == SurfaceMode::Cp;
    const ColorScale sc = p.scale;
    const float off = p.offset;
    constexpr u32 k_unknown = 0xFF8A8D93u;   // sin fluido cerca: gris neutro
    parallel_for(0, static_cast<i64>(nv), 2048, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const Vec3 n = nrm ? nrm[i] : Vec3{0, 0, 0};
            float v = k_nan;
            // Muestra a `off` celdas hacia fuera; si cae entre sólidos, prueba 1 celda más lejos.
            for (int tries = 0; tries < 2 && v != v; ++tries) {
                const Vec3 q = pos[i] + n * (off + static_cast<float>(tries));
                Vec3 u;
                float rho;
                if (sample_prims(f, q, u, rho)) v = cp ? (rho - 1.0f) * k.cp_k : length(u) * k.inv_u;
            }
            col[i] = v == v ? map_color(sc, v) : k_unknown;
        }
    });
}

// Refinamiento local: cada vértice en la rejilla más fina que lo contiene, a `offset` celdas DE ESA rejilla (junto a un
// alerón de la rejilla fina, a 1 celda fina de la pared y no a 1 celda de la base: la succión se ve donde está).
void color_mesh(Mesh& mesh, const MultiField& m, const SurfaceParams& p) {
    if (m.n <= 1 || (p.mode != SurfaceMode::Cp && p.mode != SurfaceMode::Speed)) { color_mesh(mesh, m.base(), p); return; }
    const usize nv = mesh.pos.size();
    if (mesh.color.size() != nv) mesh.color.resize(nv);
    if (nv == 0) return;
    u32* CFD_RESTRICT col = mesh.color.data();
    const Vec3* pos = mesh.pos.data();
    const Vec3* nrm = mesh.nrm.size() == nv ? mesh.nrm.data() : nullptr;
    const Norm k(m.base().u_inf);
    const bool cp = p.mode == SurfaceMode::Cp;
    const ColorScale sc = p.scale;
    const float off = p.offset;
    constexpr u32 k_unknown = 0xFF8A8D93u;
    parallel_for(0, static_cast<i64>(nv), 2048, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const Vec3 n = nrm ? nrm[i] : Vec3{0, 0, 0};
            const int g = m.finest(pos[i]);
            const GridField& G = m.g[g];
            float v = k_nan;
            for (int tries = 0; tries < 2 && v != v; ++tries) {
                const Vec3 qb = pos[i] + n * ((off + static_cast<float>(tries)) * G.scale);
                Vec3 u;
                float rho;
                if (sample_prims(G.f, g ? G.to_grid(qb) : qb, u, rho)) v = cp ? (rho - 1.0f) * k.cp_k : length(u) * k.inv_u;
            }
            col[i] = v == v ? map_color(sc, v) : k_unknown;
        }
    });
}

} // namespace cfd::flowvis
