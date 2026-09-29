// ============================================================================
//  render/flowvis_lines.cpp — rastrillos, líneas de corriente (RK4 adaptativo)
//  y partículas de humo (RK2, SoA, anillo de reciclaje).
//
//  Ambos integran sobre FlowSampler (campo AoS FP16: 4 cargas de 128 bits por
//  muestra trilineal). Toda la aritmética de posición/velocidad va en registros
//  XMM (x, y, z, ·) → sin ida y vuelta a Vec3 dentro de los bucles.
// ============================================================================
#include "flowvis.hpp"
#include "../core/util.hpp"
#include <cmath>
#include <cstring>
#include <cstddef>
#include <limits>

namespace cfd::flowvis {

using lbm::FieldView;

// ============================================================================
//  Rastrillos
// ============================================================================
Rake rake_upstream(const Aabb& obj, float gap, int nu, int nv, float margin) {
    const Vec3 s = obj.size();
    const float x = obj.lo.x - gap;
    const float y0 = obj.lo.y - margin * s.y, y1 = obj.hi.y + margin * s.y;
    const float z0 = max_(obj.lo.z - margin * s.z, 1.0f), z1 = obj.hi.z + margin * s.z;
    return Rake::grid({x, y0, z0}, {0, y1 - y0, 0}, {0, 0, z1 - z0}, nu, nv);
}
Rake rake_floor(const Aabb& obj, float gap, float z, int n, float margin) {
    const Vec3 s = obj.size();
    const float x = obj.lo.x - gap;
    return Rake::line({x, obj.lo.y - margin * s.y, z}, {x, obj.hi.y + margin * s.y, z}, n);
}
Rake rake_vertical(const Aabb& obj, float gap, float y, int n, float margin) {
    const Vec3 s = obj.size();
    const float x = obj.lo.x - gap;
    return Rake::line({x, y, max_(obj.lo.z - margin * s.z, 1.0f)}, {x, y, obj.hi.z + margin * s.z}, n);
}

namespace {

using isize = std::ptrdiff_t;

// ---------------------------------------------------------------------------
//  Utilidades XMM
// ---------------------------------------------------------------------------
CFD_INLINE __m128 xyz(__m128 v) { return _mm_blend_ps(v, _mm_setzero_ps(), 8); }       // w ← 0
CFD_INLINE __m128 dot3b(__m128 a, __m128 b) { return _mm_dp_ps(a, b, 0x7F); }            // x·x+y·y+z·z en los 4 carriles
CFD_INLINE float lane0(__m128 v) { return _mm_cvtss_f32(v); }
CFD_INLINE float lane3(__m128 v) { return _mm_cvtss_f32(_mm_permute_ps(v, 0xFF)); }
CFD_INLINE __m128 to_xmm(Vec3 p) { return _mm_setr_ps(p.x, p.y, p.z, 0.0f); }
CFD_INLINE Vec3 to_vec3(__m128 v) {
    alignas(16) float a[4];
    _mm_store_ps(a, v);
    return {a[0], a[1], a[2]};
}
// Dirección unitaria (rsqrt + 1 Newton ≈ 23 bits). Velocidad 0 → vector 0 (sin NaN).
CFD_INLINE __m128 unit(__m128 v) {
    const __m128 l2 = _mm_max_ps(dot3b(v, v), _mm_set1_ps(1e-30f));
    const __m128 y = _mm_rsqrt_ps(l2);
    const __m128 r = _mm_mul_ps(y, _mm_fnmadd_ps(_mm_mul_ps(l2, _mm_set1_ps(0.5f)), _mm_mul_ps(y, y), _mm_set1_ps(1.5f)));
    return _mm_mul_ps(xyz(v), r);
}

// (LineNorm: flowvis.cpp define otro Norm en el mismo espacio anónimo → choque en `make unity`)
struct LineNorm {
    float inv_u, inv_u2, cp_k;
    explicit LineNorm(float u) : inv_u(1.0f / u), inv_u2(1.0f / (u * u)), cp_k(2.0f / (3.0f * u * u)) {}
};

// Escalar de color de una muestra (ux, uy, uz, ρ-1).
template <Quantity Q>
CFD_INLINE float sample_scalar(__m128 s, const LineNorm& k) {
    if constexpr (Q == Quantity::Ux) return lane0(s) * k.inv_u;
    else if constexpr (Q == Quantity::Uz) return _mm_cvtss_f32(_mm_permute_ps(s, 0xAA)) * k.inv_u;
    else if constexpr (Q == Quantity::Cp) return lane3(s) * k.cp_k;
    else if constexpr (Q == Quantity::Cp0) {
        const float r1 = lane3(s);
        return r1 * k.cp_k + (1.0f + r1) * lane0(dot3b(s, s)) * k.inv_u2;
    } else return std::sqrt(lane0(dot3b(s, s))) * k.inv_u;   // Speed (y resto)
}

// Muestreadores intercambiables para la plantilla del integrador.
struct PackedSampler {
    const FlowSampler& s;
    CFD_INLINE __m128 sample(__m128 p) const { return s.sample(p); }
    CFD_INLINE bool outside(__m128 p) const { return s.outside(p); }
    CFD_INLINE bool solid(__m128 p) const { return s.solid(p); }
    CFD_INLINE void prefetch(float x, float y, float z) const { s.prefetch(x, y, z); }
    int nx() const { return s.nx; }
    float u_inf() const { return s.u_inf; }
};
struct MultiAdapter {   // refinamiento local: la rejilla más fina en cada punto
    const MultiSampler& s;
    CFD_INLINE __m128 sample(__m128 p) const { return s.sample(p); }
    CFD_INLINE bool outside(__m128 p) const { return s.outside(p); }
    CFD_INLINE bool solid(__m128 p) const { return s.solid(p); }
    CFD_INLINE void prefetch(float x, float y, float z) const { s.prefetch(x, y, z); }
    int nx() const { return s.nx; }
    float u_inf() const { return s.u_inf; }
};
struct RefSampler {   // FP32 SoA vía FieldView::velocity() (referencia)
    const FieldView& f;
    alignas(16) float mx[4];
    explicit RefSampler(const FieldView& fv) : f(fv) {
        mx[0] = static_cast<float>(f.nx - 1); mx[1] = static_cast<float>(f.ny - 1); mx[2] = static_cast<float>(f.nz - 1); mx[3] = 1.0f;
    }
    CFD_INLINE __m128 sample(__m128 p) const {
        const Vec3 q = to_vec3(p);
        const Vec3 u = f.velocity(q);
        return _mm_setr_ps(u.x, u.y, u.z, f.sample(f.rho, q) - 1.0f);
    }
    CFD_INLINE bool outside(__m128 p) const {
        const __m128 bad = _mm_or_ps(_mm_cmp_ps(p, _mm_setzero_ps(), _CMP_NGE_UQ), _mm_cmp_ps(p, _mm_load_ps(mx), _CMP_NLE_UQ));
        return (_mm_movemask_ps(bad) & 7) != 0;
    }
    CFD_INLINE bool solid(__m128 p) const {
        const Vec3 q = to_vec3(p);
        return f.solid(static_cast<int>(q.x + 0.5f), static_cast<int>(q.y + 0.5f), static_cast<int>(q.z + 0.5f));
    }
    int nx() const { return f.nx; }
    float u_inf() const { return f.u_inf; }
};

struct TraceConsts {
    float h0, hmin, hmax, cos_max, cos_grow, min_sp2;
    LineNorm k;
    ColorScale sc;
};

// ---------------------------------------------------------------------------
//  Integrador de líneas de corriente con K "carriles" entrelazados.
//
//  Cada paso RK4 son 4 muestras DEPENDIENTES (k2 ← k1, k3 ← k2, ...): una sola
//  línea es una cadena de latencia pura (~40 ns por muestra). Con K líneas
//  independientes avanzando en paralelo ETAPA A ETAPA (k2 de todas, luego k3
//  de todas, ...) el núcleo fuera de orden solapa sus cargas y FMA → más
//  rendimiento por hilo sin SIMD horizontal (entrelazado de cadenas / ILP).
//  Un "trabajo" = (semilla, sentido). Al terminar una línea, su carril toma el
//  siguiente trabajo del rango (no hay carriles ociosos hasta el final).
// ---------------------------------------------------------------------------
#ifndef FLOWVIS_SL_LANES
#define FLOWVIS_SL_LANES 4
#endif

struct JobCtx {
    const Vec3* seeds;
    usize ns, slot;
    int ms;
    bool both;
    Vec3* pts;
    u32* cols;
    u32* nb;   // puntos hacia atrás por semilla (incluida la semilla)
    u32* nf;   // puntos hacia delante por semilla (0 = semilla inválida)
};

template <Quantity CQ, int K, class S>
CFD_INLINE void trace_jobs(const S& s, const TraceConsts& c, const JobCtx& J, i64 lo, i64 hi) {
    struct Lane {
        __m128 p, smp, k1, k2, k3, k4;
        float h, sgn;
        isize center;
        int stride, n;
        usize seed;
        bool back;
    };
    Lane L[K]{};   // inicializado: evita avisos -Wmaybe-uninitialized (coste nulo: una vez por trozo)
    u32 act = 0;
    i64 next = lo;
    // Carga el siguiente trabajo válido en el carril l. false si no quedan.
    auto start = [&](int l) -> bool {
        while (next < hi) {
            const i64 j = next++;
            const bool back = static_cast<usize>(j) >= J.ns;
            const usize i = back ? static_cast<usize>(j) - J.ns : static_cast<usize>(j);
            const isize center = static_cast<isize>(i * J.slot) + (J.both ? J.ms - 1 : 0);
            const __m128 p = to_xmm(J.seeds[i]);
            if (s.outside(p) || s.solid(p)) { (back ? J.nb : J.nf)[i] = back ? 1 : 0; continue; }
            Lane& a = L[l];
            a.p = p; a.smp = s.sample(p); a.h = c.h0; a.sgn = back ? -1.0f : 1.0f;
            a.stride = back ? -1 : 1; a.center = center; a.seed = i; a.back = back; a.n = 1;
            if (!back) {   // el punto semilla lo escribe sólo el trabajo hacia delante
                J.pts[center] = to_vec3(p);
                J.cols[center] = map_color(c.sc, sample_scalar<CQ>(a.smp, c.k));
            }
            return true;
        }
        return false;
    };
    auto finish = [&](int l) { const Lane& a = L[l]; (a.back ? J.nb : J.nf)[a.seed] = static_cast<u32>(a.n); };
#pragma GCC unroll 8
    for (int l = 0; l < K; ++l) if (start(l)) act |= 1u << l;
    while (act) {
        // Condiciones de parada al inicio del paso (estancamiento / sin puntos libres) + k1.
#pragma GCC unroll 8
        for (int l = 0; l < K; ++l) {
            if (!(act >> l & 1)) continue;
            for (;;) {
                Lane& a = L[l];
                const __m128 u = xyz(a.smp);
                if (a.n < J.ms && lane0(dot3b(u, u)) >= c.min_sp2) { a.k1 = unit(u); break; }
                finish(l);
                if (!start(l)) { act &= ~(1u << l); break; }
            }
        }
        // Etapas RK4 entrelazadas; los carriles con giro excesivo repiten con medio paso.
        u32 r = act;
        while (r) {
#pragma GCC unroll 8
            for (int l = 0; l < K; ++l) if (r >> l & 1) { Lane& a = L[l]; a.k2 = unit(s.sample(_mm_fmadd_ps(a.k1, _mm_set1_ps(0.5f * a.h * a.sgn), a.p))); }
#pragma GCC unroll 8
            for (int l = 0; l < K; ++l) if (r >> l & 1) { Lane& a = L[l]; a.k3 = unit(s.sample(_mm_fmadd_ps(a.k2, _mm_set1_ps(0.5f * a.h * a.sgn), a.p))); }
#pragma GCC unroll 8
            for (int l = 0; l < K; ++l) if (r >> l & 1) { Lane& a = L[l]; a.k4 = unit(s.sample(_mm_fmadd_ps(a.k3, _mm_set1_ps(a.h * a.sgn), a.p))); }
            u32 nr = 0;
#pragma GCC unroll 8
            for (int l = 0; l < K; ++l) {
                if (!(r >> l & 1)) continue;
                Lane& a = L[l];
                if (lane0(dot3b(a.k1, a.k4)) < c.cos_max && a.h > c.hmin) { a.h = max_(a.h * 0.5f, c.hmin); nr |= 1u << l; }
            }
            r = nr;
        }
        // Avance: p += h/6 (k1 + 2k2 + 2k3 + k4); fin de línea al salir del dominio o entrar en sólido.
#pragma GCC unroll 8
        for (int l = 0; l < K; ++l) {
            if (!(act >> l & 1)) continue;
            Lane& a = L[l];
            const __m128 sum = _mm_add_ps(_mm_add_ps(a.k1, a.k4), _mm_mul_ps(_mm_add_ps(a.k2, a.k3), _mm_set1_ps(2.0f)));
            const __m128 pn = _mm_fmadd_ps(sum, _mm_set1_ps(a.h * a.sgn * (1.0f / 6.0f)), a.p);
            if (lane0(dot3b(a.k1, a.k4)) > c.cos_grow) a.h = min_(a.h * 1.25f, c.hmax);   // tramo recto: alarga el paso
            if (s.outside(pn) || s.solid(pn)) {
                finish(l);
                if (!start(l)) act &= ~(1u << l);
                continue;
            }
            a.smp = s.sample(pn);
            const isize o = a.center + static_cast<isize>(a.n) * a.stride;
            J.pts[o] = to_vec3(pn);
            J.cols[o] = map_color(c.sc, sample_scalar<CQ>(a.smp, c.k));
            a.p = pn;
            ++a.n;
        }
    }
}

} // namespace

// ============================================================================
//  Streamlines
// ============================================================================
void Streamlines::set_rakes(std::span<const Rake> rakes) {
    seeds_.clear();
    for (const Rake& r : rakes)
        for (int i = 0; i < r.count(); ++i) seeds_.push_back(r.seed(i));
}
void Streamlines::set_seeds(std::span<const Vec3> seeds) { seeds_.assign(seeds.begin(), seeds.end()); }

usize Streamlines::total_points() const {
    usize s = 0;
    for (usize i = 0; i < n_lines_; ++i) s += counts_[i];
    return s;
}

template <class S>
void Streamlines::compute_impl(const S& s) {
    const StreamlineParams& P = params;
    const int ms = max_(P.max_steps, 2);
    const bool both = P.both_directions;
    const usize slot = static_cast<usize>(both ? 2 * ms - 1 : ms);
    const usize ns = seeds_.size();
    const usize need = slot * ns;
    if (pts_.size() < need) { pts_.resize(need); col_.resize(need); }
    if (starts_.size() < ns) { starts_.resize(ns > 0 ? ns : 1); counts_.resize(ns > 0 ? ns : 1); }
    n_lines_ = 0;
    if (ns == 0) return;

    const float u = s.u_inf();
    const float hmax = max_(P.max_step, P.min_step);
    TraceConsts c{clamp_(P.step, P.min_step, hmax), max_(P.min_step, 1e-3f), hmax,
                  std::cos(P.max_turn_deg * k_deg2rad), std::cos(0.25f * P.max_turn_deg * k_deg2rad),
                  sq(P.min_speed * u), LineNorm(u), P.scale};
    // Temporales por semilla: nf en counts_, nb en starts_ (se combinan y compactan después).
    u32* st = starts_.data();
    u32* cn = counts_.data();
    const Quantity cq = P.color_by;
    const JobCtx J{seeds_.data(), ns, slot, ms, both, pts_.data(), col_.data(), st, cn};
    if (!both) for (usize i = 0; i < ns; ++i) st[i] = 1;
    const i64 njobs = static_cast<i64>(both ? 2 * ns : ns);
    constexpr int K = FLOWVIS_SL_LANES;

    auto run = [&]<Quantity CQ>() {
        parallel_for(0, njobs, 4 * K, [&](i64 lo, i64 hi) { trace_jobs<CQ, K>(s, c, J, lo, hi); });
    };
    switch (cq) {
        case Quantity::Ux: run.template operator()<Quantity::Ux>(); break;
        case Quantity::Uz: run.template operator()<Quantity::Uz>(); break;
        case Quantity::Cp: run.template operator()<Quantity::Cp>(); break;
        case Quantity::Cp0: run.template operator()<Quantity::Cp0>(); break;
        default: run.template operator()<Quantity::Speed>(); break;
    }
    // Combina ambos sentidos: inicio = centro - (nb-1), cantidad = nf + nb - 1.
    for (usize i = 0; i < ns; ++i) {
        const usize center = i * slot + (both ? static_cast<usize>(ms - 1) : 0);
        const u32 nf = cn[i], nb = st[i];
        st[i] = static_cast<u32>(center - (nb - 1));
        cn[i] = nf ? nf + nb - 1 : 0;
    }
    // Compacta la lista: sólo líneas con ≥ 2 puntos (los puntos no se mueven).
    usize m = 0;
    for (usize i = 0; i < ns; ++i)
        if (cn[i] >= 2) { st[m] = st[i]; cn[m] = cn[i]; ++m; }
    n_lines_ = m;
}

void Streamlines::compute(const FlowSampler& s) {
    if (!s.ready()) { n_lines_ = 0; return; }
    compute_impl(PackedSampler{s});
}
void Streamlines::compute(const MultiSampler& s) {
    if (!s.ready()) { n_lines_ = 0; return; }
    compute_impl(MultiAdapter{s});
}
void Streamlines::compute_reference(const FieldView& f) {
    if (!f.valid() || !f.flags) { n_lines_ = 0; return; }
    compute_impl(RefSampler(f));
}

// ============================================================================
//  Particles
// ============================================================================
#ifndef FLOWVIS_PREFETCH
#define FLOWVIS_PREFETCH 8   // distancia de prefetch en partículas (0 = desactivado)
#endif
namespace {
constexpr usize k_chunk = 4096;   // partículas por trozo de trabajo (≈ 64 KB de SoA)
constexpr float k_dead = std::numeric_limits<float>::infinity();

} // namespace

void Particles::ensure_capacity() {
    const usize cap = (static_cast<usize>(max_(params.capacity, 8)) + 7) & ~usize{7};
    if (cap == cap_) return;
    cap_ = cap;
    x_.resize(cap); y_.resize(cap); z_.resize(cap); age_.resize(cap);
    age_.fill(k_dead);
    tmp_pts_.resize(cap); out_pts_.resize(cap);
    tmp_col_.resize(cap); out_col_.resize(cap);
    const usize nch = (cap + k_chunk - 1) / k_chunk;
    chunk_cnt_.resize(nch);
    chunk_dead_.resize(nch * 3);
    head_ = 0;
    stats_ = {};
}

void Particles::set_emitters(std::span<const Rake> rakes) {
    seeds_.clear();
    for (const Rake& r : rakes)
        for (int i = 0; i < r.count(); ++i) seeds_.push_back(r.seed(i));
    emit_cursor_ = 0;
}

void Particles::reset() {
    if (cap_) age_.fill(k_dead);
    head_ = 0;
    emit_acc_ = 0.0;
    stats_ = {};
}

void Particles::step(const FlowSampler& s, float lattice_steps) {
    ensure_capacity();
    if (!s.ready()) return;
    step_impl(PackedSampler{s}, lattice_steps);
}
void Particles::step(const MultiSampler& s, float lattice_steps) {
    ensure_capacity();
    if (!s.ready()) return;
    step_impl(MultiAdapter{s}, lattice_steps);
}

template <class S>
void Particles::step_impl(const S& s, float lattice_steps) {
    if (!rng_init_) { rng_ = params.seed; rng_init_ = true; }
    const float dt = lattice_steps * params.time_scale;
    if (!(dt > 0.0f)) return;   // pausa: se conserva la salida anterior
    const float u = s.u_inf();
    age_max_eff_ = params.max_age > 0.0f ? params.max_age : 1.25f * static_cast<float>(s.nx()) / u;
    rate_eff_ = params.rate > 0.0f ? params.rate : 0.85f * static_cast<float>(cap_) / age_max_eff_;
    const float amax = age_max_eff_;

    // ---- 1. Emisión (serie; pocas por cuadro). Posición "virtual" aguas arriba:
    //         la partícula nacida en la fracción `frac` del intervalo se coloca en
    //         semilla - u·dt·(1-frac); la advección del cuadro la deja en
    //         semilla + u·dt·frac → emisión continua sin pulsos ni pasada extra.
    if (!seeds_.empty()) {
        emit_acc_ += static_cast<double>(rate_eff_) * static_cast<double>(dt);
        usize n_emit = static_cast<usize>(emit_acc_);
        emit_acc_ -= static_cast<double>(n_emit);
        if (n_emit > cap_) n_emit = cap_;
        WyRand rng(rng_);
        const float j = params.jitter;
        for (usize e = 0; e < n_emit; ++e) {
            const Vec3 sd = seeds_[static_cast<usize>(emit_cursor_++ % seeds_.size())];
            const Vec3 q{sd.x + rng.uniform(-j, j), sd.y + rng.uniform(-j, j), sd.z + rng.uniform(-j, j)};
            const float back = dt * (1.0f - rng.uniform());
            __m128 p = to_xmm(q);
            float age0 = -back;
            if (s.outside(p)) continue;
            const __m128 pv = _mm_fnmadd_ps(xyz(s.sample(p)), _mm_set1_ps(back), p);
            if (!s.outside(pv)) p = pv; else age0 = 0.0f;
            const usize slot = head_;
            head_ = head_ + 1 == cap_ ? 0 : head_ + 1;
            if (age_[slot] < amax) ++stats_.overwritten;
            const Vec3 pp = to_vec3(p);
            x_[slot] = pp.x; y_[slot] = pp.y; z_[slot] = pp.z; age_[slot] = age0;
            ++stats_.emitted;
        }
        rng_ = rng.s;
    }

    // ---- 2. Advección RK2 (punto medio) + reciclaje + color + salida por trozos.
    const int nsub = clamp_(static_cast<int>(std::ceil(dt * 1.5f * u / max_(params.max_cells_per_substep, 0.1f))), 1, max_(params.max_substeps, 1));
    const float h = dt / static_cast<float>(nsub);
    const __m128 hv = _mm_set1_ps(h), hh = _mm_set1_ps(0.5f * h);
    const usize nch = (cap_ + k_chunk - 1) / k_chunk;
    const LineNorm k(u);
    const ColorScale sc = params.scale;
    const Quantity cq = params.color_by;
    const float inten = clamp_(params.intensity, 0.0f, 1.0f);   // (u32) de un float negativo sería UB
    const float inv_amax = 1.0f / amax;
    float* CFD_RESTRICT px = x_.data();
    float* CFD_RESTRICT py = y_.data();
    float* CFD_RESTRICT pz = z_.data();
    float* CFD_RESTRICT pa = age_.data();
    Vec3* CFD_RESTRICT tp = tmp_pts_.data();
    u32* CFD_RESTRICT tc = tmp_col_.data();

    auto run = [&]<Quantity CQ>() {
        parallel_for(0, static_cast<i64>(nch), 1, [&](i64 lo, i64 hi) {
            for (i64 ch = lo; ch < hi; ++ch) {
                const usize b = static_cast<usize>(ch) * k_chunk, e = min_(b + k_chunk, cap_);
                u32 cnt = 0;
                u64 d_out = 0, d_solid = 0, d_age = 0;
                for (usize i = b; i < e; ++i) {
#if FLOWVIS_PREFETCH > 0
                    // Prefetch de la celda de la partícula i+D (su posición actual = la 1.ª muestra RK2).
                    if (i + FLOWVIS_PREFETCH < e) s.prefetch(px[i + FLOWVIS_PREFETCH], py[i + FLOWVIS_PREFETCH], pz[i + FLOWVIS_PREFETCH]);
#endif
                    float a = pa[i];
                    if (!(a < amax)) continue;                     // muerta
                    __m128 p = _mm_setr_ps(px[i], py[i], pz[i], 0.0f);
                    __m128 s2 = _mm_setzero_ps();
                    for (int sub = 0; sub < nsub; ++sub) {
                        const __m128 s1 = s.sample(p);
                        const __m128 pm = _mm_fmadd_ps(xyz(s1), hh, p);
                        s2 = s.sample(pm);
                        p = _mm_fmadd_ps(xyz(s2), hv, p);
                    }
                    if (s.outside(p)) { pa[i] = k_dead; ++d_out; continue; }
                    if (s.solid(p)) { pa[i] = k_dead; ++d_solid; continue; }
                    a += dt;
                    if (a >= amax) { pa[i] = k_dead; ++d_age; continue; }
                    pa[i] = a;
                    const Vec3 q = to_vec3(p);
                    px[i] = q.x; py[i] = q.y; pz[i] = q.z;
                    // Desvanecido: entrada rápida (5% de la vida) y salida cuadrática.
                    const float r = a * inv_amax;
                    const float fade = saturate(r * 20.0f) * (1.0f - r * r);
                    // RGB = mapa de color; A = desvanecido × intensidad (el rasterizador multiplica por A,
                    // tanto en aditivo como en mezcla alfa).
                    const u32 c = map_color(sc, sample_scalar<CQ>(s2, k));
                    const usize o = b + cnt++;
                    tp[o] = q;
                    tc[o] = (c & 0x00FFFFFFu) | (static_cast<u32>(min_(fade * inten, 1.0f) * 255.0f + 0.5f) << 24);
                }
                chunk_cnt_[static_cast<usize>(ch)] = cnt;
                chunk_dead_[3 * static_cast<usize>(ch) + 0] = d_out;
                chunk_dead_[3 * static_cast<usize>(ch) + 1] = d_solid;
                chunk_dead_[3 * static_cast<usize>(ch) + 2] = d_age;
            }
        });
    };
    switch (cq) {
        case Quantity::Ux: run.template operator()<Quantity::Ux>(); break;
        case Quantity::Uz: run.template operator()<Quantity::Uz>(); break;
        case Quantity::Cp: run.template operator()<Quantity::Cp>(); break;
        case Quantity::Cp0: run.template operator()<Quantity::Cp0>(); break;
        default: run.template operator()<Quantity::Speed>(); break;
    }

    // ---- 3. Compactación: prefijo por trozo (serie, ~100 trozos) + copias en paralelo.
    usize total = 0;
    for (usize ch = 0; ch < nch; ++ch) {
        const u32 cnt = chunk_cnt_[ch];
        chunk_cnt_[ch] = static_cast<u32>(total);   // se reutiliza como desplazamiento de salida
        total += cnt;
        stats_.died_outside += chunk_dead_[3 * ch + 0];
        stats_.died_solid += chunk_dead_[3 * ch + 1];
        stats_.died_age += chunk_dead_[3 * ch + 2];
    }
    const usize tot = total;
    parallel_for(0, static_cast<i64>(nch), 2, [&](i64 lo, i64 hi) {
        for (i64 ch = lo; ch < hi; ++ch) {
            const usize c = static_cast<usize>(ch);
            const usize off = chunk_cnt_[c];
            const usize end = c + 1 < nch ? chunk_cnt_[c + 1] : tot;
            const usize n = end - off;
            if (!n) continue;
            std::memcpy(static_cast<void*>(out_pts_.data() + off), tp + c * k_chunk, n * sizeof(Vec3));
            std::memcpy(out_col_.data() + off, tc + c * k_chunk, n * sizeof(u32));
        }
    });
    stats_.alive = tot;
}

} // namespace cfd::flowvis
