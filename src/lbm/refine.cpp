// ============================================================================
//  lbm/refine.cpp — refinamiento local por bloques (2:1, anidados) del solver D3Q19.
//
//  DISEÑO (docs/FISICA.md §1.6)
//  ----------------------------
//  * Cada nivel es una rejilla UNIFORME (otra instancia de Solver::Impl) con el mismo kernel AVX2 Esoteric-Pull,
//    las mismas clases de bloque, contorno (Bouzidi + modelo Slip), ruedas y cinta. Una caja fina cubre un bloque
//    [lo, hi] de celdas de su padre partiendo cada una en 2×2×2 (centradas: la celda fina i está en
//    lo − ¾ + ½·i en coordenadas del padre); i = 0 y i = n−1 son la capa FANTASMA.
//  * Escalado acústico: dx/2, dt/2, misma u de red, ν_red ×2 (τ − ½ ×2), Smagorinsky con Δ = 1 celda de cada nivel
//    (la escala de filtro es la dx local), ley de pared con la ν de red de cada nivel. 2 subpasos finos por paso.
//  * Acoplamiento por MOMENTOS con reescalado del no equilibrio (Dupuis & Chopard 2003; Lagrava et al. 2012, en su
//    variante centrada en celdas):
//      - grueso → fino: el post-colisión de cada fantasma se reconstruye (misma colisión regularizada que el kernel)
//        a partir de ρ, u y X = Π^neq/(ρτ) interpolados TRILINEALMENTE (2.º orden) sobre las 8 celdas del padre que lo
//        rodean: las activas del padre, interpoladas LINEALMENTE en el tiempo entre T y T+1 (subpaso ½), y las
//        cubiertas, con la media actual de sus 8 celdas finas. X es proporcional al tensor de deformación × dt →
//        X_fino = X_grueso/2 y la τ del fantasma sale de su propio Smagorinsky: τ = τ0 + ¼K|X|.
//      - fino → grueso: la capa de celdas cubiertas del padre junto a la interfaz ("esclavas") recibe en cada paso
//        del padre el post-colisión reconstruido con la media (ρ, j, X) de sus 8 celdas finas (X ×2). Las cubiertas
//        más adentro ("muertas") no se procesan ni se leen.
//      Esclavas, muertas y fantasmas son sólidos con id 0 para el kernel (no se procesan: sus conjuntos de acceso
//      Esoteric-Pull sólo los escriben estas pasadas) y NO son pared para el contorno ni para las fuerzas.
//  * Orden de un paso del padre en T (invariante: padre pre-colisión en T, hija en 2T con los fantasmas de 2T−1):
//      kernel(padre, T) → R ← medias finas (2T) → esclavas post@T → fantasmas del propio padre (si es hija) →
//      contorno del padre (T) → M_{T+1} ← padre pre@T+1 → 2 × [R (2T+s) → paso de la hija con M(T + s/2)].
//  * Fuerzas: cada rejilla calcula el intercambio de momento en sus nodos; un enlace del padre cuyo punto medio cae
//    dentro de una caja fina NO cuenta (lo cuenta la fina). Se suman en unidades de la red base (×scale², ×scale³).
// ============================================================================
#include "solver.hpp"
#include "solver_impl.hpp"
#include "../core/threadpool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace cfd::lbm {
namespace {

// Parámetros de la colisión de una rejilla que necesitan las pasadas de interfaz.
struct CollParams {
    float K = 0;        // 18√2 C_s² (Smagorinsky)
    bool bulk = false;  // viscosidad de volumen propia
    float taub = 1;     // τ_b = 1/ω_b
    float omcb = 0;     // 1 − ω_b
};
CollParams coll_params(const Config& c) {
    CollParams p;
    p.K = 18.0f * std::sqrt(2.0f) * c.cs_smag * c.cs_smag;
    p.bulk = c.bulk_omega > 0.0f && c.collision != Collision::BGK;
    const float wb = std::clamp(c.bulk_omega, 0.01f, 1.99f);
    p.taub = 1.0f / wb;
    p.omcb = 1.0f - wb;
    return p;
}

// Poblaciones PRE-colisión de la celda n tal como las ve el kernel en el paso `step` (unidades de almacenamiento): con
// las correcciones de pared móvil que el kernel suma AL CARGAR (cinta: ±6w₉u_g en las direcciones 9/16 de las celdas
// kGroundOnly; otras paredes móviles con el rebote implícito: Ladd, con la corrección de masa de las impermeables).
// Sin esto la cantidad de movimiento de las celdas sobre la cinta sale u/3 más baja (medido: el flujo uniforme sobre
// la cinta se frenaba dentro de una caja apoyada en el suelo).
struct PreLoader {
    const void* ld[L::Q];
    const u8* F;
    const u8* SI;
    const i64* off;
    int nx, ny;
    i64 nxny;
    float ug;           // velocidad de la cinta en ese paso
    bool mexp;          // paredes móviles ≠ cinta con rebote interpolado explícito (no suman Ladd en el kernel)
    Motion mot[256];
    PreLoader(const Solver::Impl& G, u64 step) {
        void* st[L::Q];
        G.step_ptrs(step, ld, st);
        F = G.flags.data(); SI = G.sid.data(); off = G.off;
        nx = G.nx; ny = G.ny; nxny = G.nxny;
        G.fill_motion(step, mot);
        ug = mot[k_ground_id].v.x;
        mexp = G.cfg.bounce == BounceBack::Interpolated;
    }
    template <Precision P>
    CFD_INLINE void load(i64 n, float* CFD_RESTRICT f) const {
#pragma GCC unroll 19
        for (int k = 0; k < L::Q; ++k) f[k] = load_s<P>(ld[k], n);
        const u8 fl = F[n];
        if (!(fl & kNearMoving)) return;
        if (fl & kGroundOnly) {
            const float d = kSc<P> * 6.0f * L::w[9] * ug;
            f[9] += d;
            f[16] -= d;
            return;
        }
        const int x = static_cast<int>(n % nx), y = static_cast<int>((n / nx) % ny), z = static_cast<int>(n / nxny);
        float S = 0.0f, Wm = 0.0f;
        u32 mv = 0;
        for (int i = 1; i < L::Q; ++i) {
            const i64 s = n - off[i];
            if (!(F[s] & kMoving)) continue;
            if (mexp && SI[s] != k_ground_id) continue;
            const Vec3 uw = wall_velocity(mot[SI[s]], float(x - L::c[i][0]), float(y - L::c[i][1]), float(z - L::c[i][2]));
            const float dl = kSc<P> * 6.0f * L::w[i] * (float(L::c[i][0]) * uw.x + float(L::c[i][1]) * uw.y + float(L::c[i][2]) * uw.z);
            f[i] += dl;
            if (mot[SI[s]].conserve) { S += dl; Wm += L::w[i]; mv |= 1u << i; }
        }
        if (Wm > 0.0f) {
            const float cS = S / Wm;
            while (mv) { const int i = std::countr_zero(mv); mv &= mv - 1; f[i] -= cS * L::w[i]; }
        }
    }
};

// Estado compacto (kMS floats) a partir de las poblaciones PRE-colisión f (unidades de almacenamiento):
// o = {ρ, u, X} con X = Π^neq_desv/(ρτ) + (tr Π^neq/3)/(ρτ_b)·δ, τ la efectiva de la cortante (con Smagorinsky, misma
// fórmula que el kernel) y todo × xs (paso a las unidades de tiempo de la rejilla fina de la interfaz).
// Estado compacto desde ρ, u y Π^neq (unidades reales).
CFD_INLINE void state_from_mom(float rho, float ux, float uy, float uz, const Neq<float>& q, float tau0, const CollParams& cp, float xs,
                               float* CFD_RESTRICT o) {
    const float inv = 1.0f / rho;
    const float qq = q.xx * q.xx + q.yy * q.yy + q.zz * q.zz + 2.0f * (q.xy * q.xy + q.xz * q.xz + q.yz * q.yz);
    const float tau = 0.5f * (tau0 + std::sqrt(tau0 * tau0 + cp.K * std::sqrt(qq) * inv));
    const float tr = (q.xx + q.yy + q.zz) * (1.0f / 3.0f);
    const float a = xs * inv / tau;
    const float b = xs * inv / (cp.bulk ? cp.taub : tau);
    o[0] = rho; o[1] = ux; o[2] = uy; o[3] = uz;
    o[4] = (q.xx - tr) * a + tr * b;
    o[5] = (q.yy - tr) * a + tr * b;
    o[6] = (q.zz - tr) * a + tr * b;
    o[7] = q.xy * a; o[8] = q.xz * a; o[9] = q.yz * a;
}

template <Precision P>
CFD_INLINE void cell_state(const float* CFD_RESTRICT f, float tau0, const CollParams& cp, float xs, float* CFD_RESTRICT o) {
    auto ldf = [&](int k) { return f[k]; };
    const Mom<float> m = moments<float>(ldf);
    const float drho = (P == Precision::FP16S) ? m.drho * kInvScale : m.drho;
    const float rho = 1.0f + drho;
    const float inv = 1.0f / rho;
    const float invj = (P == Precision::FP16S) ? inv * kInvScale : inv;
    const float ux = m.jx * invj, uy = m.jy * invj, uz = m.jz * invj;
    const Neq<float> q = noneq<kSc<P>>(m, drho, rho, ux, uy, uz);
    state_from_mom(rho, ux, uy, uz, q, tau0, cp, xs, o);
}

// Escribe el post-colisión de la celda n (punteros st del paso) reconstruido desde el estado s (X × xs → unidades de
// la rejilla destino): Π^neq = ρ(τ X_desv + τ_b X_traza), τ = τ0 + ¼K|X_desv| (inversa exacta del Smagorinsky del
// kernel), y la MISMA colisión regularizada (2.º orden o recursiva) que el kernel. layer: capa de 2.º orden (sin el
// término de 3er orden de la RR).
template <Precision P, Collision C, bool Bulk>
CFD_INLINE void write_post(void* const* st, i64 n, const float* CFD_RESTRICT s, float xs, float tau0, const CollParams& cp, bool layer, u32 dmask) {
    const float rho = s[0], ux = s[1], uy = s[2], uz = s[3];
    const float xxx = s[4] * xs, xyy = s[5] * xs, xzz = s[6] * xs, xxy = s[7] * xs, xxz = s[8] * xs, xyz = s[9] * xs;
    const float t3 = (xxx + xyy + xzz) * (1.0f / 3.0f);
    const float dxx = xxx - t3, dyy = xyy - t3, dzz = xzz - t3;
    const float nrm = std::sqrt(dxx * dxx + dyy * dyy + dzz * dzz + 2.0f * (xxy * xxy + xxz * xxz + xyz * xyz));
    const float tau = tau0 + 0.25f * cp.K * nrm;
    const float tb = Bulk ? cp.taub : tau;
    const float rt = rho * tau, rb = rho * tb * t3;
    Neq<float> q;
    q.xx = rt * dxx + rb; q.yy = rt * dyy + rb; q.zz = rt * dzz + rb;
    q.xy = rt * xxy; q.xz = rt * xxz; q.yz = rt * xyz;
    const float omc = 1.0f - 1.0f / tau;
    auto ldz = [](int) { return 0.0f; };
    auto stf = [&](int k, float v) { if ((dmask >> k) & 1u) store_s<P>(st[k], n, v); };
    collide<C, kSc<P>, Bulk>(ldz, stf, rho - 1.0f, rho, ux, uy, uz, q, omc, cp.omcb, layer ? 0.0f : omc);
}

// ---- Rutas AVX2 (8 celdas consecutivas en memoria) ------------------------------------------------------------------
// Post-colisión de 8 celdas consecutivas n0..n0+7 desde el estado SoA s[m] (mismas fórmulas que write_post); lay: máscara
// de carriles en la capa de 2.º orden (sin el término de 3er orden de la RR).
template <Precision P, Collision C, bool Bulk>
CFD_INLINE void write_post8(void* const* st, i64 n0, const f8* s, float xs, float tau0, const CollParams& cp, f8 lay, u32 dmask) {
    const f8 X(xs);
    const f8 rho = s[0], ux = s[1], uy = s[2], uz = s[3];
    const f8 xxx = s[4] * X, xyy = s[5] * X, xzz = s[6] * X, xxy = s[7] * X, xxz = s[8] * X, xyz = s[9] * X;
    const f8 t3 = (xxx + xyy + xzz) * f8(1.0f / 3.0f);
    const f8 dxx = xxx - t3, dyy = xyy - t3, dzz = xzz - t3;
    const f8 nrm = simd::sqrt(dxx * dxx + dyy * dyy + dzz * dzz + f8(2.0f) * (xxy * xxy + xxz * xxz + xyz * xyz));
    const f8 tau = f8(tau0) + f8(0.25f * cp.K) * nrm;
    const f8 tb = Bulk ? f8(cp.taub) : tau;
    const f8 rt = rho * tau, rb = rho * tb * t3;
    Neq<f8> q;
    q.xx = rt * dxx + rb; q.yy = rt * dyy + rb; q.zz = rt * dzz + rb;
    q.xy = rt * xxy; q.xz = rt * xxz; q.yz = rt * xyz;
    const f8 omc = f8(1.0f) - f8(1.0f) / tau;
    auto ldz = [](int) { return f8::zero(); };
    auto stf = [&](int k, f8 v) {
        if (!((dmask >> k) & 1u)) return;
        if constexpr (P == Precision::FP32) v.store(static_cast<float*>(st[k]) + n0);
        else v.store_h(static_cast<u16*>(st[k]) + n0);
    };
    collide<C, kSc<P>, Bulk>(ldz, stf, rho - f8(1.0f), rho, ux, uy, uz, q, omc, f8(cp.omcb), f8(_mm256_andnot_ps(lay, omc)));
}

// Llama a f.template operator()<P, C, Bulk>() con la combinación de la configuración (BGK → regularizada: las
// interfaces no tienen poblaciones pre-colisión propias que relajar).
template <class F>
void with_kernel(const Config& c, F&& f) {
    const bool fp32 = c.precision == Precision::FP32;
    const bool rec = c.collision == Collision::Recursive;
    const bool bulk = c.bulk_omega > 0.0f && c.collision != Collision::BGK;
    if (fp32) {
        if (rec) { if (bulk) f.template operator()<Precision::FP32, Collision::Recursive, true>(); else f.template operator()<Precision::FP32, Collision::Recursive, false>(); }
        else { if (bulk) f.template operator()<Precision::FP32, Collision::Regularized, true>(); else f.template operator()<Precision::FP32, Collision::Regularized, false>(); }
    } else {
        if (rec) { if (bulk) f.template operator()<Precision::FP16S, Collision::Recursive, true>(); else f.template operator()<Precision::FP16S, Collision::Recursive, false>(); }
        else { if (bulk) f.template operator()<Precision::FP16S, Collision::Regularized, true>(); else f.template operator()<Precision::FP16S, Collision::Regularized, false>(); }
    }
}

constexpr float kRest[kMS] = {1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};   // equilibrio en reposo (sin datos)

} // namespace

// ---- Planificación ------------------------------------------------------------------------------------------------
bool Solver::Impl::is_covered(int x, int y, int z) const {
    for (const LevelBox& b : cover)
        if (x >= b.lo[0] && x <= b.hi[0] && y >= b.lo[1] && y <= b.hi[1] && z >= b.lo[2] && z <= b.hi[2]) return true;
    return false;
}

// Normaliza las cajas de c (en su sitio). Reglas: la padre va antes; márgenes ≥ 3 celdas a las caras del padre (sus
// celdas 1-2 alimentan la restricción hacia el abuelo y la plantilla usa lo−1); con suelo, lo[2] ≤ 1 → apoyada
// (lo[2] = 1); nx de la hija = 2(hi−lo+1)+2 múltiplo de 8; hermanas separadas ≥ 3 celdas (si no: CFD_CHECK).
void Solver::normalize_boxes(Config& c) {
    const int nb = std::clamp(c.n_boxes, 0, k_max_boxes);
    c.n_boxes = nb;
    std::vector<std::array<int, 3>> dims(static_cast<usize>(nb) + 1);
    std::vector<char> gat(static_cast<usize>(nb) + 1, 0);
    dims[0] = {c.nx, c.ny, c.nz};
    gat[0] = c.ground != GroundMode::None;
    for (int b = 0; b < nb; ++b) {
        LevelBox& B = c.boxes[b];
        CFD_CHECK(B.parent >= 0 && B.parent <= b, "lbm::Solver::init: la rejilla padre de una caja debe ir antes");
        const auto& pd = dims[static_cast<usize>(B.parent)];
        const bool ga = gat[static_cast<usize>(B.parent)] && B.lo[2] <= 1;
        gat[static_cast<usize>(b) + 1] = ga;
        for (int a = 0; a < 3; ++a) {
            B.lo[a] = (a == 2 && ga) ? 1 : std::max(B.lo[a], 3);
            B.hi[a] = std::min(B.hi[a], pd[static_cast<usize>(a)] - 4);
            CFD_CHECK(B.hi[a] > B.lo[a], "lbm::Solver::init: caja de refinamiento vacía tras los márgenes");
        }
        // nx fino = 2(hi−lo+1)+2 múltiplo de 8 ⇔ (hi−lo+1) ≡ 3 (mód 4): se crece hacia +x (y hacia −x si no cabe); si no
        // hay sitio, se encoge por +x. (Forma cerrada: el bucle anterior oscilaba ±1 sin fin con la caja pegada a ambos
        // márgenes.)
        {
            const int len = B.hi[0] - B.lo[0] + 1;
            const int grow = ((3 - len) % 4 + 4) % 4;
            const int room_hi = pd[0] - 4 - B.hi[0], room_lo = B.lo[0] - 3;
            if (grow <= room_hi + room_lo) {
                const int gh = std::min(grow, room_hi);
                B.hi[0] += gh;
                B.lo[0] -= grow - gh;
            } else {
                B.hi[0] -= (4 - grow) % 4;
            }
            CFD_CHECK(B.hi[0] > B.lo[0] && ((B.hi[0] - B.lo[0] + 1) & 3) == 3, "lbm::Solver::init: caja de refinamiento demasiado estrecha en x");
        }
        for (int a = 0; a < 3; ++a) dims[static_cast<usize>(b) + 1][static_cast<usize>(a)] = 2 * (B.hi[a] - B.lo[a] + 1) + 2;
        for (int o = 0; o < b; ++o) {
            const LevelBox& O = c.boxes[o];
            if (O.parent != B.parent) continue;
            bool sep = false;
            for (int a = 0; a < 3; ++a) sep |= B.lo[a] > O.hi[a] + 3 || O.lo[a] > B.hi[a] + 3;
            CFD_CHECK(sep, "lbm::Solver::init: cajas de refinamiento hermanas demasiado cerca (fusionarlas)");
        }
    }
}

// Normaliza las cajas y fija `cover` de la raíz.
void Solver::Impl::refine_plan(Config& c) {
    Solver::normalize_boxes(c);
    cover.clear();
    for (int b = 0; b < c.n_boxes; ++b)
        if (c.boxes[b].parent == 0) cover.push_back(c.boxes[b]);
}

void Solver::Impl::refine_create() {
    const Config& c = cfg;
    gid = 0; depth = 0; nsub = 1; org = Vec3(0, 0, 0); scale = 1.0f;
    for (int b = 0; b < c.n_boxes; ++b) {
        const LevelBox& B = c.boxes[b];
        Impl& P = grid(B.parent);
        auto kid = std::make_unique<Impl>();
        Impl& K = *kid;
        K.par = &P;
        K.gid = b + 1;
        K.depth = P.depth + 1;
        K.nsub = P.nsub * 2;
        K.box = B;
        K.nested = true;
        K.gattached = (P.par ? P.gattached : c.ground != GroundMode::None) && B.lo[2] == 1;
        K.scale = P.scale * 0.5f;
        K.org = P.org + Vec3(static_cast<float>(B.lo[0]) - 0.75f, static_cast<float>(B.lo[1]) - 0.75f, static_cast<float>(B.lo[2]) - 0.75f) * P.scale;
        for (int o = b + 1; o < c.n_boxes; ++o)
            if (c.boxes[o].parent == b + 1) K.cover.push_back(c.boxes[o]);
        Config kc = c;
        kc.n_boxes = 0;
        kc.nx = 2 * (B.hi[0] - B.lo[0] + 1) + 2;
        kc.ny = 2 * (B.hi[1] - B.lo[1] + 1) + 2;
        kc.nz = 2 * (B.hi[2] - B.lo[2] + 1) + 2;
        const float f = static_cast<float>(K.nsub);
        kc.nu = c.nu * f;                    // τ − ½ ×2 por nivel: misma viscosidad física
        kc.wall_nu = c.wall_nu * f;
        kc.ramp_steps = c.ramp_steps * K.nsub;
        kc.sponge_frac = 0.0f;
        kc.ground = K.gattached ? c.ground : GroundMode::None;
        K.tun = tun;
        K.setup(kc);
        P.kids.push_back(&K);
        owned.push_back(std::move(kid));
    }
    iface_dirty = true;
}

WallMotion Solver::Impl::to_grid(const WallMotion& m) const {
    WallMotion r = m;
    const float is = 1.0f / scale;
    r.omega = m.omega * scale;               // rad/paso: el paso de la rejilla es scale veces el de la base
    r.center = (m.center - org) * is;
    r.contact_z = m.contact_z < 0.0f ? m.contact_z : (m.contact_z - org.z) * is;
    return r;
}

// Propiedad de las fuerzas: el enlace n → n + c_k cuenta aquí salvo que su punto medio caiga dentro (cerrado) de una
// caja hija: allí lo cuenta la rejilla fina (que cubre [lo − ½, hi + ½] con sus celdas interiores).
void Solver::Impl::build_fown() {
    fown.assign(wnodes.size(), 0u);
    const LevelBox* B = cover.data();
    const usize nb = cover.size();
    parallel_for(0, static_cast<i64>(wnodes.size()), 1024, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const WNode& nd = wnodes[static_cast<usize>(i)];
            u32 own = 0, bits = nd.mask;
            while (bits) {
                const int k = std::countr_zero(bits);
                bits &= bits - 1;
                const float mx = nd.x + 0.5f * L::c[k][0], my = nd.y + 0.5f * L::c[k][1], mz = nd.z + 0.5f * L::c[k][2];
                bool in = false;
                for (usize b = 0; b < nb && !in; ++b)
                    in = mx >= B[b].lo[0] - 0.5f && mx <= B[b].hi[0] + 0.5f && my >= B[b].lo[1] - 0.5f && my <= B[b].hi[1] + 0.5f &&
                         mz >= B[b].lo[2] - 0.5f && mz <= B[b].hi[2] + 0.5f;
                if (!in) own |= 1u << k;
            }
            fown[static_cast<usize>(i)] = own;
        }
    });
}

// ---- Interfaz de una hija con su padre -----------------------------------------------------------------------------
void Solver::Impl::iface_build() {
    Impl& P = *par;
    const LevelBox& B = box;
    const u8* PF = P.flags.data();
    const u8* PS = P.sid.data();
    const u8* F = flags.data();
    const u8* SI = sid.data();
    auto pidx = [&](int x, int y, int z) { return static_cast<u32>(x + P.nx * (y + static_cast<i64>(P.ny) * z)); };
    // 1) Esclavas: cáscara de la caja en el padre (sin la cara de abajo si está apoyada en el suelo).
    rcells.clear();
    std::unordered_map<u32, u32> rmap;
    for (int z = B.lo[2]; z <= B.hi[2]; ++z)
        for (int y = B.lo[1]; y <= B.hi[1]; ++y)
            for (int x = B.lo[0]; x <= B.hi[0]; ++x) {
                const bool shell = x == B.lo[0] || x == B.hi[0] || y == B.lo[1] || y == B.hi[1] || z == B.hi[2] || (z == B.lo[2] && !gattached);
                if (!shell) continue;
                const u32 np = pidx(x, y, z);
                if (!(PF[np] & kSolid) || PS[np] != 0) continue;   // sólido del usuario: sigue siendo pared del padre
                IRcell r;
                r.np = np;
                const int ci = 2 * (x - B.lo[0]) + 1, cj = 2 * (y - B.lo[1]) + 1, ck = 2 * (z - B.lo[2]) + 1;
                r.c0 = static_cast<u32>(ci + nx * (cj + static_cast<i64>(ny) * ck));
                int nf = 0;
                for (int d = 0; d < 8; ++d) {
                    const i64 cc = r.c0 + (d & 1) + nx * ((d >> 1) & 1) + nxny * (d >> 2);
                    nf += !(F[cc] & kSolid);
                }
                r.nfl = static_cast<u8>(nf);
                r.layer = P.lay.size() ? P.lay[np] : 0;
                for (int k = 0; k < L::Q; ++k)
                    if (!(PF[static_cast<i64>(np) + P.off[k]] & kSolid)) r.dmask |= 1u << k;
                rmap.emplace(np, static_cast<u32>(rcells.size()));
                rcells.push_back(r);
            }
    // 1b) Bloques "tap": los bloques de 8 celdas con alguna hija fluida de una esclava guardan en el kernel sus momentos
    //     pre-colisión (kBlkTap): las medias de la restricción salen de ahí sin volver a leer las 19 poblaciones (en las caras x
    //     cada hija arrastraba 19 líneas de caché para 2 valores útiles: el paso era de ancho de banda).
    tapidx.resize(static_cast<usize>(N / 8));
    tapidx.fill(~0u);
    u32 nslots = 0;
    for (const IRcell& r : rcells)
        for (int d = 0; d < 8; ++d) {
            const i64 cc = r.c0 + (d & 1) + nx * ((d >> 1) & 1) + nxny * (d >> 2);
            if (F[cc] & kSolid) continue;
            u32& ti = tapidx[static_cast<usize>(cc >> 3)];
            if (ti == ~0u) ti = nslots++;
        }
    tap.resize(static_cast<usize>(std::max<u32>(nslots, 1)) * kTapFloats, true);
    for (i64 bl = 0; bl < N / 8; ++bl)
        if (tapidx[static_cast<usize>(bl)] != ~0u) cls[bl] |= kBlkTap;
    // 2) Fantasmas: plantilla trilineal sobre las 8 celdas del padre que rodean su centro.
    ghosts.clear();
    mcells.clear();
    std::unordered_map<u32, u32> mmap;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const bool face = x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == nz - 1 || (z == 0 && !gattached);
                if (!face) continue;
                const i64 n = x + nx * (y + static_cast<i64>(ny) * z);
                if (!(F[n] & kSolid) || SI[n] != 0) continue;   // sólido del usuario (o suelo): no es fantasma
                IGhost g;
                g.n = static_cast<u32>(n);
                g.layer = lay.size() ? lay[n] : 0;
                for (int k = 0; k < L::Q; ++k) {
                    const i64 m = n + off[k];
                    const int mx = x + L::c[k][0], my = y + L::c[k][1], mz = z + L::c[k][2];
                    if (mx >= 0 && my >= 0 && mz >= 0 && mx < nx && my < ny && mz < nz && !(F[m] & kSolid)) g.dmask |= 1u << k;
                }
                const float p[3] = {static_cast<float>(B.lo[0]) - 0.75f + 0.5f * x, static_cast<float>(B.lo[1]) - 0.75f + 0.5f * y,
                                    static_cast<float>(B.lo[2]) - 0.75f + 0.5f * z};
                int b0[3];
                float tt[3];
                for (int a = 0; a < 3; ++a) { b0[a] = static_cast<int>(std::floor(p[a])); tt[a] = p[a] - static_cast<float>(b0[a]); }
                u32 mi[8], ri[8];
                float mw[8], rw[8];
                int nm = 0, nr = 0;
                float wsum = 0.0f;
                for (int d = 0; d < 8; ++d) {
                    const int q[3] = {b0[0] + (d & 1), b0[1] + ((d >> 1) & 1), b0[2] + (d >> 2)};
                    float w = 1.0f;
                    for (int a = 0; a < 3; ++a) w *= ((d >> a) & 1) ? tt[a] : 1.0f - tt[a];
                    if (w <= 0.0f) continue;
                    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= P.nx || q[1] >= P.ny || q[2] >= P.nz) continue;
                    const u32 nq = pidx(q[0], q[1], q[2]);
                    if (PF[nq] & kSolid) {
                        if (PS[nq] != 0) continue;                     // pared del padre
                        const auto it = rmap.find(nq);                 // cubierta: media actual de sus hijas
                        if (it == rmap.end() || rcells[it->second].nfl == 0) continue;
                        ri[nr] = it->second; rw[nr] = w; ++nr;
                    } else if (PF[nq] & (kInlet | kOutlet)) {
                        continue;                                      // (fuera de contrato: márgenes)
                    } else {
                        auto [it, fresh] = mmap.emplace(nq, static_cast<u32>(mcells.size()));
                        if (fresh) mcells.push_back(nq);
                        mi[nm] = it->second; mw[nm] = w; ++nm;
                    }
                    wsum += w;
                }
                const float iw = wsum > 0.0f ? 1.0f / wsum : 0.0f;
                g.nm = static_cast<u8>(nm);
                g.nr = static_cast<u8>(nr);
                for (int e = 0; e < nm; ++e) { g.idx[e] = mi[e]; g.w[e] = mw[e] * iw; }
                for (int e = 0; e < nr; ++e) { g.idx[nm + e] = ri[e]; g.w[nm + e] = rw[e] * iw; }
                ghosts.push_back(g);
            }
    // Grupos AVX2 de escritura: 8 elementos seguidos de la lista que son 8 celdas consecutivas en memoria (fantasmas: filas
    // de las caras y/z; esclavas: 8 celdas seguidas del padre con sus 8 hijas fluidas).
    gvec.clear(); gsca.clear(); rvec.clear(); rsca.clear();
    for (usize i = 0; i < ghosts.size();) {
        bool ok = i + 8 <= ghosts.size();
        for (usize j = 1; ok && j < 8; ++j) ok = ghosts[i + j].n == ghosts[i].n + j;
        if (ok) { gvec.push_back(static_cast<u32>(i)); i += 8; }
        else { gsca.push_back(static_cast<u32>(i)); ++i; }
    }
    for (usize i = 0; i < rcells.size();) {
        bool ok = i + 8 <= rcells.size();
        for (usize j = 0; ok && j < 8; ++j) {
            const IRcell& r = rcells[i + j];
            ok = r.np == rcells[i].np + j && r.nfl == 8;
        }
        if (ok) { rvec.push_back(static_cast<u32>(i)); i += 8; }
        else { rsca.push_back(static_cast<u32>(i)); ++i; }
    }
    Mprev.resize(static_cast<usize>(kMS) * std::max<usize>(mcells.size(), 1), true);
    Mnext.resize(static_cast<usize>(kMS) * std::max<usize>(mcells.size(), 1), true);
    Rst.resize(static_cast<usize>(kMS) * std::max<usize>(rcells.size(), 1), true);
}

// Estado de las esclavas en el paso actual: media de sus hijas fluidas con los momentos pre-colisión que el kernel acaba
// de guardar (bloques kBlkTap). ρ y j se promedian (conservativo); Π^neq de la media = media de Π^neq de las hijas + la
// tensión de la dispersión de velocidades entre ellas (⟨ρuu⟩ − ρ̄ūū: lo que da promediar las poblaciones); la τ de
// Smagorinsky sale del Π^neq medio.
void Solver::Impl::iface_R() {
    const CollParams cp = coll_params(cfg);
    const u8* F = flags.data();
    const float t0 = tau0[1];   // rejilla fina: sin esponja (τ0 uniforme)
    float* R = Rst.data();
    const IRcell* rc = rcells.data();
    const float* TP = tap.data();
    const u32* TI = tapidx.data();
    const i64 n = static_cast<i64>(rcells.size());
    parallel_for(0, n, 512, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const IRcell& r = rc[i];
            float* o = R + kMS * i;
            float sr = 0, sjx = 0, sjy = 0, sjz = 0;
            float q[6] = {}, uu[6] = {};
            int cnt = 0;
            for (int d = 0; d < 8; ++d) {
                const i64 c = r.c0 + (d & 1) + nx * ((d >> 1) & 1) + nxny * (d >> 2);
                if (F[c] & kSolid) continue;
                const float* b = TP + static_cast<i64>(TI[c >> 3]) * kTapFloats + (c & 7);
                const float rh = b[0], ux = b[8], uy = b[16], uz = b[24];
                sr += rh; sjx += rh * ux; sjy += rh * uy; sjz += rh * uz;
                for (int m = 0; m < 6; ++m) q[m] += b[32 + 8 * m];
                uu[0] += rh * ux * ux; uu[1] += rh * uy * uy; uu[2] += rh * uz * uz;
                uu[3] += rh * ux * uy; uu[4] += rh * ux * uz; uu[5] += rh * uy * uz;
                ++cnt;
            }
            if (!cnt) { std::memcpy(o, kRest, sizeof kRest); continue; }
            const float ic = 1.0f / static_cast<float>(cnt), ir = 1.0f / sr;
            const float rho = sr * ic, ux = sjx * ir, uy = sjy * ir, uz = sjz * ir;
            Neq<float> qa;
            qa.xx = (q[0] + uu[0]) * ic - rho * ux * ux;
            qa.yy = (q[1] + uu[1]) * ic - rho * uy * uy;
            qa.zz = (q[2] + uu[2]) * ic - rho * uz * uz;
            qa.xy = (q[3] + uu[3]) * ic - rho * ux * uy;
            qa.xz = (q[4] + uu[4]) * ic - rho * ux * uz;
            qa.yz = (q[5] + uu[5]) * ic - rho * uy * uz;
            state_from_mom(rho, ux, uy, uz, qa, t0, cp, 1.0f, o);
        }
    });
}

// Estado PRE-colisión del padre en su paso `pstep` en las celdas activas de las plantillas (X ×½ → unidades finas).
void Solver::Impl::iface_M(Buffer<float>& dst, u64 pstep) {
    Impl& P = *par;
    const PreLoader pl(P, pstep);
    const CollParams cp = coll_params(P.cfg);
    const u32* mc = mcells.data();
    float* D = dst.data();
    const i64 n = static_cast<i64>(mcells.size());
    const float* t0 = P.tau0.data();
    const int pnx = P.nx;
    auto run = [&]<Precision Pr>() {
        parallel_for(0, n, 512, [&](i64 lo, i64 hi) {
            for (i64 i = lo; i < hi; ++i) {
                float f[L::Q];
                pl.load<Pr>(mc[i], f);
                cell_state<Pr>(f, t0[mc[i] % static_cast<u32>(pnx)], cp, 0.5f, D + kMS * i);
            }
        });
    };
    if (P.cfg.precision == Precision::FP32) run.template operator()<Precision::FP32>();
    else run.template operator()<Precision::FP16S>();
}

// Post-colisión de las esclavas del padre en su paso actual (X ×2 → unidades del padre).
void Solver::Impl::iface_restrict() {
    Impl& P = *par;
    const void* ld[L::Q];
    void* st[L::Q];
    P.step_ptrs(P.t, ld, st);
    const CollParams cp = coll_params(P.cfg);
    const IRcell* rc = rcells.data();
    const float* R = Rst.data();
    const u32* RV = rvec.data();
    const u32* RS = rsca.data();
    const i64 nv = static_cast<i64>(rvec.size()), ns = static_cast<i64>(rsca.size());
    const float* t0 = P.tau0.data();
    const int pnx = P.nx;
    with_kernel(P.cfg, [&]<Precision Pr, Collision C, bool Bulk>() {
        parallel_for(0, nv, 32, [&](i64 lo, i64 hi) {
            for (i64 v = lo; v < hi; ++v) {
                const i64 i0 = RV[v];
                alignas(32) float tmp[kMS][8];
                alignas(32) i32 lay[8];
                for (int l = 0; l < 8; ++l) {
                    for (int m = 0; m < kMS; ++m) tmp[m][l] = R[kMS * (i0 + l) + m];
                    lay[l] = rc[i0 + l].layer ? -1 : 0;
                }
                f8 sv[kMS];
                for (int m = 0; m < kMS; ++m) sv[m] = f8::load(tmp[m]);
                const u32 np = rc[i0].np;
                u32 dm = 0;
                for (int l = 0; l < 8; ++l) dm |= rc[i0 + l].dmask;
                write_post8<Pr, C, Bulk>(st, np, sv, 2.0f, t0[np % static_cast<u32>(pnx)], cp, f8(_mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(lay)))), dm);
            }
        });
        parallel_for(0, ns, 256, [&](i64 lo, i64 hi) {
            for (i64 j = lo; j < hi; ++j) {
                const i64 i = RS[j];
                const IRcell& r = rc[i];
                write_post<Pr, C, Bulk>(st, r.np, r.nfl ? R + kMS * i : kRest, 2.0f, t0[r.np % static_cast<u32>(pnx)], cp, r.layer != 0, r.dmask);
            }
        });
    });
}

// Post-colisión de los fantasmas en el paso actual: plantilla trilineal, parte activa del padre interpolada en el
// tiempo con peso a (0 → T, ½ → T + ½). Con macro, también ρ,u de los fantasmas en los campos de salida.
namespace {
// Estado interpolado de un fantasma (plantilla trilineal; parte del padre interpolada en el tiempo).
CFD_INLINE void ghost_state(const IGhost& g, const float* M0, const float* M1, const float* R, float a, float* s) {
    if (g.nm + g.nr == 0) { std::memcpy(s, kRest, sizeof kRest); return; }
    const float b = 1.0f - a;
    for (int m = 0; m < kMS; ++m) s[m] = 0.0f;
    for (int e = 0; e < g.nm; ++e) {
        const float* p0 = M0 + kMS * g.idx[e];
        const float* p1 = M1 + kMS * g.idx[e];
        const float wb = g.w[e] * b, wa = g.w[e] * a;
        for (int m = 0; m < kMS; ++m) s[m] += wb * p0[m] + wa * p1[m];
    }
    for (int e = g.nm; e < g.nm + g.nr; ++e) {
        const float* pr = R + kMS * g.idx[e];
        for (int m = 0; m < kMS; ++m) s[m] += g.w[e] * pr[m];
    }
}
} // namespace

void Solver::Impl::iface_ghost(float a, bool macro) {
    const void* ld[L::Q];
    void* st[L::Q];
    step_ptrs(t, ld, st);
    const CollParams cp = coll_params(cfg);
    const IGhost* G = ghosts.data();
    const float* M0 = Mprev.data();
    const float* M1 = Mnext.data();
    const float* R = Rst.data();
    const u32* GV = gvec.data();
    const u32* GS = gsca.data();
    const i64 nv = static_cast<i64>(gvec.size()), ns = static_cast<i64>(gsca.size());
    const float t0 = tau0[1];
    float *orho = rho.data(), *oux = ux.data(), *ouy = uy.data(), *ouz = uz.data();
    with_kernel(cfg, [&]<Precision Pr, Collision C, bool Bulk>() {
        parallel_for(0, nv, 32, [&](i64 lo, i64 hi) {
            for (i64 v = lo; v < hi; ++v) {
                const i64 i0 = GV[v];
                alignas(32) float tmp[kMS][8];
                alignas(32) i32 lay[8];
                for (int l = 0; l < 8; ++l) {
                    float s[kMS];
                    ghost_state(G[i0 + l], M0, M1, R, a, s);
                    for (int m = 0; m < kMS; ++m) tmp[m][l] = s[m];
                    lay[l] = G[i0 + l].layer ? -1 : 0;
                }
                f8 sv[kMS];
                for (int m = 0; m < kMS; ++m) sv[m] = f8::load(tmp[m]);
                const u32 n0 = G[i0].n;
                u32 dm = 0;
                for (int l = 0; l < 8; ++l) dm |= G[i0 + l].dmask;
                write_post8<Pr, C, Bulk>(st, n0, sv, 1.0f, t0, cp, f8(_mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(lay)))), dm);
                if (macro) { sv[0].store(orho + n0); sv[1].store(oux + n0); sv[2].store(ouy + n0); sv[3].store(ouz + n0); }
            }
        });
        parallel_for(0, ns, 256, [&](i64 lo, i64 hi) {
            for (i64 j = lo; j < hi; ++j) {
                const IGhost& g = G[GS[j]];
                float s[kMS];
                ghost_state(g, M0, M1, R, a, s);
                write_post<Pr, C, Bulk>(st, g.n, s, 1.0f, t0, cp, g.layer != 0, g.dmask);
                if (macro) { orho[g.n] = s[0]; oux[g.n] = s[1]; ouy[g.n] = s[2]; ouz[g.n] = s[3]; }
            }
        });
    });
}

// ---- Paso recursivo --------------------------------------------------------------------------------------------------
// Fase A de un paso de esta rejilla (tiempo t): kernel (los bloques tap guardan sus momentos) y, si es una hija, las
// medias de sus esclavas en t (Rst) para la restricción hacia el padre y para sus propios fantasmas.
void Solver::Impl::step_kernel(bool macro) {
    update_motion_table();
    const double t0 = now_sec();
    run_kernel(macro);
    const double t1 = now_sec();
    if (par) iface_R();
    t_kernel += t1 - t0;
    t_iface += now_sec() - t1;
}

// Fase B (tiempo t = T): fase A del primer subpaso (2T) de cada hija → esclavas post@T → fantasmas propios post@t (antes
// del contorno: Bouzidi los lee) → contorno → M_{T+1} de cada hija (padre pre@T+1) → resto del subpaso 2T de cada hija y
// su subpaso 2T+1 completo (fantasmas en T + ½: interpolación lineal en el tiempo).
void Solver::Impl::step_rest(bool macro, float a) {
    for (Impl* k : kids) k->step_kernel(false);
    const double t1 = now_sec();
    for (Impl* k : kids) k->iface_restrict();
    if (par) iface_ghost(a, macro);
    const double t2 = now_sec();
    boundary_pass(true, t);
    const double t3 = now_sec();
    for (Impl* k : kids) k->iface_M(k->Mnext, t + 1);
    t_force += t3 - t2;
    t_iface += (t2 - t1) + (now_sec() - t3);
    for (Impl* k : kids) {
        k->step_rest(false, 0.0f);
        k->step_kernel(macro);
        k->step_rest(macro, 0.5f);
        std::swap(k->Mprev, k->Mnext);
    }
    ++t;
}

void Solver::Impl::advance(bool macro, float a) {
    step_kernel(macro);
    step_rest(macro, a);
}

// ---- Visualización ---------------------------------------------------------------------------------------------------
// Celdas cubiertas de esta rejilla en los campos de salida: media de ρ,u de sus 8 hijas (las fluidas; si todas son
// sólidas, ρ = 1 y la media de sus velocidades de pared).
void Solver::Impl::fill_vis() {
    for (Impl* k : kids) {
        const LevelBox& B = k->box;
        const u8* KF = k->vflags.data();
        const float *kr = k->rho.data(), *kx = k->ux.data(), *ky = k->uy.data(), *kz = k->uz.data();
        const i64 knx = k->nx, knxny = k->nxny;
        parallel_for(B.lo[2], B.hi[2] + 1, 1, [&](i64 z0, i64 z1) {
            for (i64 z = z0; z < z1; ++z)
                for (int y = B.lo[1]; y <= B.hi[1]; ++y)
                    for (int x = B.lo[0]; x <= B.hi[0]; ++x) {
                        const i64 n = x + nx * (y + static_cast<i64>(ny) * z);
                        const i64 c0 = (2 * (x - B.lo[0]) + 1) + knx * (2 * (y - B.lo[1]) + 1) + knxny * (2 * (z - B.lo[2]) + 1);
                        float r = 0, u = 0, v = 0, w = 0, us = 0, vs = 0, ws = 0;
                        int nf = 0;
                        for (int d = 0; d < 8; ++d) {
                            const i64 c = c0 + (d & 1) + knx * ((d >> 1) & 1) + knxny * (d >> 2);
                            if (KF[c] & kSolid) { us += kx[c]; vs += ky[c]; ws += kz[c]; continue; }
                            r += kr[c]; u += kx[c]; v += ky[c]; w += kz[c]; ++nf;
                        }
                        if (nf) { const float i = 1.0f / static_cast<float>(nf); rho[n] = r * i; ux[n] = u * i; uy[n] = v * i; uz[n] = w * i; }
                        else { rho[n] = 1.0f; ux[n] = us * 0.125f; uy[n] = vs * 0.125f; uz[n] = ws * 0.125f; }
                    }
        });
    }
}

// Todas las interfaces (tras cambiar geometría / movimientos / reinicio): plantillas, flags de visualización y M_T.
void Solver::Impl::iface_build_all() {
    for (int g = 1; g < n_grids(); ++g) grid(g).iface_build();
    // Flags de visualización, de la rejilla más fina a la base: fantasmas → fluido; cubiertas → mayoría de sus hijas.
    for (int g = n_grids() - 1; g >= 0; --g) {
        Impl& G = grid(g);
        if (!G.refined()) continue;
        const u8* F = G.flags.data();
        const u8* SI = G.sid.data();
        u8* VF = G.vflags.data();
        u8* VS = G.vsid.data();
        parallel_for(0, G.N, 1 << 15, [&](i64 lo, i64 hi) {
            for (i64 n = lo; n < hi; ++n) {
                const bool ghost = (F[n] & kSolid) && SI[n] == 0;
                VF[n] = ghost ? u8(kFluid) : F[n];
                VS[n] = SI[n];
            }
        });
        for (Impl* k : G.kids) {
            const LevelBox& B = k->box;
            const u8* KF = k->vflags.data();
            const u8* KS = k->vsid.data();
            for (int z = B.lo[2]; z <= B.hi[2]; ++z)
                for (int y = B.lo[1]; y <= B.hi[1]; ++y)
                    for (int x = B.lo[0]; x <= B.hi[0]; ++x) {
                        const i64 n = x + G.nx * (y + static_cast<i64>(G.ny) * z);
                        const i64 c0 = (2 * (x - B.lo[0]) + 1) + k->nx * (2 * (y - B.lo[1]) + 1) + k->nxny * (2 * (z - B.lo[2]) + 1);
                        int ns = 0;
                        u8 s = 0, fl = 0;
                        for (int d = 0; d < 8; ++d) {
                            const i64 c = c0 + (d & 1) + k->nx * ((d >> 1) & 1) + k->nxny * (d >> 2);
                            if (KF[c] & kSolid) { ++ns; if (!s) { s = KS[c]; fl = KF[c]; } }
                        }
                        if (ns >= 4) { VF[n] = fl; VS[n] = s; }
                        else { VF[n] = kFluid; VS[n] = 0; }
                    }
        }
    }
    for (int g = 1; g < n_grids(); ++g) {
        Impl& K = grid(g);
        K.iface_M(K.Mprev, K.par->t);   // padre pre-colisión en su paso actual
    }
    iface_dirty = false;
}

// ================================================================================================
int Solver::grids() const { return impl_->n_grids(); }

GridInfo Solver::grid_info(int g) const {
    const Impl& R = *impl_;
    CFD_CHECK(g >= 0 && g < R.n_grids(), "lbm::Solver::grid_info: rejilla inexistente");
    const Impl& I = R.grid(g);
    GridInfo gi;
    gi.nx = I.nx; gi.ny = I.ny; gi.nz = I.nz;
    gi.parent = I.par ? I.par->gid : -1;
    gi.depth = I.depth;
    gi.scale = I.scale;
    gi.org = I.org;
    gi.ground = g == 0 ? R.cfg.ground != GroundMode::None : I.gattached;
    gi.box = I.box;
    if (g == 0) {
        gi.inner = Aabb{Vec3(-0.5f, -0.5f, -0.5f), Vec3(I.nx - 0.5f, I.ny - 0.5f, I.nz - 0.5f)};
    } else {
        // Caras de las celdas interiores: [½, n − 3/2] en celdas propias.
        gi.inner = Aabb{I.org + Vec3(0.5f, 0.5f, 0.5f) * I.scale, I.org + Vec3(I.nx - 1.5f, I.ny - 1.5f, I.nz - 1.5f) * I.scale};
    }
    return gi;
}

} // namespace cfd::lbm
