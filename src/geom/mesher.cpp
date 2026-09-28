// ============================================================================
//  geom/mesher.cpp — mallas de render:
//    * mesh_scene : superficie de nivel 0 del SDF con "Surface Nets" (dual
//                   contouring sin QEF) sobre una rejilla muestreada sólo en
//                   banda estrecha; signos en BITBOARDS de 64 bits.
//    * mesh_voxels: caras sólido/fluido de la red LBM con greedy meshing.
//
//  Trucos (medidos en docs/opt/geom.md):
//   - Muestreo jerárquico 8³→4³→2³ SIN halo con RegionEval (poda exacta de grupos,
//     cotas heredadas del nodo padre): sólo se evalúan las hojas 2³ inciertas.
//     Después, una pasada de "relleno de esquinas" evalúa exactamente las muestras
//     que son esquina de un cubo con superficie y faltan: needed = dilat(M) & ~E.
//   - Signos como bitboards (1 bit por muestra, fila X = palabras u64, un byte por
//     bloque 8³ → cada bloque escribe bytes propios, sin carreras ni atómicos).
//   - Cubos mixtos de 64 en 64 con AND/OR de 4 filas + desplazamiento (SWAR).
//   - Índice de vértice por RANK (popcount del prefijo, BZHI de BMI2): sin rejilla
//     de índices de 4 bytes por cubo; salida escrita directamente en su sitio
//     (conteo → suma prefija → emisión), determinista y sin fusiones.
//   - Valores del SDF en ranuras de 2 KB sólo para bloques con superficie.
//   - Vértices por sub-bloques de 2³ cubos con poda heredada: ~1 grupo por evaluación.
//   - Rejilla desplazada φ-1 muestras: la caja de la escena toca la superficie en
//     puntos tangentes → sin caras ambiguas sistemáticas (aristas no-manifold).
//   - Tablas constexpr (aristas del cubo, tetraedro del gradiente).
//   - mesh_voxels: máscaras 2D construidas con AVX2 (32 celdas por instrucción),
//     búsqueda de tramos con movemask + tzcnt, rectángulos por plano en paralelo.
// ============================================================================
#include "voxelizer.hpp"

#include "../core/simd.hpp"
#include "../core/threadpool.hpp"
#include "../core/util.hpp"
#include "../render/colormap.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

// Conmutadores A/B (sólo para medir; por defecto la variante optimizada):
//   CFD_GEOM_SIMD = 0      → mesh_voxels con bucles escalares en vez de AVX2;
//   CFD_GEOM_VERT_SUB = 0  → vértices con el RegionEval del bloque 8³ (sin sub-bloques 2³).
#ifndef CFD_GEOM_SIMD
#define CFD_GEOM_SIMD 1
#endif
#ifndef CFD_GEOM_VERT_SUB
#define CFD_GEOM_VERT_SUB 1
#endif

namespace cfd::geom {
namespace mesh_detail {

inline constexpr float k_sqrt3 = 1.7320508075688772f;

// ---- Tablas constexpr ----------------------------------------------------------------
struct EdgeCI { u8 a, b; };
// 12 aristas del cubo como pares de esquinas (bit0 = +x, bit1 = +y, bit2 = +z).
constexpr std::array<EdgeCI, 12> make_cube_edges() {
    std::array<EdgeCI, 12> e{};
    int n = 0;
    for (int axis = 0; axis < 3; ++axis)
        for (int c = 0; c < 8; ++c)
            if (!(c & (1 << axis))) e[static_cast<usize>(n++)] = {static_cast<u8>(c), static_cast<u8>(c | (1 << axis))};
    return e;
}
inline constexpr std::array<EdgeCI, 12> k_edges = make_cube_edges();
// Tetraedro regular (Σk = 0, Σ k kᵀ = 4 I): gradiente y valor con 4 evaluaciones.
inline constexpr Vec3 k_tet[4] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};

// ---- Rejilla de muestreo ----------------------------------------------------------------
struct SGrid {
    int nx = 0, ny = 0, nz = 0;   // muestras por eje (múltiplos de 8)
    int bx = 0, by = 0, bz = 0;   // bloques 8³
    int W = 0;                    // palabras u64 por fila del bitboard
    Vec3 lo;                      // posición (m) de la muestra (0,0,0)
    float h = 0;                  // paso (m)
    CFD_INLINE usize row(int j, int k) const { return static_cast<usize>(j) + static_cast<usize>(ny) * static_cast<usize>(k); }
    CFD_INLINE Vec3 pos(float i, float j, float k) const { return Vec3(lo.x + i * h, lo.y + j * h, lo.z + k * h); }
    // Floats por bloques 8³ (2 KB contiguos) en "ranuras" asignadas sólo a los bloques que
    // evalúan muestras: memoria ∝ bloques con superficie, no ∝ volumen de la rejilla.
    CFD_INLINE usize block(int i, int j, int k) const {
        return static_cast<usize>(i >> 3) + static_cast<usize>(bx) * (static_cast<usize>(j >> 3) + static_cast<usize>(by) * static_cast<usize>(k >> 3));
    }
    CFD_INLINE static usize local(int i, int j, int k) { return static_cast<usize>((i & 7) | ((j & 7) << 3) | ((k & 7) << 6)); }
};

// Espacio de trabajo persistente (por hilo llamador): sin reservas en llamadas sucesivas.
struct Work {
    Buffer<u64> S, E, M;       // signo (d<0), evaluada, cubo mixto
    Buffer<u32> P;             // rango: nº de cubos mixtos antes de cada palabra (global)
    Buffer<float> D;           // valores del SDF (sólo hojas evaluadas), 512 por ranura
    Buffer<u32> slot;          // ranura de cada bloque 8³ (~0u = sin valores)
    Buffer<u32> rowv, rowq;    // vértices / quads por fila → sumas prefijas
    std::vector<Aabb> boxes;
};
template <class T>
static void ensure(Buffer<T>& b, usize n) { if (b.size() < n) b.resize(n); }

struct SampleCtx {
    const SGrid* G;
    u8* Sb;
    u8* Eb;
    float* D;
    u32* slot;
    std::atomic<u32>* nslot;
    float L, m;
    usize row_bytes;
    float apron;      // 1 = halo de una muestra (variante anterior), 0 = sin halo
};

// Ranura del bloque (la reserva sólo el hilo dueño del bloque; lectores tras la barrera).
CFD_INLINE float* block_values(u32* slot, std::atomic<u32>* nslot, float* D, usize b) {
    u32 sl = slot[b];
    if (sl == ~0u) { sl = nslot->fetch_add(1, std::memory_order_relaxed); slot[b] = sl; }
    return D + (static_cast<usize>(sl) << 9);
}

// Nodo de muestreo de lado s: la clasificación cubre [x0-apron, x0+s-1+apron] en cada eje.
void sample_node(const SampleCtx& X, const RegionEval& parent, int x0, int y0, int z0, int s, usize& nsampled) {
    const SGrid& G = *X.G;
    const float half = 0.5f * static_cast<float>(s - 1), reach = half + X.apron;
    const float cx = static_cast<float>(x0) + half, cy = static_cast<float>(y0) + half, cz = static_cast<float>(z0) + half;
    const Aabb reg{G.pos(cx - reach, cy - reach, cz - reach), G.pos(cx + reach, cy + reach, cz + reach)};
    const float LR = X.L * k_sqrt3 * reach * G.h + X.m;
    RegionEval re;
    re.refine(parent, reg, G.pos(cx, cy, cz), LR);
    const int cls = re.classify_sign(LR);
    if (cls < 0) return;                                      // todo fuera: bits ya a 0
    const usize col = static_cast<usize>(x0 >> 3);
    const u8 bits = static_cast<u8>(((1u << s) - 1u) << (x0 & 7));
    if (cls > 0) {                                            // todo dentro
        for (int z = z0; z < z0 + s; ++z)
            for (int y = y0; y < y0 + s; ++y) X.Sb[G.row(y, z) * X.row_bytes + col] |= bits;
        return;
    }
    if (s > 2) {
        const int hs = s >> 1;
        for (int k = 0; k < 8; ++k)
            sample_node(X, re, x0 + ((k & 1) ? hs : 0), y0 + ((k & 2) ? hs : 0), z0 + ((k & 4) ? hs : 0), hs, nsampled);
        return;
    }
    float* CFD_RESTRICT Dv = block_values(X.slot, X.nslot, X.D, G.block(x0, y0, z0));
    for (int z = z0; z < z0 + 2; ++z)
        for (int y = y0; y < y0 + 2; ++y) {
            u8 sb = 0;
            for (int x = x0; x < x0 + 2; ++x) {
                const float d = re.eval(G.pos(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
                Dv[SGrid::local(x, y, z)] = d;
                sb |= static_cast<u8>((d < 0.0f) << (x & 7));
            }
            const usize r = G.row(y, z) * X.row_bytes + col;
            X.Sb[r] |= sb;
            X.Eb[r] |= bits;
        }
    nsampled += 8;
}

// Máscara de los bits de la palabra w que caen en [lo, hi] (índices de muestra).
CFD_INLINE u64 range_bits(int w, int lo, int hi) {
    const int b0 = w * 64, b1 = b0 + 63;
    if (hi < b0 || lo > b1) return 0;
    const int a = max_(lo, b0) - b0, b = min_(hi, b1) - b0;
    const u64 upto = b == 63 ? ~0ull : ((1ull << (b + 1)) - 1);
    return upto & ~((1ull << a) - 1);
}

// Rango (índice global de vértice) del cubo i de una fila: prefijo + popcount(BZHI).
CFD_INLINE u32 rank_of(const u64* CFD_RESTRICT M, const u32* CFD_RESTRICT P, usize roww, int i) {
    const usize w = roww + static_cast<usize>(i >> 6);
    return P[w] + static_cast<u32>(std::popcount(_bzhi_u64(M[w], static_cast<unsigned>(i & 63))));
}

} // namespace mesh_detail

void mesh_scene(const sdf::Scene& scene, const LatticeMap& map, float cell_fraction, render::Mesh& out) {
    mesh_scene(scene, map, cell_fraction, out, MeshOptions{}, nullptr);
}

void mesh_scene(const sdf::Scene& scene, const LatticeMap& map, float cell_fraction, render::Mesh& out,
                const MeshOptions& mo, MeshStats* stats) {
    using namespace mesh_detail;
    Stopwatch sw_total, sw;
    MeshStats st;
    out.clear();
    const Aabb sb = scene.bounds();
    if (sb.empty() || scene.group_count() == 0) { if (stats) *stats = st; return; }

    // ---- Rejilla ------------------------------------------------------------------------
    SGrid G;
    G.h = max_(cell_fraction, 0.02f) * map.dx;
    for (int it = 0; it < 8; ++it) {
        const Vec3 ext = sb.size();
        // Margen ≥ 2 muestras (bordes siempre fuera) + desfase irracional (φ-1 ≈ 0.618): la caja
        // de la escena toca la superficie en puntos TANGENTES; si un plano de muestras coincidiera
        // con ellos aparecerían caras ambiguas (tablero de ajedrez) → aristas no-manifold.
        G.lo = sb.lo - Vec3(2.618034f * G.h);
        auto dim = [&](float e) { return ((static_cast<int>(std::ceil(e / G.h)) + 6 + 7) & ~7); };
        G.nx = dim(ext.x); G.ny = dim(ext.y); G.nz = dim(ext.z);
        const usize tot = static_cast<usize>(G.nx) * static_cast<usize>(G.ny) * static_cast<usize>(G.nz);
        if (tot <= mo.max_samples) break;
        G.h *= std::cbrt(static_cast<float>(tot) / static_cast<float>(mo.max_samples)) * 1.01f;
    }
    G.bx = G.nx >> 3; G.by = G.ny >> 3; G.bz = G.nz >> 3;
    G.W = (G.nx + 63) >> 6;
    st.n[0] = G.nx; st.n[1] = G.ny; st.n[2] = G.nz; st.h = G.h;
    const usize rows = static_cast<usize>(G.ny) * static_cast<usize>(G.nz);
    const usize W = static_cast<usize>(G.W);
    const usize nwords = rows * W;

    static thread_local Work ws_tls;
    Work& ws = ws_tls;    // referencia: los hilos del pool deben usar la instancia del llamador
    ensure(ws.S, nwords); ensure(ws.E, nwords); ensure(ws.M, nwords); ensure(ws.P, nwords);
    const usize nblocks = static_cast<usize>(G.bx) * static_cast<usize>(G.by) * static_cast<usize>(G.bz);
    ensure(ws.D, nblocks << 9);
    ensure(ws.slot, nblocks);
    std::memset(ws.slot.data(), 0xFF, nblocks * sizeof(u32));
    std::atomic<u32> nslot{0};
    ensure(ws.rowv, rows + 1); ensure(ws.rowq, rows + 1);
    std::memset(ws.S.data(), 0, nwords * 8);
    std::memset(ws.E.data(), 0, nwords * 8);
    ws.boxes.resize(scene.group_count());
    for (usize i = 0; i < ws.boxes.size(); ++i) ws.boxes[i] = scene.groups()[i].box_model;
    RegionEval root;
    root.init_all(scene, ws.boxes.data());

    u64* CFD_RESTRICT S = ws.S.data();
    u64* CFD_RESTRICT E = ws.E.data();
    u64* CFD_RESTRICT M = ws.M.data();
    u32* CFD_RESTRICT P = ws.P.data();
    float* CFD_RESTRICT D = ws.D.data();
    u32* CFD_RESTRICT slot = ws.slot.data();
    const float L = max_(mo.lipschitz, 1.0f), m = 0.01f * G.h;
    const int nth = max_(pool().size(), 1);
    auto widx = [nth] { return max_(ThreadPool::worker_index(), 0) % nth; };

    // ---- 1. Muestreo en banda estrecha -----------------------------------------------------
    {
        std::vector<Padded<usize>> cnt(static_cast<usize>(nth));
        SampleCtx X{&G, reinterpret_cast<u8*>(S), reinterpret_cast<u8*>(E), D, slot, &nslot, L, m, W * 8, mo.halo ? 1.0f : 0.0f};
        const i64 nblk = static_cast<i64>(G.bx) * G.by * G.bz;
        if (mo.block_skip) {
            parallel_for(0, nblk, 1, [&](i64 lo, i64 hi) {
                usize& c = cnt[static_cast<usize>(widx())].value;
                for (i64 b = lo; b < hi; ++b) {
                    const int ix = static_cast<int>(b % G.bx), iy = static_cast<int>((b / G.bx) % G.by), iz = static_cast<int>(b / (static_cast<i64>(G.bx) * G.by));
                    sample_node(X, root, ix * 8, iy * 8, iz * 8, 8, c);
                }
            });
        } else {
            // Referencia: todas las muestras con Scene::eval. Varias capas z comparten bloque →
            // ranuras densas asignadas de antemano (sin reserva perezosa concurrente).
            for (usize b = 0; b < nblocks; ++b) slot[b] = static_cast<u32>(b);
            nslot.store(static_cast<u32>(nblocks), std::memory_order_relaxed);
            parallel_for(0, G.nz, 1, [&](i64 lo, i64 hi) {
                usize& c = cnt[static_cast<usize>(widx())].value;
                for (int k = static_cast<int>(lo); k < static_cast<int>(hi); ++k)
                    for (int j = 0; j < G.ny; ++j) {
                        u64* srow = S + G.row(j, k) * W;
                        u64* erow = E + G.row(j, k) * W;
                        for (int i = 0; i < G.nx; ++i) {
                            const float d = scene.eval(G.pos(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)));
                            D[(static_cast<usize>(slot[G.block(i, j, k)]) << 9) | SGrid::local(i, j, k)] = d;
                            srow[i >> 6] |= static_cast<u64>(d < 0.0f) << (i & 63);
                            erow[i >> 6] |= 1ull << (i & 63);
                        }
                        c += static_cast<usize>(G.nx);
                    }
            });
        }
        for (const auto& c : cnt) st.sampled += c.value;
    }
    st.t_sample = sw.lap();

    // ---- 2. Cubos mixtos (64 por palabra) + conteo por fila ---------------------------------
    u32* CFD_RESTRICT rowv = ws.rowv.data();
    u32 V = 0;
    auto build_mixed = [&] {
        parallel_for(0, G.nz, 1, [&](i64 klo, i64 khi) {
            for (int k = static_cast<int>(klo); k < static_cast<int>(khi); ++k)
                for (int j = 0; j < G.ny; ++j) {
                    const usize r = G.row(j, k);
                    u64* mrow = M + r * W;
                    if (j >= G.ny - 1 || k >= G.nz - 1) { std::memset(mrow, 0, W * 8); rowv[r] = 0; continue; }
                    const u64* a = S + r * W;
                    const u64* b = S + G.row(j + 1, k) * W;
                    const u64* c = S + G.row(j, k + 1) * W;
                    const u64* d = S + G.row(j + 1, k + 1) * W;
                    u32 count = 0;
                    u64 A = a[0] & b[0] & c[0] & d[0], O = a[0] | b[0] | c[0] | d[0];
                    for (usize w = 0; w < W; ++w) {
                        const u64 An = w + 1 < W ? (a[w + 1] & b[w + 1] & c[w + 1] & d[w + 1]) : 0;
                        const u64 On = w + 1 < W ? (a[w + 1] | b[w + 1] | c[w + 1] | d[w + 1]) : 0;
                        const u64 all_in = A & ((A >> 1) | (An << 63));    // las 8 esquinas dentro
                        const u64 any_in = O | (O >> 1) | (On << 63);      // alguna esquina dentro
                        const u64 mixed = any_in & ~all_in & range_bits(static_cast<int>(w), 0, G.nx - 2);
                        mrow[w] = mixed;
                        count += static_cast<u32>(std::popcount(mixed));
                        A = An; O = On;
                    }
                    rowv[r] = count;
                }
        });
        // Suma prefija exclusiva (serie: ~10⁴ filas).
        V = 0;
        for (usize r = 0; r < rows; ++r) { const u32 c = rowv[r]; rowv[r] = V; V += c; }
        rowv[rows] = V;
        parallel_for(0, static_cast<i64>(rows), 256, [&](i64 lo, i64 hi) {
            for (usize r = static_cast<usize>(lo); r < static_cast<usize>(hi); ++r) {
                u32 acc = rowv[r];
                for (usize w = 0; w < W; ++w) { P[r * W + w] = acc; acc += static_cast<u32>(std::popcount(M[r * W + w])); }
            }
        });
    };
    build_mixed();
    st.t_cubes = sw.lap();

    // ---- 2b. Relleno de esquinas: muestras que son esquina de algún cubo mixto y no se
    //          evaluaron (hojas uniformes sin halo). needed = dilatación(M) & ~E, por bloques 8³:
    //          cada muestra pertenece a un único bloque → sin carreras; región acotada → RegionEval.
    //          Autocorrección: si el valor exacto contradice el signo clasificado (SDF no Lipschitz
    //          en esa zona, p.ej. culling de sdf.cpp dentro de uniones solapadas), se corrige el bit
    //          y se repite (cubos mixtos → esquinas) hasta el punto fijo. Casi nunca itera.
    auto fill_corners = [&]() -> bool {
        std::vector<Padded<usize>> cnt(static_cast<usize>(nth)), fixes(static_cast<usize>(nth));
        const u8* Eb0 = reinterpret_cast<const u8*>(E);
        u8* Ebw = reinterpret_cast<u8*>(E);
        u8* Sbw = reinterpret_cast<u8*>(S);
        const usize rb = W * 8;
        auto mword = [&](int j, int k, int w) -> u64 {
            return (j < 0 || k < 0 || w < 0) ? 0ull : M[G.row(j, k) * W + static_cast<usize>(w)];
        };
        auto needed_byte = [&](int j, int k, int bxi) -> u32 {
            const int w = bxi >> 3;
            const u64 cw = mword(j, k, w) | mword(j - 1, k, w) | mword(j, k - 1, w) | mword(j - 1, k - 1, w);
            const u64 cp = mword(j, k, w - 1) | mword(j - 1, k, w - 1) | mword(j, k - 1, w - 1) | mword(j - 1, k - 1, w - 1);
            const u64 nw = cw | (cw << 1) | (cp >> 63);                    // esquina i si cubo i o i-1
            return static_cast<u32>((nw >> (8 * (bxi & 7))) & 0xFFu) & ~static_cast<u32>(Eb0[G.row(j, k) * rb + static_cast<usize>(bxi)]);
        };
        const i64 nblk = static_cast<i64>(G.bx) * G.by * G.bz;
        parallel_for(0, nblk, 4, [&](i64 lo, i64 hi) {
            const usize wi = static_cast<usize>(widx());
            usize& c = cnt[wi].value;
            usize& fx = fixes[wi].value;
            for (i64 b = lo; b < hi; ++b) {
                const int bxi = static_cast<int>(b % G.bx), byi = static_cast<int>((b / G.bx) % G.by), bzi = static_cast<int>(b / (static_cast<i64>(G.bx) * G.by));
                const int j0 = byi * 8, k0 = bzi * 8, i0 = bxi * 8;
                bool any = false;
                for (int k = k0; k < k0 + 8 && !any; ++k)
                    for (int j = j0; j < j0 + 8 && !any; ++j) any = needed_byte(j, k, bxi) != 0;
                if (!any) continue;
                const float fi = static_cast<float>(i0), fj = static_cast<float>(j0), fk = static_cast<float>(k0);
                const Aabb reg{G.pos(fi, fj, fk), G.pos(fi + 7, fj + 7, fk + 7)};
                RegionEval re;
                re.refine(root, reg, G.pos(fi + 3.5f, fj + 3.5f, fk + 3.5f), L * k_sqrt3 * 3.5f * G.h + m);
                float* CFD_RESTRICT Dv = block_values(slot, &nslot, D, static_cast<usize>(b));
                for (int k = k0; k < k0 + 8; ++k)
                    for (int j = j0; j < j0 + 8; ++j) {
                        u32 bits = needed_byte(j, k, bxi);
                        if (!bits) continue;
                        const usize byte = G.row(j, k) * rb + static_cast<usize>(bxi);
                        Ebw[byte] |= static_cast<u8>(bits);
                        u8 sb = Sbw[byte];
                        for (; bits; bits &= bits - 1) {
                            const int bit = std::countr_zero(bits), i = i0 + bit;
                            const float d = re.eval(G.pos(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)));
                            Dv[SGrid::local(i, j, k)] = d;
                            const u8 want = static_cast<u8>((d < 0.0f) << bit);
                            if ((sb & (1u << bit)) != want) { sb = static_cast<u8>((sb & ~(1u << bit)) | want); ++fx; }
                            ++c;
                        }
                        Sbw[byte] = sb;
                    }
            }
        });
        usize nf = 0;
        for (const auto& c : cnt) st.corner_fill += c.value;
        for (const auto& f : fixes) nf += f.value;
        st.sign_fixes += nf;
        return nf != 0;
    };
    for (int it = 0; it < 8 && fill_corners(); ++it) build_mixed();
    st.t_fill = sw.lap();

    // ---- 3. Vértices ------------------------------------------------------------------------
    out.pos.resize(V);
    out.nrm.resize(V);
    out.group.resize(V);
    if (mo.fill_color) out.color.resize(V);
    std::vector<Padded<Aabb>> bacc(static_cast<usize>(nth));
    {
        const float e = clamp_(0.02f * G.h, 1e-4f, 1e-3f);       // paso del tetraedro (m)
        const float pe = e * 1.01f / G.h + 1e-3f;                  // margen de región (muestras)
        const int steps = clamp_(mo.projection_steps, 0, 3);
        const u8* Mb = reinterpret_cast<const u8*>(M);
        const u8* Eb = reinterpret_cast<const u8*>(E);
        const usize rb = W * 8;
        u32 group_color[256];
        for (int i = 0; i < 256; ++i) group_color[i] = 0xFFB8BCC4u;
        for (usize gi = 0; gi < scene.group_count() && gi < 254; ++gi)
            group_color[gi + 1] = render::component_color(scene.groups()[gi].component);
        Vec3* CFD_RESTRICT opos = out.pos.data();
        Vec3* CFD_RESTRICT onrm = out.nrm.data();
        u8* CFD_RESTRICT ogrp = out.group.data();
        u32* CFD_RESTRICT ocol = mo.fill_color ? out.color.data() : nullptr;
        const i64 nblk = static_cast<i64>(G.bx) * G.by * G.bz;
        parallel_for(0, nblk, 1, [&](i64 lo, i64 hi) {
            Aabb& bb = bacc[static_cast<usize>(widx())].value;
            for (i64 b = lo; b < hi; ++b) {
                const int cbx = static_cast<int>(b % G.bx), cby = static_cast<int>((b / G.bx) % G.by), cbz = static_cast<int>(b / (static_cast<i64>(G.bx) * G.by));
                const int j0 = cby * 8, k0 = cbz * 8, i0 = cbx * 8;
                const int j1 = min_(j0 + 8, G.ny - 1), k1 = min_(k0 + 8, G.nz - 1);
                bool any = false;
                for (int k = k0; k < k1 && !any; ++k)
                    for (int j = j0; j < j1 && !any; ++j) any = Mb[G.row(j, k) * rb + static_cast<usize>(cbx)] != 0;
                if (!any) continue;
                const float fi = static_cast<float>(i0), fj = static_cast<float>(j0), fk = static_cast<float>(k0);
                const Aabb reg{G.pos(fi - pe, fj - pe, fk - pe), G.pos(fi + 8 + pe, fj + 8 + pe, fk + 8 + pe)};
                RegionEval re8;
                re8.refine(root, reg, G.pos(fi + 4, fj + 4, fk + 4), L * k_sqrt3 * (4.0f + pe) * G.h + m);
                // Sub-bloques de 2³ cubos: poda heredada (reglas b/c de RegionEval) → casi siempre
                // queda un solo grupo candidato para las 4 evaluaciones del tetraedro.
                for (int sz = 0; sz < 4; ++sz)
                    for (int sy = 0; sy < 4; ++sy)
                        for (int sx = 0; sx < 4; ++sx) {
                            const int ka = k0 + 2 * sz, ja = j0 + 2 * sy, ia = i0 + 2 * sx;
                            u32 sub[4] = {0, 0, 0, 0};                   // bits de las 4 filas (dj,dk)
                            bool anys = false;
                            for (int q = 0; q < 4; ++q) {
                                const int j = ja + (q & 1), k = ka + (q >> 1);
                                if (j >= j1 || k >= k1) continue;
                                sub[q] = (static_cast<u32>(Mb[G.row(j, k) * rb + static_cast<usize>(cbx)]) >> (2 * sx)) & 3u;
                                anys |= sub[q] != 0;
                            }
                            if (!anys) continue;
                            const float ci0 = static_cast<float>(ia), cj0 = static_cast<float>(ja), ck0 = static_cast<float>(ka);
                            const Aabb reg2{G.pos(ci0 - pe, cj0 - pe, ck0 - pe), G.pos(ci0 + 2 + pe, cj0 + 2 + pe, ck0 + 2 + pe)};
                            RegionEval re_sub;
                            if (CFD_GEOM_VERT_SUB) re_sub.refine(re8, reg2, G.pos(ci0 + 1, cj0 + 1, ck0 + 1), L * k_sqrt3 * (1.0f + pe) * G.h + m);
                            const RegionEval& re = CFD_GEOM_VERT_SUB ? re_sub : re8;
                            for (int q = 0; q < 4; ++q)
                                for (u32 bits = sub[q]; bits; bits &= bits - 1) {
                                    const int i = ia + std::countr_zero(bits), j = ja + (q & 1), k = ka + (q >> 1);
                                    const usize r = G.row(j, k);
                                    const u32 vi = rank_of(M, P, r * W, i);
                                    // Valores en las 8 esquinas (evaluadas en muestreo o relleno de esquinas).
                                    float v[8];
                                    for (int c = 0; c < 8; ++c) {
                                        const int ci = i + (c & 1), cj = j + ((c >> 1) & 1), ck = k + (c >> 2);
                                        const usize er = G.row(cj, ck) * rb + static_cast<usize>(ci >> 3);
                                        v[c] = ((Eb[er] >> (ci & 7)) & 1)
                                                   ? D[(static_cast<usize>(slot[G.block(ci, cj, ck)]) << 9) | SGrid::local(ci, cj, ck)]
                                                   : re.eval(G.pos(static_cast<float>(ci), static_cast<float>(cj), static_cast<float>(ck)));
                                    }
                                    // Media de los cruces por cero en las aristas (Surface Nets).
                                    Vec3 sum(0.0f);
                                    int ncross = 0;
                                    for (const EdgeCI& ed : k_edges) {
                                        const float va = v[ed.a], vb = v[ed.b];
                                        if ((va < 0.0f) == (vb < 0.0f)) continue;
                                        const float t = clamp_(va / (va - vb), 0.0f, 1.0f);
                                        const Vec3 ca(static_cast<float>(ed.a & 1), static_cast<float>((ed.a >> 1) & 1), static_cast<float>(ed.a >> 2));
                                        const Vec3 cb(static_cast<float>(ed.b & 1), static_cast<float>((ed.b >> 1) & 1), static_cast<float>(ed.b >> 2));
                                        sum += ca + (cb - ca) * t;
                                        ++ncross;
                                    }
                                    const Vec3 pl = ncross ? sum * (1.0f / static_cast<float>(ncross)) : Vec3(0.5f);
                                    const float fi2 = static_cast<float>(i), fj2 = static_cast<float>(j), fk2 = static_cast<float>(k);
                                    Vec3 p = G.pos(fi2 + pl.x, fj2 + pl.y, fk2 + pl.z);
                                    constexpr float inset = 1e-3f;
                                    const Vec3 cmin = G.pos(fi2 + inset, fj2 + inset, fk2 + inset);
                                    const Vec3 cmax = G.pos(fi2 + 1 - inset, fj2 + 1 - inset, fk2 + 1 - inset);
                                    // Proyección sobre la superficie: p -= d ∇d / |∇d|² (tetraedro: 4 evals/paso).
                                    Vec3 grad(0, 0, 1);
                                    int gid = 0;
                                    const int nsteps = steps > 0 ? steps : 1;
                                    for (int it = 0; it < nsteps; ++it) {
                                        float dsum = 0;
                                        Vec3 g(0.0f);
                                        for (int t4 = 0; t4 < 4; ++t4) {
                                            int id = 0;
                                            const float d = re.eval(p + k_tet[t4] * e, t4 == 0 ? &id : nullptr);
                                            if (t4 == 0) gid = id;
                                            dsum += d;
                                            g += k_tet[t4] * d;
                                        }
                                        g = g * (0.25f / e);
                                        grad = g;
                                        if (steps > 0) {
                                            const float g2 = dot(g, g);
                                            if (g2 > 1e-12f) p = vmin(vmax(p - g * (0.25f * dsum / g2), cmin), cmax);
                                        }
                                    }
                                    const float gl = dot(grad, grad);
                                    const Vec3 n = gl > 1e-20f ? grad * (1.0f / std::sqrt(gl)) : Vec3(0, 0, 1);
                                    const Vec3 pc = map.to_cells(p);
                                    opos[vi] = pc;
                                    onrm[vi] = n;
                                    ogrp[vi] = static_cast<u8>(gid);
                                    if (ocol) ocol[vi] = group_color[gid & 255];
                                    bb.grow(pc);
                                }
                        }
            }
        });
    }
    for (const auto& a : bacc) out.bounds.grow(a.value);
    st.t_verts = sw.lap();

    // ---- 4. Quads por arista con cambio de signo → 2 triángulos (diagonal más corta) -----
    u32* CFD_RESTRICT rowq = ws.rowq.data();
    const int nx = G.nx, ny = G.ny, nz = G.nz;
    // Máscaras por palabra: aristas X válidas i∈[0,nx-2]; Y/Z válidas i∈[1,nx-2].
    u64 mask_x[16], mask_yz[16];
    const bool wide = W > 16;
    for (usize w = 0; w < min_(W, usize(16)); ++w) {
        mask_x[w] = mesh_detail::range_bits(static_cast<int>(w), 0, nx - 2);
        mask_yz[w] = mesh_detail::range_bits(static_cast<int>(w), 1, nx - 2);
    }
    auto mx = [&](usize w) { return wide ? mesh_detail::range_bits(static_cast<int>(w), 0, nx - 2) : mask_x[w]; };
    auto myz = [&](usize w) { return wide ? mesh_detail::range_bits(static_cast<int>(w), 1, nx - 2) : mask_yz[w]; };
    // Palabras de cambio de signo de la fila (j,k) para cada tipo de arista.
    auto edge_words = [&](int j, int k, usize w, u64& ex, u64& ey, u64& ez) {
        const u64* s0 = S + G.row(j, k) * W;
        const u64 nxt = w + 1 < W ? s0[w + 1] : 0;
        ex = (j >= 1 && j <= ny - 2 && k >= 1 && k <= nz - 2) ? (s0[w] ^ ((s0[w] >> 1) | (nxt << 63))) & mx(w) : 0;
        ey = (j <= ny - 2 && k >= 1 && k <= nz - 2) ? (s0[w] ^ S[G.row(j + 1, k) * W + w]) & myz(w) : 0;
        ez = (j >= 1 && j <= ny - 2 && k <= nz - 2) ? (s0[w] ^ S[G.row(j, k + 1) * W + w]) & myz(w) : 0;
    };
    parallel_for(0, static_cast<i64>(rows), 128, [&](i64 lo, i64 hi) {
        for (usize r = static_cast<usize>(lo); r < static_cast<usize>(hi); ++r) {
            const int j = static_cast<int>(r % static_cast<usize>(ny)), k = static_cast<int>(r / static_cast<usize>(ny));
            u32 q = 0;
            for (usize w = 0; w < W; ++w) {
                u64 ex, ey, ez;
                edge_words(j, k, w, ex, ey, ez);
                q += static_cast<u32>(std::popcount(ex) + std::popcount(ey) + std::popcount(ez));
            }
            rowq[r] = q;
        }
    });
    u32 Q = 0;
    for (usize r = 0; r < rows; ++r) { const u32 c = rowq[r]; rowq[r] = Q; Q += c; }
    out.tri.resize(static_cast<usize>(Q) * 6);
    {
        u32* CFD_RESTRICT tri = out.tri.data();
        const Vec3* CFD_RESTRICT ps = out.pos.data();
        parallel_for(0, static_cast<i64>(rows), 128, [&](i64 lo, i64 hi) {
            for (usize r = static_cast<usize>(lo); r < static_cast<usize>(hi); ++r) {
                const int j = static_cast<int>(r % static_cast<usize>(ny)), k = static_cast<int>(r / static_cast<usize>(ny));
                u32* t = tri + static_cast<usize>(rowq[r]) * 6;
                const u64* s0 = S + r * W;
                auto rk = [&](int ci, int cj, int ck) { return mesh_detail::rank_of(M, P, G.row(cj, ck) * W, ci); };
                // q0..q3 en orden antihorario visto desde +eje; `in` = la muestra base está dentro.
                auto emit = [&](u32 q0, u32 q1, u32 q2, u32 q3, bool in) {
                    if (!in) { const u32 tmp = q1; q1 = q3; q3 = tmp; }
                    const float d02 = length2(ps[q0] - ps[q2]), d13 = length2(ps[q1] - ps[q3]);
                    if (d02 <= d13) { t[0] = q0; t[1] = q1; t[2] = q2; t[3] = q0; t[4] = q2; t[5] = q3; }
                    else            { t[0] = q0; t[1] = q1; t[2] = q3; t[3] = q1; t[4] = q2; t[5] = q3; }
                    t += 6;
                };
                for (usize w = 0; w < W; ++w) {
                    u64 ex, ey, ez;
                    edge_words(j, k, w, ex, ey, ez);
                    const u64 sw_ = s0[w];
                    for (; ex; ex &= ex - 1) {
                        const int b = std::countr_zero(ex), i = static_cast<int>(w) * 64 + b;
                        emit(rk(i, j - 1, k - 1), rk(i, j, k - 1), rk(i, j, k), rk(i, j - 1, k), (sw_ >> b) & 1);
                    }
                    for (; ey; ey &= ey - 1) {
                        const int b = std::countr_zero(ey), i = static_cast<int>(w) * 64 + b;
                        emit(rk(i - 1, j, k - 1), rk(i - 1, j, k), rk(i, j, k), rk(i, j, k - 1), (sw_ >> b) & 1);
                    }
                    for (; ez; ez &= ez - 1) {
                        const int b = std::countr_zero(ez), i = static_cast<int>(w) * 64 + b;
                        emit(rk(i - 1, j - 1, k), rk(i, j - 1, k), rk(i, j, k), rk(i - 1, j, k), (sw_ >> b) & 1);
                    }
                }
            }
        });
    }
    st.t_quads = sw.lap();

    // ---- 5. Normales robustas ---------------------------------------------------------------
    // El gradiente del SDF es la mejor normal donde el campo es suave, pero es basura donde
    // sdf.cpp es discontinuo (restas que sobresalen: bug de culling; interior de uniones) o en
    // piezas más finas que h. flowvis muestrea Cp en pos + n·offset: una normal invertida mira
    // DENTRO del sólido. Normal de caras = Σ áreas vectoriales de los quads que tocan el vértice
    // (reunidas desde los bitboards de signo, sin scatter → sin carreras); si el gradiente se
    // aparta más de acos(normal_min_dot) de ella, se usa la de caras.
    if (mo.normal_min_dot > -1.0f && V > 0) {
        std::vector<Padded<usize>> nfix(static_cast<usize>(nth));
        const Vec3* CFD_RESTRICT ps = out.pos.data();
        Vec3* CFD_RESTRICT onrm = out.nrm.data();
        const float tau = mo.normal_min_dot;
        auto sbit = [&](int i, int j, int k) -> u32 { return static_cast<u32>((S[G.row(j, k) * W + static_cast<usize>(i >> 6)] >> (i & 63)) & 1u); };
        auto rk = [&](int ci, int cj, int ck) { return mesh_detail::rank_of(M, P, G.row(cj, ck) * W, ci); };
        // Área vectorial (×2) del quad q0..q3 (antihorario visto desde +eje), orientada por `in`.
        auto qarea = [&](u32 q0, u32 q1, u32 q2, u32 q3, u32 in) {
            const Vec3 a = cross(ps[q2] - ps[q0], ps[q3] - ps[q1]);
            return in ? a : -a;
        };
        parallel_for(0, static_cast<i64>(rows), 64, [&](i64 lo, i64 hi) {
            usize& fx = nfix[static_cast<usize>(widx())].value;
            for (usize r = static_cast<usize>(lo); r < static_cast<usize>(hi); ++r) {
                const int j = static_cast<int>(r % static_cast<usize>(ny)), k = static_cast<int>(r / static_cast<usize>(ny));
                for (usize w = 0; w < W; ++w)
                    for (u64 mb = M[r * W + w]; mb; mb &= mb - 1) {
                        const int i = static_cast<int>(w) * 64 + std::countr_zero(mb);
                        const u32 vi = P[r * W + w] + static_cast<u32>(std::popcount(_bzhi_u64(M[r * W + w], static_cast<unsigned>(i & 63))));
                        // Signos de las 8 esquinas (bit c: +x si c&1, +y si c&2, +z si c&4).
                        u32 cs = 0;
                        for (int c = 0; c < 8; ++c) cs |= sbit(i + (c & 1), j + ((c >> 1) & 1), k + (c >> 2)) << c;
                        Vec3 A(0.0f);
                        for (int e2 = 0; e2 < 4; ++e2) {
                            const int da = e2 & 1, db = e2 >> 1;
                            {   // arista X en (i, j+da, k+db): esquinas c y c|1 con c = 2da + 4db
                                const int jj = j + da, kk = k + db, c = 2 * da + 4 * db;
                                const u32 s0 = (cs >> c) & 1u;
                                if (s0 != ((cs >> (c | 1)) & 1u) && jj >= 1 && jj <= ny - 2 && kk >= 1 && kk <= nz - 2 && i <= nx - 2)
                                    A += qarea(rk(i, jj - 1, kk - 1), rk(i, jj, kk - 1), rk(i, jj, kk), rk(i, jj - 1, kk), s0);
                            }
                            {   // arista Y en (i+da, j, k+db): esquinas c y c|2 con c = da + 4db
                                const int ii = i + da, kk = k + db, c = da + 4 * db;
                                const u32 s0 = (cs >> c) & 1u;
                                if (s0 != ((cs >> (c | 2)) & 1u) && ii >= 1 && ii <= nx - 2 && j <= ny - 2 && kk >= 1 && kk <= nz - 2)
                                    A += qarea(rk(ii - 1, j, kk - 1), rk(ii - 1, j, kk), rk(ii, j, kk), rk(ii, j, kk - 1), s0);
                            }
                            {   // arista Z en (i+da, j+db, k): esquinas c y c|4 con c = da + 2db
                                const int ii = i + da, jj = j + db, c = da + 2 * db;
                                const u32 s0 = (cs >> c) & 1u;
                                if (s0 != ((cs >> (c | 4)) & 1u) && ii >= 1 && ii <= nx - 2 && jj >= 1 && jj <= ny - 2 && k <= nz - 2)
                                    A += qarea(rk(ii - 1, jj - 1, k), rk(ii, jj - 1, k), rk(ii, jj, k), rk(ii - 1, jj, k), s0);
                            }
                        }
                        const float a2 = dot(A, A);
                        if (!(a2 > 1e-30f)) continue;
                        const Vec3 nf = A * (1.0f / std::sqrt(a2));
                        if (dot(onrm[vi], nf) < tau) { onrm[vi] = nf; ++fx; }
                    }
            }
        });
        for (const auto& f : nfix) st.normal_fixes += f.value;
    }
    st.t_normals = sw.lap();
    st.vertices = V;
    st.quads = Q;
    st.mixed_blocks = nslot.load(std::memory_order_relaxed);
    st.seconds = sw_total.elapsed();
    if (stats) *stats = st;
}

// =========================================================================================
//  mesh_voxels — greedy meshing de las caras sólido/fluido
// =========================================================================================
namespace vmesh_detail {

struct Rect { u16 u0, v0, w, h; u8 id, pos; };   // pos = 1 → normal hacia +eje
struct Task { u32 worker, off, count; };

struct VWork {
    std::vector<std::vector<Rect>> per_worker;
    std::vector<Task> tasks;
    Buffer<u8> maskp, maskn;   // por hilo: nth × (W·H) bytes
};

// Celda sólida para el mallado: 1..254 (y 255 si no se ignora el suelo).
CFD_INLINE __m256i solid_mask32(__m256i v, bool skip_ground) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i s = _mm256_xor_si256(_mm256_cmpeq_epi8(v, zero), _mm256_set1_epi8(-1));
    if (skip_ground) s = _mm256_andnot_si256(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(-1)), s);
    return s;
}
CFD_INLINE bool solid1(u8 v, bool skip_ground) { return v != 0 && !(skip_ground && v == 255); }

// Máscaras de un plano a partir de dos filas contiguas a (lado -) y b (lado +):
//   pos[x] = id(a) si a sólido y b no (normal +eje);  neg[x] = id(b) si b sólido y a no.
CFD_INLINE void face_row(const u8* CFD_RESTRICT a, const u8* CFD_RESTRICT b, u8* CFD_RESTRICT pos, u8* CFD_RESTRICT neg, int n, bool sg) {
    int x = 0;
    for (; CFD_GEOM_SIMD && x + 32 <= n; x += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + x));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + x));
        const __m256i sa = solid_mask32(va, sg), sbm = solid_mask32(vb, sg);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pos + x), _mm256_and_si256(_mm256_andnot_si256(sbm, sa), va));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg + x), _mm256_and_si256(_mm256_andnot_si256(sa, sbm), vb));
    }
    for (; x < n; ++x) {
        const bool sa = solid1(a[x], sg), sbb = solid1(b[x], sg);
        pos[x] = (sa && !sbb) ? a[x] : 0;
        neg[x] = (sbb && !sa) ? b[x] : 0;
    }
}

// Primer índice ≥ u en [u, n) con m[idx] != v (o n). 32 bytes por iteración (AVX2).
CFD_INLINE int run_end(const u8* CFD_RESTRICT m, int u, int n, u8 v) {
    const __m256i vv = _mm256_set1_epi8(static_cast<char>(v));
    while (CFD_GEOM_SIMD && u + 32 <= n) {
        const u32 eq = static_cast<u32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + u)), vv)));
        if (eq != 0xFFFFFFFFu) return u + std::countr_zero(~eq);
        u += 32;
    }
    while (u < n && m[u] == v) ++u;
    return u;
}
// Siguiente índice ≥ u con m[idx] != 0 (o n).
CFD_INLINE int next_nonzero(const u8* CFD_RESTRICT m, int u, int n) {
    const __m256i z = _mm256_setzero_si256();
    while (CFD_GEOM_SIMD && u + 32 <= n) {
        const u32 isz = static_cast<u32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + u)), z)));
        if (isz != 0xFFFFFFFFu) return u + std::countr_zero(~isz);
        u += 32;
    }
    while (u < n && m[u] == 0) ++u;
    return u;
}

// Greedy meshing de una máscara W×H (fila = v, contigua en u). Destruye la máscara.
void greedy(u8* CFD_RESTRICT m, int W, int H, u8 pos, std::vector<Rect>& outv) {
    for (int v = 0; v < H; ++v) {
        u8* row = m + static_cast<usize>(v) * static_cast<usize>(W);
        int u = next_nonzero(row, 0, W);
        while (u < W) {
            const u8 id = row[u];
            const int w = run_end(row, u, W, id) - u;
            int h = 1;
            while (v + h < H) {
                const u8* r2 = row + static_cast<usize>(h) * static_cast<usize>(W);
                if (run_end(r2, u, u + w, id) != u + w) break;
                ++h;
            }
            for (int t = 0; t < h; ++t) std::memset(row + static_cast<usize>(t) * static_cast<usize>(W) + u, 0, static_cast<usize>(w));
            outv.push_back(Rect{static_cast<u16>(u), static_cast<u16>(v), static_cast<u16>(w), static_cast<u16>(h), id, pos});
            u = next_nonzero(row, u + w, W);
        }
    }
}

} // namespace vmesh_detail

void mesh_voxels(const u8* solid_id, int nx, int ny, int nz, render::Mesh& out, bool skip_ground) {
    using namespace vmesh_detail;
    out.clear();
    if (!solid_id || nx <= 0 || ny <= 0 || nz <= 0) return;
    const usize sy = static_cast<usize>(nx), sz = static_cast<usize>(nx) * static_cast<usize>(ny);
    const int nth = max_(pool().size(), 1);
    auto widx = [nth] { return max_(ThreadPool::worker_index(), 0) % nth; };

    // ---- 1. Caja de celdas sólidas (AVX2: 32 celdas por comparación) ------------------------
    struct BB { int lo[3] = {INT_MAX, INT_MAX, INT_MAX}, hi[3] = {INT_MIN, INT_MIN, INT_MIN}; };
    std::vector<Padded<BB>> bbs(static_cast<usize>(nth));
    parallel_for(0, nz, 1, [&](i64 zlo, i64 zhi) {
        BB& bb = bbs[static_cast<usize>(widx())].value;
        for (int z = static_cast<int>(zlo); z < static_cast<int>(zhi); ++z)
            for (int y = 0; y < ny; ++y) {
                const u8* row = solid_id + sy * static_cast<usize>(y) + sz * static_cast<usize>(z);
                int first = -1, last = -1, x = 0;
                for (; CFD_GEOM_SIMD && x + 32 <= nx; x += 32) {
                    const u32 mk = static_cast<u32>(_mm256_movemask_epi8(solid_mask32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x)), skip_ground)));
                    if (mk) { if (first < 0) first = x + std::countr_zero(mk); last = x + 31 - std::countl_zero(mk); }
                }
                for (; x < nx; ++x) if (solid1(row[x], skip_ground)) { if (first < 0) first = x; last = x; }
                if (first < 0) continue;
                bb.lo[0] = min_(bb.lo[0], first); bb.hi[0] = max_(bb.hi[0], last);
                bb.lo[1] = min_(bb.lo[1], y); bb.hi[1] = max_(bb.hi[1], y);
                bb.lo[2] = min_(bb.lo[2], z); bb.hi[2] = max_(bb.hi[2], z);
            }
    });
    BB bb;
    for (const auto& b : bbs)
        for (int i = 0; i < 3; ++i) { bb.lo[i] = min_(bb.lo[i], b.value.lo[i]); bb.hi[i] = max_(bb.hi[i], b.value.hi[i]); }
    if (bb.lo[0] > bb.hi[0]) return;
    const int lo[3] = {bb.lo[0], bb.lo[1], bb.lo[2]}, hi[3] = {bb.hi[0], bb.hi[1], bb.hi[2]};
    const int ext[3] = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};

    // ---- 2. Tareas = planos (eje, p): p ∈ [lo, hi+1], el plano p separa las celdas p-1 y p ----
    // Eje 0 (X): máscara (u=y, v=z); eje 1 (Y): (u=x, v=z); eje 2 (Z): (u=x, v=y).
    static thread_local VWork vw_tls;
    VWork& vw = vw_tls;   // referencia: los hilos del pool deben usar la instancia del llamador
    const int np[3] = {ext[0] + 1, ext[1] + 1, ext[2] + 1};
    const int ntask = np[0] + np[1] + np[2];
    vw.tasks.assign(static_cast<usize>(ntask), Task{0, 0, 0});
    vw.per_worker.resize(static_cast<usize>(nth));
    for (auto& v : vw.per_worker) v.clear();
    const int dimU[3] = {ext[1], ext[0], ext[0]}, dimV[3] = {ext[2], ext[2], ext[1]};
    usize mask_sz = 0;
    for (int a = 0; a < 3; ++a) mask_sz = max_(mask_sz, static_cast<usize>(dimU[a]) * static_cast<usize>(dimV[a]));
    mask_sz = (mask_sz + 63) & ~usize(63);
    if (vw.maskp.size() < mask_sz * static_cast<usize>(nth)) { vw.maskp.resize(mask_sz * static_cast<usize>(nth)); vw.maskn.resize(mask_sz * static_cast<usize>(nth)); }

    const u8* base = solid_id;
    auto cell = [&](int x, int y, int z) -> u8 { return base[static_cast<usize>(x) + sy * static_cast<usize>(y) + sz * static_cast<usize>(z)]; };
    parallel_for(0, ntask, 1, [&](i64 tlo, i64 thi) {
        const int wi = widx();
        std::vector<Rect>& outv = vw.per_worker[static_cast<usize>(wi)];
        u8* mp = vw.maskp.data() + mask_sz * static_cast<usize>(wi);
        u8* mn = vw.maskn.data() + mask_sz * static_cast<usize>(wi);
        for (int t = static_cast<int>(tlo); t < static_cast<int>(thi); ++t) {
            int axis = 0, pl = t;
            if (pl >= np[0]) { pl -= np[0]; axis = 1; if (pl >= np[1]) { pl -= np[1]; axis = 2; } }
            const int p = lo[axis] + pl;                 // plano entre p-1 y p
            const int Wd = dimU[axis], Hd = dimV[axis];
            if (axis == 0) {
                // X: lectura con paso (u=y, v=z). Fuera de la red = aire.
                for (int v = 0; v < Hd; ++v) {
                    const int z = lo[2] + v;
                    u8* rp = mp + static_cast<usize>(v) * static_cast<usize>(Wd);
                    u8* rn = mn + static_cast<usize>(v) * static_cast<usize>(Wd);
                    for (int u = 0; u < Wd; ++u) {
                        const int y = lo[1] + u;
                        const u8 a = p - 1 >= 0 ? cell(p - 1, y, z) : u8{0};
                        const u8 b = p < nx ? cell(p, y, z) : u8{0};
                        const bool sa = solid1(a, skip_ground), sbb = solid1(b, skip_ground);
                        rp[u] = (sa && !sbb) ? a : 0;
                        rn[u] = (sbb && !sa) ? b : 0;
                    }
                }
            } else {
                for (int v = 0; v < Hd; ++v) {
                    int y0, z0, y1, z1;
                    if (axis == 1) { z0 = z1 = lo[2] + v; y0 = p - 1; y1 = p; }
                    else           { y0 = y1 = lo[1] + v; z0 = p - 1; z1 = p; }
                    u8* rp = mp + static_cast<usize>(v) * static_cast<usize>(Wd);
                    u8* rn = mn + static_cast<usize>(v) * static_cast<usize>(Wd);
                    const bool va = (axis == 1 ? y0 : z0) >= 0, vb = (axis == 1 ? y1 < ny : z1 < nz);
                    const u8* ra = va ? base + static_cast<usize>(lo[0]) + sy * static_cast<usize>(y0) + sz * static_cast<usize>(z0) : nullptr;
                    const u8* rbp = vb ? base + static_cast<usize>(lo[0]) + sy * static_cast<usize>(y1) + sz * static_cast<usize>(z1) : nullptr;
                    if (ra && rbp) face_row(ra, rbp, rp, rn, Wd, skip_ground);
                    else if (ra) { for (int u = 0; u < Wd; ++u) { rp[u] = solid1(ra[u], skip_ground) ? ra[u] : 0; rn[u] = 0; } }
                    else if (rbp) { for (int u = 0; u < Wd; ++u) { rn[u] = solid1(rbp[u], skip_ground) ? rbp[u] : 0; rp[u] = 0; } }
                    else { std::memset(rp, 0, static_cast<usize>(Wd)); std::memset(rn, 0, static_cast<usize>(Wd)); }
                }
            }
            const usize off = outv.size();
            greedy(mp, Wd, Hd, 1, outv);
            greedy(mn, Wd, Hd, 0, outv);
            vw.tasks[static_cast<usize>(t)] = Task{static_cast<u32>(wi), static_cast<u32>(off), static_cast<u32>(outv.size() - off)};
        }
    });

    // ---- 3. Suma prefija por tarea (orden determinista) y emisión --------------------------
    std::vector<u32> tbase(static_cast<usize>(ntask) + 1);
    u32 R = 0;
    for (int t = 0; t < ntask; ++t) { tbase[static_cast<usize>(t)] = R; R += vw.tasks[static_cast<usize>(t)].count; }
    tbase[static_cast<usize>(ntask)] = R;
    out.pos.resize(static_cast<usize>(R) * 4);
    out.nrm.resize(static_cast<usize>(R) * 4);
    out.group.resize(static_cast<usize>(R) * 4);
    out.color.resize(static_cast<usize>(R) * 4);
    out.tri.resize(static_cast<usize>(R) * 6);
    Vec3* CFD_RESTRICT P = out.pos.data();
    Vec3* CFD_RESTRICT N = out.nrm.data();
    u8* CFD_RESTRICT G = out.group.data();
    u32* CFD_RESTRICT C = out.color.data();
    u32* CFD_RESTRICT T = out.tri.data();
    parallel_for(0, ntask, 4, [&](i64 tlo, i64 thi) {
        for (int t = static_cast<int>(tlo); t < static_cast<int>(thi); ++t) {
            const Task tk = vw.tasks[static_cast<usize>(t)];
            if (!tk.count) continue;
            int axis = 0, pl = t;
            if (pl >= np[0]) { pl -= np[0]; axis = 1; if (pl >= np[1]) { pl -= np[1]; axis = 2; } }
            const float pc = static_cast<float>(lo[axis] + pl) - 0.5f;
            const int au = axis == 0 ? 1 : 0, av = axis == 2 ? 1 : 2;      // ejes de u y v
            const float ou = static_cast<float>(lo[au]) - 0.5f, ov = static_cast<float>(lo[av]) - 0.5f;
            // Normal natural de (e_u × e_v): +X para eje 0, -Y para eje 1, +Z para eje 2.
            const float natural = axis == 1 ? -1.0f : 1.0f;
            const Rect* rs = vw.per_worker[tk.worker].data() + tk.off;
            u32 ri = tbase[static_cast<usize>(t)];
            for (u32 q = 0; q < tk.count; ++q, ++ri) {
                const Rect& rc = rs[q];
                const float u0 = ou + rc.u0, u1 = u0 + rc.w, v0 = ov + rc.v0, v1 = v0 + rc.h;
                const float sgn = rc.pos ? 1.0f : -1.0f;
                Vec3 nrm(0.0f);
                nrm[axis] = sgn;
                const float uu[4] = {u0, u1, u1, u0}, vv[4] = {v0, v0, v1, v1};
                const u32 vb = ri * 4;
                for (int c = 0; c < 4; ++c) {
                    Vec3 p;
                    p[axis] = pc; p[au] = uu[c]; p[av] = vv[c];
                    P[vb + static_cast<u32>(c)] = p;
                    N[vb + static_cast<u32>(c)] = nrm;
                    G[vb + static_cast<u32>(c)] = rc.id;
                    C[vb + static_cast<u32>(c)] = 0xFFB8BCC4u;
                }
                u32* tr = T + static_cast<usize>(ri) * 6;
                if (sgn * natural > 0) { tr[0] = vb; tr[1] = vb + 1; tr[2] = vb + 2; tr[3] = vb; tr[4] = vb + 2; tr[5] = vb + 3; }
                else                   { tr[0] = vb; tr[1] = vb + 2; tr[2] = vb + 1; tr[3] = vb; tr[4] = vb + 3; tr[5] = vb + 2; }
            }
        }
    });
    out.bounds = Aabb{Vec3(static_cast<float>(lo[0]) - 0.5f, static_cast<float>(lo[1]) - 0.5f, static_cast<float>(lo[2]) - 0.5f),
                      Vec3(static_cast<float>(hi[0]) + 0.5f, static_cast<float>(hi[1]) + 0.5f, static_cast<float>(hi[2]) + 0.5f)};
}

} // namespace cfd::geom
