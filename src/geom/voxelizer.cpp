// ============================================================================
//  geom/voxelizer.cpp — escena SDF → solid_id de la red LBM.
//
//  Algoritmo (ver docs/opt/geom.md):
//   1. Caja de la escena (+margen) → rango de celdas; lo de fuera se pone a 0
//      con memset por filas (paralelo por z). Cada celda z ≥ z_min se escribe
//      exactamente una vez.
//   2. La caja se recorre en bloques 8³ alineados a la red (fila de 8 celdas =
//      un u64), repartidos dinámicamente entre los hilos (parallel_for, grano 1).
//   3. Descenso jerárquico 8³ → 4³ → 2³ → celda. En cada nodo se poda la lista
//      de grupos candidatos (RegionEval::refine, exacto) y con d_h(centro) ± L·R
//      se decide: todo fluido, todo sólido con un único id, o subdividir.
//      Sólo los nodos 2³ que cortan la superficie se evalúan celda a celda.
//   4. Submuestreo 2×2×2 (mayoría ≥ 4/8) sólo donde puede cambiar el resultado:
//      |d - umbral| < L·√3/4·dx (y |d| < dx, la definición del contrato).
//  El resultado es idéntico al de evaluar Scene::eval en cada celda (hipótesis
//  de Lipschitz L; verificado en tests/test_geom.cpp).
// ============================================================================
#include "voxelizer.hpp"

#include "../core/threadpool.hpp"
#include "../core/util.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <vector>

namespace cfd::geom {
namespace vox_detail {

inline constexpr float k_sqrt3 = 1.7320508075688772f;

// Conmutadores A/B (sólo para medir; por defecto la variante optimizada):
//   CFD_GEOM_VOTE_EARLY = 0 → contar siempre los 8 submuestreos;
//   CFD_GEOM_SS_NARROW  = 0 → votar en toda la banda |d| < dx del contrato.
#ifndef CFD_GEOM_VOTE_EARLY
#define CFD_GEOM_VOTE_EARLY 1
#endif
#ifndef CFD_GEOM_SS_NARROW
#define CFD_GEOM_SS_NARROW 1
#endif

// Acumulador por hilo (sin false sharing gracias a Padded<>).
struct Acc {
    usize count = 0;
    usize literal = 0;     // celdas evaluadas por la ruta literal (zonas de discontinuidad)
    int lo[3] = {INT_MAX, INT_MAX, INT_MAX};
    int hi[3] = {INT_MIN, INT_MIN, INT_MIN};
    CFD_INLINE void add_box(int x0, int x1, int y0, int y1, int z0, int z1) {   // [x0,x1) ...
        count += static_cast<usize>(x1 - x0) * static_cast<usize>(y1 - y0) * static_cast<usize>(z1 - z0);
        lo[0] = min_(lo[0], x0); hi[0] = max_(hi[0], x1 - 1);
        lo[1] = min_(lo[1], y0); hi[1] = max_(hi[1], y1 - 1);
        lo[2] = min_(lo[2], z0); hi[2] = max_(hi[2], z1 - 1);
    }
    CFD_INLINE void add_cell(int x, int y, int z) {
        ++count;
        lo[0] = min_(lo[0], x); hi[0] = max_(hi[0], x);
        lo[1] = min_(lo[1], y); hi[1] = max_(hi[1], y);
        lo[2] = min_(lo[2], z); hi[2] = max_(hi[2], z);
    }
};

struct Ctx {
    const sdf::Scene* scene;
    LatticeMap map;
    u8* CFD_RESTRICT out;
    int nx, ny, nz, zmin;
    usize sy, sz;          // pasos lineales (nx, nx*ny)
    float dx, thr, L, m;   // umbral = thicken·dx, L = Lipschitz, m = margen absoluto
    float band;            // banda de submuestreo útil: L·√3/4·dx + m
    int leaf;              // tamaño de nodo hoja (2 → celdas)
    const RegionEval* root;
    // Cajas (modelo, +m) de las primitivas restadas donde el SDF de sdf.cpp es DISCONTINUO por
    // el bug de culling de Op::Subtract/SmoothSubtract (ver sdf_subtract_culling_bug). Sólo
    // se rellenan si el bug está presente y thicken > 0; ahí no valen las cotas de Lipschitz.
    const Aabb* disc;
    int ndisc;
};

// ¿Toca la región alguna caja de discontinuidad?
CFD_INLINE bool touches_disc(const Ctx& C, const Aabb& r) {
    for (int i = 0; i < C.ndisc; ++i) {
        const Aabb& b = C.disc[i];
        if (r.lo.x <= b.hi.x && r.hi.x >= b.lo.x && r.lo.y <= b.hi.y && r.hi.y >= b.lo.y && r.lo.z <= b.hi.z && r.hi.z >= b.lo.z)
            return true;
    }
    return false;
}

CFD_INLINE usize lin(const Ctx& C, int x, int y, int z) {
    return static_cast<usize>(x) + C.sy * static_cast<usize>(y) + C.sz * static_cast<usize>(z);
}

// Rellena [x0,x1)×[y0,y1)×[z0,z1) con v. Filas de ≤ 8 bytes: memset de tamaño pequeño → 1 store.
CFD_INLINE void fill_box(const Ctx& C, int x0, int x1, int y0, int y1, int z0, int z1, u8 v) {
    const usize w = static_cast<usize>(x1 - x0);
    if (w == 8) {                                       // fila completa de bloque: un u64 (SWAR)
        const u64 word = 0x0101010101010101ull * v;
        for (int z = z0; z < z1; ++z)
            for (int y = y0; y < y1; ++y) std::memcpy(C.out + lin(C, x0, y, z), &word, 8);
        return;
    }
    for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y) std::memset(C.out + lin(C, x0, y, z), v, w);
}

// Offsets del submuestreo 2×2×2 (en celdas), tabla constexpr.
struct Off3 { float x, y, z; };
inline constexpr Off3 k_sub[8] = {
    {-0.25f, -0.25f, -0.25f}, {0.25f, -0.25f, -0.25f}, {-0.25f, 0.25f, -0.25f}, {0.25f, 0.25f, -0.25f},
    {-0.25f, -0.25f, 0.25f},  {0.25f, -0.25f, 0.25f},  {-0.25f, 0.25f, 0.25f},  {0.25f, 0.25f, 0.25f}};

// Celda sólida (definición del contrato) con la escena evaluada por `E` (RegionEval o Scene).
// NARROW = false → regla literal del contrato (votar siempre que |d| < dx), sin atajo de Lipschitz.
template <bool SS, bool NARROW = true, class Eval>
CFD_INLINE u8 cell_value(const Ctx& C, const Eval& E, int x, int y, int z) {
    const Vec3 pc(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    int id = 0;
    const float d = E.eval(C.map.to_model(pc), &id);
    bool solid;
    if constexpr (SS) {
        // Sólo se vota donde el voto PUEDE diferir del centro (|d-umbral| < banda); fuera de la
        // banda los 8 submuestreos caen del mismo lado que el centro (Lipschitz L).
        if ((!NARROW || !CFD_GEOM_SS_NARROW || std::fabs(d - C.thr) < C.band) && std::fabs(d) < C.dx) {
            // Voto con salida anticipada: decide en cuanto hay 4 dentro o 5 fuera (mismo resultado
            // que contar los 8, ≈ 5-6 evaluaciones de media en vez de 8).
            int in = 0, out = 0;
            for (const Off3& o : k_sub) {
                if (E.eval(C.map.to_model(Vec3(pc.x + o.x, pc.y + o.y, pc.z + o.z)), nullptr) < C.thr) { if (++in >= 4 && CFD_GEOM_VOTE_EARLY) break; }
                else if (++out >= 5 && CFD_GEOM_VOTE_EARLY) break;
            }
            solid = in >= 4;
        } else {
            solid = d < C.thr;
        }
    } else {
        solid = d < C.thr;
    }
    return solid ? static_cast<u8>(id) : u8{0};
}

// Adaptador para usar Scene directamente en la ruta de referencia.
struct SceneEval {
    const sdf::Scene* s;
    CFD_INLINE float eval(Vec3 p, int* gid) const { return s->eval(p, gid); }
};

// Nodo cúbico de lado s (8, 4, 2) con esquina (x0,y0,z0) en celdas.
template <bool SS>
void process_node(const Ctx& C, const RegionEval& parent, int x0, int y0, int z0, int s, Acc& acc) {
    const int cx0 = max_(x0, 0), cx1 = min_(x0 + s, C.nx);
    const int cy0 = max_(y0, 0), cy1 = min_(y0 + s, C.ny);
    const int cz0 = max_(z0, C.zmin), cz1 = min_(z0 + s, C.nz);
    if (cx0 >= cx1 || cy0 >= cy1 || cz0 >= cz1) return;

    const float half = 0.5f * static_cast<float>(s - 1);
    const float reach = half + (SS ? 0.25f : 0.0f);                // alcance de los submuestreos
    const Vec3 cc(static_cast<float>(x0) + half, static_cast<float>(y0) + half, static_cast<float>(z0) + half);
    const Aabb reg{C.map.to_model(cc - Vec3(reach)), C.map.to_model(cc + Vec3(reach))};
    const float LR = C.L * k_sqrt3 * reach * C.dx + C.m;

    if (C.ndisc && touches_disc(C, reg)) {
        // Zona con SDF discontinuo (resta que sobresale + bug de sdf.cpp): ni clasificación por
        // cotas ni cotas heredadas. Los hijos que salgan de la zona parten de la raíz; las hojas
        // aplican la definición literal con Scene::eval (idéntico a la fuerza bruta).
        if (s > C.leaf) {
            const int h = s >> 1;
            for (int k = 0; k < 8; ++k)
                process_node<SS>(C, *C.root, x0 + ((k & 1) ? h : 0), y0 + ((k & 2) ? h : 0), z0 + ((k & 4) ? h : 0), h, acc);
            return;
        }
        const SceneEval se{C.scene};
        for (int z = cz0; z < cz1; ++z)
            for (int y = cy0; y < cy1; ++y) {
                u8* CFD_RESTRICT row = C.out + lin(C, 0, y, z);
                for (int x = cx0; x < cx1; ++x) {
                    const u8 v = cell_value<SS, false>(C, se, x, y, z);
                    row[x] = v;
                    if (v) acc.add_cell(x, y, z);
                }
            }
        acc.literal += static_cast<usize>(cx1 - cx0) * static_cast<usize>(cy1 - cy0) * static_cast<usize>(cz1 - cz0);
        return;
    }

    RegionEval re;
    re.refine(parent, reg, C.map.to_model(cc), LR);
    const int cls = re.classify_id(LR, C.thr);
    if (cls < 0) {                                                  // todo fluido
        fill_box(C, cx0, cx1, cy0, cy1, cz0, cz1, 0);
        return;
    }
    if (cls > 0) {                                                  // todo sólido con id único
        fill_box(C, cx0, cx1, cy0, cy1, cz0, cz1, static_cast<u8>(cls));
        acc.add_box(cx0, cx1, cy0, cy1, cz0, cz1);
        return;
    }
    if (s > C.leaf) {
        const int h = s >> 1;
        for (int k = 0; k < 8; ++k)
            process_node<SS>(C, re, x0 + ((k & 1) ? h : 0), y0 + ((k & 2) ? h : 0), z0 + ((k & 4) ? h : 0), h, acc);
        return;
    }
    for (int z = cz0; z < cz1; ++z)
        for (int y = cy0; y < cy1; ++y) {
            u8* CFD_RESTRICT row = C.out + lin(C, 0, y, z);
            for (int x = cx0; x < cx1; ++x) {
                const u8 v = cell_value<SS>(C, re, x, y, z);
                row[x] = v;
                if (v) acc.add_cell(x, y, z);
            }
        }
}

} // namespace vox_detail

// Sonda (una vez por proceso, inicialización estática hilo-segura) del bug de culling de
// sdf.cpp: Scene::eval_group descarta Op::Subtract/SmoothSubtract si el punto está FUERA de la
// pieza base aunque esté dentro de lo restado → SDF discontinuo (positivo a ambos lados) en la
// caja de la primitiva restada. Cuando sdf.cpp se corrija, la sonda da false y el voxelizador
// deja de pagar la ruta literal en esas cajas.
bool sdf_subtract_culling_bug() {
    static const bool bug = [] {
        sdf::Scene s;
        s.begin_group("sonda", sdf::Component::Object, sdf::Frame::Fixed);
        s.sphere({0, 0, 0}, 1.0f);
        s.round_box({0, 0, 1.0f}, {0.3f, 0.3f, 0.5f}, 0.0f, {}, sdf::Op::Subtract);
        s.end_group();
        s.begin_group("sonda", sdf::Component::Object, sdf::Frame::Fixed);
        s.sphere({10, 0, 0}, 1.0f);
        s.round_box({10, 0, 1.0f}, {0.3f, 0.3f, 0.5f}, 0.0f, {}, sdf::Op::SmoothSubtract, 0.05f);
        s.end_group();
        // Valores exactos: max(0.02, 0.3) = 0.3 y smax(0.2, 0.3, 0.05) = 0.3.
        return std::fabs(s.eval_group(0, {0, 0, 1.02f}) - 0.3f) > 1e-3f || std::fabs(s.eval_group(1, {10, 0, 1.2f}) - 0.3f) > 1e-3f;
    }();
    return bug;
}

VoxelStats voxelize(const sdf::Scene& scene, const LatticeMap& map, int nx, int ny, int nz, u8* solid_id,
                    const VoxelOptions& opt) {
    using namespace vox_detail;
    const Stopwatch sw;
    VoxelStats st;
    const int zmin = clamp_(opt.z_min, 0, nz);
    if (nx <= 0 || ny <= 0 || nz <= 0 || zmin >= nz) { st.seconds = sw.elapsed(); return st; }

    Ctx C;
    C.scene = &scene; C.map = map; C.out = solid_id;
    C.nx = nx; C.ny = ny; C.nz = nz; C.zmin = zmin;
    C.sy = static_cast<usize>(nx); C.sz = static_cast<usize>(nx) * static_cast<usize>(ny);
    C.dx = map.dx; C.thr = opt.thicken * map.dx; C.L = max_(opt.lipschitz, 1.0f); C.m = 0.01f * map.dx;
    C.band = C.L * k_sqrt3 * 0.25f * map.dx + C.m;
    C.leaf = 2;
    C.root = nullptr;
    // Zonas de discontinuidad (sólo importan con umbral > 0: con thr ≤ 0 basta el signo, que
    // sdf.cpp sí calcula bien, y valores exteriores ≤ distancia real → la clasificación es válida).
    // Cajas ajustadas: box_frame (marco del grupo; con espejo ya recortada a y ≥ 0) y su reflejo
    // por separado, cada una al espacio modelo. box_model de una pieza con espejo abarcaría las
    // dos mitades (p.ej. llantas a ±1 m → todo el ancho del coche) y encarecería la ruta literal.
    std::vector<Aabb> disc;
    if (C.thr > 0.0f && opt.block_skip && sdf_subtract_culling_bug())
        for (const sdf::Group& g : scene.groups()) {
            const Xform F = g.frame == sdf::Frame::Body ? scene.body_frame() : (g.frame == sdf::Frame::Wheels ? scene.wheel_frame() : Xform{});
            for (u32 i = g.first; i < g.first + g.count; ++i) {
                const sdf::Prim& pr = scene.prims()[i];
                if (pr.op != sdf::Op::Subtract && pr.op != sdf::Op::SmoothSubtract) continue;
                const Aabb& b = pr.box_frame;
                disc.push_back(b.transformed(F).expanded(C.m));
                if (pr.mirror_y) disc.push_back(Aabb{Vec3(b.lo.x, -b.hi.y, b.lo.z), Vec3(b.hi.x, -b.lo.y, b.hi.z)}.transformed(F).expanded(C.m));
            }
        }
    C.disc = disc.data();
    C.ndisc = static_cast<int>(disc.size());

    // --- 1. Rango de celdas a recorrer (caja de la escena + margen) -----------------------
    int X0 = 0, X1 = 0, Y0 = 0, Y1 = 0, Z0 = zmin, Z1 = zmin;   // cobertura en bloques 8³
    const Aabb sb = scene.bounds();
    if (!sb.empty() && scene.group_count() > 0) {
        const float marg = max_(C.thr, 0.0f) + map.dx;            // fuera: d > marg ≥ umbral + alcance
        const Vec3 cl = map.to_cells(sb.lo - Vec3(marg)), ch = map.to_cells(sb.hi + Vec3(marg));
        auto lo_i = [](float v, int n) { return static_cast<int>(std::ceil(clamp_(v, -1.0f, static_cast<float>(n)))); };
        auto hi_i = [](float v, int n) { return static_cast<int>(std::floor(clamp_(v, -1.0f, static_cast<float>(n)))); };
        const int lx = max_(lo_i(cl.x, nx), 0), hx = min_(hi_i(ch.x, nx), nx - 1);
        const int ly = max_(lo_i(cl.y, ny), 0), hy = min_(hi_i(ch.y, ny), ny - 1);
        const int lz = max_(lo_i(cl.z, nz), zmin), hz = min_(hi_i(ch.z, nz), nz - 1);
        if (lx <= hx && ly <= hy && lz <= hz) {
            X0 = lx & ~7; X1 = min_((hx | 7) + 1, nx);
            Y0 = ly & ~7; Y1 = min_((hy | 7) + 1, ny);
            Z0 = max_(lz & ~7, zmin); Z1 = min_((hz | 7) + 1, nz);
        }
    }

    // --- 2. Cero fuera de la cobertura (cada celda z ≥ z_min se escribe una sola vez) -----
    parallel_for(zmin, nz, 4, [&](i64 zlo, i64 zhi) {
        for (int z = static_cast<int>(zlo); z < static_cast<int>(zhi); ++z) {
            u8* slab = solid_id + C.sz * static_cast<usize>(z);
            if (z < Z0 || z >= Z1 || X0 >= X1) { std::memset(slab, 0, C.sz); continue; }
            if (Y0 > 0) std::memset(slab, 0, C.sy * static_cast<usize>(Y0));
            if (Y1 < ny) std::memset(slab + C.sy * static_cast<usize>(Y1), 0, C.sy * static_cast<usize>(ny - Y1));
            for (int y = Y0; y < Y1; ++y) {
                u8* row = slab + C.sy * static_cast<usize>(y);
                if (X0 > 0) std::memset(row, 0, static_cast<usize>(X0));
                if (X1 < nx) std::memset(row + X1, 0, static_cast<usize>(nx - X1));
            }
        }
    });

    // --- 3. Bloques ---------------------------------------------------------------------------
    const int nth = max_(pool().size(), 1);
    std::vector<Padded<Acc>> accs(static_cast<usize>(nth));
    auto slot_acc = [&]() -> Acc& { return accs[static_cast<usize>(max_(ThreadPool::worker_index(), 0) % nth)].value; };

    if (X0 < X1 && Y0 < Y1 && Z0 < Z1) {
        const int bx = (X1 - X0 + 7) >> 3, by = (Y1 - Y0 + 7) >> 3, bz = ((Z1 - (Z0 & ~7)) + 7) >> 3;
        const int zb0 = Z0 & ~7;
        const i64 nblk = static_cast<i64>(bx) * by * bz;
        if (opt.block_skip) {
            std::vector<Aabb> boxes(scene.group_count());
            for (usize i = 0; i < boxes.size(); ++i) boxes[i] = scene.groups()[i].box_model;
            RegionEval root;
            root.init_all(scene, boxes.data());
            C.root = &root;
            parallel_for(0, nblk, 1, [&](i64 lo, i64 hi) {
                Acc& acc = slot_acc();
                for (i64 b = lo; b < hi; ++b) {
                    const int ix = static_cast<int>(b % bx), iy = static_cast<int>((b / bx) % by), iz = static_cast<int>(b / (static_cast<i64>(bx) * by));
                    const int x0 = X0 + 8 * ix, y0 = Y0 + 8 * iy, z0 = zb0 + 8 * iz;
                    if (opt.supersample) process_node<true>(C, root, x0, y0, z0, 8, acc);
                    else                 process_node<false>(C, root, x0, y0, z0, 8, acc);
                }
            });
        } else {
            // Referencia: definición literal (Scene::eval en cada celda de la cobertura, voto en
            // toda la banda |d| < dx) sin salto de bloques ni atajos de Lipschitz.
            const SceneEval se{&scene};
            parallel_for(Z0, Z1, 1, [&](i64 zlo, i64 zhi) {
                Acc& acc = slot_acc();
                for (int z = static_cast<int>(zlo); z < static_cast<int>(zhi); ++z)
                    for (int y = Y0; y < Y1; ++y) {
                        u8* row = solid_id + lin(C, 0, y, z);
                        for (int x = X0; x < X1; ++x) {
                            const u8 v = opt.supersample ? cell_value<true, false>(C, se, x, y, z) : cell_value<false>(C, se, x, y, z);
                            row[x] = v;
                            if (v) acc.add_cell(x, y, z);
                        }
                    }
            });
        }
    }

    // --- 4. Estadísticas ------------------------------------------------------------------
    Acc tot;
    for (const auto& a : accs) {
        tot.count += a.value.count;
        tot.literal += a.value.literal;
        for (int i = 0; i < 3; ++i) { tot.lo[i] = min_(tot.lo[i], a.value.lo[i]); tot.hi[i] = max_(tot.hi[i], a.value.hi[i]); }
    }
    st.solid_cells = tot.count;
    st.literal_cells = tot.literal;
    if (tot.count) for (int i = 0; i < 3; ++i) { st.lo[i] = tot.lo[i]; st.hi[i] = tot.hi[i]; }
    st.seconds = sw.elapsed();
    return st;
}

} // namespace cfd::geom
