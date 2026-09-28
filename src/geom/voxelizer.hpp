// ============================================================================
//  geom/voxelizer.hpp — SDF → celdas sólidas del LBM, y SDF → mallas de render
//  (contrato público; implementación en geom/voxelizer.cpp y geom/mesher.cpp).
//
//  Extensiones compatibles añadidas por el módulo geom (ver docs/opt/geom.md):
//   * VoxelOptions::block_skip / lipschitz  (el resto de campos no cambia)
//   * VoxelStats::literal_cells y sdf_subtract_culling_bug() (ruta literal mientras
//     sdf.cpp tenga el bug de culling de restas)
//   * MeshOptions + MeshStats y una sobrecarga de mesh_scene con opciones
//     (MeshOptions::normal_min_dot / MeshStats::normal_fixes: normales robustas)
//   * RegionEval: evaluador de la escena restringido a una caja, EXACTAMENTE
//     equivalente a Scene::eval dentro de ella pero podando grupos lejanos.
// ============================================================================
#pragma once

#include "../core/mem.hpp"
#include "../render/mesh.hpp"
#include "lattice_map.hpp"
#include "sdf.hpp"

#include <atomic>

namespace cfd::geom {

struct VoxelStats {
    usize solid_cells = 0;
    int lo[3] = {0, 0, 0}, hi[3] = {-1, -1, -1};   // caja de celdas sólidas (inclusiva)
    double seconds = 0;
    // --- Extensión (geom) ---------------------------------------------------------------------
    // Celdas evaluadas por la ruta literal (Scene::eval por celda, sin cotas) dentro de las cajas
    // de primitivas restadas mientras sdf.cpp tenga el bug de culling de restas (thicken > 0).
    usize literal_cells = 0;
};

struct VoxelOptions {
    // Engrosamiento: celda sólida si sdf(centro) < thicken * dx. Un valor pequeño (≈0.1–0.2)
    // evita que bordes de salida y elementos finos (flaps) desaparezcan en redes gruesas.
    float thicken = 0.12f;
    // Submuestreo 2×2×2 en celdas cercanas a la superficie (|sdf| < dx): sólida si ≥ 4/8 dentro.
    bool supersample = true;
    int z_min = 1;          // no tocar la capa del suelo (z < z_min) — la gestiona el solver
    // --- Extensiones (geom) -------------------------------------------------------------
    // Salto jerárquico de bloques 8³→4³→2³ (false = evaluación celda a celda: referencia lenta).
    bool block_skip = true;
    // Constante de Lipschitz supuesta del SDF (|∇d| ≤ L). Las primitivas "cota" (elipsoide,
    // ala con flecha/estrechamiento, taper box) superan ligeramente 1; 1.5 deja margen.
    float lipschitz = 1.5f;
};

// Escribe solid_id (tamaño nx*ny*nz): 0 = fluido, 1 + índice de grupo = sólido.
// Sólo recorre la caja de la escena (+margen); fuera de ella pone 0. Paralelo (pool).
// Resultado idéntico celda a celda a aplicar la definición con Scene::eval en cada celda para
// thicken ≥ 0 (hipótesis: |∇d| ≤ lipschitz fuera de las cajas de primitivas restadas y SDF ≥
// distancia a la caja de su grupo). Con thicken < 0 no se garantiza: el culling de uniones de
// sdf.cpp hace el SDF interior discontinuo (ver docs/opt/geom.md).
VoxelStats voxelize(const sdf::Scene& scene, const LatticeMap& map, int nx, int ny, int nz, u8* solid_id,
                    const VoxelOptions& opt = {});

// Extensión: true si el Scene::eval enlazado tiene el bug de culling de restas (sonda de una vez).
// Mientras sea true, voxelize usa la ruta literal dentro de las cajas de las primitivas restadas.
bool sdf_subtract_culling_bug();

// Malla suave de la escena (render): superficie de nivel 0 del SDF muestreado con paso
// `cell_fraction`·dx (p.ej. 0.5 = el doble de fino que la red) sobre la caja de la escena.
// Salida en coordenadas de CELDAS de la red (map.to_cells), normales = gradiente SDF (o la
// media de caras donde el gradiente es incoherente, ver MeshOptions::normal_min_dot),
// group = id del grupo más cercano. Paralelo. Algoritmo libre (surface nets / marching cubes),
// requisitos: sin grietas, orientación consistente (normales hacia fuera), sin triángulos degenerados.
void mesh_scene(const sdf::Scene& scene, const LatticeMap& map, float cell_fraction, render::Mesh& out);

// --- Extensión: opciones y estadísticas del mallador ------------------------------------
struct MeshOptions {
    // Pasos de proyección p -= d·∇d/|∇d|² (0..3) con un tetraedro de 4 evaluaciones por paso;
    // el gradiente del último paso da la normal. 1 paso: error medio ≈ 0.02 mm con h = 15 mm.
    int projection_steps = 1;
    bool block_skip = true;     // muestreo sólo en banda estrecha (bloques 8³→4³→2³)
    // true: clasificación con halo de 1 muestra (variante anterior, sólo para comparar);
    // false: sin halo + pasada que evalúa las esquinas que falten de los cubos con superficie.
    bool halo = false;
    // Ver VoxelOptions::lipschitz. Aquí basta 1.25: la pasada de esquinas corrige cualquier
    // signo mal clasificado junto a la superficie (malla idéntica al muestreo completo en las
    // 19 escenas de prueba incluso con 1.0; 1.25 deja margen para burbujas lejos de ella).
    float lipschitz = 1.25f;
    bool fill_color = true;     // rellena Mesh::color con el color del componente del grupo
    usize max_samples = usize(96) << 20;   // tope de muestras; si se supera se engrosa el paso
    // Normales robustas: si el gradiente del SDF forma con la normal media de los quads del
    // vértice un coseno < normal_min_dot (SDF discontinuo, piezas más finas que h), se usa esta
    // última. ≤ -1 desactiva la corrección (normales = gradiente puro).
    float normal_min_dot = 0.5f;
};
struct MeshStats {
    int n[3] = {0, 0, 0};       // muestras por eje
    float h = 0;                // paso de muestreo (m)
    usize sampled = 0;          // evaluaciones del SDF en la fase de muestreo (hojas)
    usize corner_fill = 0;      // evaluaciones extra de esquinas de cubos mixtos (sin halo)
    usize sign_fixes = 0;       // signos clasificados corregidos por el valor exacto (autocorrección)
    usize vertices = 0, quads = 0;
    usize mixed_blocks = 0;     // bloques 8³ con valores guardados (ranuras de 2 KB)
    double t_sample = 0, t_cubes = 0, t_fill = 0, t_verts = 0, t_quads = 0, seconds = 0;
    usize normal_fixes = 0;     // normales de vértice sustituidas por la media de caras
    double t_normals = 0;
};
void mesh_scene(const sdf::Scene& scene, const LatticeMap& map, float cell_fraction, render::Mesh& out,
                const MeshOptions& mo, MeshStats* stats = nullptr);

// Malla "de bloques" de lo que realmente ve el solver: caras entre celdas sólidas y fluidas.
// Caras coplanares fusionadas (greedy meshing) para no generar millones de triángulos.
// skip_ground: ignora id 255 (suelo).
void mesh_voxels(const u8* solid_id, int nx, int ny, int nz, render::Mesh& out, bool skip_ground = true);

// ======================================================================================
//  RegionEval — evaluación de la escena restringida a una caja (espacio modelo).
//
//  Reproduce EXACTAMENTE el pliegue de Scene::eval (grupos en orden de índice, salto si
//  dist(caja_grupo, p) ≥ mejor, actualización si d < mejor) sobre una lista de candidatos
//  podada para la región (reglas a/b/c en refine). Valor E id devueltos idénticos a
//  Scene::eval para todo p de la región, bajo dos hipótesis: |∇d_h| ≤ L en la región y
//  d_h ≥ distancia a su caja FUERA de ella (la misma que usa Scene::eval para descartar).
//  Ojo a la semántica de Scene::eval: si p ya está dentro de un grupo (mejor < 0), los
//  siguientes se saltan → gana el PRIMER grupo que contiene p, no el más profundo.
//  dc[t] guarda d_h(c) de cada candidato: sirve para clasificar la región completa
//  (classify_id / classify_sign) y como cota heredada por las regiones hijas.
//  Uso típico: raíz init_all → refine por bloque → refine por sub-bloque → eval por punto.
// ======================================================================================
// Instrumentación (sólo para medir, docs/opt/geom.md): con -DCFD_GEOM_COUNTING se cuentan
// las llamadas a Scene::eval_group (k = 0: centros en refine, 1: puntos en eval).
#ifdef CFD_GEOM_COUNTING
inline std::atomic<u64> g_eval_count[2];
#define CFD_GEOM_COUNT(k) ::cfd::geom::g_eval_count[k].fetch_add(1, std::memory_order_relaxed)
#else
#define CFD_GEOM_COUNT(k) ((void)0)
#endif
// Conmutador A/B de las podas de RegionEval::refine (3 = a+b+c, 2 = a+b, 1 = sólo a).
#ifndef CFD_GEOM_PRUNE
#define CFD_GEOM_PRUNE 3
#endif

CFD_INLINE float aabb_gap(const Aabb& a, const Aabb& b) {
    const Vec3 d = vmax(vmax(a.lo - b.hi, b.lo - a.hi), Vec3(0.0f));
    return std::sqrt(dot(d, d));
}

struct RegionEval {
    static constexpr int k_max = 256;
    const sdf::Scene* scene = nullptr;
    const Aabb* boxes = nullptr;   // caja (modelo) por índice de grupo
    int n = 0;                     // candidatos conservados
    float lr = 3.0e38f;            // L·R + margen de ESTA región (cota heredable por las hijas)
    u8 g[k_max];                   // índices de grupo (0-based), en orden creciente
    float dc[k_max];               // d_g(centro) de cada candidato (válido tras refine)

    // Raíz: todos los grupos. `boxes` debe vivir mientras se use el evaluador.
    void init_all(const sdf::Scene& s, const Aabb* group_boxes) {
        scene = &s; boxes = group_boxes;
        n = static_cast<int>(s.group_count());
        lr = 3.0e38f;
        for (int i = 0; i < n; ++i) { g[i] = static_cast<u8>(i); dc[i] = 0.0f; }
    }
    // Poda los candidatos de `parent` para `region` ⊂ región del padre (centro c, radio R ya
    // multiplicado por L y con margen: LR) y evalúa d_h(c) de los conservados.
    // Dos podas exactas para el grupo g (h recorre los conservados anteriores, h < g):
    //  (a) dist(caja_g, región) ≥ min UB_h          → Scene::eval SALTA g (test de caja);
    //  (b) LB_g ≥ min max(UB_h, 0), con LB_g = dc_padre - lr_padre (cota heredada, sin evaluar)
    //      → g se evaluaría pero nunca bajaría el "mejor" del pliegue.
    //  (c) LB_g > max(UB_w, 0) para un testigo w CONSERVADO de cualquier índice (el de menor UB
    //      heredada): por inducción sobre el pliegue, con d_g > 0 quitar g sólo cambia el
    //      resultado si d_g fuese el mínimo final, y el final nunca supera max(d_w, 0).
    //      Es la única poda que puede quitar al PRIMER grupo (Scene::eval siempre lo evalúa).
    void refine(const RegionEval& parent, const Aabb& region, Vec3 c, float LR) {
        scene = parent.scene; boxes = parent.boxes;
        int w = -1;
        float W0 = 3.0e38f;
        if (parent.lr < 1.0e38f)
            for (int t = 0; t < parent.n; ++t) {
                const float ub = max_(parent.dc[t] + parent.lr, 0.0f);
                if (ub < W0) { W0 = ub; w = t; }
            }
        int m = 0;
        float M = 3.0e38f, M0 = 3.0e38f;
        for (int t = 0; t < parent.n; ++t) {
            const int h = parent.g[t];
            const float lb = parent.dc[t] - parent.lr;
            if (CFD_GEOM_PRUNE >= 3 && t != w && lb > W0) continue;   // (c)
            if (CFD_GEOM_PRUNE >= 2 && lb >= M0) continue;            // (b)
            if (aabb_gap(boxes[h], region) >= M) continue;        // (a)
            const float d = scene->eval_group(h, c);
            CFD_GEOM_COUNT(0);
            g[m] = static_cast<u8>(h); dc[m] = d; ++m;
            M = min_(M, d + LR);
            M0 = min_(M0, max_(d + LR, 0.0f));
        }
        n = m;
        lr = LR;
    }
    // == Scene::eval(p, gid) para p dentro de la región con la que se hizo refine.
    CFD_INLINE float eval(Vec3 p, int* gid = nullptr) const {
        float best = 1e30f;
        int bi = 0;
        for (int t = 0; t < n; ++t) {
            const int h = g[t];
            if (boxes[h].distance(p) >= best) continue;
            const float d = scene->eval_group(h, p);
            CFD_GEOM_COUNT(1);
            if (d < best) { best = d; bi = h + 1; }
        }
        if (gid) *gid = bi;
        return best;
    }
    // Clasificación de TODA la región (cotas LB_h = dc-LR ≤ d_h(p) ≤ UB_h = dc+LR) frente a un
    // umbral: devuelve -1 si Scene::eval(p) ≥ thr ∀p; id ≥ 1 si Scene::eval(p) < thr con ese id
    // ∀p; 0 si no se puede garantizar (hay que subdividir). Reproduce la semántica exacta del
    // pliegue: un grupo anterior con d < 0 hace que se salten los posteriores (su distancia
    // de caja ≥ 0 ≥ mejor), así que "gana" el PRIMERO en orden de índice, no el más profundo.
    CFD_INLINE int classify_id(float LR, float thr) const {
        float mn = 1e30f;
        for (int t = 0; t < n; ++t) mn = min_(mn, dc[t]);
        if (mn - LR >= thr) return -1;                     // mejor final ≥ min LB ≥ thr
        int b = -1;
        for (int t = 0; t < n; ++t) if (dc[t] + LR < thr) { b = t; break; }
        if (b < 0) return 0;
        const float ub = dc[b] + LR, gate = max_(ub, 0.0f);
        for (int t = 0; t < b; ++t) if (dc[t] - LR <= gate) return 0;   // h<b: nunca debe bajar de max(UB_b,0)
        if (ub >= 0.0f)                                                 // si UB_b<0 los g>b se saltan siempre
            for (int t = b + 1; t < n; ++t) if (dc[t] - LR <= ub) return 0;
        return g[b] + 1;
    }
    // Sólo el signo (d < 0 = dentro): -1 fuera ∀p, +1 dentro ∀p, 0 indeterminado.
    CFD_INLINE int classify_sign(float LR) const {
        float mn = 1e30f;
        for (int t = 0; t < n; ++t) mn = min_(mn, dc[t]);
        if (mn - LR >= 0.0f) return -1;
        for (int t = 0; t < n; ++t) {
            if (dc[t] - LR > 0.0f) continue;               // siempre fuera: deja mejor > 0
            return dc[t] + LR < 0.0f ? 1 : 0;              // primero que puede estar dentro
        }
        return 0;
    }
};

} // namespace cfd::geom
