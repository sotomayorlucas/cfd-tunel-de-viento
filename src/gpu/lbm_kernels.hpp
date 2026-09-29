// ============================================================================
//  gpu/lbm_kernels.hpp — generación (con gpu/spirv.hpp) de los compute shaders
//  del LBM D3Q19 en la iGPU. Uso interno de gpu/lbm_gpu.cpp.
//
//  Mismo algoritmo que lbm/solver.cpp (ver allí la física): Esoteric-Pull
//  in-place, poblaciones desplazadas f̃ = f − w (FP16S: ×2^15 en IEEE half),
//  colisión BGK/Regularizada + Smagorinsky, esponja τ0(x), entrada/campo lejano
//  y salida de equilibrio, rebote implícito en sólidos, cinta móvil y paredes
//  móviles con Ladd (kNearMoving), ley de pared/amortiguamiento (kNearWall),
//  y la pasada de contorno por nodo de pared: Bouzidi, modelo Slip, ruedas
//  interpoladas y fuerzas por intercambio de momento (manométricas + Wen).
//
//  Tres kernels, todos especializados AL GENERAR (constantes literales):
//   * step(paridad, macro): 1 hilo por celda (x contiguo → accesos coalescidos).
//   * boundary(paridad):    1 hilo por nodo de pared; escribe la población
//                           entrante y la fuerza/momento por (nodo, id).
//   * reduce:               1 grupo por trozo de registros de un mismo id →
//                           suma (subgrupo + memoria compartida) por paso.
// ============================================================================
#pragma once

#include "../core/config.hpp"

#include <vector>

namespace cfd::gpu::lbmk {

// Mapa de bindings (conjunto 0) compartido por TODOS los pipelines: un único conjunto de
// descriptores se enlaza una vez por búfer de comandos.
enum Binding : u32 {
    // 0..18: vista de la UBICACIÓN de la población i en el paso de esta paridad (ranura ls[i], desplazamiento
    //        lo[i] ya sumados en la dirección base del descriptor): todas las cargas/escrituras del kernel de
    //        celdas usan el MISMO índice (n o el par m) → sin 19 registros de dirección vivos (ver docs/opt/gpu.md).
    //        Hay un conjunto de descriptores por paridad. Modo par FP16S: palabras u32; si no, f16/f32.
    kBindLoc = 0,
    kBindFlags = 19,     // u8 [Npad]
    kBindSid = 20,       // u8 [Npad]
    kBindClass = 21,     // u32 por grupo de trabajo: 1 = "puro" (todos sus flags se deducen de las coordenadas)
    kBindMacro = 22,     // f32 [2 mitades × 4 campos × N]: ρ, ux, uy, uz (doble búfer)
    kBindMotion = 23,    // f32 [256 × 12]: v, ω, centro, contact_z, impermeable, (libre)
    kBindParams = 24,    // f32 [cabecera 16 + 8 por paso]
    kBindBad = 25,       // u32 [4]: divergencia
    kBindNodes = 26,     // u32 [nodos × kNodeWords]
    kBindRecs = 27,      // f32 [registros × 6]: fuerza y momento por (nodo, id)
    kBindChunks = 28,    // u32 [trozos × 2]: (primer registro, cuántos)
    kBindForces = 29,    // f32 [pasos × trozos × 6]
    kBindLocAlt = 30,    // 30..48: las mismas ubicaciones en f16 (modo par: escrituras de desplazamiento impar)
    kBindFlags32 = 49,   // u32 [Npad/4]: 4 flags por palabra (modo par)
    kBindSlot = 50,      // 50..68: ranura completa k (f16/f32; kernel de nodos de pared, índice P + n + o)
    kNumBindings = 69,
};
inline constexpr u32 kPushBytes = 16;        // [0] índice del paso dentro del lote
inline constexpr u32 kMaxBatch = 2048;       // pasos por lote (tabla de parámetros; la app pide ≤ 2000)
inline constexpr u32 kParamHeader = 16;
inline constexpr u32 kParamPerStep = 8;
// Cabecera de parámetros (índices f32; los marcados u32 se reinterpretan)
enum Param : u32 {
    kPK = 0,            // 18√2·C_s²
    kPWallC3 = 1,       // 3·C de la ley de pared
    kPWallFloor = 2,
    kPMacroBase = 3,    // (u32) desplazamiento de la mitad de campos macro que escribe este lote
    kPNuW = 4, kPInvNuW = 5, kPWwA0 = 6,   // ley de pared del modelo Slip
    kPNu = 7, kPXs = 8, kPNuMax = 9, kPSponge = 10,   // τ0(x) analítico (= Solver::Impl::build_tau)
    kPOmcb = 11,        // 1 − ω_b de la traza de Π^neq (viscosidad de volumen; sólo con Spec::bulk)
};
// Por paso (base kParamHeader + i·kParamPerStep)
enum StepParam : u32 { kSUin = 0, kSR = 1, kSUg = 2, kSR1 = 3, kSUg1 = 4, kSFout = 5 /* u32 */ };
// Nodo de pared: 0 n, 1 máscara de enlaces, 2 x|y<<16, 3 z|kind<<16, 4-6 normal, 7 yw, 8 yw^(1/7), 9 sgeo,
// 10-14 q[20] (u8), 15 ids de las 4 ranuras (u8), 16-19 registro por ranura.
inline constexpr u32 kNodeWords = 20;
inline constexpr u32 kSlots = 4;             // ids distintos por nodo de pared
inline constexpr u32 kMotionWords = 12;

struct Spec {
    // Dominio
    int nx = 0, ny = 0, nz = 0;
    u64 N = 0, S = 0, P = 0;
    i64 off[19] = {};
    // Física / almacenamiento
    bool fp16 = true;
    bool regularized = true;     // proyección de 2º orden (PR)
    bool rr = false;             // regularización recursiva de 3er orden (lbm::Collision::Recursive)
    bool bulk = false;           // traza de Π^neq con relajación propia (lbm::Config::bulk_omega > 0)
    bool wall = true;            // kNearWall presente (modelo de pared ≠ None)
    bool interp = true;          // rebote interpolado (Bouzidi): pasada de contorno escribe
    bool slip = true;            // modelo Slip (requiere interp)
    bool gauge = true, galilean = true;
    // Ejecución
    u32 wg = 256, sg = 16;       // grupo de trabajo / subgrupo del kernel de celdas
    u32 wg_nodes = 64;           // grupo de trabajo del kernel de nodos
    u32 gx = 1;                  // grupos en X del despacho 2D del kernel de celdas
    bool pair = true;            // 2 celdas contiguas en x por hilo con accesos u32 (f16×2) — ver docs/opt/gpu.md
    int ground = 2;              // 0 sin suelo, 1 fijo, 2 cinta móvil (flags deducibles de las coordenadas)
    bool classify = false;       // experimento: grupos "puros" sin carga de flags (medido: más lento, ver docs/opt/gpu.md)
    bool rte16 = true, denorm16 = true;
};

// Ubicación (ranura, índice de elemento base) del descriptor kBindLoc + i para la paridad p. En FP16S con
// desplazamiento impar la base se alinea a 4 B (elemento par) y el kernel suma 1 al índice.
void location(const Spec& s, int parity, int i, int* slot, u64* elem);
// Ranura y desplazamiento (sin P) de donde se CARGA la población i en un paso de paridad p (= step_ptrs).
void load_slot(const Spec& s, int parity, int i, int* slot, i64* off);
std::vector<u32> gen_step(const Spec& s, int parity, bool macro);
// Flags que tendría la celda (x, y, z) sin geometría (caras del túnel, suelo, capa sobre el suelo): los grupos de
// trabajo cuyas celdas coinciden TODAS con esto se marcan "puros" y el kernel no carga flags en ellos.
u8 expected_flags(const Spec& s, int x, int y, int z);
std::vector<u32> gen_boundary(const Spec& s, int parity);
std::vector<u32> gen_reduce(const Spec& s);

} // namespace cfd::gpu::lbmk
