// ============================================================================
//  lbm/solver.hpp — solver Lattice Boltzmann D3Q19 (contrato público).
//
//  Implementación en lbm/solver.cpp (retículo en lbm/lattice.hpp). Resumen del diseño:
//   * D3Q19, streaming in-place "Esoteric-Pull" (Lehmann 2022): UNA sola copia
//     de las poblaciones, mitad de memoria y de tráfico que A-B.
//   * Almacenamiento FP16S (poblaciones desplazadas f_i - w_i, escaladas, en
//     IEEE half con F16C) o FP32; aritmética siempre en FP32.
//   * Colisión BGK, Regularizada (2º orden) o Regularizada RECURSIVA (3er orden, defecto) + viscosidad de
//     volumen propia (bulk_omega) + Smagorinsky (LES) para Re altos (ver docs/FISICA.md §1.5).
//   * Kernel AVX2 sobre filas en X (8 celdas por registro). Entrada, salida,
//     campo lejano, sólidos y la cinta móvil van por la ruta vectorial (con
//     máscaras); la ruta escalar sólo atiende a vecinos de paredes móviles
//     que no son la cinta (ruedas girando).
//   * Rebote (bounce-back) implícito en sólidos (full-way: la población vuelve
//     2 pasos después; en estado estacionario la pared queda a mitad de enlace)
//     o INTERPOLADO (Bouzidi, con la superficie real; defecto de la app) con ley
//     de pared; cinta del suelo con Ladd; ruedas con Ladd (implícito) o con el
//     término de pared móvil de Bouzidi (interpolado).
//   * Fuerzas por intercambio de momento, desglosadas por id de sólido.
//  Unidades de red: celdas, pasos; ρ∞ = 1; u∞ ≤ ~0.1.
//
//  Notas de comportamiento (para otros módulos):
//   * Las celdas de las 6 caras son SIEMPRE fronteras (kInlet en x=0, y=0,
//     y=ny-1, z=nz-1 y z=0 sin suelo; kOutlet en x=nx-1): los ids de sólido que
//     el voxelizador ponga en las caras se ignoran. Con suelo, la capa z=0 es
//     sólido id 255 (y kMoving si la cinta se mueve). Un sólido con id 255 fuera
//     de z=0 (fuera de contrato) se trata como parte del suelo: se mueve con la
//     cinta (ruta escalar, Ladd exacto) y su fuerza se suma al id 255.
//   * forces_mean() promedia TODOS los pasos del último step(n): para medir un
//     estado estacionario, no incluir el transitorio en ese step(n).
//   * flags puede llevar además el bit interno 1<<5 (vecino móvil = sólo la
//     cinta). Usar siempre máscaras (flags & kSolid, ...), nunca igualdades.
//   * Las velocidades de pared (set_wall_motion y cinta) se escalan con la rampa
//     u(t)/u∞ para que ruedas, cinta y túnel arranquen juntos.
//   * Hilos: usa el pool global (cfd::pool()). Recomendado en el 155H: ver
//     docs/BENCHMARKS_LBM.md (Tuning::max_threads permite limitar sólo el LBM).
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "field.hpp"
#include <array>

namespace cfd::lbm {

enum class GroundMode : u8 {
    None,      // sin suelo: la capa z=0 es campo lejano (objeto en aire libre)
    Static,    // suelo fijo sin deslizamiento (túnel antiguo: crece capa límite)
    Moving,    // cinta móvil a u∞ (rolling road): el suelo real visto desde el coche
};
// Colisión (todas admiten Smagorinsky, cs_smag > 0):
//  BGK          → relajación simple (inestable a τ → ½).
//  Regularized  → proyección de Hermite de 2º orden del no equilibrio (PR).
//  Recursive    → regularización RECURSIVA de 3er orden (RR, Malaspinas 2015 / Coreixas 2017) con las 6 combinaciones
//                 de 3er orden que soporta D3Q19, en el equilibrio y en el no equilibrio. Defecto de la app: con PR a
//                 ν = 1e-4 y u∞ = 0.09 un túnel VACÍO se llenaba de ruido (±40 % de u∞, ver docs/FISICA.md §1.5).
enum class Collision : u8 { BGK, Regularized, Recursive };
enum class Precision : u8 { FP32, FP16S };
// Modelo de pared en las paredes FIJAS (ver docs/FISICA.md). Con dx de centímetros la capa límite real
// (≈ 1 % de la cuerda) es mucho más fina que una celda: un rebote "no deslizante" por enlace la convierte
// en una capa artificial de 2-5 celdas (fricción ×10, separación prematura, poca sustentación).
//  None   → rebote puro (no deslizante) con la viscosidad LES (Smagorinsky, Δ = 1 celda).
//  LogLaw → rebote no deslizante + ley de pared por VISCOSIDAD en la 1.ª celda junto a paredes FIJAS:
//           τ = max(3ν_w + ½, ½ + ¼(τ_LES − ½)) con ν_w = C·|u|^{3/4} ≥ ν (Werner-Wengle, u⁺ = 8.3·y⁺^{1/7};
//           y = ½ celda). El suelo de ¼ de la viscosidad LES es de estabilidad (sin él diverge en huecos de 1-2
//           celdas) y en la práctica domina: el efecto neto es moderado.
//  Slip   → (defecto; requiere BounceBack::Interpolated) IMPOSICIÓN EXACTA de la tensión de pared: en cada nodo
//           junto a una pared fija, los enlaces reciben el término de pared móvil de Bouzidi con una velocidad
//           tangente u_s = s·t̂ (t̂ = dirección de la velocidad tangente del nodo) elegida para que la fuerza
//           tangencial que el nodo transmite a la pared sea τ_w·A (Werner-Wengle con la distancia real y ν_w:
//           Reynolds físico), con s ∈ [0, |u_t|]. No deslizante en aristas vivas (condición de Kutta) y si la
//           capa límite está resuelta (y⁺ < 5). Conservativo en masa por nodo. Además amortigua Smagorinsky en
//           la 1.ª celda (τ = max(τ0, ½ + ¼(τ_LES − ½))). Placa plana a Re_x 10⁶-10⁷: cf 0.5-1.1× la turbulenta
//           (tests/test_aero.cpp [6]); NACA 0012 AR 3 a 6° con 42 celdas de cuerda y C_s = 0.10: CL 0.30.
enum class WallModel : u8 { None, LogLaw, Slip };
// Condición de rebote en sólidos FIJOS:
//  Implicit     → rebote implícito de Esoteric-Pull (full-way: la población vuelve 2 pasos después;
//                 pared a mitad de enlace → superficie en escalera).
//  Interpolated → pasada explícita tras cada paso que escribe la población entrante: half-way (q = ½) y,
//                 si set_geometry recibe la distancia a la superficie real (WallSdf), rebote interpolado
//                 lineal de Bouzidi, Firdaouss y Lallemand (2001) con q = fracción del enlace fluido→sólido
//                 hasta la pared: la pared deja de ser una escalera (esfera a Re = 100: Cd 1.30 → 1.20 frente a
//                 1.09 de Schiller-Naumann con D = 10). Paredes MÓVILES que no son el suelo (ruedas): también
//                 interpoladas, con el término de pared móvil g_q·6w(c·u_w) escrito en la pasada de contorno (el
//                 kernel no les suma Ladd): esfera giratoria Re 100 α 0.5, CL 0.62 → 0.51 y CD 1.46 → 1.31 frente
//                 al rebote implícito en escalera. La CINTA (id 255) sigue con rebote implícito full-way + Ladd.
//                 Verificado contra una referencia independiente A-B (tests/test_lbm.cpp, 2h-2j): |dρ|, |du| ~1e-7.
enum class BounceBack : u8 { Implicit, Interpolated };

// Distancia con signo a la superficie REAL del sólido (en celdas, < 0 dentro), en un punto de la red
// (coordenadas de celda). Debe ser segura entre hilos y seguir viva mientras el solver la use (hasta el
// siguiente set_geometry: set_wall_motion/set_ground también reconstruyen los enlaces).
struct WallSdf {
    float (*fn)(const void* ctx, Vec3 p_cells) = nullptr;
    const void* ctx = nullptr;
};

struct Config {
    int nx = 256, ny = 128, nz = 96;   // nx múltiplo de 8 (ruta vectorial)
    float u_inf = 0.08f;               // velocidad de entrada (red)
    float nu = 2e-4f;                  // viscosidad cinemática (red): τ = 3ν + ½
    float cs_smag = 0.16f;             // constante de Smagorinsky (0 = sin LES)
    Collision collision = Collision::Recursive;
    Precision precision = Precision::FP16S;
    GroundMode ground = GroundMode::Moving;
    float sponge_frac = 0.12f;         // fracción final del dominio (en X) con viscosidad creciente
    int ramp_steps = 600;              // rampa suave de u∞ tras reset/cambio de velocidad
    // ---- Extensiones compatibles (fase 2: física) ------------------------------------------
    WallModel wall_model = WallModel::Slip;
    float wall_nu = 0.0f;              // ν (red) con la que se calcula y⁺ en la ley de pared; 0 → nu.
                                       // La app pasa la del aire real → fricción al Reynolds físico.
    bool force_gauge = true;           // fuerzas MANOMÉTRICAS: se integra p - p∞ (ρ∞ = 1), no p = ρ/3.
                                       // Sin esto un sólido que no es superficie cerrada para el fluido
                                       // (rueda apoyada en el suelo, grupos que se tocan) recibe p∞·A_contacto,
                                       // ~80× la presión dinámica ½ρu² con u = 0.09.
    bool force_galilean = true;        // intercambio de momento galileanamente invariante en paredes móviles
                                       // (Wen et al., J. Comput. Phys. 266, 2014): resta u_w·(f_out - f_in).
    BounceBack bounce = BounceBack::Interpolated;
    // Viscosidad de VOLUMEN (colisiones regularizadas): la traza de Π^neq se relaja con su propio ω_b ∈ (0, 2)
    // (0 = con la ω de la cortante, como antes). Con ω_b = 1 (defecto) ν_b = (2/9)(1/ω_b − ½) = 1/9 en red:
    // amortigua las ondas acústicas y el modo par/impar de periodo 2 que con ν = 1e-4 (τ = 0.5003) apenas se
    // amortiguaban (pulso del arranque, rebotes en las caras). Sólo actúa sobre ∇·u (≈ 0 en flujo incompresible).
    // BGK la ignora. Ver docs/FISICA.md §1.5.
    float bulk_omega = 1.0f;
    // (Sólo con Collision::Recursive) CAPA junto a los cuerpos, en celdas (distancia de Chebyshev a un sólido que no es el
    // suelo, extendida a bloques completos de 8 en x), donde la colisión NO añade el término de 3er orden (queda la
    // proyección de 2º orden, con la que se calibró la física de pared). Con RR también en la capa límite los perfiles se
    // desprendían (NACA 0012 a 6°, Media: CL 0.30 → 0.13, desprendimiento desde el borde de ataque; Ahmed: CL −0.29 →
    // +0.82). Medido (docs/FISICA.md §1.5): 6-8 celdas recuperan las fuerzas y el campo lejano sigue limpio; con 12 el
    // ruido empieza a reaparecer junto al coche y con 24 vuelve. 0 = RR en todo el dominio.
    int rr_wall_layer = 8;
};

// Movimiento de pared en unidades de red (igual que LatticeMap::LatticeMotion).
struct WallMotion {
    Vec3 v{0, 0, 0};        // celdas/paso
    Vec3 omega{0, 0, 0};    // rad/paso
    Vec3 center{0, 0, 0};   // celdas
    // (fase 2) Zona de contacto con el suelo: las celdas de este id con z ≤ contact_z (celdas) se mueven con
    // la CINTA (u_g x̂, 0 si el suelo es fijo) en vez de con el movimiento rígido. Modela la huella aplanada
    // del neumático: sin ella, la cuña rueda-cinta de 1-2 celdas "bombea" (Cp ±13) y da una sustentación
    // espuria de la rueda (medido: SCz −0.8 m² en el eje delantero del F1 2022; con contact_z = 2: ≈ 0).
    float contact_z = -1.0f;
    // (fase 2) Pared impermeable: la velocidad real es TANGENTE a la superficie (rotación de un cuerpo de
    // revolución: ruedas). En la escalera de vóxeles u_w tiene componente normal a las caras y el rebote de
    // Ladd bombearía masa nodo a nodo; con esto se anula la masa neta que la pared inyecta en cada nodo.
    // false (defecto) = Ladd clásico (válido para paredes que deslizan sobre sí mismas, p.ej. la cinta).
    bool impermeable = false;
};

// Fuerzas del FLUIDO SOBRE cada sólido (unidades de red, por paso), por id (255 = suelo).
// Por enlace fluido n → sólido s = n + c_k (sólo fluido "verdadero": ni sólido ni frontera de equilibrio):
//   F_k = (f̃_k(τ) + f̃_k(τ-1) + 2w_k·[!force_gauge] - 6w_k c_k·u_w)·c_k  - [force_galilean]·6w_k (c_k·u_w)·u_w
// con f̃ = f - w las poblaciones desplazadas que guarda el solver. Con force_gauge (defecto) la referencia
// es el fluido en reposo a ρ∞ = 1: un sólido en fluido quieto no recibe fuerza aunque no esté cerrado.
struct ForceSample {
    std::array<Vec3, 256> force{};
    std::array<Vec3, 256> moment{};   // respecto a la referencia fijada con set_moment_reference
    Vec3 total{0, 0, 0};              // suma de ids 1..254 (excluye el suelo)
    Vec3 total_moment{0, 0, 0};
    u64 step = 0;
};

// ---- (fase 3) Backend externo (iGPU: src/gpu/lbm_gpu.*) -----------------------------------------
// Nodo de fluido "verdadero" junto a una pared, tal como lo usa la pasada de contorno (rebote interpolado,
// modelo Slip, paredes móviles y fuerzas). Público sólo para que un backend externo reutilice el
// preprocesado geométrico del solver en vez de duplicarlo.
struct WallNode {
    u32 n;               // celda de fluido
    u32 mask;            // bit k (1..18): n + c_k es sólido
    u16 x, y, z;         // coordenadas (momentos)
    u16 kind;            // bit 0: algún enlace a sólido MÓVIL; bit 1: algún enlace a sólido FIJO
    float nx, ny, nz;    // normal de la pared (unitaria, hacia el fluido)
    float yw;            // distancia del nodo a la pared (celdas) para la ley de pared
    float yw17;          // yw^(1/7)
    float area;          // área de pared asociada al nodo (celdas²)
    float sgeo;          // factor geométrico del deslizamiento (0 en aristas vivas)
    u8 q[20];            // q por enlace (código/254; ½ = 127 exacto)
};
// Vista del estado interno para el backend externo (válida hasta el siguiente cambio de revision()).
struct ExternalView {
    void* ddf = nullptr;              // poblaciones: dirección k en ddf + (k·S + P)·esize (FP16S: half de f̃·2^15)
    i64 S = 0, P = 0, N = 0;
    int esize = 4;
    i64 off[19] = {};                 // desplazamiento lineal de n + c_k
    const u8* flags = nullptr;        // incluye los bits internos 1<<5 (vecino móvil = sólo cinta) y 1<<6 (junto a pared)
    const u8* sid = nullptr;
    float *rho = nullptr, *ux = nullptr, *uy = nullptr, *uz = nullptr;   // campos macro propios de la CPU
    const float *tau0 = nullptr, *tau0sq = nullptr, *omc0 = nullptr;   // τ0(x) (esponja)
    const WallNode* nodes = nullptr;
    usize n_nodes = 0;
    const WallMotion* motion = nullptr;   // 256 movimientos de pared SIN rampa
    const bool* moving = nullptr;         // 256: id con movimiento de pared
    float wall_c3 = 0.0f;             // 3·C de la ley de pared (0 con el modelo Slip)
    float wall_floor = 0.25f;         // suelo de estabilidad de la ley de pared
};
// Resultado de n pasos hechos por el backend externo.
struct ExternalStep {
    int steps = 0;
    ForceSample last, mean;
    bool diverged = false;
    double mlups = 0, kernel_s = 0, force_s = 0;
};
// Llamado ANTES de que el solver lea o modifique su estado interno (poblaciones, flags, nodos, tablas):
// el backend debe terminar el trabajo en vuelo y dejar las poblaciones de la CPU al día.
using ExternalSync = void (*)(void* ctx);

class Solver {
public:
    Solver();
    ~Solver();
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    void init(const Config& cfg);              // reserva memoria, inicializa flujo uniforme
    const Config& config() const;

    // Geometría: solid_id[nx*ny*nz], 0 = fluido, 1..254 = grupo. El suelo lo gestiona el solver.
    // Conserva el flujo: las celdas que pasan a fluido se inicializan en equilibrio (ρ=1, u=0).
    void set_geometry(const u8* solid_id);
    // Igual, con la distancia a la superficie real para el rebote interpolado (Config::bounce).
    void set_geometry(const u8* solid_id, const WallSdf& sdf);
    void set_wall_motion(u8 id, const WallMotion& m);   // sólidos móviles (ruedas)
    void clear_wall_motions();
    void set_ground(GroundMode g);
    void set_inflow(float u_inf);              // rampa suave hacia el nuevo valor
    void set_viscosity(float nu);
    void set_smagorinsky(float cs);
    // Modelo de pared y ν de la ley de pared (0 → nu) en caliente (reclasifica bloques: O(N)).
    void set_wall_model(WallModel m, float wall_nu);
    void set_moment_reference(Vec3 cells);
    // ρ=1 en todo el dominio y reinicia la rampa: con ramp_steps > 0 el fluido parte del REPOSO
    // (u=0) y u∞ (entrada, campo lejano, cinta y ruedas) sube con smoothstep en ramp_steps pasos;
    // con ramp_steps = 0, arranque impulsivo con u=u∞ en todo el fluido. Reinicia steps() a 0.
    void reset_flow();

    // Avanza n pasos. Si update_macro, el ÚLTIMO paso escribe ρ,u en los campos de salida
    // (fusionado en el kernel: sin pasada extra). Las fuerzas se calculan cada paso.
    void step(int n, bool update_macro = true);

    FieldView field() const;                   // campos del último update_macro
    const ForceSample& forces() const;         // fuerzas del último paso
    u64 steps() const;
    double last_mlups() const;                 // millones de celdas·paso por segundo (último step())
    usize memory_bytes() const;
    bool diverged() const;                     // NaN / densidad fuera de rango detectada
    float current_u_inf() const;               // valor instantáneo de la rampa

    // ---- Extensiones (compatibles) -------------------------------------------------------
    // Media de las fuerzas sobre los n pasos del último step(n): menos ruido que forces()
    // (el rebote implícito de Esoteric-Pull tiene un leve vaivén par/impar).
    const ForceSample& forces_mean() const;
    // Masa total conservada (diagnóstico, O(N), no usar por cuadro): poblaciones de todas las
    // celdas de fluido (sin fronteras de equilibrio) + las que están "en vuelo" dentro de sólidos.
    double total_mass() const;
    // Ajustes finos del kernel (benchmarks). Valores 0 = automático.
    struct Tuning {
        int row_grain = 0;        // filas (y,z) por trozo dinámico (0 = ~4096 celdas: medido +13 % vs 4 filas)
        int nt_macro = -1;        // stores no temporales para ρ,u: -1 auto (no: medido -2.5 %), 0 no, 1 sí
        int prefetch = -1;        // distancia de prefetch software en bloques de 8 celdas (-1 auto = off: sin efecto medido)
        int pair_blocks = -1;     // pares de bloques puros entrelazados (16 celdas): -1 auto (no), 0 no, 1 sí
        int max_threads = 0;      // tope de hilos del pool que usa el kernel (0 = todos). Medido: con un pool de 20,
                                  // limitar a 14 es PEOR que un pool de 14 (no elige qué núcleos) → no recomendado.
        int ftz = -1;             // MXCSR FTZ|DAZ (subnormales → 0) durante el kernel: -1 auto (no), 0 no, 1 sí
    };
    void set_tuning(const Tuning& t);
    const Tuning& tuning() const;
    // Tiempo (s) de la última pasada de fuerzas y del último kernel (para perfiles).
    double last_force_seconds() const;
    double last_kernel_seconds() const;

    // ---- (fase 3) Backend externo: el solver sigue siendo el dueño del estado (config, geometría, rampa,
    // tiempo, fuerzas); el backend avanza los pasos con su copia de las poblaciones y publica el resultado.
    void set_external(ExternalSync sync, void* ctx);   // nullptr = desenganchar
    ExternalView external_view() const;
    u64 revision() const;                              // cambia si cambian poblaciones/geometría/tablas en la CPU
    void ramp_at(u64 step, float* u_in, float* wall_scale) const;   // u∞(t) y factor de las paredes móviles
    Vec3 moment_reference() const;
    void external_commit(const ExternalStep& r);       // t += steps, fuerzas, divergencia, MLUPS
    // Campos macro que devuelve field() (memoria del backend); nullptr = los propios.
    void set_field_override(const float* rho, const float* ux, const float* uy, const float* uz);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace cfd::lbm
