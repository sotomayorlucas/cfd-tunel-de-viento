// ============================================================================
//  lbm/solver.cpp — solver Lattice Boltzmann D3Q19 con streaming in-place
//  "Esoteric-Pull" (M. Lehmann, Computation 2022, 10(6):92), kernel AVX2.
//
//  ESQUEMA DE MEMORIA (una sola copia de las poblaciones, SoA por dirección)
//  -------------------------------------------------------------------------
//  f_k[n] vive en  buf[k·S + P + n]:  S = zancada por dirección (N + 2P + sesgo
//  anti-aliasing de caché), P = relleno delante/detrás (≥ nx·ny + nx + 8) que
//  absorbe los accesos fuera de rango de las caras del dominio SIN comprobar
//  índices ni envolver: sólo las celdas de frontera de equilibrio (que ignoran
//  lo que cargan) pueden salirse, y nadie lee lo que escriben ahí.
//
//  Por cada par opuesto (i, i+1) (i impar) y paridad p = t & 1, la celda n:
//     p = 0:  carga f_i ← [i+1][n],  f_{i+1} ← [i][n+c_i];  guarda f_i → [i][n+c_i],   f_{i+1} → [i+1][n]
//     p = 1:  carga f_i ← [i][n],    f_{i+1} ← [i+1][n+c_i]; guarda f_i → [i+1][n+c_i], f_{i+1} → [i][n]
//  Cada celda escribe EXACTAMENTE las posiciones que lee (conjunto de acceso):
//  sin carreras entre hilos y sin segunda copia. El puntero de escritura de f_i
//  es el de lectura de f_{i+1} y viceversa (ver StepPtrs).
//
//  REBOTE IMPLÍCITO: las celdas sólidas no se procesan. Demostrable: en un
//  esquema in-place sin carreras la población que sale de n hacia un sólido
//  vuelve a n DOS pasos después (rebote "full-way": el nodo sólido la guarda un
//  paso y la devuelve invertida). En estado estacionario la pared queda a mitad
//  de camino, igual que el rebote "half-way". tests/test_lbm.cpp lo verifica
//  contra una referencia A-B independiente con nodos sólidos de inversión.
//
//  Poblaciones DESPLAZADAS: se guarda f̃_k = f_k - w_k (en FP16S además ×32768).
//  ρ-1 = Σ f̃ sin cancelación catastrófica → más precisión en FP16 y en FP32.
// ============================================================================
#include "solver.hpp"
#include "lattice.hpp"
#include "../core/mem.hpp"
#include "../core/simd.hpp"
#include "../core/threadpool.hpp"
#include "../core/util.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

namespace cfd::lbm {
namespace {

namespace L = d3q19;
using simd::f8;

constexpr float kScale = 32768.0f;           // FP16S: f̃·2^15 en IEEE half (exacto: potencia de 2)
constexpr float kInvScale = 1.0f / 32768.0f;
constexpr float kSpongeNu = 0.12f;           // viscosidad máxima al final de la esponja (τ ≈ 0.86)

// Clase de cada bloque de 8 celdas (precalculada con SWAR sobre los flags en set_geometry).
enum BlockClass : u8 {
    kBlkSkip = 0,     // 8 sólidos: no se procesa (sólo se rellenan ρ,u en pasos macro)
    kBlkPure = 1,     // 8 fluidos normales: ruta AVX2 sin máscaras
    kBlkMasked = 2,   // hay sólidos o fronteras de equilibrio: AVX2 con mezclas por carril
    kBlkScalar = 3,   // pared móvil cerca (kNearMoving / kMoving) que no es sólo la cinta: ruta escalar
    kBlkGround = 4,   // capa z=1 sobre la cinta móvil: AVX2 con máscaras + corrección de Ladd constante
    kBlkWall = 8,     // BIT añadido a Pure/Masked/Ground: algún carril junto a pared con ley de pared
    kBlkLayer = 16,   // BIT: bloque en la capa junto a los cuerpos (kLayer en sus 8 carriles): sin término de 3er orden
};
// Bit interno de flags (no forma parte de field.hpp): celda kNearMoving cuyo ÚNICO vecino móvil
// es el suelo (cinta a u_g x̂). La corrección de pared móvil sobre ella es la misma en todas:
// sólo las direcciones 9 = (1,0,1) y 16 = (-1,0,1) reciben ±6 w u_g (c_x u_g ≠ 0 y vienen de z-1).
constexpr u8 kGroundOnly = 1u << 5;
// Bit interno de flags: celda de fluido (no frontera) con algún vecino sólido en las 18 direcciones
// → primera celda junto a la pared: ley de pared (WallModel::LogLaw). Los consumidores de FieldView
// usan máscaras (flags & kSolid), así que un bit interno más no les afecta.
constexpr u8 kNearWall = 1u << 6;
// Bit interno de flags: celda en la CAPA junto a los cuerpos (Config::rr_wall_layer celdas de un sólido que no es el
// suelo, extendida a bloques completos de 8 en x) donde la colisión recursiva NO añade su término de 3er orden (= la
// proyección de 2º orden). Ver collide y docs/FISICA.md §1.5: con RR también en la capa límite los perfiles se
// desprendían (NACA 0012 a 6°, Media: CL 0.30 → 0.13) y cambiaba toda la calibración.
constexpr u8 kLayer = 1u << 7;
// Distancia de la primera celda a la pared (rebote a mitad de enlace) en la ley de pared (modelo LogLaw).
constexpr float kWallY = 0.5f;
// Suelo de estabilidad de la ley de pared: τ ≥ ½ + kWallFloor·(τ_LES − ½) (ver wall_omc). Medido en el F1 2022
// a 2.5 M celdas: sin suelo diverge en ~400 pasos (huecos de 1-2 celdas bajo el fondo y junto a las ruedas);
// con 0.25 y 0.5 estable 2.5 pasos de flujo. Se toma el menor (menos fricción espuria).
constexpr float kWallFloor = 0.25f;

struct Motion { Vec3 v{0, 0, 0}, omega{0, 0, 0}, center{0, 0, 0}; float contact_z = -1.0f; Vec3 vc{0, 0, 0}; bool conserve = false; };

CFD_INLINE Vec3 wall_velocity(const Motion& m, float x, float y, float z) {
    if (z <= m.contact_z) return m.vc;   // huella de contacto: se mueve con la cinta
    return m.v + cross(m.omega, Vec3(x, y, z) - m.center);
}

// ---- Operaciones genéricas (float escalar o f8 AVX2): UN solo código de colisión -------------
CFD_INLINE float vfma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
CFD_INLINE f8 vfma(f8 a, f8 b, f8 c) { return simd::fmadd(a, b, c); }
CFD_INLINE float vfnma(float a, float b, float c) { return __builtin_fmaf(-a, b, c); }
CFD_INLINE f8 vfnma(f8 a, f8 b, f8 c) { return simd::fnmadd(a, b, c); }
CFD_INLINE float vsqrt(float a) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(a))); }
CFD_INLINE f8 vsqrt(f8 a) { return simd::sqrt(a); }

// Dos bloques de 8 celdas "entrelazados" (16 carriles): el MISMO código genérico de colisión
// genera dos cadenas de dependencias independientes que el núcleo fuera de orden solapa
// (segmentación software). Medido con llvm-mca: ver docs/opt/lbm.md.
struct f8x2 {
    f8 a, b;
    f8x2() = default;
    CFD_INLINE f8x2(f8 x, f8 y) : a(x), b(y) {}
    CFD_INLINE explicit f8x2(float s) : a(s), b(s) {}
    CFD_INLINE f8x2 operator+(f8x2 o) const { return {a + o.a, b + o.b}; }
    CFD_INLINE f8x2 operator-(f8x2 o) const { return {a - o.a, b - o.b}; }
    CFD_INLINE f8x2 operator*(f8x2 o) const { return {a * o.a, b * o.b}; }
    CFD_INLINE f8x2 operator/(f8x2 o) const { return {a / o.a, b / o.b}; }
    CFD_INLINE f8x2& operator+=(f8x2 o) { a += o.a; b += o.b; return *this; }
    CFD_INLINE f8x2& operator-=(f8x2 o) { a -= o.a; b -= o.b; return *this; }
    CFD_INLINE f8x2& operator*=(f8x2 o) { a *= o.a; b *= o.b; return *this; }
};
CFD_INLINE f8x2 vfma(f8x2 x, f8x2 y, f8x2 z) { return {vfma(x.a, y.a, z.a), vfma(x.b, y.b, z.b)}; }
CFD_INLINE f8x2 vfnma(f8x2 x, f8x2 y, f8x2 z) { return {vfnma(x.a, y.a, z.a), vfnma(x.b, y.b, z.b)}; }
CFD_INLINE f8x2 vsqrt(f8x2 x) { return {vsqrt(x.a), vsqrt(x.b)}; }

// Momentos "en flujo": las poblaciones se consumen POR PARES opuestos según se cargan
// (s = f_i + f_{i+1} → ρ, Π;  d = f_i - f_{i+1} → j) y se acumulan en 10 registros.
// Así nunca hay 19 valores vivos a la vez → sin derrames a la pila (medido: -80 vmovaps/bloque).
// ld(k) devuelve la población k (desplazada; en FP16S aún escalada por 2^15: el reescalado se
// pliega después en 1/ρ y en las FMAs de Π, nunca en las 19 poblaciones).
template <class V> struct Mom { V drho, jx, jy, jz, pxx, pyy, pzz, pxy, pxz, pyz; };

template <class V, class Ld>
CFD_INLINE Mom<V> moments(const Ld& ld) {
    // Dos juegos de acumuladores INDEPENDIENTES (ejes / diagonales) que se suman al final:
    // la cadena de dependencias de ρ baja de 10 a ~6 sumas (latencia 4 c/u) → más solape entre bloques.
    Mom<V> m, g;
    V a = ld(1), b = ld(2);                                   // ±x
    V s = a + b, d = a - b;
    m.drho = ld(0) + s; m.jx = d; m.pxx = s;
    a = ld(3); b = ld(4); s = a + b; d = a - b;               // ±y
    m.drho += s; m.jy = d; m.pyy = s;
    a = ld(5); b = ld(6); s = a + b; d = a - b;               // ±z
    m.drho += s; m.jz = d; m.pzz = s;
    a = ld(7); b = ld(8); s = a + b; d = a - b;               // ±(1,1,0)
    g.drho = s; g.jx = d; g.jy = d; g.pxx = s; g.pyy = s; m.pxy = s;
    a = ld(9); b = ld(10); s = a + b; d = a - b;              // ±(1,0,1)
    g.drho += s; g.jx += d; g.jz = d; g.pxx += s; g.pzz = s; m.pxz = s;
    a = ld(11); b = ld(12); s = a + b; d = a - b;             // ±(0,1,1)
    g.drho += s; g.jy += d; g.jz += d; g.pyy += s; g.pzz += s; m.pyz = s;
    a = ld(13); b = ld(14); s = a + b; d = a - b;             // ±(1,-1,0)
    g.drho += s; g.jx += d; g.jy -= d; g.pxx += s; g.pyy += s; m.pxy -= s;
    a = ld(15); b = ld(16); s = a + b; d = a - b;             // ±(1,0,-1)
    g.drho += s; g.jx += d; g.jz -= d; g.pxx += s; g.pzz += s; m.pxz -= s;
    a = ld(17); b = ld(18); s = a + b; d = a - b;             // ±(0,1,-1)
    g.drho += s; g.jy += d; g.jz -= d; g.pyy += s; g.pzz += s; m.pyz -= s;
    m.drho += g.drho; m.jx += g.jx; m.jy += g.jy; m.jz += g.jz;
    m.pxx += g.pxx; m.pyy += g.pyy; m.pzz += g.pzz;
    return m;
}

// Tensor de tensiones de no equilibrio Π^neq_ab = Σ f̃ c_a c_b - (ρ-1) c_s² δ_ab - ρ u_a u_b.
template <class V> struct Neq { V xx, yy, zz, xy, xz, yz; };

// Π_ab de m viene en unidades de ALMACENAMIENTO (×Sc); el reescalado 1/Sc se pliega en FMAs.
template <float Sc, class V>
CFD_INLINE Neq<V> noneq(const Mom<V>& m, V drho, V rho, V ux, V uy, V uz) {
    const V d3 = drho * V(1.0f / 3.0f);
    const V rx = rho * ux, ry = rho * uy, rz = rho * uz;
    Neq<V> q;
    if constexpr (Sc == 1.0f) {
        q.xx = vfnma(rx, ux, m.pxx - d3);
        q.yy = vfnma(ry, uy, m.pyy - d3);
        q.zz = vfnma(rz, uz, m.pzz - d3);
        q.xy = vfnma(rx, uy, m.pxy);
        q.xz = vfnma(rx, uz, m.pxz);
        q.yz = vfnma(ry, uz, m.pyz);
    } else {
        const V is(1.0f / Sc);
        q.xx = vfnma(rx, ux, vfma(m.pxx, is, V(0.0f) - d3));
        q.yy = vfnma(ry, uy, vfma(m.pyy, is, V(0.0f) - d3));
        q.zz = vfnma(rz, uz, vfma(m.pzz, is, V(0.0f) - d3));
        q.xy = vfnma(rx, uy, m.pxy * is);
        q.xz = vfnma(rx, uz, m.pxz * is);
        q.yz = vfnma(ry, uz, m.pyz * is);
    }
    return q;
}

// Smagorinsky (Hou et al. 1996): τ = ½(τ0 + sqrt(τ0² + 18√2 C_s² |Π^neq|/ρ)).
// Devuelve 1 - ω = 1 - 2/(τ0 + sqrt(...)) con UNA sola división. K = 18√2 C_s² (0 → τ = τ0).
template <class V>
CFD_INLINE V one_minus_omega(const Neq<V>& q, V inv_rho, V tau0, V tau0sq, V K) {
    // |Π|² en árbol (profundidad 3 en vez de 7 operaciones encadenadas).
    const V dxy = vfma(q.xx, q.xx, q.yy * q.yy);
    const V dz = vfma(q.zz, q.zz, V(2.0f) * (q.xy * q.xy));
    const V oo = V(2.0f) * vfma(q.xz, q.xz, q.yz * q.yz);
    const V qq = (dxy + dz) + oo;
    const V den = tau0 + vsqrt(vfma(K * vsqrt(qq), inv_rho, tau0sq));
    return V(1.0f) - V(2.0f) / den;
}

// Ley de pared de equilibrio (Werner & Wengle 1991) en la primera celda de fluido (a y = ½ de la pared):
//   u⁺ = 8.3·(y⁺)^{1/7}  →  u_τ = (|u| / (8.3·(y/ν_w)^{1/7}))^{7/8}
//   la tensión τ_w = ρu_τ² la transmite la celda si ν_ef·|u|/y = u_τ²  →  ν_ef = C·|u|^{3/4},
//   C = y·(8.3·(y/ν_w)^{1/7})^{-7/4} (constante por paso: C3 = 3C).
//   τ = max(3ν_ef + ½, τ0(x)): ν_ef ≥ ν → subcapa viscosa lineal (y⁺ < 11.8) y esponja respetadas.
// |u|^{3/4} con 3 raíces (vsqrtps): s1 = |u|, s2 = |u|^{1/2}, √(s1·s2) = |u|^{3/4}.
CFD_INLINE float vmax(float a, float b) { return a > b ? a : b; }
CFD_INLINE f8 vmax(f8 a, f8 b) { return simd::max(a, b); }
// Suelo de estabilidad: τ ≥ ½ + f·(τ_LES − ½), una fracción f de la viscosidad turbulenta de Smagorinsky
// (omc_s = 1 − 1/τ_LES de la celda). Sin él la celda junto a la pared queda con τ − ½ ~ 10⁻⁴ y diverge en
// huecos de 1-2 celdas (bajo el fondo, junto a las ruedas).
template <class V>
CFD_INLINE V wall_omc(V u2, V tau0, V C3, V omc_s, V ff) {
    const V s1 = vsqrt(u2);
    const V s2 = vsqrt(s1);
    const V u34 = vsqrt(s1 * s2);
    const V tw = vmax(vfma(C3, u34, V(0.5f)), tau0);
    const V ts = V(1.0f) / (V(1.0f) - omc_s);
    const V tf = vfma(ff, ts - V(0.5f), V(0.5f));
    return V(1.0f) - V(1.0f) / vmax(tw, tf);
}
// C3 = 3·C de la ley de pared para la viscosidad ν_w (red).
inline float wall_c3(float nu_w) {
    const float A = 8.3f * std::pow(kWallY / std::max(nu_w, 1e-9f), 1.0f / 7.0f);
    return 3.0f * kWallY * std::pow(A, -1.75f);
}

// Colisión + escritura por pares: st(k, v) guarda la población post-colisión k.
//   Equilibrio desplazado: f̃eq = w (ρ-1 + ρ(a + a²/2 - 3u²/2)),  a = 3 c·u.
//   BGK:         f̃* = f̃eq + (1-ω)(f̃ - f̃eq)          (ld(k) relee f̃_k: línea aún en L1)
//   Regularizado: f̃* = f̃eq + (1-ω)·w·4.5·(c_a c_b - δ/3)Π^neq_ab  (proyección Hermite de 2º orden)
// Con (1-ω) = 0 el resultado es EXACTAMENTE f̃eq (fronteras de equilibrio).
// Sc = escala de almacenamiento (1 en FP32, 2^15 en FP16S): se pliega en los pesos → 0 mul extra.
// ORDEN: en cada par se leen f_i y f_{i+1} ANTES de escribir, porque st(i) pisa la posición de
// la que se leyó f_{i+1} (y viceversa).
template <Collision C, float Sc, bool Bulk, class V, class Ld, class St>
CFD_INLINE void collide(const Ld& ld, const St& st, V drho, V rho, V ux, V uy, V uz, const Neq<V>& q, V omc, V omcb, V om3) {
    const V ux3 = V(3.0f) * ux, uy3 = V(3.0f) * uy, uz3 = V(3.0f) * uz;
    const V c3 = V(0.0f) - vfma(ux3, ux, vfma(uy3, uy, uz3 * uz));   // -3u²
    const V hr = V(0.5f) * rho;
    // Pesos (×Sc) plegados en ρ, ρ/2 y ρ-1 por clase de peso (ejes / diagonales), estilo FluidX3D:
    // f̃eq± = W(ρ-1) + (Wρ/2)(a² - 3u²) ± (Wρ) a   →  5 operaciones por par (antes 7).
    constexpr float W0 = L::w[0] * Sc, W1 = L::w[1] * Sc, W2 = L::w[7] * Sc;
    const V r1 = V(W1) * rho, h1 = V(W1) * hr, d1 = V(W1) * drho;
    const V r2 = V(W2) * rho, h2 = V(W2) * hr, d2 = V(W2) * drho;
    if constexpr (C != Collision::BGK) {
        // post = f̃eq + (1-ω)·[W·4.5·(c c - δ/3):Π^neq]. La proyección NO depende de ω → se calcula en
        // paralelo con la cadena larga de Smagorinsky (2 sqrt + div) y (1-ω) entra con UNA FMA final
        // por dirección: la ruta crítica tras ω pasa de ~6 operaciones a 1.
        const V mxx = V(4.5f) * q.xx, myy = V(4.5f) * q.yy, mzz = V(4.5f) * q.zz;
        const V mxy = V(9.0f) * q.xy, mxz = V(9.0f) * q.xz, myz = V(9.0f) * q.yz;
        // Traza (parte de VOLUMEN) separada de la desviadora: 4.5·Q_i:Π = 4.5·Q_i:Π_dev + trm·(|c_i|² − 1) con
        // trm = 1.5·tr Π. Ejes (|c|² = 1): sólo desviadora; diagonales (|c|² = 2): + trm; reposo: − trm. La traza se
        // relaja con (1 − ω_b) (omcb; = omc sin viscosidad de volumen propia) y se pliega en el término constante de
        // cada clase de peso → 1 FMA por celda en vez de 1 por dirección.
        const V trm = (mxx + myy + mzz) * V(1.0f / 3.0f);
        const V rx = mxx - trm, ry = myy - trm, rz = mzz - trm;
        // Con Bulk (plantilla) omcb es una constante del paso: la traza NO depende de la cadena de Smagorinsky y no
        // alarga la ruta crítica (llvm-mca: con una mezcla en tiempo de ejecución, +15 % de ciclos por bloque).
        const V tb = (Bulk ? omcb : omc) * trm;
        const V d2b = vfma(V(W2), tb, d2);
        st(0, vfnma(V(W0), tb, V(W0) * vfma(hr, c3, drho)));
        const V rxy = rx + ry, rxz = rx + rz, ryz = ry + rz;
        if constexpr (C == Collision::Regularized) {
            auto pair = [&](int i, V a, V R, float W, V rW, V hW, V dW) {
                const V t = vfma(hW, vfma(a, a, c3), dW);
                const V ra = rW * a;
                const V pr = V(W) * R;
                st(i, vfma(omc, pr, t + ra));
                st(i + 1, vfma(omc, pr, t - ra));
            };
            pair(1, ux3, rx, W1, r1, h1, d1);
            pair(3, uy3, ry, W1, r1, h1, d1);
            pair(5, uz3, rz, W1, r1, h1, d1);
            pair(7, ux3 + uy3, rxy + mxy, W2, r2, h2, d2b);
            pair(9, ux3 + uz3, rxz + mxz, W2, r2, h2, d2b);
            pair(11, uy3 + uz3, ryz + myz, W2, r2, h2, d2b);
            pair(13, ux3 - uy3, rxy - mxy, W2, r2, h2, d2b);
            pair(15, ux3 - uz3, rxz - mxz, W2, r2, h2, d2b);
            pair(17, uy3 - uz3, ryz - myz, W2, r2, h2, d2b);
        } else {
            // REGULARIZACIÓN RECURSIVA (Malaspinas 2015; Coreixas et al., PRE 96, 033306, 2017): el no equilibrio de
            // 3er orden se reconstruye de Π^neq, a3neq_αβγ = u_α Π_βγ + u_β Π_αγ + u_γ Π_αβ, proyectado sobre las 6
            // combinaciones de polinomios de Hermite de 3er orden que D3Q19 soporta (normas de la red 2/27 la suma y
            // 6/27 la diferencia → coeficientes 1/(2c_s⁶) y 1/(6c_s⁶)). Por dirección queda un término IMPAR en c:
            //   ejes ±x: ∓9w(a_xyy + a_xzz) (y cíclicos);  diagonales (c_x, c_y, 0): 9w(c_y a_xxy + c_x a_xyy) (y cíclicos).
            // Se relaja con la ω de la cortante. Con la proyección de 2º orden esos momentos quedan libres (se ponen a 0
            // cada paso) y a ν → 0 con Ma ≈ 0.16 el esquema es LINEALMENTE INESTABLE: un túnel VACÍO se llenaba de ruido
            // de ±40 % de u∞ (docs/FISICA.md §1.5). El equilibrio sigue siendo de 2º orden: su término de 3er orden
            // (ρuuu) se probó y no aporta estabilidad (medido), y cuesta ~15 operaciones más por celda.
            // Forma "par ± impar": E = f̃eq_par + (1−ω)·Π-término, O = f̃eq_impar + (1−ω)·a3neq-término; st = E ± O.
            // llvm-mca (Golden Cove, ciclos por bloque de 8, FP16S / FP32): proyección de 2º orden 129 / 107; esta forma
            // 131 / 123; con el término sumado a ra (forma de la proyección) 148 / 121; recalculando a3neq de m_αβ por
            // par (menos valores vivos, pensado para la iGPU) 135-137 / 123-132 → se queda esta (también en la iGPU,
            // donde las tres dan el mismo tiempo: docs/opt/gpu.md).
            const V ux2 = ux + ux, uy2 = uy + uy, uz2 = uz + uz;
            const V k9 = V(9.0f * W2);   // 9·w_diag (con la escala de almacenamiento); ejes: 9·w_eje = 2·(9·w_diag)
            const V nxxy = k9 * vfma(ux2, q.xy, uy * q.xx), nxyy = k9 * vfma(uy2, q.xy, ux * q.yy);
            const V nxzz = k9 * vfma(uz2, q.xz, ux * q.zz), nxxz = k9 * vfma(ux2, q.xz, uz * q.xx);
            const V nyzz = k9 * vfma(uz2, q.yz, uy * q.zz), nyyz = k9 * vfma(uy2, q.yz, uz * q.yy);
            auto pair = [&](int i, V a, V R, float W, V rW, V hW, V dW, V p3) {
                const V t = vfma(hW, vfma(a, a, c3), dW);
                const V ra = rW * a;
                const V pr = V(W) * R;
                const V E = vfma(omc, pr, t);
                const V O = vfma(om3, p3, ra);
                st(i, E + O);
                st(i + 1, E - O);
            };
            const V m2 = V(-2.0f);
            pair(1, ux3, rx, W1, r1, h1, d1, m2 * (nxyy + nxzz));
            pair(3, uy3, ry, W1, r1, h1, d1, m2 * (nxxy + nyzz));
            pair(5, uz3, rz, W1, r1, h1, d1, m2 * (nyyz + nxxz));
            pair(7, ux3 + uy3, rxy + mxy, W2, r2, h2, d2b, nxxy + nxyy);
            pair(9, ux3 + uz3, rxz + mxz, W2, r2, h2, d2b, nxzz + nxxz);
            pair(11, uy3 + uz3, ryz + myz, W2, r2, h2, d2b, nyzz + nyyz);
            pair(13, ux3 - uy3, rxy - mxy, W2, r2, h2, d2b, nxyy - nxxy);
            pair(15, ux3 - uz3, rxz - mxz, W2, r2, h2, d2b, nxzz - nxxz);
            pair(17, uy3 - uz3, ryz - myz, W2, r2, h2, d2b, nyzz - nyyz);
        }
    } else {
        (void)omcb; (void)om3;   // BGK: la traza se relaja con ω (sin viscosidad de volumen propia)
        auto pair = [&](int i, V a, V rW, V hW, V dW) {
            const V fi = ld(i), fj = ld(i + 1);
            const V t = vfma(hW, vfma(a, a, c3), dW);
            const V ra = rW * a;
            const V e1 = t + ra, e2 = t - ra;
            st(i, vfma(omc, fi - e1, e1));
            st(i + 1, vfma(omc, fj - e2, e2));
        };
        {
            const V f0 = ld(0);
            const V e0 = V(W0) * vfma(hr, c3, drho);
            st(0, vfma(omc, f0 - e0, e0));
        }
        pair(1, ux3, r1, h1, d1);
        pair(3, uy3, r1, h1, d1);
        pair(5, uz3, r1, h1, d1);
        pair(7, ux3 + uy3, r2, h2, d2);
        pair(9, ux3 + uz3, r2, h2, d2);
        pair(11, uy3 + uz3, r2, h2, d2);
        pair(13, ux3 - uy3, r2, h2, d2);
        pair(15, ux3 - uz3, r2, h2, d2);
        pair(17, uy3 - uz3, r2, h2, d2);
    }
}

// ---- Contexto de un paso (todo lo que lee el kernel; constante durante el paso) -------------
struct alignas(64) KCtx {
    const void* ld[L::Q];     // fhn[k] = ld[k][n]   (ya incluye relleno, paridad y desplazamiento al vecino)
    void* st[L::Q];           // fhn[k] → st[k][n]   (= ld[par(k)])
    const u8* flags;
    const u8* sid;
    const u8* cls;            // clase por bloque de 8
    const u32* rows;          // filas a procesar (índice fila = y + ny·z)
    const float* tau0;        // τ0(x) (esponja)
    const float* tau0sq;
    const float* omc0;        // 1 - 1/τ0(x): sin LES (K = 0) se usa directamente (sin sqrt/div en la ruta crítica)
    float K;                  // 18√2 C_s²
    float wallC3;             // 3·C de la ley de pared (ver wall_omc)
    float wall_floor;         // fracción mínima de la viscosidad LES en celdas junto a pared
    float omcb;               // 1 − ω_b de la traza de Π^neq (viscosidad de volumen), si bulk
    bool bulk;                // viscosidad de volumen propia (si no, la traza se relaja con la omc de la celda)
    float u_in;               // u∞ instantánea (rampa)
    float u_ground;           // velocidad de la cinta (0 si el suelo no es móvil)
    int nx, ny, nz, nbx;
    i64 nxny;
    i64 off[L::Q];
    float *rho, *ux, *uy, *uz;
    const Motion* motion;     // 256 entradas (ya escaladas por la rampa)
    std::atomic<u32>* bad;
    bool nt;                  // stores no temporales para ρ,u
    bool pair;                // procesar pares de bloques puros entrelazados (16 celdas)
    bool ftz;                 // activar FTZ|DAZ durante el kernel
    bool mexp;                // paredes móviles ≠ suelo con rebote interpolado explícito (sin Ladd en el kernel)
    int pf;                   // distancia de prefetch (bloques), 0 = off
};

template <Precision P> struct Store;
template <> struct Store<Precision::FP32> { using T = float; };
template <> struct Store<Precision::FP16S> { using T = u16; };

CFD_INLINE void store_macro(const KCtx& k, i64 n0, f8 r, f8 x, f8 y, f8 z) {
    if (k.nt) {
        r.stream(k.rho + n0); x.stream(k.ux + n0); y.stream(k.uy + n0); z.stream(k.uz + n0);
    } else {
        r.store_a(k.rho + n0); x.store_a(k.ux + n0); y.store_a(k.uy + n0); z.store_a(k.uz + n0);
    }
}

// ---- Ruta AVX2: 8 celdas contiguas en x ------------------------------------------------------
template <Precision P> inline constexpr float kSc = P == Precision::FP32 ? 1.0f : kScale;

template <Precision P, Collision C, bool Macro, bool Bulk, bool Masked, bool Ground = false, bool Wall = false>
CFD_INLINE void block_vec(const KCtx& k, i64 n0, int x0, __m256& bad, __m256 m3) {
    using T = typename Store<P>::T;
    // Cinta móvil (sólo clase Ground): δ = 6 w_9 u_g en unidades de almacenamiento, en los carriles kGroundOnly.
    __m256 gdelta = _mm256_setzero_ps(), mg = _mm256_setzero_ps();
    if constexpr (Ground) {
        const __m256i fl = simd::load_u8x8(k.flags + n0).v;
        mg = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_and_si256(fl, _mm256_set1_epi32(kGroundOnly)), _mm256_setzero_si256()));
        gdelta = _mm256_and_ps(mg, _mm256_set1_ps(kSc<P> * 6.0f * L::w[9] * k.u_ground));
    }
    // Carga de la población i para los 8 carriles: FP32 directo; FP16S vcvtph2ps SIN reescalar.
    auto ld = [&](int i) -> f8 {
        f8 v;
        if constexpr (P == Precision::FP32) v = f8::load(static_cast<const float*>(k.ld[i]) + n0);
        else v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const u16*>(k.ld[i]) + n0)));
        if constexpr (Ground) {   // i es constante tras el inlining → sin ramas
            if (i == 9) v = v + f8(gdelta);
            if (i == 16) v = v - f8(gdelta);
        }
        return v;
    };
    if (k.pf) {   // prefetch software opcional (medido: sin efecto claro → desactivado por defecto)
#pragma GCC unroll 19
        for (int i = 0; i < L::Q; ++i)
            _mm_prefetch(reinterpret_cast<const char*>(static_cast<const T*>(k.ld[i]) + n0 + 8 * k.pf), _MM_HINT_T0);
    }
    // Momentos en unidades de almacenamiento: 1/Sc se pliega en ρ-1, en 1/ρ para u y en las FMAs de Π.
    const Mom<f8> m = moments<f8>(ld);
    f8 drho = (P == Precision::FP16S) ? m.drho * f8(kInvScale) : m.drho;
    f8 rho = f8(1.0f) + drho;
    const f8 inv = f8(1.0f) / rho;
    const f8 invj = (P == Precision::FP16S) ? inv * f8(kInvScale) : inv;
    f8 ux = m.jx * invj, uy = m.jy * invj, uz = m.jz * invj;

    __m256 msol = _mm256_setzero_ps(), meq = _mm256_setzero_ps();
    if constexpr (Masked) {
        // Flags de 8 celdas → máscaras de carril (1 carga de 64 bits + vpmovzxbd).
        const __m256i fl = simd::load_u8x8(k.flags + n0).v;
        const __m256i zero = _mm256_setzero_si256();
        auto bits = [&](int b) {
            return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_and_si256(fl, _mm256_set1_epi32(b)), zero));
        };
        msol = bits(kSolid);
        const __m256 min = bits(kInlet), mout = bits(kOutlet);
        meq = _mm256_or_ps(min, mout);
        // Salida (x = nx-1, siempre carril 7): u copiada del carril 6 (x-1) con un vpermps.
        const __m256i sh = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        const __m256 uxs = _mm256_permutevar8x32_ps(_mm256_andnot_ps(msol, ux), sh);
        const __m256 uys = _mm256_permutevar8x32_ps(_mm256_andnot_ps(msol, uy), sh);
        const __m256 uzs = _mm256_permutevar8x32_ps(_mm256_andnot_ps(msol, uz), sh);
        ux = _mm256_blendv_ps(_mm256_blendv_ps(ux, uxs, mout), _mm256_set1_ps(k.u_in), min);
        uy = _mm256_andnot_ps(min, _mm256_blendv_ps(uy, uys, mout));
        uz = _mm256_andnot_ps(min, _mm256_blendv_ps(uz, uzs, mout));
        rho = _mm256_blendv_ps(rho, _mm256_set1_ps(1.0f), meq);
        drho = _mm256_andnot_ps(meq, drho);
    }
    const Neq<f8> q = noneq<kSc<P>>(m, drho, rho, ux, uy, uz);
    // Rama uniforme en todo el paso (predicción perfecta): sin LES, 1-ω sale de tabla por plano x.
    f8 omc = k.K > 0.0f ? one_minus_omega(q, inv, f8::load(k.tau0 + x0), f8::load(k.tau0sq + x0), f8(k.K))
                        : f8::load(k.omc0 + x0);
    if constexpr (Wall) {
        // Ley de pared en los carriles kNearWall (sustituye a Smagorinsky): velocidad relativa a la pared
        // (la cinta en los carriles kGroundOnly; paredes fijas en el resto).
        const __m256i fl = simd::load_u8x8(k.flags + n0).v;
        const __m256 mw = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_and_si256(fl, _mm256_set1_epi32(kNearWall)), _mm256_setzero_si256()));
        f8 urx = ux;
        if constexpr (Ground) urx = ux - f8(_mm256_and_ps(mg, _mm256_set1_ps(k.u_ground)));
        const f8 u2 = vfma(urx, urx, vfma(uy, uy, uz * uz));
        const f8 omw = wall_omc(u2, f8::load(k.tau0 + x0), f8(k.wallC3), omc, f8(k.wall_floor));
        omc = _mm256_blendv_ps(omc, omw, mw);
    }
    f8 omcb = f8(k.omcb);   // (sólo con Bulk)
    if constexpr (Masked) { omc = _mm256_andnot_ps(meq, omc); if constexpr (Bulk) omcb = _mm256_andnot_ps(meq, omcb); }

    // Detección barata de divergencia: ρ fuera de (0.2, 5) o NaN (comparaciones no ordenadas).
    {
        __m256 b = _mm256_or_ps(_mm256_cmp_ps(rho, _mm256_set1_ps(0.2f), _CMP_NGT_UQ),
                                _mm256_cmp_ps(rho, _mm256_set1_ps(5.0f), _CMP_NLT_UQ));
        if constexpr (Masked) b = _mm256_andnot_ps(msol, b);
        bad = _mm256_or_ps(bad, b);
    }
    if constexpr (Macro) {
        if constexpr (Masked) {
            store_macro(k, n0, _mm256_blendv_ps(rho, _mm256_set1_ps(1.0f), msol), _mm256_andnot_ps(msol, ux),
                        _mm256_andnot_ps(msol, uy), _mm256_andnot_ps(msol, uz));
        } else {
            store_macro(k, n0, rho, ux, uy, uz);
        }
    }

    // Escritura. Carriles sólidos: se reescribe el valor original (misma posición que se leyó
    // como población opuesta en ESTE paso; nadie más la toca) → store completo sin vmaskmov.
    __m128i m16 = _mm_setzero_si128();
    if constexpr (Masked && P == Precision::FP16S) {
        const __m256i mi = _mm256_castps_si256(msol);
        m16 = _mm_packs_epi32(_mm256_castsi256_si128(mi), _mm256_extracti128_si256(mi, 1));
    }
    auto st = [&](int i, f8 v) {
        if constexpr (P == Precision::FP32) {
            float* dst = static_cast<float*>(k.st[i]) + n0;
            __m256 x = v;
            if constexpr (Masked) x = _mm256_blendv_ps(x, _mm256_loadu_ps(dst), msol);
            _mm256_storeu_ps(dst, x);
        } else {
            u16* dst = static_cast<u16*>(k.st[i]) + n0;
            __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            if constexpr (Masked) h = _mm_blendv_epi8(h, _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst)), m16);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), h);
        }
    };
    // Término de 3er orden de la RR: (1−ω) salvo en la capa junto a los cuerpos (m3 = 0 en todo el bloque).
    collide<C, kSc<P>, Bulk>(ld, st, drho, rho, ux, uy, uz, q, omc, omcb, f8(_mm256_and_ps(omc, m3)));
}

// ---- Par de bloques puros (16 celdas) con cadenas entrelazadas -------------------------------
template <Precision P, Collision C, bool Macro, bool Bulk>
CFD_INLINE void block_vec2(const KCtx& k, i64 n0, int x0, __m256& bad) {
    auto ld1 = [&](int i, i64 n) -> f8 {
        if constexpr (P == Precision::FP32) return f8::load(static_cast<const float*>(k.ld[i]) + n);
        else return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const u16*>(k.ld[i]) + n)));
    };
    auto ld = [&](int i) -> f8x2 { return {ld1(i, n0), ld1(i, n0 + 8)}; };
    const Mom<f8x2> m = moments<f8x2>(ld);
    const f8x2 drho = (P == Precision::FP16S) ? m.drho * f8x2(kInvScale) : m.drho;
    const f8x2 rho = f8x2(1.0f) + drho;
    const f8x2 inv = f8x2(1.0f) / rho;
    const f8x2 invj = (P == Precision::FP16S) ? inv * f8x2(kInvScale) : inv;
    const f8x2 ux = m.jx * invj, uy = m.jy * invj, uz = m.jz * invj;
    const Neq<f8x2> q = noneq<kSc<P>>(m, drho, rho, ux, uy, uz);
    const f8x2 omc = k.K > 0.0f ? one_minus_omega(q, inv, f8x2(f8::load(k.tau0 + x0), f8::load(k.tau0 + x0 + 8)),
                                                  f8x2(f8::load(k.tau0sq + x0), f8::load(k.tau0sq + x0 + 8)), f8x2(k.K))
                                : f8x2(f8::load(k.omc0 + x0), f8::load(k.omc0 + x0 + 8));
    {
        const __m256 lo = _mm256_set1_ps(0.2f), hi = _mm256_set1_ps(5.0f);
        const __m256 b0 = _mm256_or_ps(_mm256_cmp_ps(rho.a, lo, _CMP_NGT_UQ), _mm256_cmp_ps(rho.a, hi, _CMP_NLT_UQ));
        const __m256 b1 = _mm256_or_ps(_mm256_cmp_ps(rho.b, lo, _CMP_NGT_UQ), _mm256_cmp_ps(rho.b, hi, _CMP_NLT_UQ));
        bad = _mm256_or_ps(bad, _mm256_or_ps(b0, b1));
    }
    if constexpr (Macro) {
        store_macro(k, n0, rho.a, ux.a, uy.a, uz.a);
        store_macro(k, n0 + 8, rho.b, ux.b, uy.b, uz.b);
    }
    auto st1 = [&](int i, i64 n, f8 v) {
        if constexpr (P == Precision::FP32) _mm256_storeu_ps(static_cast<float*>(k.st[i]) + n, v);
        else _mm_storeu_si128(reinterpret_cast<__m128i*>(static_cast<u16*>(k.st[i]) + n),
                              _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    };
    auto st = [&](int i, f8x2 v) { st1(i, n0, v.a); st1(i, n0 + 8, v.b); };
    collide<C, kSc<P>, Bulk>(ld, st, drho, rho, ux, uy, uz, q, omc, f8x2(k.omcb), omc);
}

// ---- Ruta escalar (bloques con paredes móviles cerca; poco frecuente) ------------------------
// Carga/escritura escalar en unidades de ALMACENAMIENTO (FP16S: f̃·2^15, sin reescalar).
template <Precision P>
CFD_INLINE float load_s(const void* p, i64 n) {
    if constexpr (P == Precision::FP32) return static_cast<const float*>(p)[n];
    else return f16_to_f32(static_cast<const u16*>(p)[n]);
}
template <Precision P>
CFD_INLINE void store_s(void* p, i64 n, float v) {
    if constexpr (P == Precision::FP32) static_cast<float*>(p)[n] = v;
    else static_cast<u16*>(p)[n] = f32_to_f16(v);
}

CFD_INLINE void macro_solid(const KCtx& k, i64 n, u8 fl, int x, int y, int z) {
    Vec3 u{0, 0, 0};
    if (fl & kMoving) u = wall_velocity(k.motion[k.sid[n]], float(x), float(y), float(z));
    k.rho[n] = 1.0f; k.ux[n] = u.x; k.uy[n] = u.y; k.uz[n] = u.z;
}

template <Precision P, Collision C, bool Macro, bool Bulk>
CFD_NOINLINE u32 block_scalar(const KCtx& k, i64 n0, int x0, int y, int z) {
    u32 bad = 0;
    float pux = 0.0f, puy = 0.0f, puz = 0.0f;   // u (pre-colisión) de la celda x-1
    for (int l = 0; l < 8; ++l) {
        const i64 n = n0 + l;
        const int x = x0 + l;
        const u8 fl = k.flags[n];
        if (fl & kSolid) {
            if constexpr (Macro) macro_solid(k, n, fl, x, y, z);
            pux = puy = puz = 0.0f;
            continue;
        }
        float f[L::Q];   // en unidades de almacenamiento (FP16S: escaladas por 2^15)
#pragma GCC unroll 19
        for (int i = 0; i < L::Q; ++i) f[i] = load_s<P>(k.ld[i], n);
        Vec3 uwall{0, 0, 0};   // suma de velocidades de las paredes móviles vecinas (ley de pared)
        int nmov = 0;
        if (fl & kNearMoving) {
            // Pared móvil (Ladd): la población que llega desde el sólido s = n - c_i recibe
            // +6 w_i ρ_w (c_i·u_w), ρ_w = 1, u_w = movimiento rígido evaluado en el centro de s.
            // IMPERMEABLE (fase 2, WallMotion::impermeable): en una superficie en escalera u_w (tangente a la
            // superficie real, p.ej. una rueda que gira) tiene componente normal a las caras de los vóxeles y
            // Ladd inyecta/extrae masa Σ_i 6w_i c_i·u_w por nodo. Se resta esa masa repartida con los pesos w_i.
            float S = 0.0f, Wm = 0.0f;
            u32 mv = 0;
            for (int i = 1; i < L::Q; ++i) {
                const i64 s = n - k.off[i];
                const u8 fs = k.flags[s];
                if (fs & kMoving) {
                    const Vec3 uw = wall_velocity(k.motion[k.sid[s]], float(x - L::c[i][0]), float(y - L::c[i][1]),
                                                  float(z - L::c[i][2]));
                    uwall += uw;
                    ++nmov;
                    // Rebote interpolado explícito (ruedas): la pasada de contorno ya escribió la población con el
                    // término de pared móvil.
                    if (k.mexp && k.sid[s] != k_ground_id) continue;
                    const float dl = kSc<P> * 6.0f * L::w[i] * (float(L::c[i][0]) * uw.x + float(L::c[i][1]) * uw.y + float(L::c[i][2]) * uw.z);
                    f[i] += dl;
                    if (k.motion[k.sid[s]].conserve) { S += dl; Wm += L::w[i]; mv |= 1u << i; }
                }
            }
            if (Wm > 0.0f) {
                const float cS = S / Wm;
                while (mv) { const int i = std::countr_zero(mv); mv &= mv - 1; f[i] -= cS * L::w[i]; }
            }
        }
        auto ldf = [&](int i) { return f[i]; };
        const Mom<float> m = moments<float>(ldf);
        float drho = (P == Precision::FP16S) ? m.drho * kInvScale : m.drho;
        float rho = 1.0f + drho;
        const float inv = 1.0f / rho;
        const float invj = (P == Precision::FP16S) ? inv * kInvScale : inv;
        float ux = m.jx * invj, uy = m.jy * invj, uz = m.jz * invj;
        const float cux = ux, cuy = uy, cuz = uz;
        bool eq = false;
        if (fl & kInlet) { ux = k.u_in; uy = uz = 0.0f; eq = true; }
        else if (fl & kOutlet) { ux = pux; uy = puy; uz = puz; eq = true; }
        if (eq) { rho = 1.0f; drho = 0.0f; }
        pux = cux; puy = cuy; puz = cuz;
        const Neq<float> q = noneq<kSc<P>>(m, drho, rho, ux, uy, uz);
        float omc = k.K > 0.0f ? one_minus_omega(q, inv, k.tau0[x], k.tau0sq[x], k.K) : k.omc0[x];
        if (fl & kNearWall) {
            // Ley de pared: velocidad relativa a la media de las paredes MÓVILES vecinas (0 si todas fijas).
            Vec3 ur(cux, cuy, cuz);
            if (nmov) ur -= uwall * (1.0f / static_cast<float>(nmov));
            omc = wall_omc(dot(ur, ur), k.tau0[x], k.wallC3, omc, k.wall_floor);
        }
        float omcb = k.omcb;
        if (eq) omc = omcb = 0.0f;
        bad |= !(rho > 0.2f && rho < 5.0f);
        if constexpr (Macro) { k.rho[n] = rho; k.ux[n] = ux; k.uy[n] = uy; k.uz[n] = uz; }
        auto stf = [&](int i, float v) { store_s<P>(k.st[i], n, v); };
        collide<C, kSc<P>, Bulk>(ldf, stf, drho, rho, ux, uy, uz, q, omc, omcb, (fl & kLayer) ? 0.0f : omc);
    }
    return bad;
}

// Bloques 100% sólidos en pasos macro: ρ = 1, u = velocidad de pared (o 0).
CFD_INLINE void block_skip_macro(const KCtx& k, i64 n0, int x0, int y, int z) {
    u64 w;
    std::memcpy(&w, k.flags + n0, 8);
    if (!(w & (0x0101010101010101ull * kMoving))) {
        const f8 one(1.0f), zero = f8::zero();
        store_macro(k, n0, one, zero, zero, zero);
        return;
    }
    for (int l = 0; l < 8; ++l) macro_solid(k, n0 + l, k.flags[n0 + l], x0 + l, y, z);
}

// Recorre un rango de filas (y,z). Una fila = nx/8 bloques; la clase de cada bloque decide la ruta.
template <Precision P, Collision C, bool Macro, bool Bulk>
CFD_HOT void rows_kernel(const KCtx& k, i64 r0, i64 r1) {
    // FTZ|DAZ: los subnormales disparan asistencias de microcódigo (~100+ ciclos). MXCSR es por
    // hilo: se activa al entrar y se restaura al salir (2 ldmxcsr por trozo de ~4096 celdas).
    const unsigned csr = _mm_getcsr();
    if (k.ftz) _mm_setcsr(csr | 0x8040u);
    u32 badflag = 0;
    for (i64 r = r0; r < r1; ++r) {
        const i64 row = k.rows[r];
        const int y = static_cast<int>(row % k.ny), z = static_cast<int>(row / k.ny);
        const i64 nrow = row * k.nx;
        const u8* CFD_RESTRICT cls = k.cls + row * k.nbx;
        __m256 bad = _mm256_setzero_ps();
        for (int b = 0; b < k.nbx; ++b) {
            const i64 n0 = nrow + 8 * b;
            const int x0 = 8 * b;
            if (k.pair && cls[b] == kBlkPure && b + 1 < k.nbx && cls[b + 1] == kBlkPure) {
                block_vec2<P, C, Macro, Bulk>(k, n0, x0, bad);
                ++b;
                continue;
            }
            const u8 c = cls[b];
            const __m256 m3 = (c & kBlkLayer) ? _mm256_setzero_ps() : _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            switch (c & ~kBlkLayer) {
            case kBlkPure: block_vec<P, C, Macro, Bulk, false>(k, n0, x0, bad, m3); break;
            case kBlkMasked: block_vec<P, C, Macro, Bulk, true>(k, n0, x0, bad, m3); break;
            case kBlkGround: block_vec<P, C, Macro, Bulk, true, true>(k, n0, x0, bad, m3); break;
            case kBlkPure | kBlkWall: block_vec<P, C, Macro, Bulk, false, false, true>(k, n0, x0, bad, m3); break;
            case kBlkMasked | kBlkWall: block_vec<P, C, Macro, Bulk, true, false, true>(k, n0, x0, bad, m3); break;
            case kBlkGround | kBlkWall: block_vec<P, C, Macro, Bulk, true, true, true>(k, n0, x0, bad, m3); break;
            case kBlkScalar: badflag |= block_scalar<P, C, Macro, Bulk>(k, n0, x0, y, z); break;
            default:
                if constexpr (Macro) block_skip_macro(k, n0, x0, y, z);
                break;
            }
        }
        badflag |= static_cast<u32>(_mm256_movemask_ps(bad));
    }
    if (CFD_UNLIKELY(badflag)) k.bad->store(1, std::memory_order_relaxed);
    if constexpr (Macro) if (k.nt) _mm_sfence();   // los stores NT son débilmente ordenados
    if (k.ftz) _mm_setcsr(csr);
}

using RowsFn = void (*)(const KCtx&, i64, i64);

// Bulk: viscosidad de volumen propia (constante del paso) → plantilla, para que la traza no dependa de ω (ver collide).
// BGK no la usa: sólo se instancia sin ella.
template <Precision P, Collision C>
constexpr RowsFn pick_rows(bool macro, bool bulk) {
    if constexpr (C == Collision::BGK) return macro ? &rows_kernel<P, C, true, false> : &rows_kernel<P, C, false, false>;
    else if (bulk) return macro ? &rows_kernel<P, C, true, true> : &rows_kernel<P, C, false, true>;
    else return macro ? &rows_kernel<P, C, true, false> : &rows_kernel<P, C, false, false>;
}

// ---- Fuerzas por intercambio de momento --------------------------------------------------------
// Lista compacta de sólidos con vecinos fluidos: máscara de enlaces (bit k = el fluido está en s - c_k)
// + id en los bits 24..31. Ordenada por (id, n) → cada trozo acumula corridas de un mismo id.
// Nodo de fluido "verdadero" (ni sólido ni frontera de equilibrio) con algún vecino sólido: sus enlaces
// de rebote n → s = n + c_k (bit k de mask). Lista compacta ordenada por n (acceso a memoria creciente).
// (fase 3) El nodo de pared es público (lbm::WallNode, solver.hpp) para que el backend GPU reutilice este
// preprocesado; mismos campos que la versión interna anterior.
using WNode = WallNode;
constexpr u8 kQHalf = 127;
// Acumulador de fuerza/momento por id y trozo (reducción determinista).
struct FAcc { double f[3], m[3]; };
constexpr int kMaxForceChunks = 256;
// Tablas float (evitan cvtsi2ss por enlace en la pasada de fuerzas).
struct DirTabF { float c[L::Q][3]; float w2[L::Q]; float w6[L::Q]; float rinv[L::Q]; };
constexpr DirTabF make_dirtab() {
    DirTabF t{};
    for (int k = 0; k < L::Q; ++k) {
        int n2 = 0;
        for (int a = 0; a < 3; ++a) { t.c[k][a] = static_cast<float>(L::c[k][a]); n2 += L::c[k][a] * L::c[k][a]; }
        t.w2[k] = 2.0f * L::w[k];
        t.w6[k] = 6.0f * L::w[k];
        t.rinv[k] = n2 == 0 ? 0.0f : (n2 == 1 ? 1.0f : 0.70710678f);   // 1/|c_k|
    }
    return t;
}
constexpr DirTabF kDir = make_dirtab();

} // namespace

// ================================================================================================
struct Solver::Impl {
    Config cfg;
    Tuning tun;
    int nx = 0, ny = 0, nz = 0, nbx = 0;
    i64 N = 0, nxny = 0, S = 0, P = 0;       // celdas, zancada por dirección, relleno
    i64 off[L::Q] = {};
    Buffer<u8> ddf;                          // 19·S elementos de float o u16
    Buffer<u8> flags, flags_old, sid, user_sid, cls;
    Buffer<float> rho, ux, uy, uz;
    Buffer<float> tau0, tau0sq, omc0;
    Buffer<u32> rows_all, rows_active;
    i64 n_rows_active = 0;
    std::vector<WNode> wnodes;               // nodos junto a paredes (rebote, fuerzas, modelo de pared)
    WallSdf sdf;                             // distancia a la superficie real (rebote interpolado)
    Buffer<FAcc> facc;                       // kMaxForceChunks × 256 acumuladores
    Buffer<u8> ftouch;                       // kMaxForceChunks × 256: id tocado en el trozo
    WallMotion user_motion[256];
    bool user_moving[256] = {};
    Motion motion_now[256];
    Vec3 moment_ref{0, 0, 0};
    ForceSample fs, fs_mean;
    double acc_f[256][3] = {}, acc_m[256][3] = {};
    u64 t = 0;
    float u_from = 0, u_to = 0;
    u64 ramp_t0 = 0;
    double mlups = 0, t_kernel = 0, t_force = 0;
    std::atomic<u32> bad{0};
    // (fase 3) Backend externo: sincronización previa a leer/modificar el estado y campos macro sustitutos.
    u64 revision = 1;
    ExternalSync ext = nullptr;
    void* ext_ctx = nullptr;
    const float* ov[4] = {nullptr, nullptr, nullptr, nullptr};
    void sync_ext() const { if (ext) ext(ext_ctx); }

    usize esize() const { return cfg.precision == Precision::FP32 ? 4 : 2; }
    void* dir_base(int k) { return ddf.data() + static_cast<usize>((static_cast<i64>(k) * S + P)) * esize(); }
    const void* dir_base(int k) const { return ddf.data() + static_cast<usize>((static_cast<i64>(k) * S + P)) * esize(); }

    float u_at(u64 step) const {
        if (cfg.ramp_steps <= 0 || step >= ramp_t0 + static_cast<u64>(cfg.ramp_steps)) return u_to;
        const float s = smoothstep(0.0f, static_cast<float>(cfg.ramp_steps), static_cast<float>(step - ramp_t0));
        return u_from + (u_to - u_from) * s;
    }

    // Punteros de carga/escritura por paridad.
    void step_ptrs(u64 step, const void* ld[L::Q], void* st[L::Q]) const {
        const int p = static_cast<int>(step & 1);
        const usize es = esize();
        auto base = [&](int k, i64 o) -> const u8* {
            return ddf.data() + static_cast<usize>(static_cast<i64>(k) * S + P + o) * es;
        };
        ld[0] = base(0, 0);
        for (int i = 1; i < L::Q; i += 2) {
            ld[i] = p ? base(i, 0) : base(i + 1, 0);
            ld[i + 1] = p ? base(i + 1, off[i]) : base(i, off[i]);
        }
        st[0] = const_cast<void*>(ld[0]);
        for (int i = 1; i < L::Q; i += 2) {
            st[i] = const_cast<void*>(ld[i + 1]);
            st[i + 1] = const_cast<void*>(ld[i]);
        }
    }

    void build_tau() {
        ++revision;
        const float nu = std::max(cfg.nu, 1e-7f);
        const int xs = static_cast<int>(static_cast<float>(nx) * (1.0f - std::clamp(cfg.sponge_frac, 0.0f, 0.9f)));
        const float nu_max = std::max(nu, kSpongeNu);
        for (int x = 0; x < nx; ++x) {
            float v = nu;
            if (cfg.sponge_frac > 0.0f && x > xs && nx - 1 > xs) {
                const float s = smoothstep(static_cast<float>(xs), static_cast<float>(nx - 1), static_cast<float>(x));
                v = nu + (nu_max - nu) * s * s;
            }
            tau0[x] = 3.0f * v + 0.5f;
            tau0sq[x] = tau0[x] * tau0[x];
            omc0[x] = 1.0f - 2.0f / (tau0[x] + tau0[x]);   // misma expresión que con K=0 en one_minus_omega
        }
    }

    void update_motion_table() { fill_motion(t, motion_now); }
    void fill_motion(u64 step, Motion* dst) const {
        // Las velocidades de pared se dan a la u∞ objetivo: durante la rampa se escalan con u(t)/u∞.
        // Rampa hacia 0 (parar el túnel): se escalan respecto a la velocidad de partida → paran también.
        const float uc = u_at(step);
        const float r = u_to != 0.0f ? uc / u_to : (u_from != 0.0f ? uc / u_from : 1.0f);
        const Vec3 vg = cfg.ground == GroundMode::Moving ? Vec3(uc, 0, 0) : Vec3(0, 0, 0);
        for (int id = 0; id < 255; ++id) {
            const WallMotion& m = user_motion[id];
            dst[id].v = m.v * r;
            dst[id].omega = m.omega * r;
            dst[id].center = m.center;
            dst[id].contact_z = m.contact_z;
            dst[id].vc = vg;
            dst[id].conserve = m.impermeable;
        }
        dst[255] = Motion{};
        dst[255].v = vg;
    }

    bool id_moving(int id) const {
        if (id == k_ground_id) return cfg.ground == GroundMode::Moving;
        return user_moving[id];
    }

    void reset_fill();
    void rebuild(bool transitions);
    void run_kernel(bool macro);
    void compute_wall_geometry();
    // Rebote explícito (+ fuerzas) tras el paso `tt` (el que dejó en memoria su post-colisión). write_only:
    // sólo escribe las poblaciones entrantes (arranque / tras cambiar la geometría), sin fuerzas.
    void boundary_pass(bool accumulate, u64 tt, bool write_only = false);
};

// ---- Relleno uniforme (reset): f̃ = feq(ρ=1, u0 x̂) - w en la semántica de ranura de la paridad 0.
void Solver::Impl::reset_fill() {
    ++revision;
    const float u0 = u_at(0);
    float val[L::Q];   // valor f̃ de la población k
    for (int k = 0; k < L::Q; ++k) {
        const float a = 3.0f * static_cast<float>(L::c[k][0]) * u0;
        val[k] = L::w[k] * (a + 0.5f * a * a - 1.5f * u0 * u0);
    }
    // Paridad 0: la ranura i (impar) guarda f_{i+1}, la ranura i+1 guarda f_i.
    float slot[L::Q];
    slot[0] = val[0];
    for (int i = 1; i < L::Q; i += 2) { slot[i] = val[i + 1]; slot[i + 1] = val[i]; }
    const i64 total = S * L::Q;
    const bool fp32 = cfg.precision == Precision::FP32;
    void* d = ddf.data();
    parallel_for(0, total, 1 << 16, [&](i64 lo, i64 hi) {
        for (i64 e = lo; e < hi;) {
            const int k = static_cast<int>(e / S);
            const i64 end = std::min(hi, (static_cast<i64>(k) + 1) * S);
            if (fp32) { float* p = static_cast<float*>(d); const float v = slot[k]; for (i64 i = e; i < end; ++i) p[i] = v; }
            else { u16* p = static_cast<u16*>(d); const u16 v = f32_to_f16(slot[k] * kScale); for (i64 i = e; i < end; ++i) p[i] = v; }
            e = end;
        }
    });
    // Campos macro iniciales.
    const u8* fl = flags.data();
    parallel_for(0, N, 1 << 15, [&](i64 lo, i64 hi) {
        for (i64 n = lo; n < hi; ++n) {
            const bool s = fl[n] & kSolid;
            rho[n] = 1.0f; ux[n] = s ? 0.0f : u0; uy[n] = 0.0f; uz[n] = 0.0f;
        }
    });
}

// ---- Reconstrucción de flags, clases de bloque, filas activas y lista de fronteras -----------
void Solver::Impl::rebuild(bool transitions) {
    ++revision;
    const bool ground_solid = cfg.ground != GroundMode::None;
    std::memcpy(flags_old.data(), flags.data(), static_cast<usize>(N));
    u8* F = flags.data();
    u8* SI = sid.data();
    const u8* U = user_sid.data();
    bool moving_id[256];
    bool any_moving = false;
    for (int id = 0; id < 256; ++id) { moving_id[id] = id_moving(id); any_moving |= (id > 0 && moving_id[id]); }

    // 1) Flags base.
    parallel_for(0, nz, 1, [&](i64 z0, i64 z1) {
        for (i64 z = z0; z < z1; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const i64 n = x + nx * (y + static_cast<i64>(ny) * z);
                    u8 fl = kFluid, s = 0;
                    if (z == 0 && ground_solid) {
                        s = k_ground_id;
                        fl = kSolid | (moving_id[k_ground_id] ? kMoving : 0);
                    } else if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
                        const bool far = (y == 0 || y == ny - 1 || z == 0 || z == nz - 1);
                        fl = (x == nx - 1 && !far) ? kOutlet : kInlet;
                    } else {
                        s = U[n];
                        if (s) fl = kSolid | (moving_id[s] ? kMoving : 0);
                    }
                    F[n] = fl;
                    SI[n] = s;
                }
    });
    // 2) kNearMoving: fluido (no frontera) con algún vecino sólido móvil.
    //    Dos pasadas por paridad de z (revisión: TSan detectó una carrera). Con una sola pasada, el hilo del
    //    plano z escribía F[n] |= kNearMoving mientras el del plano z±1 leía ese mismo byte como vecino F[m]
    //    (UB formal aunque sólo se mire kMoving). Ahora cada pasada escribe planos de UNA paridad y sólo lee
    //    los de la otra (que nadie escribe en esa pasada) y el propio plano (mismo hilo).
    //    "Sólo cinta" (kGroundOnly → atajo vectorial con corrección constante en dirs 9/16) exige que el
    //    vecino móvil sea la capa z=0 (m < nx·ny), no cualquier celda con id 255 (revisión: prueba 2f).
    // (fase 2) En la misma pasada: kNearWall = fluido con algún vecino sólido (ley de pared), sólo si el
    // modelo de pared está activo (el bit presente ⇔ ley de pared en esa celda).
    // (El modelo Slip NO usa la ley de pared por viscosidad —combinada con el deslizamiento oscilaba en z en una
    // placa plana y divergía bajo una caja a 2 celdas de la cinta— pero sí marca kNearWall para AMORTIGUAR la
    // viscosidad de Smagorinsky en la 1.ª celda: τ = max(τ0, ½ + ¼(τ_LES − ½)), como el amortiguamiento de van
    // Driest. Sin él la viscosidad turbulenta de la celda de pared, alimentada por la cizalla de la propia pared,
    // engorda la capa límite y la despega: NACA 0012 AR 3 a 6° con 42 celdas de cuerda, CL 0.15 → 0.24 (0.30
    // con C_s = 0.10).)
    const bool wall = cfg.wall_model != WallModel::None;
    if (any_moving || wall) {
        for (int par = 0; par < 2; ++par) {
            const i64 zfirst = 1 + par;
            const i64 nplanes = (nz - 1 - zfirst + 1) / 2;   // z = zfirst, zfirst+2, … ≤ nz-2
            parallel_for(0, nplanes, 1, [&](i64 j0, i64 j1) {
                for (i64 j = j0; j < j1; ++j) {
                    const i64 z = zfirst + 2 * j;
                    for (int y = 1; y < ny - 1; ++y)
                        for (int x = 1; x < nx - 1; ++x) {
                            const i64 n = x + nx * (y + static_cast<i64>(ny) * z);
                            if (F[n] != kFluid) continue;
                            bool near_ground = false, near_other = false;
                            bool near_static = false;
                            for (int i = 1; i < L::Q; ++i) {
                                const i64 m = n + off[i];
                                if (F[m] & kMoving) (SI[m] == k_ground_id && m < nxny ? near_ground : near_other) = true;
                                else if (F[m] & kSolid) near_static = true;
                            }
                            if (near_ground || near_other) F[n] |= kNearMoving;
                            if (near_ground && !near_other) F[n] |= kGroundOnly;
                            // Ley de pared sólo junto a paredes FIJAS y lejos de paredes móviles: en las cuñas
                            // rueda-cinta la viscosidad baja de la ley de pared divergía (prueba 7 de test_lbm).
                            if (wall && near_static && !near_ground && !near_other) F[n] |= kNearWall;
                        }
                }
            });
        }
    }
    // 3) Transiciones sólido → no sólido: equilibrio en reposo (f̃ = 0) en su conjunto de acceso actual.
    if (transitions) {
        const void* ld[L::Q];
        void* st[L::Q];
        step_ptrs(t, ld, st);
        const u8* FO = flags_old.data();
        const bool fp32 = cfg.precision == Precision::FP32;
        parallel_for(0, N, 1 << 15, [&](i64 lo, i64 hi) {
            for (i64 n = lo; n < hi; ++n) {
                if (!(FO[n] & kSolid) || (F[n] & kSolid)) continue;
                for (int k = 0; k < L::Q; ++k) {
                    if (fp32) const_cast<float*>(static_cast<const float*>(ld[k]))[n] = 0.0f;
                    else const_cast<u16*>(static_cast<const u16*>(ld[k]))[n] = 0;
                }
            }
        });
    }
    // 3b) Capa junto a los cuerpos (sólo con la colisión recursiva): dilatación de Chebyshev de radio L de los sólidos
    //     que no son el suelo (3 pasadas separables x, y, z sobre una máscara de bytes), extendida a bloques completos de
    //     8 celdas en x (el kernel AVX2 decide por bloque; la iGPU lee el bit por celda: mismo resultado).
    if (cfg.collision == Collision::Recursive && cfg.rr_wall_layer > 0) {
        const int Lw = cfg.rr_wall_layer;
        std::vector<u8> a(static_cast<usize>(N)), b(static_cast<usize>(N));
        parallel_for(0, N, 1 << 15, [&](i64 lo, i64 hi) {
            for (i64 n = lo; n < hi; ++n) a[static_cast<usize>(n)] = (F[n] & kSolid) && !(SI[n] == k_ground_id && n < nxny);
        });
        // Máximo deslizante de radio Lw a lo largo de un eje (zancada st, longitud len): dst[j] = OR de src[j−Lw..j+Lw]
        // (distancia a la última celda marcada por delante y por detrás).
        auto pass = [&](const std::vector<u8>& src, std::vector<u8>& dst, i64 st, int len, i64 nlines, auto base_of) {
            parallel_for(0, nlines, 16, [&](i64 l0, i64 l1) {
                std::vector<int> fw(static_cast<usize>(len));
                for (i64 l = l0; l < l1; ++l) {
                    const i64 b0 = base_of(l);
                    int d = 1 << 20;
                    for (int i = 0; i < len; ++i) { d = src[static_cast<usize>(b0 + i * st)] ? 0 : d + 1; fw[static_cast<usize>(i)] = d; }
                    d = 1 << 20;
                    for (int i = len - 1; i >= 0; --i) {
                        d = src[static_cast<usize>(b0 + i * st)] ? 0 : d + 1;
                        dst[static_cast<usize>(b0 + i * st)] = std::min(d, fw[static_cast<usize>(i)]) <= Lw;
                    }
                }
            });
        };
        const i64 NX = nx, NY = ny;
        pass(a, b, 1, nx, static_cast<i64>(ny) * nz, [&](i64 l) { return l * NX; });
        pass(b, a, NX, ny, static_cast<i64>(nx) * nz, [&](i64 l) { return (l % NX) + (l / NX) * NX * NY; });
        pass(a, b, nxny, nz, nxny, [&](i64 l) { return l; });
        parallel_for(0, N / 8, 1 << 12, [&](i64 lo, i64 hi) {
            for (i64 bl = lo; bl < hi; ++bl) {
                bool any = false;
                for (int l = 0; l < 8; ++l) any |= b[static_cast<usize>(8 * bl + l)] != 0;
                if (any)
                    for (int l = 0; l < 8; ++l)
                        if (!(F[8 * bl + l] & kSolid)) F[8 * bl + l] |= kLayer;
            }
        });
    }
    // 4) Clases de bloque (SWAR sobre 8 flags) y filas activas.
    const i64 nblocks = N / 8;
    u8* CL = cls.data();
    parallel_for(0, nblocks, 1 << 12, [&](i64 lo, i64 hi) {
        constexpr u64 ones = 0x0101010101010101ull;
        for (i64 b = lo; b < hi; ++b) {
            u64 w;
            std::memcpy(&w, F + 8 * b, 8);
            u8 c;
            // kNearMoving sin kGroundOnly en algún carril: byte & (kNearMoving|kGroundOnly) == kNearMoving.
            // SWAR: (w & NM) sin el bit GO desplazado → "near-moving no sólo-suelo" por byte.
            const u64 nm = (w >> 4) & ones;           // bit kNearMoving (4) → bit 0 de cada byte
            const u64 go = (w >> 5) & ones;           // bit kGroundOnly (5) → bit 0
            const u8 wl = (w & (ones * kNearWall)) ? u8(kBlkWall) : u8(0);   // algún carril con ley de pared
            if ((w & (ones * kSolid)) == ones * kSolid) c = kBlkSkip;
            else if ((w & (ones * kMoving)) || (nm & ~go)) c = kBlkScalar;   // la ruta escalar mira kNearWall por celda
            else if (nm) c = kBlkGround | wl;
            else if (w & (ones * (kSolid | kInlet | kOutlet))) c = kBlkMasked | wl;
            else c = kBlkPure | wl;
            if (c != kBlkSkip && c != kBlkScalar && (w & (ones * kLayer))) c |= kBlkLayer;   // (escalar: por celda)
            CL[b] = c;
        }
    });
    // Filas en orden ASCENDENTE de memoria. (Medido: recorrer en z descendente para que el suelo
    // quede en L3 al calcular fuerzas es PEOR: -15..20% MLUPS y fuerzas más lentas, porque rompe
    // los flujos ascendentes que detecta el prefetcher L2 entre filas consecutivas.)
    n_rows_active = 0;
    const i64 nrows = static_cast<i64>(ny) * nz;
    for (i64 r = 0; r < nrows; ++r) {
        bool act = false;
        for (int b = 0; b < nbx && !act; ++b) act = CL[r * nbx + b] != kBlkSkip;
        if (act) rows_active[n_rows_active++] = static_cast<u32>(r);
    }
    // 5) Nodos de fluido "verdadero" (sin fronteras de equilibrio) con enlaces a sólidos. Todos los enlaces
    //    fluido→sólido del dominio están aquí (los de las celdas de frontera usan el rebote implícito).
    std::vector<std::vector<WNode>> per_z(static_cast<usize>(nz));
    parallel_for(1, nz - 1, 1, [&](i64 z0, i64 z1) {
        for (i64 z = z0; z < z1; ++z) {
            auto& v = per_z[static_cast<usize>(z)];
            v.clear();
            for (int y = 1; y < ny - 1; ++y)
                for (int x = 1; x < nx - 1; ++x) {
                    const i64 n = x + nx * (y + static_cast<i64>(ny) * z);
                    if (F[n] & (kSolid | kInlet | kOutlet)) continue;
                    u32 mask = 0;
                    u16 kind = 0;
                    for (int k = 1; k < L::Q; ++k) {
                        const u8 fm = F[n + off[k]];
                        if (fm & kSolid) { mask |= 1u << k; kind |= (fm & kMoving) ? 1 : 2; }
                    }
                    if (!mask) continue;
                    WNode w{};
                    w.n = static_cast<u32>(n); w.mask = mask;
                    w.x = static_cast<u16>(x); w.y = static_cast<u16>(y); w.z = static_cast<u16>(z);
                    w.kind = kind;
                    v.push_back(w);
                }
        }
    });
    usize total = 0;
    for (auto& v : per_z) total += v.size();
    wnodes.resize(total);
    total = 0;
    for (auto& v : per_z) { std::copy(v.begin(), v.end(), wnodes.begin() + static_cast<i64>(total)); total += v.size(); }
    compute_wall_geometry();
    // Poblaciones entrantes del rebote explícito para la NUEVA geometría a partir del post-colisión del
    // último paso (o del relleno, que equivale a un paso "−1" en equilibrio).
    // (v2) Con la velocidad de pared ACTUAL: las paredes móviles interpoladas también se escriben aquí.
    update_motion_table();
    if (facc.size()) boundary_pass(false, t - 1, true);
}

// ---- Geometría de pared por nodo: q por enlace, normal, distancia y área --------------------------------
// q = fracción del enlace fluido n → sólido s = n + c_k hasta la superficie real: interpolación lineal de la
// distancia con signo entre los centros (exacta para una pared plana). Casos degenerados de la voxelización
// (engrosamiento 0.12·dx, voto de mayoría): centro fluido DENTRO de la superficie → q = 0; centro sólido
// FUERA → q = 1. Suelo (capa z = 0), paredes móviles o sin WallSdf → q = ½ (half-way).
// Normal: gradiente de la distancia ajustado por mínimos cuadrados a las diferencias d(s) - d(n) de los
// enlaces al sólido (exacto para una pared plana si los enlaces generan R³); si no, la del conjunto de
// enlaces: -Σ w_k c_k. Con suelo fijo, se mezcla con +ẑ según el peso de los enlaces al suelo.
void Solver::Impl::compute_wall_geometry() {
    if (wnodes.empty()) return;
    const u8* F = flags.data();
    const u8* SI = sid.data();
    const bool interp = cfg.bounce == BounceBack::Interpolated;
    const WallSdf w = sdf;
    const bool use_sdf = w.fn && interp;
    parallel_for(0, static_cast<i64>(wnodes.size()), 64, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            WNode& nd = wnodes[static_cast<usize>(i)];
            for (int k = 0; k < 20; ++k) nd.q[k] = kQHalf;
            const Vec3 pn(nd.x, nd.y, nd.z);
            const float dn = use_sdf ? w.fn(w.ctx, pn) : 0.0f;
            // Normal por mínimos cuadrados (enlaces a sólidos fijos que no son la capa de suelo).
            double AtA[3][3] = {}, Atb[3] = {};
            Vec3 lsum_b{0, 0, 0}, lsum_all{0, 0, 0};
            float wb = 0.0f, wg = 0.0f;
            int nb = 0;
            u32 bits = nd.mask;
            while (bits) {
                const int k = std::countr_zero(bits);
                bits &= bits - 1;
                const i64 sidx = static_cast<i64>(nd.n) + off[k];
                const Vec3 ck(float(L::c[k][0]), float(L::c[k][1]), float(L::c[k][2]));
                lsum_all += ck * L::w[k];
                const bool ground = SI[sidx] == k_ground_id && sidx < nxny;
                if (ground) { wg += L::w[k]; continue; }
                wb += L::w[k];
                lsum_b += ck * L::w[k];
                // (v2) Las paredes móviles que no son el suelo (ruedas) también usan el rebote interpolado.
                if ((F[sidx] & kMoving) && SI[sidx] == k_ground_id) continue;
                if (!use_sdf) continue;
                const float ds = w.fn(w.ctx, pn + ck);
                float qq;
                if (!(dn > 0.0f) && !(ds < 0.0f)) qq = 0.5f;      // ambos al revés: sin información fiable
                else if (!(dn > 0.0f)) qq = 0.0f;
                else if (!(ds < 0.0f)) qq = 1.0f;
                else qq = dn / (dn - ds);
                nd.q[k] = static_cast<u8>(std::lround(std::clamp(qq, 0.0f, 1.0f) * 254.0f));
                const double dd = static_cast<double>(ds) - dn;
                for (int a = 0; a < 3; ++a) {
                    Atb[a] += L::c[k][a] * dd;
                    for (int b = 0; b < 3; ++b) AtA[a][b] += L::c[k][a] * L::c[k][b];
                }
                ++nb;
            }
            // Normal del cuerpo (hacia el fluido).
            Vec3 nbody = length2(lsum_b) > 0.0f ? normalize(lsum_b * -1.0f) : Vec3(0, 0, 1);
            if (use_sdf && nb >= 3) {
                const double (&M)[3][3] = AtA;
                const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                                   M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
                if (std::fabs(det) > 1e-6) {
                    double g[3];
                    for (int a = 0; a < 3; ++a) {   // Cramer
                        double C[3][3];
                        for (int r = 0; r < 3; ++r) for (int c2 = 0; c2 < 3; ++c2) C[r][c2] = (c2 == a) ? Atb[r] : M[r][c2];
                        g[a] = (C[0][0] * (C[1][1] * C[2][2] - C[1][2] * C[2][1]) - C[0][1] * (C[1][0] * C[2][2] - C[1][2] * C[2][0]) +
                                C[0][2] * (C[1][0] * C[2][1] - C[1][1] * C[2][0])) / det;
                    }
                    const Vec3 gv{static_cast<float>(g[0]), static_cast<float>(g[1]), static_cast<float>(g[2])};
                    // El gradiente de la distancia apunta hacia FUERA del sólido (hacia el fluido).
                    if (length2(gv) > 1e-6f && dot(gv, nbody) > 0.0f) nbody = normalize(gv);
                }
            }
            Vec3 nrm = nbody * wb + Vec3(0, 0, 1) * wg;
            nrm = length2(nrm) > 0.0f ? normalize(nrm) : Vec3(0, 0, 1);
            nd.nx = nrm.x; nd.ny = nrm.y; nd.nz = nrm.z;
            float yw = 0.5f;
            if (use_sdf && wb > 0.0f && dn > 0.0f) yw = dn;
            if (wg > 0.0f) yw = std::min(yw, static_cast<float>(nd.z) - 0.5f);
            nd.yw = std::clamp(yw, 0.25f, 1.5f);
            nd.yw17 = std::pow(nd.yw, 1.0f / 7.0f);
            nd.area = 6.0f * length(lsum_all);
            // Curvatura media local κ ≈ ∇²d (diferencias centradas con h = 1 celda; d > 0 fuera): en una arista
            // viva (borde de salida, arista de 90°) κ ~ 1/r con r < 1 celda; en una pared suave κ ≈ 1/R pequeño.
            // Deslizamiento sólo donde R ≳ 3 celdas: las aristas conservan el no deslizamiento (condición de
            // Kutta, separación fija en aristas). Sin WallSdf: pared plana (curvatura 0).
            // (v2) Planitud UNILATERAL: en una pared lisa de radio R, d(p + e) = d(p) + n̂·e + |e_t|²/(2R) + …; el
            // residuo máximo sobre las direcciones de los ejes que NO entran en el cuerpo (n̂·e > −0.3) mide 1/(2R)
            // y en una arista viva (borde de salida) es ~1. No mira a través del cuerpo: el laplaciano centrado de
            // la v1 cruzaba perfiles de 3-4 celdas de espesor y apagaba el modelo en TODO el ala.
            // R ≥ 3 celdas → 1; R ≤ 1.25 celdas → 0.
            nd.sgeo = 1.0f;
            if (use_sdf && wb > 0.0f) {
                float res = 0.0f;
                for (int a = 0; a < 3; ++a)
                    for (int sg = -1; sg <= 1; sg += 2) {
                        Vec3 e{0, 0, 0};
                        (a == 0 ? e.x : a == 1 ? e.y : e.z) = static_cast<float>(sg);
                        const float ne = dot(nrm, e);
                        if (ne < -0.3f) continue;
                        res = std::max(res, std::fabs(w.fn(w.ctx, pn + e) - dn - ne));
                    }
                nd.sgeo = std::clamp((0.4f - res) / 0.233f, 0.0f, 1.0f);
            }
        }
    });
}

// ---- Kernel de un paso --------------------------------------------------------------------------
void Solver::Impl::run_kernel(bool macro) {
    KCtx k;
    step_ptrs(t, k.ld, k.st);
    k.flags = flags.data();
    k.sid = sid.data();
    k.cls = cls.data();
    k.rows = macro ? rows_all.data() : rows_active.data();
    const i64 nrows = macro ? static_cast<i64>(ny) * nz : n_rows_active;
    k.tau0 = tau0.data();
    k.tau0sq = tau0sq.data();
    k.omc0 = omc0.data();
    k.K = 18.0f * std::sqrt(2.0f) * cfg.cs_smag * cfg.cs_smag;
    k.wallC3 = wall_c3(cfg.wall_nu > 0.0f ? cfg.wall_nu : cfg.nu);
    k.wall_floor = kWallFloor;
    if (cfg.wall_model == WallModel::Slip) k.wallC3 = 0.0f;   // Slip: sólo el amortiguamiento del LES (ver rebuild)
    k.bulk = cfg.bulk_omega > 0.0f;
    k.omcb = 1.0f - std::clamp(cfg.bulk_omega, 0.0f, 1.99f);
    k.u_in = u_at(t);
    k.u_ground = motion_now[k_ground_id].v.x;
    k.nx = nx; k.ny = ny; k.nz = nz; k.nbx = nbx;
    k.nxny = nxny;
    for (int i = 0; i < L::Q; ++i) k.off[i] = off[i];
    k.rho = rho.data(); k.ux = ux.data(); k.uy = uy.data(); k.uz = uz.data();
    k.motion = motion_now;
    k.bad = &bad;
    // Stores NT para ρ,u: medido (3 rondas intercaladas, 1 paso macro de cada 5) ~2.5 % PEOR que stores
    // normales (el sfence por trozo y que el consumidor —visualización— relee los campos enseguida).
    k.nt = tun.nt_macro > 0;
    k.pf = tun.prefetch < 0 ? 0 : tun.prefetch;
    // Pares entrelazados: llvm-mca (Golden Cove) predice -10% ciclos sólo en FP32-Regularizado (que ya
    // está limitado por memoria) y +3..+25% en los demás por derrames de registros → por defecto NO.
    k.pair = tun.pair_blocks > 0;
    k.ftz = tun.ftz > 0;   // medido: sin efecto (no hay subnormales en este kernel) → por defecto no se toca MXCSR
    k.mexp = cfg.bounce == BounceBack::Interpolated;

    RowsFn fn;
    const bool fp32 = cfg.precision == Precision::FP32;
    const bool reg = cfg.collision == Collision::Regularized;
    const bool rec = cfg.collision == Collision::Recursive;
    const bool bk = k.bulk;
    if (fp32) fn = rec ? pick_rows<Precision::FP32, Collision::Recursive>(macro, bk)
                       : reg ? pick_rows<Precision::FP32, Collision::Regularized>(macro, bk) : pick_rows<Precision::FP32, Collision::BGK>(macro, bk);
    else fn = rec ? pick_rows<Precision::FP16S, Collision::Recursive>(macro, bk)
                  : reg ? pick_rows<Precision::FP16S, Collision::Regularized>(macro, bk) : pick_rows<Precision::FP16S, Collision::BGK>(macro, bk);

    i64 grain = tun.row_grain;
    if (grain <= 0) grain = std::max<i64>(1, 4096 / nx);   // ~4096 celdas por trozo (medido: 1 fila -23 %, 4 filas -7..-12 %)
    const int nt = pool().size();
    if (tun.max_threads > 0 && tun.max_threads < nt) {
        // Sólo max_threads "ranuras" toman trabajo; dentro, reparto dinámico por trozos con un contador propio.
        alignas(64) std::atomic<i64> next{0};
        pool().run_slots([&](int, int) {
            for (;;) {
                const i64 lo = next.fetch_add(grain, std::memory_order_relaxed);
                if (lo >= nrows) break;
                fn(k, lo, std::min(nrows, lo + grain));
            }
        }, tun.max_threads);
    } else {
        parallel_for(0, nrows, grain, [&](i64 lo, i64 hi) { fn(k, lo, hi); });
    }
}

// ---- Fuerzas: F_enlace = (g_k(τ) + g_k(τ-1) [+ 2w_k] - 6 w_k c_k·u_w) c_k [- 6w_k (c_k·u_w) u_w] ----
// (entre corchetes: sin force_gauge / con force_galilean; ver solver.hpp)
// Tras el paso τ, las DOS posiciones de un enlace fluido→sólido guardan las dos últimas
// poblaciones salientes g_k(τ) y g_k(τ-1) (rebote full-way de Esoteric-Pull), independientemente
// de la paridad. Momento: (s - c_k/2) × F_k = s × F_k porque F_k ∥ c_k.
void Solver::Impl::boundary_pass(bool accumulate, u64 tt, bool write_only) {
    const i64 L_ = static_cast<i64>(wnodes.size());
    if (write_only && cfg.bounce != BounceBack::Interpolated) return;
    // Trozos fijos (independientes del nº de hilos → suma determinista) y finos (~256 nodos)
    // para repartir bien entre núcleos P y E.
    const int nch = static_cast<int>(std::clamp<i64>(L_ / 256, 1, kMaxForceChunks));
    const bool fp32 = cfg.precision == Precision::FP32;
    const bool interp = cfg.bounce == BounceBack::Interpolated;
    const bool slip = interp && cfg.wall_model == WallModel::Slip;
    // Enlace fluido n → sólido s = n + c_k (k = dirección hacia el sólido, kb = opuesta). Tras el paso t:
    //   A  = f_k*(n, t)       (lo que n acaba de mandar hacia la pared)      → stc[k][n]
    //   B  = f_kb*(n, t)      (lo que n mandó alejándose de la pared)         → stc[kb][n]
    //   C  = f_k*(n - c_k, t) (lo que n cargará como f_k en t+1)              → ldn[k][n]
    //   destino: lo que n cargará como f_kb en t+1                            → ldn[kb][n]
    // A y destino son las DOS posiciones del enlace (las del rebote full-way). Rebote implícito: el
    // destino ya contiene f_k*(n, t-1). Interpolado (Bouzidi et al. 2001, lineal):
    //   q < ½: destino = 2q·A + (1-2q)·C          (sólo si n - c_k no es sólido: si lo fuera, C sería el
    //                                              destino de OTRO enlace del mismo nodo)
    //   q ≥ ½: destino = A/(2q) + (1 - 1/(2q))·B       (q = ½ → half-way: destino = A)
    //   + pared "móvil" u_s (modelo Slip): - g_q·6w_k (c_k·u_s),  g_q = 1 (q < ½) o 1/(2q) (q ≥ ½).
    // Pesos convexos (estable) y de suma 1: vale igual sobre poblaciones desplazadas/escaladas. Cada nodo
    // escribe sólo destinos de SUS enlaces y lee sus 19 post-colisión + C (que nadie escribe) → sin carreras.
    // Enlaces a sólidos MÓVILES: rebote implícito full-way (sin escribir) y el kernel suma Ladd al cargar.
    const void* ldc[L::Q]; void* stc[L::Q];
    const void* ldn[L::Q]; void* stn[L::Q];
    step_ptrs(tt, ldc, stc);
    step_ptrs(tt + 1, ldn, stn);
    (void)ldc; (void)stn;
    const WNode* W = wnodes.data();
    const u8* F = flags.data();
    const u8* SI = sid.data();
    const Motion* mot = motion_now;
    // Paredes móviles interpoladas: la población que se escribe aquí se carga en el paso tt+1 → velocidad de pared
    // de ese paso (durante una rampa difiere de la de tt; el Ladd del kernel ya usa la del paso en curso).
    Motion mnx[256];
    fill_motion(tt + 1, mnx);
    // Ley de pared (Werner-Wengle): τ_w = max(ν_w·|u_t|/y, (|u_t|/(8.3 (y/ν_w)^{1/7}))^{7/4}).
    const float nu_w = std::max(cfg.wall_nu > 0.0f ? cfg.wall_nu : cfg.nu, 1e-9f);
    const float inv_nu_w = 1.0f / nu_w;
    const float ww_a0 = 8.3f * std::pow(inv_nu_w, 1.0f / 7.0f);   // A = 8.3·(y/ν_w)^{1/7} = ww_a0·y^{1/7}
    // Referencia manométrica: con force_gauge el término 2w_k (fluido en reposo, ρ = 1) desaparece
    // (las poblaciones ya se guardan desplazadas f̃ = f - w). gi = término galileano de Wen et al.
    const float wref = cfg.force_gauge ? 0.0f : 1.0f;
    const float gi = cfg.force_galilean ? 1.0f : 0.0f;
    const float isc = fp32 ? 1.0f : kInvScale;           // almacenamiento → f̃
    const float sc = fp32 ? 1.0f : kScale;               // f̃ → almacenamiento
    FAcc* ACC = facc.data();
    u8* TCH = ftouch.data();
    pool().run_slots([&](int c, int ncs) {
        const i64 lo = L_ * c / ncs, hi = L_ * (c + 1) / ncs;
        FAcc* acc = ACC + static_cast<i64>(c) * 256;
        u8* tch = TCH + static_cast<i64>(c) * 256;
        std::memset(tch, 0, 256);
        auto ld = [&](const void* base, i64 n) -> float {
            return fp32 ? static_cast<const float*>(base)[n] : f16_to_f32(static_cast<const u16*>(base)[n]);
        };
        auto st = [&](const void* base, i64 n, float v) -> float {   // devuelve lo almacenado
            if (fp32) { static_cast<float*>(const_cast<void*>(base))[n] = v; return v; }
            const u16 h = f32_to_f16(v);
            static_cast<u16*>(const_cast<void*>(base))[n] = h;
            return f16_to_f32(h);
        };
        for (i64 i = lo; i < hi; ++i) {
            const WNode& nd = W[i];
            const i64 n = nd.n;
            float g[L::Q];
            for (int k = 0; k < L::Q; ++k) g[k] = ld(stc[k], n);
            // Destino SIN deslizamiento de cada enlace fijo (rebote interpolado): se calcula antes porque el
            // deslizamiento se elige a partir de la fricción que daría el rebote no deslizante.
            float v0[L::Q];
            if (interp) {
                u32 bits = nd.mask;
                while (bits) {
                    const int k = std::countr_zero(bits);
                    bits &= bits - 1;
                    if ((F[n + off[k]] & kMoving) && SI[n + off[k]] == k_ground_id) continue;
                    const int kb = L::opp[k];
                    const u8 qc = nd.q[k];
                    const float q = static_cast<float>(qc) * (1.0f / 254.0f);
                    const float A = g[k];
                    if (qc == kQHalf) v0[k] = A;
                    else if (q < 0.5f && !(F[n - off[k]] & kSolid)) {
                        const float C = ld(ldn[k], n);
                        v0[k] = vfma(2.0f * q, A - C, C);                      // 2q·A + (1-2q)·C
                    } else {
                        const float h = 0.5f / std::max(q, 0.5f);              // 1/(2q)
                        v0[k] = vfma(h, A - g[kb], g[kb]);                     // A/(2q) + (1 - 1/(2q))·B
                    }
                }
            }
            // Velocidad de deslizamiento (modelo Slip) respecto a las paredes FIJAS del nodo: IMPOSICIÓN EXACTA de
            // la tensión de pared de la ley logarítmica. Con u_s = s·t̂ la fuerza tangencial que el nodo transmite
            // a la pared es F_t(s) = F_ns − G·s (lineal en s; F_ns = la del rebote no deslizante calculada con las
            // poblaciones del paso, G = conductancia de los enlaces con peso p_k = g_q w_k). Se elige s para que
            // F_t = τ_w·A_nodo (Werner-Wengle con la distancia real y la ν del aire real), acotado a [0, f·|u_t|]:
            // la pared sólo REDUCE la fricción del rebote (nunca empuja) y como mucho desliza a la velocidad del
            // fluido (con s > |u_t| un NACA 0012 a 0° perdía la simetría: CL −0.09). F_ns incluye TODO el
            // intercambio tangencial del nodo (también la presión que en una escalera de vóxeles "se filtra" en la
            // dirección tangente porque el vector de área del nodo no es normal a la superficie real): en una placa
            // inclinada 1:10 así la velocidad a ½ celda pasa de 0.26 U a 0.5 U. Medido en una placa plana alineada
            // (Re_x 10⁶-10⁷, y⁺ ~ 300): cf 0.5-1.1 × la turbulenta (antes, con la v1 de este modelo, ×4-8).
            // Sin pesos φ(ĉ·n̂) por enlace (la v1 los usaba para la condición de Kutta): la condición de Kutta la
            // dan las aristas vivas (sgeo = 0) y la imposición exacta de τ_w.
            Vec3 us{0, 0, 0};
            float mcorr = 0.0f;
            bool slip_on = false;
            if (slip && (nd.kind & 2)) {
                float r = 0, jx = 0, jy = 0, jz = 0;
                for (int k = 0; k < L::Q; ++k) {
                    r += g[k];
                    jx += g[k] * float(L::c[k][0]); jy += g[k] * float(L::c[k][1]); jz += g[k] * float(L::c[k][2]);
                }
                const float rho = 1.0f + r * isc;
                const Vec3 u = Vec3(jx, jy, jz) * (isc / rho);
                const Vec3 nr(nd.nx, nd.ny, nd.nz);
                const Vec3 ut = u - nr * dot(u, nr);
                const float ut2 = length2(ut);
                if (ut2 > 1e-14f) {
                    const float um = std::sqrt(ut2);
                    const float y = nd.yw;
                    // τ_w/ρ = max(ν_w|u|/y, (|u|/A)^{7/4}),  x^{7/4} = x·√x·⁴√x (sin pow por paso).
                    const float xa = um / (ww_a0 * nd.yw17);
                    const float sx = std::sqrt(xa);
                    const float tw = std::max(nu_w * um / y, xa * sx * std::sqrt(sx));
                    // Deslizamiento sólo si la capa límite NO está resuelta: y⁺ de la 1.ª celda ≥ ~30 (capa
                    // logarítmica). Con y⁺ ≲ 5 (subcapa viscosa resuelta, Re bajos) el rebote no deslizante es lo correcto.
                    // En aristas vivas (sgeo → 0: bordes de salida) también: condición de Kutta.
                    const float yplus = y * std::sqrt(tw) * inv_nu_w;
                    const float slip_f = std::clamp((yplus - 5.0f) / 25.0f, 0.0f, 1.0f) * nd.sgeo;
                    if (slip_f > 0.0f) {
                        const Vec3 th = ut * (1.0f / um);
                        float G = 0.0f, Fns = 0.0f, P1 = 0.0f, W1 = 0.0f;
                        Vec3 asum{0, 0, 0};
                        u32 bits = nd.mask;
                        while (bits) {
                            const int k = std::countr_zero(bits);
                            bits &= bits - 1;
                            if (F[n + off[k]] & kMoving) continue;
                            const float q = static_cast<float>(nd.q[k]) * (1.0f / 254.0f);
                            const float gq = q < 0.5f ? 1.0f : 0.5f / q;
                            const float cx = kDir.c[k][0], cy = kDir.c[k][1], cz = kDir.c[k][2];
                            const float ct = cx * th.x + cy * th.y + cz * th.z;
                            const float pw = gq * L::w[k];
                            G += pw * ct * ct;
                            P1 += pw * ct;
                            W1 += pw;
                            Fns += (g[k] + v0[k]) * ct;
                            asum += Vec3(cx, cy, cz) * L::w[k];
                        }
                        // La corrección de masa (mcorr, abajo) devuelve 6 s (Σp_k c_k·t̂)²/Σp_k de la cantidad de
                        // movimiento tangencial en nodos asimétricos (escalones): conductancia efectiva
                        // G = 6 [Σ p_k (c_k·t̂)² − (Σ p_k c_k·t̂)²/Σ p_k] ≥ 0 (Cauchy-Schwarz; = la de antes en paredes planas).
                        G = 6.0f * (G - (W1 > 0.0f ? P1 * P1 / W1 : 0.0f));
                        const float area = 6.0f * length(asum);
                        if (G > 1e-6f) {
                            const float s = std::clamp((Fns * isc - tw * area) / G, 0.0f, slip_f * um);
                            us = th * s;
                            // Corrección de masa (conservativa como Ladd): los términos m_k = p_k·6 c_k·u_s no suman 0 en
                            // aristas y escalones y la pared inyectaba masa ∝ u_s cada paso (medido: chorro de 5·u∞ bajo
                            // una caja a 2 celdas de la cinta, prueba 7 de test_lbm). Se resta Σ m_k repartida con p_k.
                            mcorr = W1 > 0.0f ? 6.0f * s * P1 / W1 : 0.0f;
                            slip_on = s > 0.0f;
                        }
                    }
                }
            }
            // Ladd conservativo (ver block_scalar): Σ_móviles l_k y Σ w_k del nodo → corrección (Lm/Wm)·w_k.
            // (Los nodos "sólo cinta" tienen Lm = 0 exacto: la cinta es plana y u_g ∥ x̂.)
            // Paredes móviles con rebote INTERPOLADO (v2: ruedas; todo lo móvil salvo el suelo, id 255): destino =
            // Bouzidi + término de pared móvil g_q·6w_k(c_kb·u_w) (Bouzidi et al. 2001), escrito aquí; el kernel ya
            // no les suma Ladd. Con el rebote implícito en escalera la velocidad tangente de una superficie curva
            // tiene componente normal a las caras de los vóxeles y "sopla" en ellas: esfera giratoria Re = 100,
            // α = 0.5: CL 0.62 / CD 1.46 (CD sin giro 1.19); ruedas de F1 con MÁS resistencia girando que paradas.
            // Impermeables (conserve): la masa neta Σ m_k del nodo se resta repartida con g_q w_k (como Ladd).
            float Lm = 0.0f, Wm = 0.0f, Le = 0.0f, We = 0.0f;
            if (nd.kind & 1) {
                u32 mb = nd.mask;
                while (mb) {
                    const int k = std::countr_zero(mb);
                    mb &= mb - 1;
                    const i64 sidx = n + off[k];
                    if (!(F[sidx] & kMoving) || !mot[SI[sidx]].conserve) continue;
                    const float cx = kDir.c[k][0], cy = kDir.c[k][1], cz = kDir.c[k][2];
                    const bool ex = interp && SI[sidx] != k_ground_id;
                    const Vec3 uw = wall_velocity((ex ? mnx : mot)[SI[sidx]], float(nd.x) + cx, float(nd.y) + cy, float(nd.z) + cz);
                    const float lk = kDir.w6[k] * (cx * uw.x + cy * uw.y + cz * uw.z);
                    if (ex) {
                        const float q = static_cast<float>(nd.q[k]) * (1.0f / 254.0f);
                        const float gq = q < 0.5f ? 1.0f : 0.5f / q;
                        Le += gq * lk;
                        We += gq * L::w[k];
                    } else {
                        Lm += lk;
                        Wm += L::w[k];
                    }
                }
            }
            const float lcorr = Wm > 0.0f ? Lm / Wm : 0.0f;
            const float ecorr = We > 0.0f ? Le / We : 0.0f;
            int cur = -1;
            float nf[3] = {0, 0, 0}, nm[3] = {0, 0, 0};
            auto flush = [&]() {
                if (cur < 0) return;
                FAcc& o = acc[cur];
                if (!tch[cur]) { tch[cur] = 1; o = FAcc{}; }
                const double px = nd.x, py = nd.y, pz = nd.z;
                o.f[0] += nf[0]; o.f[1] += nf[1]; o.f[2] += nf[2];
                o.m[0] += py * nf[2] - pz * nf[1] + nm[0];
                o.m[1] += pz * nf[0] - px * nf[2] + nm[1];
                o.m[2] += px * nf[1] - py * nf[0] + nm[2];
                nf[0] = nf[1] = nf[2] = nm[0] = nm[1] = nm[2] = 0.0f;
            };
            u32 bits = nd.mask;
            while (bits) {
                const int k = std::countr_zero(bits);
                bits &= bits - 1;
                const int kb = L::opp[k];
                const i64 sidx = n + off[k];
                const u8 fs = F[sidx];
                const u32 id = SI[sidx];
                const float A = g[k];
                const float cx = kDir.c[k][0], cy = kDir.c[k][1], cz = kDir.c[k][2];
                float in, l = 0.0f, lg = 0.0f;   // l: Ladd que suma el kernel (implícito); lg: f_out − f_in (Wen)
                Vec3 uw{0, 0, 0};
                if ((fs & kMoving) && interp && id != k_ground_id) {
                    uw = wall_velocity(mnx[id], float(nd.x) + cx, float(nd.y) + cy, float(nd.z) + cz);
                    const float q = static_cast<float>(nd.q[k]) * (1.0f / 254.0f);
                    const float gq = q < 0.5f ? 1.0f : 0.5f / q;
                    const float mk = gq * (kDir.w6[k] * (cx * uw.x + cy * uw.y + cz * uw.z) - (mnx[id].conserve ? ecorr * L::w[k] : 0.0f));
                    in = st(ldn[kb], n, v0[k] - sc * mk);
                    lg = (A - in) * isc;
                } else if (fs & kMoving) {
                    uw = wall_velocity(mot[id], float(nd.x) + cx, float(nd.y) + cy, float(nd.z) + cz);
                    l = kDir.w6[k] * (cx * uw.x + cy * uw.y + cz * uw.z) - (mot[id].conserve ? lcorr * L::w[k] : 0.0f);   // f_out - f_in (Ladd, en el kernel)
                    lg = l;
                    // Suelo móvil (cinta): rebote implícito full-way (el destino ya guarda f_k*(n, t-1)) + Ladd en el kernel.
                    in = ld(ldn[kb], n);
                } else if (!interp) {
                    in = ld(ldn[kb], n);
                } else {
                    const float q = static_cast<float>(nd.q[k]) * (1.0f / 254.0f);
                    float v = v0[k];
                    if (slip_on) {
                        // Pared "móvil" a u_s por enlace con peso p_k = g_q w_k, menos la corrección de masa.
                        const float gq = q < 0.5f ? 1.0f : 0.5f / q;
                        v -= sc * gq * L::w[k] * (6.0f * (cx * us.x + cy * us.y + cz * us.z) - mcorr);
                    }
                    in = st(ldn[kb], n, v);
                }
                const float a = A * isc, b = in * isc;
                const float fv = (a + b + wref * kDir.w2[k]) - l;
                // Wen et al. 2014: F = Σ c_k(f_out + f_in) - u_w Σ (f_out - f_in)  (0 en paredes fijas).
                const float gx = gi * lg * uw.x, gy = gi * lg * uw.y, gz = gi * lg * uw.z;
                const float fx = fv * cx - gx, fy = fv * cy - gy, fz = fv * cz - gz;
                // Acumulación por nodo (float) y volcado por id: momento (x_n + c_k) × F_k = x_n × F_k + c_k × F_k y
                // c_k × F_k = −c_k × (gi·lg·u_w) (c_k × c_k = 0): sólo los enlaces móviles aportan el segundo término.
                if (static_cast<int>(id) != cur) { flush(); cur = static_cast<int>(id); }
                nf[0] += fx; nf[1] += fy; nf[2] += fz;
                if (fs & kMoving) {
                    nm[0] -= cy * gz - cz * gy;
                    nm[1] -= cz * gx - cx * gz;
                    nm[2] -= cx * gy - cy * gx;
                }
            }
            flush();
        }
    }, nch);
    if (write_only) return;
    // Reducción determinista (orden fijo de trozos, independiente del nº de hilos).
    double f[256][3] = {}, m[256][3] = {};
    for (int c = 0; c < nch; ++c) {
        const FAcc* acc = ACC + static_cast<i64>(c) * 256;
        const u8* tch = TCH + static_cast<i64>(c) * 256;
        for (int id = 0; id < 256; ++id)
            if (tch[id])
                for (int a = 0; a < 3; ++a) { f[id][a] += acc[id].f[a]; m[id][a] += acc[id].m[a]; }
    }
    const double r[3] = {moment_ref.x, moment_ref.y, moment_ref.z};
    double tf[3] = {0, 0, 0}, tm[3] = {0, 0, 0};
    for (int id = 0; id < 256; ++id) {
        // M_ref = Σ s×F - ref×F
        const double mx = m[id][0] - (r[1] * f[id][2] - r[2] * f[id][1]);
        const double my = m[id][1] - (r[2] * f[id][0] - r[0] * f[id][2]);
        const double mz = m[id][2] - (r[0] * f[id][1] - r[1] * f[id][0]);
        fs.force[id] = Vec3(float(f[id][0]), float(f[id][1]), float(f[id][2]));
        fs.moment[id] = Vec3(float(mx), float(my), float(mz));
        if (id >= 1 && id <= 254) {
            for (int a = 0; a < 3; ++a) tf[a] += f[id][a];
            tm[0] += mx; tm[1] += my; tm[2] += mz;
        }
        if (accumulate) {
            acc_f[id][0] += f[id][0]; acc_f[id][1] += f[id][1]; acc_f[id][2] += f[id][2];
            acc_m[id][0] += mx; acc_m[id][1] += my; acc_m[id][2] += mz;
        }
    }
    fs.total = Vec3(float(tf[0]), float(tf[1]), float(tf[2]));
    fs.total_moment = Vec3(float(tm[0]), float(tm[1]), float(tm[2]));
    fs.step = t + 1;
}

// ================================================================================================
Solver::Solver() : impl_(new Impl) {}
Solver::~Solver() { delete impl_; }

void Solver::init(const Config& cfg_in) {
    Impl& I = *impl_;
    I.sync_ext();
    I.ov[0] = I.ov[1] = I.ov[2] = I.ov[3] = nullptr;   // el dominio cambia: los campos del backend dejan de valer
    Config cfg = cfg_in;
    CFD_CHECK(cfg.nx >= 16 && cfg.ny >= 4 && cfg.nz >= 4, "lbm::Solver::init: dominio demasiado pequeño");
    CFD_CHECK(cfg.nx % 8 == 0, "lbm::Solver::init: nx debe ser múltiplo de 8");
    I.cfg = cfg;
    I.nx = cfg.nx; I.ny = cfg.ny; I.nz = cfg.nz; I.nbx = cfg.nx / 8;
    I.nxny = static_cast<i64>(cfg.nx) * cfg.ny;
    I.N = I.nxny * cfg.nz;
    for (int k = 0; k < L::Q; ++k) I.off[k] = L::offset(k, cfg.nx, I.nxny);
    // Relleno ≥ el mayor desplazamiento (nx·ny + nx) + 8 de la carga vectorial, múltiplo de 64.
    I.P = static_cast<i64>(round_up(static_cast<u64>(I.nxny + cfg.nx + 64), 64));
    // Zancada: múltiplo de 4 KiB + 3 líneas de sesgo → las 19 corrientes caen en conjuntos
    // distintos de L1/L2 (sin sesgo, N·es múltiplo de 4096 las apila en el mismo conjunto).
    const usize es = cfg.precision == Precision::FP32 ? 4 : 2;
    // CFD_LBM_SKEW (sólo benchmarks): nº de líneas de sesgo (defecto 3).
    usize skew = 3;
    if (const char* e = std::getenv("CFD_LBM_SKEW")) skew = static_cast<usize>(std::atoi(e));
    const usize bytes = round_up(static_cast<u64>(I.N + 2 * I.P) * es, 4096) + skew * 64;
    I.S = static_cast<i64>(bytes / es);
    I.ddf.resize(bytes * L::Q);
    const usize n = static_cast<usize>(I.N);
    I.flags.resize(n, true);
    I.flags_old.resize(n, true);
    I.sid.resize(n, true);
    I.user_sid.resize(n, true);
    I.cls.resize(n / 8, true);
    I.rho.resize(n); I.ux.resize(n); I.uy.resize(n); I.uz.resize(n);
    I.tau0.resize(static_cast<usize>(cfg.nx) + 8);
    I.tau0sq.resize(static_cast<usize>(cfg.nx) + 8);
    I.omc0.resize(static_cast<usize>(cfg.nx) + 8);
    const usize nrows = static_cast<usize>(cfg.ny) * static_cast<usize>(cfg.nz);
    I.rows_all.resize(nrows);
    I.rows_active.resize(nrows);
    for (usize r = 0; r < nrows; ++r) I.rows_all[r] = static_cast<u32>(r);
    I.facc.resize(static_cast<usize>(kMaxForceChunks) * 256);
    I.ftouch.resize(static_cast<usize>(kMaxForceChunks) * 256, true);
    for (int id = 0; id < 256; ++id) { I.user_motion[id] = WallMotion{}; I.user_moving[id] = false; }
    I.build_tau();
    I.t = 0;
    I.u_to = cfg.u_inf;
    I.rebuild(false);
    reset_flow();
}

const Config& Solver::config() const { return impl_->cfg; }

void Solver::set_geometry(const u8* solid_id) { set_geometry(solid_id, WallSdf{}); }

void Solver::set_geometry(const u8* solid_id, const WallSdf& sdf) {
    Impl& I = *impl_;
    I.sync_ext();
    if (solid_id) std::memcpy(I.user_sid.data(), solid_id, static_cast<usize>(I.N));
    else I.user_sid.zero();
    I.sdf = sdf;
    I.rebuild(true);
}

void Solver::set_wall_motion(u8 id, const WallMotion& m) {
    Impl& I = *impl_;
    if (id == 0 || id == k_ground_id) return;
    I.user_motion[id] = m;
    const bool mv = length2(m.v) + length2(m.omega) > 0.0f;
    const bool changed = mv != I.user_moving[id];
    if (changed) I.sync_ext();
    I.user_moving[id] = mv;
    if (changed) I.rebuild(true);
}

void Solver::clear_wall_motions() {
    Impl& I = *impl_;
    bool changed = false;
    for (int id = 0; id < 256; ++id) changed |= I.user_moving[id];
    if (changed) I.sync_ext();
    for (int id = 0; id < 256; ++id) { I.user_motion[id] = WallMotion{}; I.user_moving[id] = false; }
    if (changed) I.rebuild(true);
}

void Solver::set_ground(GroundMode g) {
    Impl& I = *impl_;
    if (g == I.cfg.ground) return;
    I.sync_ext();
    I.cfg.ground = g;
    I.rebuild(true);
}

void Solver::set_inflow(float u_inf) {
    Impl& I = *impl_;
    I.u_from = I.u_at(I.t);
    I.u_to = u_inf;
    I.ramp_t0 = I.t;
    I.cfg.u_inf = u_inf;
}

void Solver::set_viscosity(float nu) { impl_->sync_ext(); impl_->cfg.nu = nu; impl_->build_tau(); }
void Solver::set_wall_model(WallModel m, float wall_nu) {
    Impl& I = *impl_;
    I.cfg.wall_nu = std::max(wall_nu, 0.0f);
    if (m == I.cfg.wall_model) return;
    I.sync_ext();
    I.cfg.wall_model = m;
    I.rebuild(false);   // el bit kNearWall y las clases de bloque dependen del modelo
}
void Solver::set_smagorinsky(float cs) { impl_->cfg.cs_smag = std::max(cs, 0.0f); }
void Solver::set_moment_reference(Vec3 cells) { impl_->moment_ref = cells; }

void Solver::reset_flow() {
    Impl& I = *impl_;
    I.sync_ext();
    I.ov[0] = I.ov[1] = I.ov[2] = I.ov[3] = nullptr;   // reset_fill escribe los campos propios (los del backend quedan viejos)
    I.t = 0;
    I.ramp_t0 = 0;
    I.u_to = I.cfg.u_inf;
    I.u_from = I.cfg.ramp_steps > 0 ? 0.0f : I.cfg.u_inf;
    I.bad.store(0);
    I.fs = ForceSample{};
    I.fs_mean = ForceSample{};
    I.reset_fill();
    // El relleno equivale al post-colisión de un paso "−1" en equilibrio: se escriben ya las poblaciones
    // entrantes del rebote explícito (si no, el primer paso vería el valor full-way del relleno).
    I.update_motion_table();
    I.boundary_pass(false, I.t - 1, true);
}

void Solver::step(int n, bool update_macro) {
    Impl& I = *impl_;
    if (n <= 0) return;
    I.sync_ext();      // (fase 3) con un backend externo enganchado: poblaciones de la CPU al día
    ++I.revision;      // ... y las que resulten de este paso en la CPU invalidan las del backend
    const double t0 = now_sec();
    double tk = 0, tf = 0;
    for (int id = 0; id < 256; ++id)
        for (int a = 0; a < 3; ++a) { I.acc_f[id][a] = 0; I.acc_m[id][a] = 0; }
    for (int s = 0; s < n; ++s) {
        const bool macro = update_macro && s == n - 1;
        I.update_motion_table();
        const double a = now_sec();
        I.run_kernel(macro);
        const double b = now_sec();
        I.boundary_pass(true, I.t);
        const double c = now_sec();
        tk += b - a; tf += c - b;
        ++I.t;
    }
    const double dt = now_sec() - t0;
    I.mlups = dt > 0 ? static_cast<double>(I.N) * n / dt * 1e-6 : 0.0;
    I.t_kernel = tk; I.t_force = tf;
    // Media de fuerzas.
    const double inv = 1.0 / n;
    Vec3 tf3{0, 0, 0}, tm3{0, 0, 0};
    for (int id = 0; id < 256; ++id) {
        I.fs_mean.force[id] = Vec3(float(I.acc_f[id][0] * inv), float(I.acc_f[id][1] * inv), float(I.acc_f[id][2] * inv));
        I.fs_mean.moment[id] = Vec3(float(I.acc_m[id][0] * inv), float(I.acc_m[id][1] * inv), float(I.acc_m[id][2] * inv));
        if (id >= 1 && id <= 254) { tf3 += I.fs_mean.force[id]; tm3 += I.fs_mean.moment[id]; }
    }
    I.fs_mean.total = tf3;
    I.fs_mean.total_moment = tm3;
    I.fs_mean.step = I.t;
}

FieldView Solver::field() const {
    const Impl& I = *impl_;
    FieldView v;
    v.nx = I.nx; v.ny = I.ny; v.nz = I.nz;
    v.rho = I.rho.data(); v.ux = I.ux.data(); v.uy = I.uy.data(); v.uz = I.uz.data();
    if (I.ov[0]) { v.rho = I.ov[0]; v.ux = I.ov[1]; v.uy = I.ov[2]; v.uz = I.ov[3]; }
    v.flags = I.flags.data();
    v.solid_id = I.sid.data();
    v.u_inf = I.cfg.u_inf;
    return v;
}

const ForceSample& Solver::forces() const { return impl_->fs; }
const ForceSample& Solver::forces_mean() const { return impl_->fs_mean; }
u64 Solver::steps() const { return impl_->t; }
double Solver::last_mlups() const { return impl_->mlups; }
double Solver::last_force_seconds() const { return impl_->t_force; }
double Solver::last_kernel_seconds() const { return impl_->t_kernel; }
bool Solver::diverged() const { return impl_->bad.load(std::memory_order_relaxed) != 0; }
float Solver::current_u_inf() const { return impl_->u_at(impl_->t); }
void Solver::set_tuning(const Tuning& t) { impl_->tun = t; }
const Solver::Tuning& Solver::tuning() const { return impl_->tun; }

usize Solver::memory_bytes() const {
    const Impl& I = *impl_;
    return I.ddf.size() + I.flags.size() * 4 + I.cls.size() + (I.rho.size() + I.ux.size() + I.uy.size() + I.uz.size()) * 4 +
           I.wnodes.size() * sizeof(WNode) + I.facc.size() * sizeof(FAcc) + (I.rows_all.size() + I.rows_active.size()) * 4;
}

double Solver::total_mass() const {
    const Impl& I = *impl_;
    I.sync_ext();
    const void* ld[L::Q];
    void* st[L::Q];
    const void* ld_prev[L::Q];
    void* st_prev[L::Q];
    I.step_ptrs(I.t, ld, st);                  // lo que se cargará en el próximo paso
    I.step_ptrs(I.t + 1, ld_prev, st_prev);    // paridad del paso anterior (misma que t+1)
    (void)st; (void)ld_prev;
    const bool fp32 = I.cfg.precision == Precision::FP32;
    auto val = [&](const void* p, i64 n, int k) -> double {
        const double v = fp32 ? static_cast<const float*>(p)[n] : f16_to_f32(static_cast<const u16*>(p)[n]) * kInvScale;
        return v + L::wd[k];
    };
    const u8* F = I.flags.data();
    double total = 0;
    for (i64 n = 0; n < I.N; ++n) {
        if (F[n] & (kSolid | kInlet | kOutlet)) continue;
        for (int k = 0; k < L::Q; ++k) total += val(ld[k], n, k);
        // En vuelo (rebote implícito full-way): lo que n empujó hacia un sólido en el último paso (llegará
        // dentro de 2 pasos). Con el rebote interpolado la población entrante ya está escrita en ld (half-way).
        if (I.cfg.bounce == BounceBack::Implicit)
            for (int k = 1; k < L::Q; ++k)
                if (F[n + I.off[k]] & kSolid) total += val(st_prev[k], n, k);
    }
    return total;
}

// ---- (fase 3) Backend externo --------------------------------------------------------------------
void Solver::set_external(ExternalSync sync, void* ctx) { impl_->ext = sync; impl_->ext_ctx = ctx; }

ExternalView Solver::external_view() const {
    const Impl& I = *impl_;
    ExternalView v;
    v.ddf = const_cast<u8*>(I.ddf.data());
    v.S = I.S; v.P = I.P; v.N = I.N;
    v.esize = static_cast<int>(I.esize());
    for (int k = 0; k < L::Q; ++k) v.off[k] = I.off[k];
    v.flags = I.flags.data();
    v.sid = I.sid.data();
    v.rho = const_cast<float*>(I.rho.data()); v.ux = const_cast<float*>(I.ux.data());
    v.uy = const_cast<float*>(I.uy.data()); v.uz = const_cast<float*>(I.uz.data());
    v.tau0 = I.tau0.data(); v.tau0sq = I.tau0sq.data(); v.omc0 = I.omc0.data();
    v.nodes = I.wnodes.data();
    v.n_nodes = I.wnodes.size();
    v.motion = I.user_motion;
    v.moving = I.user_moving;
    v.wall_c3 = I.cfg.wall_model == WallModel::Slip ? 0.0f : wall_c3(I.cfg.wall_nu > 0.0f ? I.cfg.wall_nu : I.cfg.nu);
    v.wall_floor = kWallFloor;
    return v;
}

u64 Solver::revision() const { return impl_->revision; }

void Solver::ramp_at(u64 step, float* u_in, float* wall_scale) const {
    const Impl& I = *impl_;
    const float uc = I.u_at(step);
    if (u_in) *u_in = uc;
    if (wall_scale) *wall_scale = I.u_to != 0.0f ? uc / I.u_to : (I.u_from != 0.0f ? uc / I.u_from : 1.0f);   // = fill_motion
}

Vec3 Solver::moment_reference() const { return impl_->moment_ref; }

void Solver::external_commit(const ExternalStep& r) {
    Impl& I = *impl_;
    I.t += static_cast<u64>(r.steps);
    I.fs = r.last;
    I.fs.step = I.t;
    I.fs_mean = r.mean;
    I.fs_mean.step = I.t;
    if (r.diverged) I.bad.store(1, std::memory_order_relaxed);
    I.mlups = r.mlups;
    I.t_kernel = r.kernel_s;
    I.t_force = r.force_s;
}

void Solver::set_field_override(const float* rho, const float* ux, const float* uy, const float* uz) {
    Impl& I = *impl_;
    I.ov[0] = rho; I.ov[1] = ux; I.ov[2] = uy; I.ov[3] = uz;
}

} // namespace cfd::lbm
