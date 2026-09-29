// ============================================================================
//  lbm/kernel.hpp — piezas COMPARTIDAS (privadas del módulo lbm) del kernel D3Q19: constantes de
//  almacenamiento, bits internos de flags, momentos en flujo, tensor de no equilibrio, Smagorinsky, ley de
//  pared y la colisión genérica (float / f8 / f8x2). Las usan lbm/solver.cpp (kernel AVX2, contorno) y
//  lbm/refine.cpp (pasadas de interfaz del refinamiento local). Nada de esto forma parte del contrato público
//  (lbm/solver.hpp): espacio de nombres detail.
// ============================================================================
#pragma once

#include "solver.hpp"
#include "lattice.hpp"
#include "../core/simd.hpp"
#include "../core/util.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace cfd::lbm::detail {

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
    kBlkTap = 32,     // BIT (refinamiento): el kernel guarda ρ, u y Π^neq de sus 8 celdas (restricción hacia el padre)
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

template <Precision P> struct Store;
template <> struct Store<Precision::FP32> { using T = float; };
template <> struct Store<Precision::FP16S> { using T = u16; };

// (refinamiento) Floats por bloque "tap": 10 magnitudes (ρ, u_x, u_y, u_z, Π^neq xx yy zz xy xz yz) × 8 carriles (SoA).
inline constexpr int kTapFloats = 80;

// Escala de almacenamiento como constante de plantilla (1 en FP32, 2^15 en FP16S).
template <Precision P> inline constexpr float kSc = P == Precision::FP32 ? 1.0f : kScale;

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

} // namespace cfd::lbm::detail
