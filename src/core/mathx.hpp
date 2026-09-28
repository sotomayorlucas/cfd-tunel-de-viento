// ============================================================================
//  core/mathx.hpp — álgebra 3D mínima, constexpr y sin dependencias.
//
//  Convención global: mano derecha, Z hacia arriba.
//    X = dirección del flujo (el aire viaja hacia +X; el coche mira hacia -X)
//    Y = lateral (+Y = DERECHA del piloto: el coche mira hacia -X con Z arriba)
//    Z = vertical
//  Matrices en orden fila-mayor (row-major); vectores columna: v' = M * v.
// ============================================================================
#pragma once

#include "config.hpp"
#include <bit>
#include <cmath>
#include <cstring>

namespace cfd {

inline constexpr float k_pi = 3.14159265358979323846f;
inline constexpr float k_deg2rad = k_pi / 180.0f;
inline constexpr float k_rad2deg = 180.0f / k_pi;

template <class T> CFD_INLINE constexpr T min_(T a, T b) { return b < a ? b : a; }
template <class T> CFD_INLINE constexpr T max_(T a, T b) { return a < b ? b : a; }
template <class T> CFD_INLINE constexpr T clamp_(T v, T lo, T hi) { return min_(max_(v, lo), hi); }
template <class T> CFD_INLINE constexpr T sq(T v) { return v * v; }
CFD_INLINE constexpr float lerp(float a, float b, float t) { return a + (b - a) * t; }
CFD_INLINE constexpr float saturate(float v) { return clamp_(v, 0.0f, 1.0f); }
CFD_INLINE constexpr float smoothstep(float e0, float e1, float x) {
    const float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}

// --- Trucos numéricos clásicos -------------------------------------------------
// Raíz cuadrada inversa rápida (Quake III) + 1 iteración de Newton. Error < 0.2%.
// Sólo por tradición: en rutas SIMD usar _mm256_rsqrt_ps + Newton (core/simd.hpp).
CFD_INLINE constexpr float fast_rsqrt(float x) {
    const float xh = 0.5f * x;
    float y = std::bit_cast<float>(0x5f3759dfu - (std::bit_cast<u32>(x) >> 1));
    return y * (1.5f - xh * y * y);
}
// floor → int sin llamar a floorf (rama-libre para x en rango int).
CFD_INLINE constexpr int ifloor(float x) { const int i = static_cast<int>(x); return i - (x < static_cast<float>(i)); }

// --- Vectores ------------------------------------------------------------------
struct Vec2 {
    float x = 0, y = 0;
    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
    constexpr Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
    constexpr Vec2 operator-() const { return {-x, -y}; }
};
CFD_INLINE constexpr float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
CFD_INLINE constexpr float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
CFD_INLINE float length(Vec2 a) { return std::sqrt(dot(a, a)); }

struct Vec3 {
    float x = 0, y = 0, z = 0;
    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr explicit Vec3(float s) : x(s), y(s), z(s) {}
    constexpr Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(Vec3 o) const { return {x * o.x, y * o.y, z * o.z}; }
    constexpr Vec3 operator/(Vec3 o) const { return {x / o.x, y / o.y, z / o.z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(float s) const { const float r = 1.0f / s; return {x * r, y * r, z * r}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr Vec3& operator+=(Vec3 o) { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr Vec3& operator-=(Vec3 o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    constexpr float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    constexpr float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};
CFD_INLINE constexpr Vec3 operator*(float s, Vec3 v) { return v * s; }
CFD_INLINE constexpr float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
CFD_INLINE constexpr Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
CFD_INLINE float length(Vec3 a) { return std::sqrt(dot(a, a)); }
CFD_INLINE constexpr float length2(Vec3 a) { return dot(a, a); }
CFD_INLINE Vec3 normalize(Vec3 a) { const float l2 = dot(a, a); return l2 > 0 ? a * (1.0f / std::sqrt(l2)) : Vec3{0, 0, 0}; }
CFD_INLINE constexpr Vec3 vmin(Vec3 a, Vec3 b) { return {min_(a.x, b.x), min_(a.y, b.y), min_(a.z, b.z)}; }
CFD_INLINE constexpr Vec3 vmax(Vec3 a, Vec3 b) { return {max_(a.x, b.x), max_(a.y, b.y), max_(a.z, b.z)}; }
CFD_INLINE Vec3 vabs(Vec3 a) { return {std::fabs(a.x), std::fabs(a.y), std::fabs(a.z)}; }
CFD_INLINE constexpr Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
CFD_INLINE constexpr float max_comp(Vec3 a) { return max_(a.x, max_(a.y, a.z)); }
CFD_INLINE constexpr float min_comp(Vec3 a) { return min_(a.x, min_(a.y, a.z)); }

struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
    constexpr Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    constexpr Vec3 xyz() const { return {x, y, z}; }
};

// --- Matriz 3x3 (rotaciones) ----------------------------------------------------
struct Mat3 {
    float m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static constexpr Mat3 identity() { return {}; }
    constexpr Vec3 operator*(Vec3 v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    constexpr Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
        return r;
    }
    constexpr Mat3 transposed() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
        return r;
    }
    // Rotaciones activas (ángulo en radianes, regla de la mano derecha).
    static Mat3 rot_x(float a) { const float c = std::cos(a), s = std::sin(a); Mat3 r; r.m[1][1] = c; r.m[1][2] = -s; r.m[2][1] = s; r.m[2][2] = c; return r; }
    static Mat3 rot_y(float a) { const float c = std::cos(a), s = std::sin(a); Mat3 r; r.m[0][0] = c; r.m[0][2] = s; r.m[2][0] = -s; r.m[2][2] = c; return r; }
    static Mat3 rot_z(float a) { const float c = std::cos(a), s = std::sin(a); Mat3 r; r.m[0][0] = c; r.m[0][1] = -s; r.m[1][0] = s; r.m[1][1] = c; return r; }
};

// Transformación rígida: p_mundo = R * p_local + t
struct Xform {
    Mat3 R;
    Vec3 t;
    constexpr Vec3 apply(Vec3 p) const { return R * p + t; }
    constexpr Vec3 apply_dir(Vec3 d) const { return R * d; }
    constexpr Vec3 inverse_apply(Vec3 p) const { return R.transposed() * (p - t); }
    constexpr Xform inverse() const { const Mat3 Rt = R.transposed(); return {Rt, -(Rt * t)}; }
    constexpr Xform operator*(const Xform& o) const { return {R * o.R, R * o.t + t}; } // this ∘ o
};

// --- Matriz 4x4 (cámara / proyección) -------------------------------------------
struct Mat4 {
    float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    constexpr Vec4 operator*(Vec4 v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w};
    }
    constexpr Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j] + m[i][3] * o.m[3][j];
        return r;
    }
    // Vista "look-at": cámara en eye mirando a target; espacio de cámara: +X derecha, +Y arriba, -Z adelante.
    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        const Vec3 f = normalize(target - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r;
        r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
        r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
        r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
        return r;
    }
    // Perspectiva estilo OpenGL: NDC z en [-1,1], w_clip = -z_cam (distancia de vista).
    static Mat4 perspective(float fovy_rad, float aspect, float znear, float zfar) {
        const float f = 1.0f / std::tan(fovy_rad * 0.5f);
        Mat4 r;
        r.m[0][0] = f / aspect; r.m[1][1] = f;
        r.m[2][2] = (zfar + znear) / (znear - zfar); r.m[2][3] = 2.0f * zfar * znear / (znear - zfar);
        r.m[3][2] = -1.0f; r.m[3][3] = 0.0f;
        return r;
    }
};

// --- Caja envolvente --------------------------------------------------------------
struct Aabb {
    Vec3 lo{ 1e30f,  1e30f,  1e30f};
    Vec3 hi{-1e30f, -1e30f, -1e30f};
    constexpr bool empty() const { return lo.x > hi.x || lo.y > hi.y || lo.z > hi.z; }
    constexpr void grow(Vec3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    constexpr void grow(const Aabb& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    constexpr Aabb expanded(float r) const { return {lo - Vec3(r), hi + Vec3(r)}; }
    constexpr Vec3 center() const { return (lo + hi) * 0.5f; }
    constexpr Vec3 size() const { return hi - lo; }
    constexpr bool contains(Vec3 p) const { return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z; }
    // Distancia (≥0) de p a la caja; 0 si está dentro. Cota inferior del SDF de lo que contenga.
    CFD_INLINE float distance(Vec3 p) const {
        const Vec3 d = vmax(vmax(lo - p, p - hi), Vec3(0.0f));
        return std::sqrt(dot(d, d));
    }
    // Caja que envuelve la caja transformada (8 esquinas).
    Aabb transformed(const Xform& X) const {
        Aabb r;
        for (int i = 0; i < 8; ++i)
            r.grow(X.apply({(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z}));
        return r;
    }
};

} // namespace cfd
