// ============================================================================
//  render/camera.hpp — cámara orbital en perspectiva.
//
//  Mundo de render = coordenadas de celdas de la red LBM (1 unidad = 1 celda),
//  Z arriba, el aire fluye hacia +X. La cámara orbita alrededor de `target`.
//  yaw = 0 mira desde -Y hacia +Y (vista lateral del lado derecho del coche);
//  pitch > 0 = desde arriba.
//  Salida de project(): coordenadas de píxel dentro del framebuffer completo
//  (ya desplazadas al viewport) y profundidad de vista lineal (> 0 delante).
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include "framebuffer.hpp"

namespace cfd::render {

struct Camera {
    // Estado controlable
    Vec3 target{0, 0, 0};
    float yaw = -0.6f;           // rad
    float pitch = 0.35f;         // rad (limitado a ±1.5)
    float distance = 400.0f;     // unidades de mundo (celdas)
    float fov_y = 38.0f * k_deg2rad;
    float znear = 1.0f, zfar = 20000.0f;
    bool ortho = false;          // proyección ortográfica (vistas técnicas)

    // Derivado (update)
    Rect vp;                     // viewport en el framebuffer
    Vec3 eye, fwd, right, up;    // base ortonormal de la cámara
    Mat4 view, proj, viewproj;
    float focal = 1.0f;          // píxeles por unidad en el plano z=1 (perspectiva)
    float ortho_scale = 1.0f;    // píxeles por unidad (ortográfica)

    void update(Rect viewport) {
        vp = viewport;
        pitch = clamp_(pitch, -1.5f, 1.5f);
        distance = clamp_(distance, 2.0f, 50000.0f);
        const float cp = std::cos(pitch), sp = std::sin(pitch), cy = std::cos(yaw), sy = std::sin(yaw);
        const Vec3 offs{sy * cp, -cy * cp, sp};           // de target hacia el ojo
        eye = target + offs * distance;
        fwd = normalize(target - eye);
        right = normalize(cross(fwd, Vec3(0, 0, 1)));
        up = cross(right, fwd);
        const float aspect = vp.h > 0 ? static_cast<float>(vp.w) / static_cast<float>(vp.h) : 1.0f;
        view = Mat4::look_at(eye, target, Vec3(0, 0, 1));
        proj = Mat4::perspective(fov_y, aspect, znear, zfar);
        viewproj = proj * view;
        focal = 0.5f * static_cast<float>(vp.h) / std::tan(0.5f * fov_y);
        ortho_scale = focal / distance;
    }
    // Mundo → píxel. Devuelve false si está detrás de la cámara (o antes de znear).
    CFD_INLINE bool project(Vec3 p, float& sx, float& sy, float& depth) const {
        const Vec3 d = p - eye;
        depth = dot(d, fwd);
        if (!ortho && depth < znear) return false;
        const float s = ortho ? ortho_scale : focal / depth;
        sx = static_cast<float>(vp.x) + 0.5f * static_cast<float>(vp.w) + dot(d, right) * s;
        sy = static_cast<float>(vp.y) + 0.5f * static_cast<float>(vp.h) - dot(d, up) * s;
        return true;
    }
    // Píxel → rayo en mundo (dir normalizada).
    CFD_INLINE void ray(float sx, float sy, Vec3& origin, Vec3& dir) const {
        const float px = sx - (static_cast<float>(vp.x) + 0.5f * static_cast<float>(vp.w));
        const float py = (static_cast<float>(vp.y) + 0.5f * static_cast<float>(vp.h)) - sy;
        if (ortho) {
            origin = eye + right * (px / ortho_scale) + up * (py / ortho_scale);
            dir = fwd;
        } else {
            origin = eye;
            dir = normalize(fwd * focal + right * px + up * py);
        }
    }
    // Controles
    void orbit(float dyaw, float dpitch) { yaw += dyaw; pitch = clamp_(pitch + dpitch, -1.5f, 1.5f); }
    void pan(float dx_px, float dy_px) {
        const float s = distance / focal;
        target += right * (-dx_px * s) + up * (dy_px * s);
    }
    void zoom(float factor) { distance = clamp_(distance * factor, 2.0f, 50000.0f); }
};

} // namespace cfd::render
