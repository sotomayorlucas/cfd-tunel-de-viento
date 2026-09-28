// ============================================================================
//  render/flowvis_draw.cpp — llamadas de dibujo de flowvis sobre el rasterizador
//  (render/raster.hpp). Separado del resto del módulo para que los cálculos
//  (tests numéricos, herramientas sin ventana) enlacen sin render/raster.cpp.
// ============================================================================
#include "flowvis.hpp"
#include "raster.hpp"

namespace cfd::flowvis {

void SliceView::draw(Framebuffer& fb, const Camera& cam) const {
    if (tex_.empty()) return;
    Vec3 c[4];
    corners(c);
    const bool opaque = params.opacity >= 0.999f && fade_ <= 0.0f;
    render::draw_textured_quad(fb, cam, c, tex_.data(), lw_, lh_, params.opacity, true, opaque);
}

void GroundFootprint::draw(Framebuffer& fb, const Camera& cam, float spacing, float offset_x, u32 base, u32 line) const {
    if (!texture()) return;
    const Aabb e = extent();
    render::draw_ground(fb, cam, e.lo.z, e, spacing, offset_x, base, line, texture(), tex_w(), tex_h());
}

namespace {
bool any_plane(std::span<const TranslucentPlane> behind) {
    for (const TranslucentPlane& p : behind) if (p.opacity > 0.0f) return true;
    return false;
}
// Copia de colores con el canal A escalado por la transmisión de los planos (paralelo).
void attenuate(const Camera& cam, std::span<const TranslucentPlane> behind, const Vec3* pts, const u32* in, u32* out, usize n) {
    parallel_for(0, static_cast<i64>(n), 1 << 13, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const u32 c = in[i];
            const float k = plane_transmission(cam, pts[i], behind);
            out[i] = k >= 1.0f ? c : (c & 0x00FFFFFFu) | (static_cast<u32>(static_cast<float>(c >> 24) * k + 0.5f) << 24);
        }
    });
}
} // namespace

void Streamlines::draw(Framebuffer& fb, const Camera& cam, bool depth_test, std::span<const TranslucentPlane> behind) const {
    if (n_lines_ == 0) return;
    if (!any_plane(behind)) {
        render::draw_polylines(fb, cam, points(), colors(), starts(), counts(), params.width, depth_test);
        return;
    }
    // Sólo se atenúan los rangos usados (las ranuras de cada línea tienen huecos).
    if (draw_col_.size() < col_.size()) draw_col_.resize(col_.size());
    parallel_for(0, static_cast<i64>(n_lines_), 16, [&](i64 lo, i64 hi) {
        for (i64 l = lo; l < hi; ++l) {
            const usize s0 = starts_[static_cast<usize>(l)], c = counts_[static_cast<usize>(l)];
            for (usize i = s0; i < s0 + c; ++i) {
                const u32 col = col_[i];
                const float k = plane_transmission(cam, pts_[i], behind);
                draw_col_[i] = k >= 1.0f ? col : (col & 0x00FFFFFFu) | (static_cast<u32>(static_cast<float>(col >> 24) * k + 0.5f) << 24);
            }
        }
    });
    render::draw_polylines(fb, cam, points(), {draw_col_.data(), col_.size()}, starts(), counts(), params.width, depth_test);
}

void Particles::draw(Framebuffer& fb, const Camera& cam, std::span<const TranslucentPlane> behind) const {
    if (stats_.alive == 0) return;
    if (!any_plane(behind)) {
        render::draw_points(fb, cam, points(), colors(), params.point_size, params.additive);
        return;
    }
    if (draw_col_.size() < stats_.alive) draw_col_.resize(cap_);
    attenuate(cam, behind, out_pts_.data(), out_col_.data(), draw_col_.data(), stats_.alive);
    render::draw_points(fb, cam, points(), {draw_col_.data(), stats_.alive}, params.point_size, params.additive);
}

} // namespace cfd::flowvis
