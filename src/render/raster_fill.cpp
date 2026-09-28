// ============================================================================
//  render/raster_fill.cpp — primitivas "de relleno":
//   * draw_textured_quad: recorte 3D + triángulos 28.4, UV perspectiva-correctas,
//     bilineal SWAR con 4 gathers, paralelo por tiles 64×64 del rectángulo.
//   * draw_ground: intersección rayo/plano por píxel (8 carriles), rejilla con
//     antialias analítico (ancho de línea por derivadas en pantalla), damero
//     sutil, cinta animada, textura opcional sobre la extensión, niebla.
// ============================================================================
#include "raster_internal.hpp"

namespace cfd::render {
namespace rz {

// Vértice proyectado del cuadrilátero.
struct QV { i32 fx, fy; float q; float u, v; };

template <bool Persp, bool DepthTest, bool DepthWrite>
static void quad_tile(Framebuffer& fb, const TriFx* T, const QV (*tv)[3], int ntri, int cx0, int cy0, int cx1, int cy1,
                      const u32* tex, int tw, int th, float alpha) {
    const Surf SF = Surf::of(fb);
    const __m256 vtw = _mm256_set1_ps(static_cast<float>(tw)), vth = _mm256_set1_ps(static_cast<float>(th));
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 va = _mm256_set1_ps(alpha * (1.0f / 255.0f));
    for (int i = 0; i < ntri; ++i) {
        const QV& a = tv[i][0];
        const QV& b = tv[i][1];
        const QV& c = tv[i][2];
        const __m256 q0 = _mm256_set1_ps(a.q), dq1 = _mm256_set1_ps(b.q - a.q), dq2 = _mm256_set1_ps(c.q - a.q);
        const __m256 q1 = _mm256_set1_ps(b.q), q2 = _mm256_set1_ps(c.q);
        const __m256 u0 = _mm256_set1_ps(a.u), du1 = _mm256_set1_ps(b.u - a.u), du2 = _mm256_set1_ps(c.u - a.u);
        const __m256 v0 = _mm256_set1_ps(a.v), dv1 = _mm256_set1_ps(b.v - a.v), dv2 = _mm256_set1_ps(c.v - a.v);
        scan_tri_in(SF, T[i], cx0, cy0, cx1, cy1, [&](const auto& blk, int bits, __m256 l1, __m256 l2) RZ_LAMBDA_INLINE {
            const __m256 q = _mm256_fmadd_ps(l1, dq1, _mm256_fmadd_ps(l2, dq2, q0));
            const __m256 w = Persp ? simd::rcp_nr(q).v : q;
            __m256 zb;
            if constexpr (DepthTest || DepthWrite) zb = blk.load_z();
            if constexpr (DepthTest) {
                bits &= _mm256_movemask_ps(_mm256_cmp_ps(w, zb, _CMP_LT_OQ));
                if (!bits) return;
            }
            __m256 p1 = l1, p2 = l2;
            if constexpr (Persp) { p1 = _mm256_mul_ps(l1, _mm256_mul_ps(q1, w)); p2 = _mm256_mul_ps(l2, _mm256_mul_ps(q2, w)); }
            const __m256 u = _mm256_fmadd_ps(p1, du1, _mm256_fmadd_ps(p2, du2, u0));
            const __m256 v = _mm256_fmadd_ps(p1, dv1, _mm256_fmadd_ps(p2, dv2, v0));
            const __m256i texel = bilinear8(tex, tw, th, _mm256_fmsub_ps(u, vtw, half), _mm256_fmsub_ps(v, vth, half));
            const __m256 a01 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(texel, 24)), va);
            const __m256i dst = blk.load_c();
            const __m256i out = _mm256_or_si256(blend_swar(dst, texel, alpha16(a01)), _mm256_set1_epi32(static_cast<int>(0xFF000000u)));
            const __m256i m = mask_si(bits);
            blk.store_c(_mm256_blendv_epi8(dst, out, m));
            if constexpr (DepthWrite) {
                const __m256 wm = _mm256_and_ps(_mm256_castsi256_ps(m), _mm256_cmp_ps(a01, half, _CMP_GE_OQ));
                const __m256 zn = DepthTest ? w : _mm256_min_ps(w, zb);
                blk.store_z(_mm256_blendv_ps(zb, zn, wm));
            }
        });
    }
}

} // namespace rz

// =============================================================================================
void draw_textured_quad(Framebuffer& fb, const Camera& cam, const Vec3 corners[4], const u32* tex, int tw, int th,
                        float alpha, bool depth_test, bool depth_write) {
    using namespace rz;
    if (!tex || tw <= 0 || th <= 0 || !(alpha > 0.0f) || fb.color.empty()) return;
    alpha = min_(alpha, 1.0f);
    if (fb.depth.empty()) depth_test = depth_write = false;
    const View V = make_view(cam, fb);
    if (V.clip.empty()) return;
    CVert poly[16], tmp[16];
    const float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    u32 all = OC_PLANES | OC_BAD, any = 0;
    for (int i = 0; i < 4; ++i) {
        poly[i].c = to_view(V, corners[i]);
        poly[i].a[0] = uv[i][0]; poly[i].a[1] = uv[i][1];
        poly[i].orig = -1;
        const u32 oc = outcode(V, poly[i].c);
        if (oc & OC_BAD) return;
        all &= oc; any |= oc;
    }
    if (all & OC_PLANES) return;   // todo fuera de un mismo plano
    int n = 4;
    if (any) n = clip_polygon(V, poly, 4, tmp, any & OC_PLANES, 2);
    if (n < 3) return;
    QV qv[16];
    for (int i = 0; i < n; ++i) {
        if (!project_fx(V, poly[i].c, qv[i].fx, qv[i].fy, qv[i].q)) return;
        qv[i].u = poly[i].a[0]; qv[i].v = poly[i].a[1];
    }
    TriFx T[14];
    QV tv[14][3];
    int nt = 0;
    int bx0 = V.clip.x1, by0 = V.clip.y1, bx1 = V.clip.x0, by1 = V.clip.y0;
    for (int k = 1; k + 1 < n; ++k) {
        QV a = qv[0], b = qv[k], c = qv[k + 1];
        i64 area = static_cast<i64>(b.fx - a.fx) * (c.fy - a.fy) - static_cast<i64>(c.fx - a.fx) * (b.fy - a.fy);
        if (area == 0) continue;
        if (area < 0) { const QV t = b; b = c; c = t; area = -area; }   // doble cara
        TriFx& t = T[nt];
        t.x[0] = a.fx; t.x[1] = b.fx; t.x[2] = c.fx;
        t.y[0] = a.fy; t.y[1] = b.fy; t.y[2] = c.fy;
        t.area2 = area;
        t.inv_area2 = 1.0f / static_cast<float>(area);
        int px0, py0, px1, py1;
        if (!tri_pixel_box(t.x, t.y, V.clip, px0, py0, px1, py1)) continue;
        bx0 = min_(bx0, px0); by0 = min_(by0, py0); bx1 = max_(bx1, px1); by1 = max_(by1, py1);
        tv[nt][0] = a; tv[nt][1] = b; tv[nt][2] = c;
        ++nt;
    }
    if (nt == 0 || bx0 > bx1 || by0 > by1) return;
    const int gx0 = bx0 >> k_tile_shift, gy0 = by0 >> k_tile_shift;
    const int ntx = (bx1 >> k_tile_shift) - gx0 + 1, nty = (by1 >> k_tile_shift) - gy0 + 1;
    using Fn = void (*)(Framebuffer&, const TriFx*, const QV (*)[3], int, int, int, int, int, const u32*, int, int, float);
    Fn fn;
    const int sel = (depth_test ? 1 : 0) | (depth_write ? 2 : 0);
    if (V.ortho) {
        constexpr Fn t[4] = {&quad_tile<false, false, false>, &quad_tile<false, true, false>, &quad_tile<false, false, true>, &quad_tile<false, true, true>};
        fn = t[sel];
    } else {
        constexpr Fn t[4] = {&quad_tile<true, false, false>, &quad_tile<true, true, false>, &quad_tile<true, false, true>, &quad_tile<true, true, true>};
        fn = t[sel];
    }
    parallel_for(0, static_cast<i64>(ntx) * nty, 1, [&](i64 lo, i64 hi) {
        for (i64 k = lo; k < hi; ++k) {
            const int tx = static_cast<int>(k % ntx) + gx0, ty = static_cast<int>(k / ntx) + gy0;
            const int cx0 = max_(tx << k_tile_shift, bx0), cy0 = max_(ty << k_tile_shift, by0);
            const int cx1 = min_((tx << k_tile_shift) + k_tile - 1, bx1), cy1 = min_((ty << k_tile_shift) + k_tile - 1, by1);
            fn(fb, T, tv, nt, cx0, cy0, cx1, cy1, tex, tw, th, alpha);
        }
    });
}

// ---------------------------------------------------------------------------------------------
void draw_ground(Framebuffer& fb, const Camera& cam, float z0, Aabb extent, float spacing, float offset_x,
                 u32 base, u32 line, const u32* tex, int tw, int th) {
    using namespace rz;
    if (fb.color.empty() || !std::isfinite(z0)) return;
    const View V = make_view(cam, fb);
    if (V.clip.empty()) return;
    const bool has_depth = !fb.depth.empty();
    const bool has_ext = !extent.empty() && std::isfinite(extent.lo.x) && std::isfinite(extent.hi.x) &&
                         std::isfinite(extent.lo.y) && std::isfinite(extent.hi.y);
    const bool grid = std::isfinite(spacing) && spacing > 1e-6f;
    const bool has_tex = tex && tw > 0 && th > 0 && has_ext && extent.hi.x > extent.lo.x && extent.hi.y > extent.lo.y;
    const float inv_sp = grid ? 1.0f / spacing : 0.0f;
    const float offx = std::isfinite(offset_x) ? offset_x : 0.0f;
    // Niebla: el suelo se funde con el fondo lejos del centro de la escena.
    const float fog0 = cam.distance * 1.6f, fog1 = cam.distance * 5.0f;
    const float base_a = static_cast<float>(base >> 24) * (1.0f / 255.0f);
    const float line_a = static_cast<float>(line >> 24) * (1.0f / 255.0f);
    if (!(base_a > 0.0f)) return;
    const Vec3 E = V.eye, R = V.right, U = V.up, F = V.fwd;
    const float f = V.f;
    const float cx = V.cx, cy = V.cy;
    const float br = static_cast<float>((base >> 16) & 255), bg = static_cast<float>((base >> 8) & 255), bb = static_cast<float>(base & 255);
    const __m256i vline = _mm256_set1_epi32(static_cast<int>(line));
    const float tsx = has_tex ? static_cast<float>(tw) / (extent.hi.x - extent.lo.x) : 0.0f;
    const float tsy = has_tex ? static_cast<float>(th) / (extent.hi.y - extent.lo.y) : 0.0f;

    parallel_for(V.clip.y0, V.clip.y1 + 1, 8, [&](i64 lo, i64 hi) {
        const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f);
        const __m256 sgn = _mm256_set1_ps(-0.0f);
        const __m256 lc = lane_c();
        auto vabs = [&](__m256 v) RZ_LAMBDA_INLINE { return _mm256_andnot_ps(sgn, v); };
        auto sat = [&](__m256 v) RZ_LAMBDA_INLINE { return _mm256_min_ps(_mm256_max_ps(v, zero), one); };
        auto dist_int = [&](__m256 v) RZ_LAMBDA_INLINE {   // |v - round(v)|
            return vabs(_mm256_sub_ps(v, _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
        };
        for (i64 yy = lo; yy < hi; ++yy) {
            const int y = static_cast<int>(yy);
            const float py = cy - (static_cast<float>(y) + 0.5f);
            // Rayo por píxel: persp  D = F·f + U·py + R·px, origen E ;  ortho  D = F, origen E + (R·px + U·py)/s
            const float px0 = static_cast<float>(V.clip.x0) + 0.5f - cx, px1 = static_cast<float>(V.clip.x1) + 0.5f - cx;
            float dz0, dz1, oz;
            if (V.ortho) { dz0 = dz1 = F.z; oz = E.z + U.z * py / f; }
            else { dz0 = F.z * f + U.z * py + R.z * px0; dz1 = F.z * f + U.z * py + R.z * px1; oz = E.z; }
            const float num = z0 - oz;
            // fila entera por encima del horizonte (t ≤ 0 en ambos extremos) → nada
            if (!(num * dz0 > 0.0f) && !(num * dz1 > 0.0f)) continue;
            u32* crow = fb.color.data() + static_cast<usize>(y) * static_cast<usize>(fb.stride);
            float* zrow = has_depth ? fb.depth.data() + static_cast<usize>(y) * static_cast<usize>(fb.stride) : nullptr;
            const __m256 vnum = _mm256_set1_ps(num);
            for (int x = V.clip.x0 & ~7; x <= V.clip.x1; x += 8) {
                int bits = lane_bits(x, V.clip.x0, V.clip.x1);
                const __m256 px = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x) - cx), lc);
                __m256 Dx, Dy, Dz, Ox, Oy;
                if (V.ortho) {
                    Dx = _mm256_set1_ps(F.x); Dy = _mm256_set1_ps(F.y); Dz = _mm256_set1_ps(F.z);
                    Ox = _mm256_fmadd_ps(px, _mm256_set1_ps(R.x / f), _mm256_set1_ps(E.x + U.x * py / f));
                    Oy = _mm256_fmadd_ps(px, _mm256_set1_ps(R.y / f), _mm256_set1_ps(E.y + U.y * py / f));
                } else {
                    Dx = _mm256_fmadd_ps(px, _mm256_set1_ps(R.x), _mm256_set1_ps(F.x * f + U.x * py));
                    Dy = _mm256_fmadd_ps(px, _mm256_set1_ps(R.y), _mm256_set1_ps(F.y * f + U.y * py));
                    Dz = _mm256_fmadd_ps(px, _mm256_set1_ps(R.z), _mm256_set1_ps(F.z * f + U.z * py));
                    Ox = _mm256_set1_ps(E.x); Oy = _mm256_set1_ps(E.y);
                }
                // (medido: sustituir estas divisiones por vrcpps(+Newton) no acelera nada → se deja exacto)
                const __m256 iDz = _mm256_div_ps(one, Dz);
                const __m256 t = _mm256_mul_ps(vnum, iDz);
                const __m256 depth = V.ortho ? t : _mm256_mul_ps(t, _mm256_set1_ps(f));
                // válido: delante de la cámara, más allá de znear, finito
                __m256 ok = _mm256_and_ps(_mm256_cmp_ps(t, zero, _CMP_GT_OQ), _mm256_cmp_ps(depth, _mm256_set1_ps(V.ortho ? -1e30f : V.znear), _CMP_GE_OQ));
                ok = _mm256_and_ps(ok, _mm256_cmp_ps(depth, _mm256_set1_ps(3.0e37f), _CMP_LT_OQ));
                __m256 zb = _mm256_set1_ps(rz::k_inf);
                if (zrow) { zb = _mm256_load_ps(zrow + x); ok = _mm256_and_ps(ok, _mm256_cmp_ps(depth, zb, _CMP_LT_OQ)); }
                bits &= _mm256_movemask_ps(ok);
                if (!bits) continue;
                const __m256 gx = _mm256_fmadd_ps(t, Dx, Ox), gy = _mm256_fmadd_ps(t, Dy, Oy);
                // Derivadas del punto del suelo respecto a (sx, sy):
                //   dp/dsx = t'·(R - D·R.z/D.z)   dp/dsy = -t'·(U - D·U.z/D.z)   (t' = t en persp, 1/s en ortho)
                const __m256 ts = V.ortho ? _mm256_set1_ps(1.0f / f) : t;
                const __m256 kR = _mm256_mul_ps(_mm256_set1_ps(R.z), iDz), kU = _mm256_mul_ps(_mm256_set1_ps(U.z), iDz);
                const __m256 dxdx = _mm256_mul_ps(ts, _mm256_fnmadd_ps(Dx, kR, _mm256_set1_ps(R.x)));
                const __m256 dydx = _mm256_mul_ps(ts, _mm256_fnmadd_ps(Dy, kR, _mm256_set1_ps(R.y)));
                const __m256 dxdy = _mm256_mul_ps(ts, _mm256_fnmadd_ps(Dx, kU, _mm256_set1_ps(U.x)));
                const __m256 dydy = _mm256_mul_ps(ts, _mm256_fnmadd_ps(Dy, kU, _mm256_set1_ps(U.y)));
                const __m256 fwx = _mm256_add_ps(_mm256_add_ps(vabs(dxdx), vabs(dxdy)), _mm256_set1_ps(1e-6f));   // mundo/píxel
                const __m256 fwy = _mm256_add_ps(_mm256_add_ps(vabs(dydx), vabs(dydy)), _mm256_set1_ps(1e-6f));
                const __m256 ifwx = _mm256_div_ps(one, fwx), ifwy = _mm256_div_ps(one, fwy);   // píxeles/mundo
                // Cobertura del borde de la extensión (AA analítico) y niebla
                __m256 a = _mm256_set1_ps(base_a);
                if (has_ext) {
                    const __m256 ex = _mm256_min_ps(_mm256_sub_ps(gx, _mm256_set1_ps(extent.lo.x)), _mm256_sub_ps(_mm256_set1_ps(extent.hi.x), gx));
                    const __m256 ey = _mm256_min_ps(_mm256_sub_ps(gy, _mm256_set1_ps(extent.lo.y)), _mm256_sub_ps(_mm256_set1_ps(extent.hi.y), gy));
                    a = _mm256_mul_ps(a, _mm256_mul_ps(sat(_mm256_fmadd_ps(ex, ifwx, half)), sat(_mm256_fmadd_ps(ey, ifwy, half))));
                }
                a = _mm256_mul_ps(a, sat(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(fog1), depth), _mm256_set1_ps(1.0f / (fog1 - fog0)))));
                bits &= _mm256_movemask_ps(_mm256_cmp_ps(a, zero, _CMP_GT_OQ));
                if (!bits) continue;
                // Color base con damero sutil
                __m256 shade = one;
                __m256 la = zero;
                if (grid) {
                    const __m256 u = _mm256_mul_ps(_mm256_sub_ps(gx, _mm256_set1_ps(offx)), _mm256_set1_ps(inv_sp));
                    const __m256 v = _mm256_mul_ps(gy, _mm256_set1_ps(inv_sp));
                    const __m256 fu = _mm256_mul_ps(fwx, _mm256_set1_ps(inv_sp)), fv = _mm256_mul_ps(fwy, _mm256_set1_ps(inv_sp));   // celdas/píxel
                    const __m256 fmax = _mm256_max_ps(fu, fv);
                    const __m256i par = _mm256_and_si256(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_floor_ps(u)), _mm256_cvtps_epi32(_mm256_floor_ps(v))), _mm256_set1_epi32(1));
                    const __m256 chk = _mm256_sub_ps(_mm256_cvtepi32_ps(par), half);   // ±0.5
                    const __m256 cfade = sat(_mm256_fnmadd_ps(fmax, _mm256_set1_ps(2.5f), one));
                    shade = _mm256_fmadd_ps(_mm256_mul_ps(chk, cfade), _mm256_set1_ps(0.07f), one);
                    // líneas finas (1 px) en cada celda y mayores (1.6 px) cada 5, desvanecidas cuando la celda < ~4 px
                    const __m256 ifu = _mm256_mul_ps(ifwx, _mm256_set1_ps(spacing)), ifv = _mm256_mul_ps(ifwy, _mm256_set1_ps(spacing));   // píxeles/celda
                    const __m256 lu = sat(_mm256_fnmadd_ps(dist_int(u), ifu, one));
                    const __m256 lv = sat(_mm256_fnmadd_ps(dist_int(v), ifv, one));
                    const __m256 fade = sat(_mm256_fnmadd_ps(fmax, _mm256_set1_ps(3.0f), _mm256_set1_ps(1.4f)));
                    const __m256 k5 = _mm256_set1_ps(0.2f);
                    const __m256 k5i = _mm256_set1_ps(5.0f);
                    const __m256 mu = sat(_mm256_fnmadd_ps(dist_int(_mm256_mul_ps(u, k5)), _mm256_mul_ps(ifu, k5i), _mm256_set1_ps(1.3f)));
                    const __m256 mv = sat(_mm256_fnmadd_ps(dist_int(_mm256_mul_ps(v, k5)), _mm256_mul_ps(ifv, k5i), _mm256_set1_ps(1.3f)));
                    const __m256 mfade = sat(_mm256_fnmadd_ps(fmax, _mm256_set1_ps(0.6f), _mm256_set1_ps(1.4f)));
                    la = _mm256_max_ps(_mm256_mul_ps(_mm256_max_ps(lu, lv), _mm256_mul_ps(fade, _mm256_set1_ps(0.55f))),
                                       _mm256_mul_ps(_mm256_max_ps(mu, mv), mfade));
                    la = _mm256_mul_ps(la, _mm256_set1_ps(line_a));
                }
                __m256i col = pack_rgb(clamp255(_mm256_mul_ps(_mm256_set1_ps(br), shade)), clamp255(_mm256_mul_ps(_mm256_set1_ps(bg), shade)),
                                       clamp255(_mm256_mul_ps(_mm256_set1_ps(bb), shade)));
                if (has_tex) {
                    const __m256 tu = _mm256_fmsub_ps(_mm256_sub_ps(gx, _mm256_set1_ps(extent.lo.x)), _mm256_set1_ps(tsx), half);
                    const __m256 tv = _mm256_fmsub_ps(_mm256_sub_ps(gy, _mm256_set1_ps(extent.lo.y)), _mm256_set1_ps(tsy), half);
                    const __m256i texel = bilinear8(tex, tw, th, tu, tv);
                    const __m256 ta = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(texel, 24)), _mm256_set1_ps(1.0f / 255.0f));
                    col = blend_swar(col, texel, alpha16(ta));
                    la = _mm256_mul_ps(la, _mm256_fnmadd_ps(ta, _mm256_set1_ps(0.6f), one));   // líneas más tenues sobre la textura
                }
                if (grid) col = blend_swar(col, vline, alpha16(la));
                const __m256i dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(crow + x));
                const __m256i out = _mm256_or_si256(blend_swar(dst, col, alpha16(a)), _mm256_set1_epi32(static_cast<int>(0xFF000000u)));
                const __m256i m = mask_si(bits);
                _mm256_store_si256(reinterpret_cast<__m256i*>(crow + x), _mm256_blendv_epi8(dst, out, m));
                if (zrow) {
                    const __m256 wm = _mm256_and_ps(_mm256_castsi256_ps(m), _mm256_cmp_ps(a, half, _CMP_GT_OQ));
                    _mm256_store_ps(zrow + x, _mm256_blendv_ps(zb, depth, wm));
                }
            }
        }
    });
}

} // namespace cfd::render
