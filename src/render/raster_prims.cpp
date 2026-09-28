// ============================================================================
//  render/raster_prims.cpp — líneas gruesas antialias, polilíneas, puntos
//  (splats suaves), caja en alambre y flechas 3D.
//
//  Líneas y puntos comparten el esquema de la malla:
//   1. pasada paralela por slots: transformar, recortar (plano cercano en 3D,
//      Liang–Barsky en 2D contra la tijera ampliada), guardar el registro en
//      pantalla y contar tiles (las líneas largas sólo cuentan los tiles que su
//      "cápsula" toca de verdad, no toda su caja);
//   2. prefijos + segunda pasada que reparte ids (orden de envío estable);
//   3. rasterizado paralelo por tile (un tile = un hilo → sin carreras):
//      cobertura analítica por distancia al segmento / al centro del splat,
//      8 píxeles por iteración, mezcla SWAR y suma saturada (vpaddusb).
// ============================================================================
#include "raster_internal.hpp"

namespace cfd::render {
namespace rz {

// ---------------------------------------------------------------------------------------------
// Segmentos
// ---------------------------------------------------------------------------------------------
struct SegOut {        // 32 B, ya en píxeles y recortado
    float x0, y0, x1, y1;
    float q0, q1;      // perspectiva: 1/Z ; ortográfica: Z
    u32 c0, c1;
};
struct SegScratch {
    Buffer<SegOut> segs;
    Buffer<u32> rects;
    Buffer<u32> first;     // polilíneas: índice del primer punto de cada segmento
    Buffer<u32> poly_off;  // polilíneas: segmentos acumulados por polilínea
    Bins bins;
};
static SegScratch& seg_scratch() { static SegScratch s; return s; }

CFD_INLINE u32 lerp_argb(u32 a, u32 b, float t) {
    const u32 w = static_cast<u32>(clamp_(t, 0.0f, 1.0f) * 256.0f + 0.5f);
    const u32 iw = 256 - w;
    const u32 rb = (((a & 0x00FF00FFu) * iw + (b & 0x00FF00FFu) * w) >> 8) & 0x00FF00FFu;
    const u32 ag = ((((a >> 8) & 0x00FF00FFu) * iw + ((b >> 8) & 0x00FF00FFu) * w)) & 0xFF00FF00u;
    return rb | ag;
}

struct SegParams {
    float hw;          // semiancho (px)
    float R;           // radio de cobertura = hw + 0.5
    float amul;        // multiplicador de alfa (líneas más finas que 1 px)
    bool depth_test;
};
CFD_INLINE SegParams seg_params(float width, bool depth_test) {
    SegParams p;
    float w = std::isfinite(width) ? width : 1.0f;
    p.amul = 1.0f;
    if (w < 1.0f) { p.amul = max_(w, 0.0f); w = 1.0f; }   // más fina que un píxel: se atenúa el alfa
    w = min_(w, 256.0f);
    p.hw = 0.5f * w;
    p.R = p.hw + 0.5f;
    p.depth_test = depth_test;
    return p;
}

// Transforma, recorta y proyecta un segmento. false si no queda nada visible.
CFD_INLINE bool make_seg(const View& V, const SegParams& P, Vec3 a, Vec3 b, u32 ca, u32 cb, SegOut& o) {
    Vec3 va = to_view(V, a), vb = to_view(V, b);
    if (!finite3(va) || !finite3(vb)) return false;
    if (!V.ortho) {
        const float zn = V.znear;
        if (va.z < zn && vb.z < zn) return false;
        if (va.z < zn || vb.z < zn) {   // recorte en 3D (desde el extremo interior)
            const bool ain = va.z >= zn;
            const Vec3 I = ain ? va : vb, O = ain ? vb : va;
            const float t = (I.z - zn) / (I.z - O.z);
            const Vec3 c = I + (O - I) * t;
            const u32 cI = ain ? ca : cb, cO = ain ? cb : ca;
            const u32 cc = lerp_argb(cI, cO, t);
            if (ain) { vb = c; cb = cc; } else { va = c; ca = cc; }
        }
    }
    float x0, y0, x1, y1, q0, q1;
    if (V.ortho) {
        x0 = V.cx + V.f * va.x; y0 = V.cy - V.f * va.y; q0 = va.z;
        x1 = V.cx + V.f * vb.x; y1 = V.cy - V.f * vb.y; q1 = vb.z;
    } else {
        q0 = 1.0f / va.z; q1 = 1.0f / vb.z;
        x0 = V.cx + V.f * va.x * q0; y0 = V.cy - V.f * va.y * q0;
        x1 = V.cx + V.f * vb.x * q1; y1 = V.cy - V.f * vb.y * q1;
    }
    // Liang–Barsky contra la tijera ampliada por el radio (en pantalla q = 1/Z es lineal).
    const float m = P.R + 2.0f;
    const float xmin = static_cast<float>(V.clip.x0) - m, xmax = static_cast<float>(V.clip.x1 + 1) + m;
    const float ymin = static_cast<float>(V.clip.y0) - m, ymax = static_cast<float>(V.clip.y1 + 1) + m;
    const float dx = x1 - x0, dy = y1 - y0;
    float t0 = 0.0f, t1 = 1.0f;
    const float pp[4] = {-dx, dx, -dy, dy};
    const float qq[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};
    for (int i = 0; i < 4; ++i) {
        if (pp[i] == 0.0f) { if (qq[i] < 0.0f) return false; continue; }
        const float r = qq[i] / pp[i];
        if (pp[i] < 0.0f) { if (r > t1) return false; if (r > t0) t0 = r; }
        else { if (r < t0) return false; if (r < t1) t1 = r; }
    }
    if (!(t0 <= t1)) return false;
    o.x0 = x0 + dx * t0; o.y0 = y0 + dy * t0; o.x1 = x0 + dx * t1; o.y1 = y0 + dy * t1;
    o.q0 = q0 + (q1 - q0) * t0; o.q1 = q0 + (q1 - q0) * t1;
    o.c0 = t0 > 0.0f ? lerp_argb(ca, cb, t0) : ca;
    o.c1 = t1 < 1.0f ? lerp_argb(ca, cb, t1) : cb;
    return std::isfinite(o.x0) && std::isfinite(o.y0) && std::isfinite(o.x1) && std::isfinite(o.y1);
}

// Caja de píxeles del segmento engordado (∩ tijera). false si vacía.
CFD_INLINE bool seg_box(const SegOut& s, float R, const Clip& c, int& x0, int& y0, int& x1, int& y1) {
    x0 = max_(static_cast<int>(std::floor(min_(s.x0, s.x1) - R)), c.x0);
    y0 = max_(static_cast<int>(std::floor(min_(s.y0, s.y1) - R)), c.y0);
    x1 = min_(static_cast<int>(std::floor(max_(s.x0, s.x1) + R)), c.x1);
    y1 = min_(static_cast<int>(std::floor(max_(s.y0, s.y1) + R)), c.y1);
    return x0 <= x1 && y0 <= y1;
}
// ¿Toca la cápsula del segmento el tile (tx,ty) relativo a la rejilla? Distancia centro→segmento.
// noipa (ni inlining ni clonado IPA): el conteo (pasada 1) y el reparto (pasada 2) DEBEN tomar la
// misma decisión bit a bit. Inlineada en dos sitios, con -ffp-contract=fast / LTO / PGO cada copia
// podría fusionar FMAs (o vectorizarse) de forma distinta → conteo ≠ reparto → listas corruptas
// (ids basura leídos en el rasterizado). Con una única copia compilada es imposible. Los rects de
// un solo tile ni la llaman (Bins::count_if / emit_if), así que el coste es despreciable.
__attribute__((noipa)) static bool seg_touches_tile(const Bins& B, const SegOut& s, float R, int tx, int ty) {
    const float cx = static_cast<float>(((tx + B.gx0) << k_tile_shift) + k_tile / 2);
    const float cy = static_cast<float>(((ty + B.gy0) << k_tile_shift) + k_tile / 2);
    const float dx = s.x1 - s.x0, dy = s.y1 - s.y0;
    const float l2 = dx * dx + dy * dy;
    float t = l2 > 1e-12f ? ((cx - s.x0) * dx + (cy - s.y0) * dy) / l2 : 0.0f;
    t = clamp_(t, 0.0f, 1.0f);
    const float ex = cx - (s.x0 + t * dx), ey = cy - (s.y0 + t * dy);
    const float lim = R + 0.7072f * k_tile + 1.0f;
    return ex * ex + ey * ey <= lim * lim;
}


template <bool Persp, bool DepthTest>
static void raster_seg_tile(Framebuffer& fb, const SegScratch& S, const SegParams& P, int tile) {
    const Bins& B = S.bins;
    int tx0, ty0, tx1, ty1;
    B.tile_rect(tile, tx0, ty0, tx1, ty1);
    const u32* it = B.items.data() + B.start.data()[tile];
    const u32 nit = B.start.data()[tile + 1] - B.start.data()[tile];
    const usize stride = static_cast<usize>(fb.stride);
    const __m256 lc = lane_c();
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    const __m256 vR = _mm256_set1_ps(P.R);
    const __m256 vam = _mm256_set1_ps(P.amul * (1.0f / 255.0f));
    for (u32 e = 0; e < nit; ++e) {
        const SegOut& s = S.segs.data()[it[e]];
        RZ_COUNT(seg_refs, 1);
        const float dx = s.x1 - s.x0, dy = s.y1 - s.y0;
        const float l2 = dx * dx + dy * dy;
        const float il2 = l2 > 1e-12f ? 1.0f / l2 : 0.0f;
        const float len = std::sqrt(l2);
        // normal unitaria (para el intervalo de x de la banda en cada fila)
        const float nx = len > 1e-6f ? -dy / len : 0.0f, ny = len > 1e-6f ? dx / len : 1.0f;
        const bool steep = std::fabs(nx) > 1e-6f;
        const float inx = steep ? 1.0f / nx : 0.0f;   // sin divisiones por fila
        const float bx0 = min_(s.x0, s.x1) - P.R, bx1 = max_(s.x0, s.x1) + P.R;
        const int ry0 = max_(static_cast<int>(std::floor(min_(s.y0, s.y1) - P.R)), ty0);
        const int ry1 = min_(static_cast<int>(std::floor(max_(s.y0, s.y1) + P.R)), ty1);
        const __m256 vdx = _mm256_set1_ps(dx), vdy = _mm256_set1_ps(dy), vil2 = _mm256_set1_ps(il2);
        const __m256 vq0 = _mm256_set1_ps(s.q0), vdq = _mm256_set1_ps(s.q1 - s.q0);
        const __m256i c0 = _mm256_set1_epi32(static_cast<int>(s.c0)), c1 = _mm256_set1_epi32(static_cast<int>(s.c1));
        for (int y = ry0; y <= ry1; ++y) {
            const float yc = static_cast<float>(y) + 0.5f;
            const float uy = yc - s.y0;
            // intervalo de x de la banda |n·(p - a)| ≤ R en esta fila, ∩ caja de la cápsula
            float xa = bx0, xb = bx1;
            if (steep) {
                const float k0 = (-P.R - ny * uy) * inx, k1 = (P.R - ny * uy) * inx;
                xa = max_(xa, s.x0 + min_(k0, k1)); xb = min_(xb, s.x0 + max_(k0, k1));
            } else if (std::fabs(uy * ny) > P.R) continue;
            int xs = max_(static_cast<int>(std::floor(xa)), tx0), xe = min_(static_cast<int>(std::floor(xb)), tx1);
            if (xs > xe) continue;
            RZ_COUNT(seg_rows, 1);
            const __m256 vuy = _mm256_set1_ps(uy);
            // sin plano de profundidad (DepthTest = false) no se toca depth.data() (nullptr + desplazamiento = UB)
            float* zrow = DepthTest ? fb.depth.data() + static_cast<usize>(y) * stride : nullptr;
            u32* crow = fb.color.data() + static_cast<usize>(y) * stride;
            for (int x = xs & ~7; x <= xe; x += 8) {
                const int lb = lane_bits(x, xs, xe);
                const __m256 ux = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x) - s.x0), lc);
                __m256 t = _mm256_mul_ps(_mm256_fmadd_ps(ux, vdx, _mm256_mul_ps(vuy, vdy)), vil2);
                t = _mm256_min_ps(_mm256_max_ps(t, zero), one);
                const __m256 ex = _mm256_fnmadd_ps(t, vdx, ux), ey = _mm256_fnmadd_ps(t, vdy, vuy);
                const __m256 d2 = _mm256_fmadd_ps(ex, ex, _mm256_mul_ps(ey, ey));
                // (medido: d²·rsqrt(d²) no es más rápido que vsqrtps aquí → se deja la raíz exacta)
                const __m256 d = _mm256_sqrt_ps(d2);
                const __m256 cov = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(vR, d), zero), one);
                int bits = lb & _mm256_movemask_ps(_mm256_cmp_ps(cov, zero, _CMP_GT_OQ));
                RZ_COUNT(seg_blocks, 1);
                if (!bits) continue;
                RZ_COUNT(seg_cov, 1);
                if constexpr (DepthTest) {
                    const __m256 q = _mm256_fmadd_ps(t, vdq, vq0);
                    const __m256 z = Persp ? simd::rcp_nr(q).v : q;
                    const __m256 zb = _mm256_load_ps(zrow + x);
                    // sesgo: 0.2 % relativo + 0.05 celdas (líneas sobre superficies)
                    const __m256 zt = _mm256_fmsub_ps(z, _mm256_set1_ps(0.998f), _mm256_set1_ps(0.05f));
                    bits &= _mm256_movemask_ps(_mm256_cmp_ps(zt, zb, _CMP_LT_OQ));
                    if (!bits) continue;
                }
                RZ_COUNT(seg_pass, 1);
                const __m256i col = blend_swar(c0, c1, alpha16(t));   // color interpolado (incl. alfa)
                const __m256 a = _mm256_mul_ps(_mm256_mul_ps(cov, vam), _mm256_cvtepi32_ps(_mm256_srli_epi32(col, 24)));
                const __m256i dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(crow + x));
                const __m256i out = blend_swar(dst, col, alpha16(a));
                _mm256_store_si256(reinterpret_cast<__m256i*>(crow + x), _mm256_blendv_epi8(dst, _mm256_or_si256(out, _mm256_set1_epi32(static_cast<int>(0xFF000000u))), mask_si(bits)));
            }
        }
    }
}

// Tubería común de segmentos. Src::get(k, a, b, ca, cb) da el segmento k.
template <class Src>
static void draw_segments(Framebuffer& fb, const Camera& cam, const Src& src, i64 nseg, float width, bool depth_test) {
    if (nseg <= 0 || fb.color.empty()) return;
    SegScratch& S = seg_scratch();
    const View V = make_view(cam, fb);
    if (V.clip.empty()) return;
    const SegParams P = seg_params(width, depth_test && !fb.depth.empty());
    if (!(P.amul > 0.0f)) return;
    const int nslots = bin_slots(nseg, 1024);
    Bins& B = S.bins;
    if (!B.setup(V.clip, nseg, nslots)) return;
    SegOut* segs = grow(S.segs, static_cast<usize>(nseg));
    u32* rects = grow(S.rects, static_cast<usize>(nseg));
    const float R = P.R;
    pool().run_slots([&](int s, int ns) {
        const i64 lo = nseg * s / ns, hi = nseg * (s + 1) / ns;
        u32* cnt = B.slot_counts(s);
        for (i64 k = lo; k < hi; ++k) {
            Vec3 a, b; u32 ca, cb;
            src.get(k, a, b, ca, cb);
            u32 r = k_no_rect;
            int x0, y0, x1, y1;
            if (((ca | cb) >> 24) && make_seg(V, P, a, b, ca, cb, segs[k]) && seg_box(segs[k], R, B.clip, x0, y0, x1, y1)) {
                r = B.pack_rect(x0, y0, x1, y1);
                const SegOut& so = segs[k];
                B.count_if(cnt, r, [&](int tx, int ty) { return seg_touches_tile(B, so, R, tx, ty); });
            }
            rects[k] = r;
        }
    }, nslots);
    B.finalize();
    pool().run_slots([&](int s, int ns) {
        const i64 lo = nseg * s / ns, hi = nseg * (s + 1) / ns;
        u32* cur = B.slot_counts(s);
        for (i64 k = lo; k < hi; ++k)
            if (rects[k] != k_no_rect) {
                const SegOut& so = segs[k];
                B.emit_if(cur, rects[k], static_cast<u32>(k), [&](int tx, int ty) { return seg_touches_tile(B, so, R, tx, ty); });
            }
    }, nslots);
    using Fn = void (*)(Framebuffer&, const SegScratch&, const SegParams&, int);
    Fn fn;
    if (V.ortho) fn = P.depth_test ? &raster_seg_tile<false, true> : &raster_seg_tile<false, false>;
    else fn = P.depth_test ? &raster_seg_tile<true, true> : &raster_seg_tile<true, false>;
    const u32* ord = B.order.data();
    parallel_for(0, B.busy, 1, [&](i64 lo, i64 hi) {
        for (i64 k = lo; k < hi; ++k) fn(fb, S, P, static_cast<int>(ord[k]));
    });
}

struct LineSrc {
    const Vec3* p; const u32* c; usize nc; bool per_vertex;
    CFD_INLINE void get(i64 k, Vec3& a, Vec3& b, u32& ca, u32& cb) const {
        a = p[2 * k]; b = p[2 * k + 1];
        if (per_vertex) {
            const usize i = static_cast<usize>(2 * k);
            ca = i < nc ? c[i] : (nc ? c[nc - 1] : 0xFFFFFFFFu);
            cb = i + 1 < nc ? c[i + 1] : ca;
        } else {
            const usize i = static_cast<usize>(k);
            ca = cb = i < nc ? c[i] : (nc ? c[nc - 1] : 0xFFFFFFFFu);
        }
    }
};
struct PolySrc {
    const Vec3* p; const u32* c; usize nc; const u32* first;
    CFD_INLINE void get(i64 k, Vec3& a, Vec3& b, u32& ca, u32& cb) const {
        const u32 j = first[k];
        a = p[j]; b = p[j + 1];
        ca = j < nc ? c[j] : (nc ? c[nc - 1] : 0xFFFFFFFFu);
        cb = j + 1 < nc ? c[j + 1] : ca;
    }
};

// ---------------------------------------------------------------------------------------------
// Puntos
// ---------------------------------------------------------------------------------------------
struct PtOut { float x, y, z; u32 c; };
struct PtScratch {
    Buffer<PtOut> pts;
    Buffer<u32> rects;
    Bins bins;
};
static PtScratch& pt_scratch() { static PtScratch s; return s; }

template <bool Additive>
static void raster_pt_tile(Framebuffer& fb, const PtScratch& S, float R, float amul, int tile) {
    const Bins& B = S.bins;
    int tx0, ty0, tx1, ty1;
    B.tile_rect(tile, tx0, ty0, tx1, ty1);
    const u32* it = B.items.data() + B.start.data()[tile];
    const u32 nit = B.start.data()[tile + 1] - B.start.data()[tile];
    const Surf SF = Surf::of(fb);
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    const __m256 viR2 = _mm256_set1_ps(1.0f / (R * R));
    const bool has_depth = !fb.depth.empty();
    for (u32 e = 0; e < nit; ++e) {
        const PtOut& p = S.pts.data()[it[e]];
        // caja de píxeles del splat ∩ tile
        const int x0 = max_(static_cast<int>(std::floor(p.x - R)), tx0), x1 = min_(static_cast<int>(std::floor(p.x + R)), tx1);
        const int y0 = max_(static_cast<int>(std::floor(p.y - R)), ty0), y1 = min_(static_cast<int>(std::floor(p.y + R)), ty1);
        if (x0 > x1 || y0 > y1) continue;
        const __m256 va = _mm256_set1_ps(amul * static_cast<float>(p.c >> 24) * (1.0f / 255.0f));
        const __m256i col = _mm256_set1_epi32(static_cast<int>(p.c | 0xFF000000u));
        const __m256 vz = _mm256_set1_ps(p.z * 0.998f - 0.05f);   // sesgo relativo (partículas sobre superficies)
        const __m256 vpx = _mm256_set1_ps(p.x), vpy = _mm256_set1_ps(p.y);
        auto splat = [&](const auto& blk, int bits) RZ_LAMBDA_INLINE {
            if (has_depth) bits &= _mm256_movemask_ps(_mm256_cmp_ps(vz, blk.load_z(), _CMP_LT_OQ));
            if (!bits) return;
            const __m256 dx = _mm256_sub_ps(blk.px(), vpx), dy = _mm256_sub_ps(blk.py(), vpy);
            // núcleo suave (1 - r²/R²)²: sin sqrt ni exp
            __m256 k = _mm256_max_ps(_mm256_fnmadd_ps(_mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)), viR2, one), zero);
            bits &= _mm256_movemask_ps(_mm256_cmp_ps(k, zero, _CMP_GT_OQ));
            if (!bits) return;
            k = _mm256_mul_ps(_mm256_mul_ps(k, k), va);
            const __m256i a16 = alpha16(k);
            const __m256i dst = blk.load_c();
            __m256i out;
            if constexpr (Additive) out = _mm256_adds_epu8(dst, scale_swar(col, a16));   // suma saturada por canal
            else out = blend_swar(dst, col, a16);
            blk.store_c(_mm256_blendv_epi8(dst, out, mask_si(bits)));
        };
        const int w = x1 - x0 + 1, h = y1 - y0 + 1;
        if (w <= 8 && h <= 8) {   // sellos 4×2 dentro del tile (sin solapes ni salirse): splats pequeños
            const int nsx = (w + 3) >> 2, nsy = (h + 1) >> 1;
            const int sx0 = min_(x0, tx1 - 4 * nsx + 1), sy0 = min_(y0, ty1 - 2 * nsy + 1);
            if (sx0 >= tx0 && sy0 >= ty0 && !RZ_EXP(7)) {
                for (int j = 0; j < nsy; ++j)
                    for (int i = 0; i < nsx; ++i) splat(StampBlk(SF, sx0 + 4 * i, sy0 + 2 * j), 0xFF);
                continue;
            }
        }
        for (int y = y0; y <= y1; ++y)
            for (int x = x0 & ~7; x <= x1; x += 8) splat(RowBlk(SF, x, y), lane_bits(x, x0, x1));
    }
}

// Arista de cono (flecha): malla persistente reutilizada.
static Mesh& arrow_mesh() { static Mesh m; return m; }

} // namespace rz

// =============================================================================================
void draw_lines(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                bool per_vertex_color, float width, bool depth_test) {
    const i64 nseg = static_cast<i64>(pts.size() / 2);
    if (nseg <= 0) return;
    rz::LineSrc src{pts.data(), colors.data(), colors.size(), per_vertex_color};
    rz::draw_segments(fb, cam, src, nseg, width, depth_test);
}

void draw_polylines(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                    std::span<const u32> starts, std::span<const u32> counts, float width, bool depth_test) {
    using namespace rz;
    const usize npl = min_(starts.size(), counts.size());
    if (npl == 0 || pts.size() < 2) return;
    SegScratch& S = seg_scratch();
    u32* off = grow(S.poly_off, npl + 1);
    // Prefijo de segmentos por polilínea (validando rangos: las que se salen se ignoran).
    u64 total = 0;
    for (usize i = 0; i < npl; ++i) {
        off[i] = static_cast<u32>(total);
        const u64 s0 = starts[i], c = counts[i];
        if (c >= 2 && s0 + c <= pts.size()) total += c - 1;
        if (total >= 0xFFFFFFF0ull) { total = off[i]; break; }
    }
    off[npl] = static_cast<u32>(total);
    if (total == 0) return;
    u32* first = grow(S.first, static_cast<usize>(total));
    parallel_for(0, static_cast<i64>(npl), 64, [&](i64 lo, i64 hi) {
        for (i64 i = lo; i < hi; ++i) {
            const u32 n = off[i + 1] - off[i];
            const u32 s0 = starts[static_cast<usize>(i)];
            for (u32 k = 0; k < n; ++k) first[off[i] + k] = s0 + k;
        }
    });
    PolySrc src{pts.data(), colors.data(), colors.size(), first};
    draw_segments(fb, cam, src, static_cast<i64>(total), width, depth_test);
}

void draw_points(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                 float size, bool additive) {
    using namespace rz;
    const i64 n = static_cast<i64>(pts.size());
    if (n <= 0 || fb.color.empty()) return;
    const View V = make_view(cam, fb);
    if (V.clip.empty()) return;
    float sz = std::isfinite(size) ? size : 1.0f;
    float amul = 1.0f;
    if (sz < 1.0f) { amul = max_(sz, 0.0f); sz = 1.0f; }
    if (!(amul > 0.0f)) return;
    sz = min_(sz, 256.0f);
    const float R = 0.5f * sz + 0.5f;
    PtScratch& S = pt_scratch();
    const int nslots = bin_slots(n, 4096);
    Bins& B = S.bins;
    if (!B.setup(V.clip, n, nslots)) return;
    PtOut* po = grow(S.pts, static_cast<usize>(n));
    u32* rects = grow(S.rects, static_cast<usize>(n));
    const usize nc = colors.size();
    const u32* cs = colors.data();
    const Vec3* ps = pts.data();
    const bool per_point = nc >= static_cast<usize>(n);
    const u32 c_uni = nc ? cs[0] : 0xFFFFFFFFu;
    const float xmin = static_cast<float>(V.clip.x0) - R, xmax = static_cast<float>(V.clip.x1 + 1) + R;
    const float ymin = static_cast<float>(V.clip.y0) - R, ymax = static_cast<float>(V.clip.y1 + 1) + R;
    pool().run_slots([&](int s, int ns) {
        const i64 lo = n * s / ns, hi = n * (s + 1) / ns;
        u32* cnt = B.slot_counts(s);
        for (i64 k = lo; k < hi; ++k) {
            u32 r = k_no_rect;
            const Vec3 c = to_view(V, ps[k]);
            const u32 col = per_point ? cs[k] : c_uni;
            float sx, sy;
            bool ok = (col >> 24) != 0 && (V.ortho || c.z >= V.znear);
            if (ok) {
                if (V.ortho) { sx = V.cx + V.f * c.x; sy = V.cy - V.f * c.y; }
                else { const float iz = 1.0f / c.z; sx = V.cx + V.f * c.x * iz; sy = V.cy - V.f * c.y * iz; }
                ok = sx > xmin && sx < xmax && sy > ymin && sy < ymax;   // falso también con NaN
                if (ok) {
                    po[k] = PtOut{sx, sy, c.z, col};
                    const int x0 = max_(static_cast<int>(std::floor(sx - R)), B.clip.x0), x1 = min_(static_cast<int>(std::floor(sx + R)), B.clip.x1);
                    const int y0 = max_(static_cast<int>(std::floor(sy - R)), B.clip.y0), y1 = min_(static_cast<int>(std::floor(sy + R)), B.clip.y1);
                    if (x0 <= x1 && y0 <= y1) { r = B.pack_rect(x0, y0, x1, y1); B.count(cnt, r); }
                }
            }
            rects[k] = r;
        }
    }, nslots);
    B.finalize();
    pool().run_slots([&](int s, int ns) {
        const i64 lo = n * s / ns, hi = n * (s + 1) / ns;
        u32* cur = B.slot_counts(s);
        for (i64 k = lo; k < hi; ++k)
            if (rects[k] != k_no_rect) B.emit(cur, rects[k], static_cast<u32>(k));
    }, nslots);
    const u32* ord = B.order.data();
    parallel_for(0, B.busy, 1, [&](i64 lo, i64 hi) {
        for (i64 k = lo; k < hi; ++k) {
            if (additive) raster_pt_tile<true>(fb, S, R, amul, static_cast<int>(ord[k]));
            else raster_pt_tile<false>(fb, S, R, amul, static_cast<int>(ord[k]));
        }
    });
}

void draw_box_wire(Framebuffer& fb, const Camera& cam, const Aabb& b, u32 color, float width) {
    if (b.empty()) return;
    Vec3 c[8];
    for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? b.hi.x : b.lo.x, (i & 2) ? b.hi.y : b.lo.y, (i & 4) ? b.hi.z : b.lo.z};
    static constexpr int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    Vec3 p[24];
    u32 cols[12];
    for (int e = 0; e < 12; ++e) { p[2 * e] = c[E[e][0]]; p[2 * e + 1] = c[E[e][1]]; cols[e] = color; }
    draw_lines(fb, cam, std::span<const Vec3>(p, 24), std::span<const u32>(cols, 12), false, width, true);
}

void draw_arrow(Framebuffer& fb, const Camera& cam, Vec3 from, Vec3 to, u32 color, float width) {
    const Vec3 d = to - from;
    const float len = length(d);
    if (!(len > 1e-6f) || !std::isfinite(len)) return;
    const Vec3 dir = d * (1.0f / len);
    // Radio de la cabeza: ≥ ~2.2 anchos de trazo en píxeles a la profundidad de la punta.
    float depth = dot(to - cam.eye, cam.fwd);
    const float px_world = cam.ortho ? 1.0f / max_(cam.ortho_scale, 1e-6f) : max_(depth, cam.znear) / max_(cam.focal, 1e-6f);
    float rh = max_(0.07f * len, 2.4f * max_(width, 1.0f) * px_world);
    rh = min_(rh, 0.3f * len);
    const float hl = min_(2.6f * rh, 0.45f * len);
    const Vec3 base = to - dir * hl;
    // Asta: línea gruesa hasta un poco dentro del cono.
    const Vec3 sp[2] = {from, base + dir * (0.3f * hl)};
    const u32 sc[1] = {color};
    draw_lines(fb, cam, std::span<const Vec3>(sp, 2), std::span<const u32>(sc, 1), false, width, true);
    // Cabeza: cono 3D sombreado (24 lados + tapa).
    Mesh& m = rz::arrow_mesh();
    m.clear();
    const Vec3 ref = std::fabs(dir.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    const Vec3 u = normalize(cross(dir, ref)), v = cross(dir, u);
    constexpr int K = 24;
    const float slope = rh / hl;   // normal del cono: (radial + dir·slope) normalizada
    for (int i = 0; i < K; ++i) {
        const float a0 = 2.0f * k_pi * static_cast<float>(i) / K, a1 = 2.0f * k_pi * (static_cast<float>(i) + 0.5f) / K;
        const Vec3 r0 = u * std::cos(a0) + v * std::sin(a0);
        const Vec3 r1 = u * std::cos(a1) + v * std::sin(a1);
        const u32 b = static_cast<u32>(m.pos.size());
        m.pos.push_back(base + r0 * rh); m.nrm.push_back(normalize(r0 + dir * slope));
        m.pos.push_back(to);             m.nrm.push_back(normalize(r1 + dir * slope));
        m.pos.push_back(base + r0 * rh); m.nrm.push_back(-dir);
        m.color.insert(m.color.end(), {color, color, color});
        (void)b;
    }
    const u32 cbase = static_cast<u32>(m.pos.size());
    m.pos.push_back(base); m.nrm.push_back(-dir); m.color.push_back(color);
    for (int i = 0; i < K; ++i) {
        const u32 a = static_cast<u32>(3 * i), n = static_cast<u32>(3 * ((i + 1) % K));
        m.tri.insert(m.tri.end(), {a, n, a + 1});                 // lateral (antihorario desde fuera)
        m.tri.insert(m.tri.end(), {cbase, n + 2, a + 2});         // tapa
    }
    m.group.assign(m.pos.size(), 0);
    m.recompute_bounds();
    MeshStyle st;
    st.two_sided = false;
    st.use_vertex_color = true;
    rz::draw_mesh_impl(fb, cam, m, Light{}, st, false);   // no pisa last_mesh_stats()
}

} // namespace cfd::render
