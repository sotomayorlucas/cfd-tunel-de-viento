// ============================================================================
//  render/raster_mesh.cpp — draw_mesh: rasterizador de mallas por tiles.
//
//  Tubería (todo en paralelo con el pool, sin atómicos en bucles internos):
//   1. Transformación de vértices AVX2, 8 a la vez: AoS→SoA con 6 cargas de 128 bits
//      + 5 shuffles, vista + proyección + outcodes, y transposición 8×8 → registro
//      VOut de 32 B por vértice (una sola carga YMM por vértice en el setup).
//   2. Binning por ordenación por conteo (raster_internal.hpp: Bins), pasada 1 en AVX2
//      (8 triángulos con gathers) → cada tile de 64×64 recibe dos listas contiguas
//      (caras frontales y traseras) en orden de envío. Los triángulos que cruzan el
//      plano cercano / la banda de guarda se recortan en espacio de vista.
//   3. Rasterizado paralelo por tile (cada tile lo posee un único hilo → sin carreras),
//      orden LPT, funciones de arista enteras 28.4 (regla top-left), sellos 4×2 para
//      triángulos diminutos, Hi-Z por bloques 8×8 para las caras traseras, test de
//      profundidad de 8 píxeles antes de interpolar, interpolación perspectiva-correcta
//      y sombreado DIFERIDO por tile (G-buffer FP16): Blinn-Phong por píxel con ambiente
//      hemisférico + relleno + especular + borde, una vez por píxel visible.
// ============================================================================
#include "raster_internal.hpp"
#include <type_traits>

namespace cfd::render {
namespace rz {

struct alignas(32) VOut {
    i32 fx, fy;          // 28.4 absoluto del framebuffer
    float q;             // perspectiva: 1/Z ; ortográfica: Z
    u32 code;            // outcode
    float nx, ny, nz;    // normal de mundo
    u32 color;
};
static_assert(sizeof(VOut) == 32);
struct ClipTri { VOut v[3]; };
struct alignas(64) ClipList {
    std::vector<ClipTri> tris;
    std::vector<u32> rects;
};

// G-buffer por hilo para el sombreado diferido de un tile opaco: normales FP16 (F16C) sin
// normalizar + máscara de cobertura (1 bit por píxel). El color base va directo al framebuffer.
struct alignas(64) GBuf {
    u16 n[3][k_tile * k_tile];
    u64 rows[k_tile];
};
static_assert(k_tile == 64, "GBuf::rows usa una palabra de 64 bits por fila del tile");

struct MeshScratch {
    Buffer<VOut> vout;
    Buffer<GBuf> gbuf;         // uno por hilo del pool (sombreado diferido)
    Buffer<u32> rects;
    Bins bins;
    ClipList clip[128];
    Padded<u64> binned[128];
    RasterStats stats;
};
static MeshScratch& mesh_scratch() { static MeshScratch s; return s; }

// ---------------------------------------------------------------------------------------------
// 1) Transformación de vértices
// ---------------------------------------------------------------------------------------------
// AoS (x,y,z)×8 → x[8], y[8], z[8]: 6 cargas de 128 bits + 5 shuffles (sin gathers).
CFD_INLINE void load_xyz8(const float* p, __m256& x, __m256& y, __m256& z) {
    const __m256 m03 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(p + 0)), _mm_loadu_ps(p + 12), 1);
    const __m256 m14 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(p + 4)), _mm_loadu_ps(p + 16), 1);
    const __m256 m25 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(p + 8)), _mm_loadu_ps(p + 20), 1);
    const __m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2));
    const __m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1));
    x = _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0));
    y = _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
    z = _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1));
}
// Transpuesta 8×8 de floats (r[i] = atributo i de 8 vértices → r[v] = registro del vértice v).
CFD_INLINE void transpose8(__m256 r[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20); r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20); r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31); r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31); r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

static void vertex_scalar(const View& V, const Mesh& m, usize i, bool has_n, bool has_c, u32 base, Vec3 nd, VOut& o) {
    const Vec3 c = to_view(V, m.pos[i]);
    o.code = outcode(V, c);
    o.fx = o.fy = 0; o.q = 1.0f;
    if (!(o.code & OC_BAD)) {
        float sx, sy;
        if (V.ortho) { sx = V.cx + V.f * c.x; sy = V.cy - V.f * c.y; o.q = c.z; }
        else { const float iz = 1.0f / c.z; sx = V.cx + V.f * c.x * iz; sy = V.cy - V.f * c.y * iz; o.q = iz; }
        if (!o.code) {
            o.fx = static_cast<i32>(std::lrintf(sx * static_cast<float>(k_sub)));
            o.fy = static_cast<i32>(std::lrintf(sy * static_cast<float>(k_sub)));
        }
    }
    const Vec3 n = has_n ? m.nrm[i] : nd;
    o.nx = n.x; o.ny = n.y; o.nz = n.z;
    o.color = has_c ? m.color[i] : base;
}

template <bool Ortho>
static void transform_vertices(const View& V, const Mesh& m, bool use_vc, u32 base, VOut* CFD_RESTRICT out) {
    const usize n = m.pos.size();
    const bool has_n = m.nrm.size() >= n;
    const bool has_c = use_vc && m.color.size() >= n;
    const Vec3 nd = -V.fwd;
    const i64 nblk = RZ_EXP(10) ? 0 : static_cast<i64>(n / 8);   // (A/B: todo escalar)
    parallel_for(0, nblk, 512, [&](i64 lo, i64 hi) {
        const __m256 m00 = _mm256_set1_ps(V.m[0][0]), m01 = _mm256_set1_ps(V.m[0][1]), m02 = _mm256_set1_ps(V.m[0][2]), m03 = _mm256_set1_ps(V.m[0][3]);
        const __m256 m10 = _mm256_set1_ps(V.m[1][0]), m11 = _mm256_set1_ps(V.m[1][1]), m12 = _mm256_set1_ps(V.m[1][2]), m13 = _mm256_set1_ps(V.m[1][3]);
        const __m256 m20 = _mm256_set1_ps(V.m[2][0]), m21 = _mm256_set1_ps(V.m[2][1]), m22 = _mm256_set1_ps(V.m[2][2]), m23 = _mm256_set1_ps(V.m[2][3]);
        const __m256 vf = _mm256_set1_ps(V.f), vcx = _mm256_set1_ps(V.cx), vcy = _mm256_set1_ps(V.cy);
        const __m256 vg = _mm256_set1_ps(k_guard), vzn = _mm256_set1_ps(V.znear), vsub = _mm256_set1_ps(static_cast<float>(k_sub));
        const __m256 inf = _mm256_set1_ps(k_inf), sgn = _mm256_set1_ps(-0.0f);
        const Vec3* P = m.pos.data();
        const Vec3* N = m.nrm.data();
        const u32* C = m.color.data();
        for (i64 b = lo; b < hi; ++b) {
            const usize i = static_cast<usize>(b) * 8;
            __m256 px, py, pz;
            load_xyz8(&P[i].x, px, py, pz);
            const __m256 X = _mm256_fmadd_ps(m00, px, _mm256_fmadd_ps(m01, py, _mm256_fmadd_ps(m02, pz, m03)));
            const __m256 Y = _mm256_fmadd_ps(m10, px, _mm256_fmadd_ps(m11, py, _mm256_fmadd_ps(m12, pz, m13)));
            const __m256 Z = _mm256_fmadd_ps(m20, px, _mm256_fmadd_ps(m21, py, _mm256_fmadd_ps(m22, pz, m23)));
            // finito ⇔ |v| < inf (comparación ordenada: NaN → falso)
            const __m256 ok = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(_mm256_andnot_ps(sgn, X), inf, _CMP_LT_OQ),
                                                          _mm256_cmp_ps(_mm256_andnot_ps(sgn, Y), inf, _CMP_LT_OQ)),
                                            _mm256_cmp_ps(_mm256_andnot_ps(sgn, Z), inf, _CMP_LT_OQ));
            const __m256 fX = _mm256_mul_ps(vf, X), fY = _mm256_mul_ps(vf, Y);
            __m256 sx, sy, q, g;
            if constexpr (Ortho) {
                sx = _mm256_add_ps(vcx, fX); sy = _mm256_sub_ps(vcy, fY); q = Z; g = vg;
            } else {
                const __m256 iz = _mm256_div_ps(_mm256_set1_ps(1.0f), Z);
                sx = _mm256_fmadd_ps(fX, iz, vcx); sy = _mm256_fnmadd_ps(fY, iz, vcy); q = iz;
                g = _mm256_mul_ps(vg, Z);
            }
            const __m256 ng = _mm256_xor_ps(g, sgn);
            __m256i code = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(fX, ng, _CMP_LT_OQ)), _mm256_set1_epi32(OC_LEFT));
            code = _mm256_or_si256(code, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(fX, g, _CMP_GT_OQ)), _mm256_set1_epi32(OC_RIGHT)));
            code = _mm256_or_si256(code, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(fY, g, _CMP_GT_OQ)), _mm256_set1_epi32(OC_TOP)));
            code = _mm256_or_si256(code, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(fY, ng, _CMP_LT_OQ)), _mm256_set1_epi32(OC_BOTTOM)));
            if constexpr (!Ortho)
                code = _mm256_or_si256(code, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(Z, vzn, _CMP_LT_OQ)), _mm256_set1_epi32(OC_NEAR)));
            code = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(_mm256_set1_epi32(OC_BAD)), _mm256_castsi256_ps(code), ok));
            const __m256i fx = _mm256_cvtps_epi32(_mm256_mul_ps(sx, vsub));
            const __m256i fy = _mm256_cvtps_epi32(_mm256_mul_ps(sy, vsub));
            __m256 r[8];
            r[0] = _mm256_castsi256_ps(fx); r[1] = _mm256_castsi256_ps(fy); r[2] = q; r[3] = _mm256_castsi256_ps(code);
            if (has_n) load_xyz8(&N[i].x, r[4], r[5], r[6]);
            else { r[4] = _mm256_set1_ps(nd.x); r[5] = _mm256_set1_ps(nd.y); r[6] = _mm256_set1_ps(nd.z); }
            r[7] = has_c ? _mm256_loadu_ps(reinterpret_cast<const float*>(C + i)) : _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(base)));
            transpose8(r);
            float* o = reinterpret_cast<float*>(out + i);
            for (int k = 0; k < 8; ++k) _mm256_store_ps(o + 8 * k, r[k]);
        }
    });
    if (RZ_EXP(10)) {
        parallel_for(0, static_cast<i64>(n), 4096, [&](i64 lo, i64 hi) { for (i64 i = lo; i < hi; ++i) vertex_scalar(V, m, static_cast<usize>(i), has_n, has_c, base, nd, out[i]); });
        return;
    }
    for (usize i = static_cast<usize>(nblk) * 8; i < n; ++i) vertex_scalar(V, m, i, has_n, has_c, base, nd, out[i]);
}

// ---------------------------------------------------------------------------------------------
// 2) Binning
// ---------------------------------------------------------------------------------------------
struct MeshCtx {
    const View* V;
    const Mesh* mesh;
    const VOut* vo;
    const u32* idx;
    u32 nv;
    bool cull_back;
    bool has_n;          // la malla trae normales (si no, la normal por defecto mira a la cámara)
    Bins* bins;
    Vec3 nd;
};

// Rect de tiles de un triángulo en 28.4 (ya sin outcodes) o k_no_rect si se descarta.
// Área en pantalla (y hacia abajo): < 0 ⇔ antihorario visto desde la cámara ⇔ cara frontal.
CFD_INLINE u32 tri_rect(const MeshCtx& C, const VOut& a, const VOut& b, const VOut& c) {
    const i32 x[3] = {a.fx, b.fx, c.fx}, y[3] = {a.fy, b.fy, c.fy};
    const i64 area = static_cast<i64>(x[1] - x[0]) * (y[2] - y[0]) - static_cast<i64>(x[2] - x[0]) * (y[1] - y[0]);
    if (area == 0 || (C.cull_back && area > 0)) return k_no_rect;
    int px0, py0, px1, py1;
    if (!tri_pixel_box(x, y, C.bins->clip, px0, py0, px1, py1)) return k_no_rect;
    return C.bins->pack_rect(px0, py0, px1, py1, area > 0);
}

CFD_COLD CFD_NOINLINE static void clip_mesh_tri(const MeshCtx& C, const u32 id[3], ClipList& cl, u32* cnt) {
    const View& V = *C.V;
    const Mesh& m = *C.mesh;
    const bool has_n = m.nrm.size() >= m.pos.size();
    CVert poly[16], tmp[16];
    u32 planes = 0;
    for (int k = 0; k < 3; ++k) {
        const VOut& o = C.vo[id[k]];
        planes |= o.code;
        CVert& cv = poly[k];
        cv.c = to_view(V, m.pos[id[k]]);
        const Vec3 n = has_n ? m.nrm[id[k]] : C.nd;
        cv.a[0] = n.x; cv.a[1] = n.y; cv.a[2] = n.z;
        cv.a[3] = static_cast<float>((o.color >> 16) & 255);
        cv.a[4] = static_cast<float>((o.color >> 8) & 255);
        cv.a[5] = static_cast<float>(o.color & 255);
        cv.orig = o.code == 0 ? static_cast<int>(id[k]) : -1;
    }
    const int n = clip_polygon(V, poly, 3, tmp, planes & OC_PLANES, 6);
    if (n < 3) return;
    VOut ov[16];
    for (int k = 0; k < n; ++k) {
        if (poly[k].orig >= 0) { ov[k] = C.vo[poly[k].orig]; continue; }
        VOut& o = ov[k];
        if (!project_fx(V, poly[k].c, o.fx, o.fy, o.q)) return;
        o.code = 0;
        o.nx = poly[k].a[0]; o.ny = poly[k].a[1]; o.nz = poly[k].a[2];
        auto ch = [](float v) { return static_cast<u32>(clamp_(v, 0.0f, 255.0f) + 0.5f); };
        o.color = 0xFF000000u | (ch(poly[k].a[3]) << 16) | (ch(poly[k].a[4]) << 8) | ch(poly[k].a[5]);
    }
    for (int k = 1; k + 1 < n; ++k) {
        const u32 r = tri_rect(C, ov[0], ov[k], ov[k + 1]);
        if (r == k_no_rect || cl.tris.size() >= (1u << 24)) continue;
        cl.tris.push_back(ClipTri{{ov[0], ov[k], ov[k + 1]}});
        cl.rects.push_back(r);
        C.bins->count(cnt, r);
    }
}

// Pasada 1 escalar de un triángulo (referencia exacta; también la usa la versión SIMD para los
// carriles "raros": recorte, índices inválidos o triángulos grandes).
CFD_INLINE void bin_tri_scalar(const MeshCtx& C, i64 t, u32* rects, u32* cnt, ClipList& cl, u64& nb) {
    const u32* ix = C.idx;
    const u32 id[3] = {ix[3 * t], ix[3 * t + 1], ix[3 * t + 2]};
    u32 r = k_no_rect;
    if ((id[0] < C.nv) & (id[1] < C.nv) & (id[2] < C.nv)) {
        const VOut& a = C.vo[id[0]];
        const VOut& b = C.vo[id[1]];
        const VOut& c = C.vo[id[2]];
        const u32 oc = a.code | b.code | c.code;
        if (CFD_LIKELY(oc == 0)) r = tri_rect(C, a, b, c);
        else if (!((a.code & b.code & c.code) & OC_PLANES) && !(oc & OC_BAD)) clip_mesh_tri(C, id, cl, cnt);
    }
    rects[t] = r;
    if (r != k_no_rect) { C.bins->count(cnt, r); ++nb; }
}

// Pasada 1 vectorizada: 8 triángulos por iteración. Índices AoS→SoA con el mismo truco de 3 vías
// que las posiciones, campos de VOut con vpgatherdd enmascarado (índices inválidos no se leen),
// área en float (EXACTA si |Δ| < 2^12: productos < 2^24; si no, carril escalar), caja de píxeles
// y rect de tiles en SIMD; sólo el conteo por tile queda escalar. Devuelve dónde se quedó.
static i64 bin_pass1_simd(const MeshCtx& C, i64 lo, i64 hi, u32* rects, u32* cnt, ClipList& cl, u64& nb) {
    const Bins& B = *C.bins;
    const int* vb = reinterpret_cast<const int*>(C.vo);
    const __m256i nvm1 = _mm256_set1_epi32(static_cast<int>(C.nv - 1));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i lim = _mm256_set1_epi32(4095);
    const __m256i cx0 = _mm256_set1_epi32(B.clip.x0), cy0 = _mm256_set1_epi32(B.clip.y0);
    const __m256i cx1 = _mm256_set1_epi32(B.clip.x1), cy1 = _mm256_set1_epi32(B.clip.y1);
    const __m256i gx = _mm256_set1_epi32(B.gx0), gy = _mm256_set1_epi32(B.gy0);
    const __m256i seven = _mm256_set1_epi32(7), eight = _mm256_set1_epi32(8);
    const __m256i none = _mm256_set1_epi32(static_cast<int>(k_no_rect));
    const bool cull = C.cull_back;
    i64 t = lo;
    for (; t + 8 <= hi; t += 8) {
        __m256 fa, fb, fc;
        load_xyz8(reinterpret_cast<const float*>(C.idx + 3 * t), fa, fb, fc);
        const __m256i i0 = _mm256_castps_si256(fa), i1 = _mm256_castps_si256(fb), i2 = _mm256_castps_si256(fc);
        // i ≤ nv-1 (sin signo) ⇔ max_epu32(i, nv-1) == nv-1
        const __m256i ok = _mm256_and_si256(_mm256_and_si256(_mm256_cmpeq_epi32(_mm256_max_epu32(i0, nvm1), nvm1),
                                                             _mm256_cmpeq_epi32(_mm256_max_epu32(i1, nvm1), nvm1)),
                                            _mm256_cmpeq_epi32(_mm256_max_epu32(i2, nvm1), nvm1));
        const __m256i o0 = _mm256_slli_epi32(i0, 3), o1 = _mm256_slli_epi32(i1, 3), o2 = _mm256_slli_epi32(i2, 3);   // VOut = 8 enteros
        auto G = [&](__m256i off, int field) RZ_LAMBDA_INLINE { return _mm256_mask_i32gather_epi32(zero, vb + field, off, ok, 4); };
        const __m256i x0 = G(o0, 0), y0 = G(o0, 1), c0 = G(o0, 3);
        const __m256i x1 = G(o1, 0), y1 = G(o1, 1), c1 = G(o1, 3);
        const __m256i x2 = G(o2, 0), y2 = G(o2, 1), c2 = G(o2, 3);
        const __m256i oc = _mm256_or_si256(_mm256_or_si256(c0, c1), c2);
        const __m256i dx1 = _mm256_sub_epi32(x1, x0), dy1 = _mm256_sub_epi32(y1, y0);
        const __m256i dx2 = _mm256_sub_epi32(x2, x0), dy2 = _mm256_sub_epi32(y2, y0);
        const __m256i mag = _mm256_max_epi32(_mm256_max_epi32(_mm256_abs_epi32(dx1), _mm256_abs_epi32(dy1)),
                                             _mm256_max_epi32(_mm256_abs_epi32(dx2), _mm256_abs_epi32(dy2)));
        const __m256i small = _mm256_cmpgt_epi32(_mm256_add_epi32(lim, _mm256_set1_epi32(1)), mag);   // |Δ| ≤ 4095
        const __m256 area = _mm256_fmsub_ps(_mm256_cvtepi32_ps(dx1), _mm256_cvtepi32_ps(dy2),
                                            _mm256_mul_ps(_mm256_cvtepi32_ps(dx2), _mm256_cvtepi32_ps(dy1)));
        const __m256 back = _mm256_cmp_ps(area, _mm256_setzero_ps(), _CMP_GT_OQ);
        __m256 keep = _mm256_cmp_ps(area, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        if (cull) keep = _mm256_andnot_ps(back, keep);
        // caja de píxeles ∩ tijera (desplazamientos aritméticos = floor para negativos)
        const __m256i xmin = _mm256_min_epi32(x0, _mm256_min_epi32(x1, x2)), xmax = _mm256_max_epi32(x0, _mm256_max_epi32(x1, x2));
        const __m256i ymin = _mm256_min_epi32(y0, _mm256_min_epi32(y1, y2)), ymax = _mm256_max_epi32(y0, _mm256_max_epi32(y1, y2));
        const __m256i px0 = _mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(xmin, seven), k_sub_shift), cx0);
        const __m256i py0 = _mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(ymin, seven), k_sub_shift), cy0);
        const __m256i px1 = _mm256_min_epi32(_mm256_srai_epi32(_mm256_sub_epi32(xmax, eight), k_sub_shift), cx1);
        const __m256i py1 = _mm256_min_epi32(_mm256_srai_epi32(_mm256_sub_epi32(ymax, eight), k_sub_shift), cy1);
        const __m256i empty = _mm256_or_si256(_mm256_cmpgt_epi32(px0, px1), _mm256_cmpgt_epi32(py0, py1));
        __m256i rect = _mm256_or_si256(_mm256_sub_epi32(_mm256_srai_epi32(px0, k_tile_shift), gx),
                                       _mm256_slli_epi32(_mm256_sub_epi32(_mm256_srai_epi32(py0, k_tile_shift), gy), 8));
        rect = _mm256_or_si256(rect, _mm256_slli_epi32(_mm256_sub_epi32(_mm256_srai_epi32(px1, k_tile_shift), gx), 16));
        rect = _mm256_or_si256(rect, _mm256_slli_epi32(_mm256_sub_epi32(_mm256_srai_epi32(py1, k_tile_shift), gy), 24));
        rect = _mm256_or_si256(rect, _mm256_and_si256(_mm256_castps_si256(back), _mm256_set1_epi32(static_cast<int>(0x80000000u))));
        // carriles normales: índices válidos, sin outcodes, pequeños; los demás van por la ruta escalar
        const __m256i normal = _mm256_and_si256(_mm256_and_si256(ok, small), _mm256_cmpeq_epi32(oc, zero));
        const __m256i good = _mm256_andnot_si256(empty, _mm256_and_si256(normal, _mm256_castps_si256(keep)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rects + t), _mm256_blendv_epi8(none, rect, good));
        const u32 special = static_cast<u32>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_andnot_si256(normal, ok))));
        for_each_bit(special, [&](int k) { bin_tri_scalar(C, t + k, rects, cnt, cl, nb); });
        u32 g = static_cast<u32>(_mm256_movemask_ps(_mm256_castsi256_ps(good)));
        nb += static_cast<u64>(std::popcount(g));
        for_each_bit(g, [&](int k) { B.count(cnt, rects[t + k]); });
    }
    return t;
}

// ---------------------------------------------------------------------------------------------
// 3) Rasterizado por tile + sombreado
// ---------------------------------------------------------------------------------------------
struct ShadeConst {
    float Lx, Ly, Lz;
    float amb, kd, ks, shin8, krim;
    float Kx, Ky, Kz;            // F·f - R·cx + U·cy
    float Rx, Ry, Rz, Ux, Uy, Uz;
    float Vox, Voy, Voz;         // vector de vista constante (ortográfica)
    float alpha;
    u32 base;
};

enum class MeshMode { Opaque, Blend, Overdraw };

// Atributos por triángulo (se calculan PEREZOSAMENTE: sólo si algún bloque pasa el test de
// profundidad; ~55 % de los triángulos de una malla densa quedan ocultos y no pagan esto).
struct TriAttr {
    float q0, dq1, dq2, q1, q2;
    float n0[3], dn1[3], dn2[3];
    float c0[3], dc1[3], dc2[3];
    float h[3];
};
CFD_INLINE void tri_attr(TriAttr& A, const VOut* v0, const VOut* v1, const VOut* v2, float s) {
    A.q0 = v0->q; A.q1 = v1->q; A.q2 = v2->q;
    A.dq1 = v1->q - v0->q; A.dq2 = v2->q - v0->q;
    const float n0[3] = {v0->nx * s, v0->ny * s, v0->nz * s};
    const float n1[3] = {v1->nx * s, v1->ny * s, v1->nz * s};
    const float n2[3] = {v2->nx * s, v2->ny * s, v2->nz * s};
    for (int k = 0; k < 3; ++k) { A.n0[k] = n0[k]; A.dn1[k] = n1[k] - n0[k]; A.dn2[k] = n2[k] - n0[k]; }
    const u32 c[3] = {v0->color, v1->color, v2->color};
    for (int k = 0; k < 3; ++k) {
        const int sh = 16 - 8 * k;
        const float a = static_cast<float>((c[0] >> sh) & 255), b = static_cast<float>((c[1] >> sh) & 255), d = static_cast<float>((c[2] >> sh) & 255);
        A.c0[k] = a; A.dc1[k] = b - a; A.dc2[k] = d - a;
    }
}

// Iluminación de 8 píxeles: n sin normalizar, (px,py) centros de píxel, c = color base 0..255.
// Blinn-Phong con ambiente hemisférico, relleno desde la cámara, especular y borde (fresnel falso).
template <bool Persp>
CFD_INLINE void shade8(const ShadeConst& K, __m256 nx, __m256 ny, __m256 nz, __m256 px, __m256 py,
                       __m256& r, __m256& g, __m256& b) {
    // +1e-20: una normal nula (malla con normales degeneradas, o que se anula al interpolar / en FP16)
    // daría rsqrt(0) = inf y 0·inf = NaN → píxel negro. Así queda n = 0 (sólo ambiente + borde).
    const __m256 inl = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_fmadd_ps(nx, nx, _mm256_fmadd_ps(ny, ny, _mm256_mul_ps(nz, nz))), _mm256_set1_ps(1e-20f)));
    nx = _mm256_mul_ps(nx, inl); ny = _mm256_mul_ps(ny, inl); nz = _mm256_mul_ps(nz, inl);
    __m256 vx, vy, vz;   // vector hacia el ojo
    if constexpr (Persp) {
        // D = F·f + R·(px - cx) + U·(cy - py) = K + R·px - U·py ;  V = -D/|D|
        const __m256 dx = _mm256_fmadd_ps(_mm256_set1_ps(K.Rx), px, _mm256_fnmadd_ps(_mm256_set1_ps(K.Ux), py, _mm256_set1_ps(K.Kx)));
        const __m256 dy = _mm256_fmadd_ps(_mm256_set1_ps(K.Ry), px, _mm256_fnmadd_ps(_mm256_set1_ps(K.Uy), py, _mm256_set1_ps(K.Ky)));
        const __m256 dz = _mm256_fmadd_ps(_mm256_set1_ps(K.Rz), px, _mm256_fnmadd_ps(_mm256_set1_ps(K.Uz), py, _mm256_set1_ps(K.Kz)));
        const __m256 ind = _mm256_xor_ps(_mm256_rsqrt_ps(_mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)))), _mm256_set1_ps(-0.0f));
        vx = _mm256_mul_ps(dx, ind); vy = _mm256_mul_ps(dy, ind); vz = _mm256_mul_ps(dz, ind);
    } else {
        (void)px; (void)py;
        vx = _mm256_set1_ps(K.Vox); vy = _mm256_set1_ps(K.Voy); vz = _mm256_set1_ps(K.Voz);
    }
    const __m256 Lx = _mm256_set1_ps(K.Lx), Ly = _mm256_set1_ps(K.Ly), Lz = _mm256_set1_ps(K.Lz);
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    const __m256 ndl = _mm256_fmadd_ps(nx, Lx, _mm256_fmadd_ps(ny, Ly, _mm256_mul_ps(nz, Lz)));
    const __m256 ndv = _mm256_fmadd_ps(nx, vx, _mm256_fmadd_ps(ny, vy, _mm256_mul_ps(nz, vz)));
    const __m256 hx = _mm256_add_ps(Lx, vx), hy = _mm256_add_ps(Ly, vy), hz = _mm256_add_ps(Lz, vz);
    const __m256 ndh = _mm256_mul_ps(_mm256_fmadd_ps(nx, hx, _mm256_fmadd_ps(ny, hy, _mm256_mul_ps(nz, hz))),
                                     _mm256_rsqrt_ps(_mm256_fmadd_ps(hx, hx, _mm256_fmadd_ps(hy, hy, _mm256_mul_ps(hz, hz)))));
    // pow(ndh, n) ≈ max(0, 1 - n(1-x)/8)^8  (3 cuadrados en vez de exp/log)
    __m256 sp = _mm256_max_ps(_mm256_fmadd_ps(_mm256_sub_ps(_mm256_max_ps(ndh, zero), one), _mm256_set1_ps(K.shin8), one), zero);
    sp = _mm256_mul_ps(sp, sp); sp = _mm256_mul_ps(sp, sp); sp = _mm256_mul_ps(sp, sp);
    sp = _mm256_mul_ps(sp, _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(ndl, _mm256_set1_ps(4.0f)), zero), one));
    __m256 rim = _mm256_sub_ps(one, _mm256_min_ps(_mm256_max_ps(ndv, zero), one));
    rim = _mm256_mul_ps(rim, _mm256_mul_ps(rim, rim));
    const __m256 hemi = _mm256_fmadd_ps(nz, _mm256_set1_ps(0.25f), _mm256_set1_ps(0.8f));
    // luz de relleno desde la cámara, sólo donde la principal no llega (no sobreexpone)
    const __m256 dl = _mm256_max_ps(ndl, zero);
    const __m256 fill = _mm256_mul_ps(_mm256_max_ps(ndv, zero), _mm256_fnmadd_ps(dl, _mm256_set1_ps(K.kd), _mm256_set1_ps(K.kd)));
    const __m256 lit = _mm256_fmadd_ps(dl, _mm256_set1_ps(K.kd), _mm256_fmadd_ps(fill, _mm256_set1_ps(0.35f), _mm256_mul_ps(hemi, _mm256_set1_ps(K.amb))));
    const __m256 spc = _mm256_mul_ps(sp, _mm256_set1_ps(K.ks * 255.0f));
    const __m256 rmc = _mm256_mul_ps(rim, _mm256_set1_ps(K.krim * 255.0f));
    r = _mm256_fmadd_ps(r, lit, _mm256_fmadd_ps(rmc, _mm256_set1_ps(0.80f), spc));
    g = _mm256_fmadd_ps(g, lit, _mm256_fmadd_ps(rmc, _mm256_set1_ps(0.90f), spc));
    b = _mm256_fmadd_ps(b, lit, _mm256_add_ps(rmc, spc));
}

template <bool Persp, MeshMode Mode, bool Wire>
static void raster_mesh_tile(Framebuffer& fb, const MeshCtx& C, MeshScratch& S, const ShadeConst& K, int tile, bool two_sided) {
    constexpr bool Deferred = Mode == MeshMode::Opaque;
    const Bins& B = S.bins;
    int tx0, ty0, tx1, ty1;
    B.tile_rect(tile, tx0, ty0, tx1, ty1);
    const Surf SF = Surf::of(fb);
    const __m256i a16 = _mm256_set1_epi32(static_cast<int>(static_cast<u32>(clamp_(K.alpha, 0.0f, 1.0f) * 256.0f + 0.5f) * 0x00010001u));
    GBuf& G = S.gbuf.data()[max_(ThreadPool::worker_index(), 0)];
    const int tgx = tx0 & ~(k_tile - 1), tgy = ty0 & ~(k_tile - 1);   // origen del tile

    // Lista 0 = caras frontales, lista 1 = traseras (se dibujan después: casi todas quedan tapadas).
    // Z jerárquico (Hi-Z) del tile: profundidad máxima por bloque de 8×8, calculada UNA vez tras
    // las caras frontales. Una cara trasera cuya profundidad mínima queda detrás del máximo de los
    // bloques que toca no puede pasar el test de profundidad → se descarta sin recorrerla.
    // Translúcido (Blend): al revés, primero las traseras y luego las frontales encima (composición
    // "over" correcta en una malla cerrada: la cara cercana domina). La malla no escribe depth en
    // este modo, así que el Hi-Z de la escena opaca sigue siendo válido aunque la lista 1 vaya primero.
    alignas(32) float zmax[k_tile / 8][k_tile / 8];
    for (int pass = 0; pass < B.lists; ++pass) {
    const int list = Mode == MeshMode::Blend ? B.lists - 1 - pass : pass;
    const int li = tile + list * B.ntiles;
    const u32* it = B.items.data() + B.start.data()[li];
    const u32 nit = B.start.data()[li + 1] - B.start.data()[li];
    const bool hiz = list == 1 && Mode != MeshMode::Overdraw && nit > 0 && !RZ_EXP(2);
    if (hiz) {
        for (int j = 0; j < k_tile / 8; ++j)
            for (int i = 0; i < k_tile / 8; ++i) {
                const int x = tgx + 8 * i, y = tgy + 8 * j;
                if (x < tx0 || x + 7 > tx1 || y < ty0 || y + 7 > ty1) { zmax[j][i] = k_inf; continue; }   // bloque parcial
                const float* zp = SF.z + static_cast<usize>(y) * SF.stride + x;
                __m256 m = _mm256_load_ps(zp);
                for (int r = 1; r < 8; ++r) m = _mm256_max_ps(m, _mm256_load_ps(zp + static_cast<usize>(r) * SF.stride));
                zmax[j][i] = simd::hmax(m);
            }
    }
    for (u32 e = 0; e < nit; ++e) {
        const u32 ref = it[e];
        const VOut *v0, *v1, *v2;
        if (CFD_UNLIKELY(ref & 0x80000000u)) {
            const ClipTri& ct = S.clip[(ref >> 24) & 127].tris[ref & 0xFFFFFFu];
            v0 = &ct.v[0]; v1 = &ct.v[1]; v2 = &ct.v[2];
        } else {
            const u32* ix = C.idx + 3 * static_cast<usize>(ref);
            v0 = C.vo + ix[0]; v1 = C.vo + ix[1]; v2 = C.vo + ix[2];
        }
        if (e + 2 < nit && !(it[e + 2] & 0x80000000u)) {   // prefetch dos triángulos por delante
            const u32* nx = C.idx + 3 * static_cast<usize>(it[e + 2]);
            CFD_PREFETCH_R(C.vo + nx[0]); CFD_PREFETCH_R(C.vo + nx[1]); CFD_PREFETCH_R(C.vo + nx[2]);
        }
        if (hiz) {   // antes de orientar/dividir: un triángulo descartado cuesta ~20 instrucciones
            const i32 xs[3] = {v0->fx, v1->fx, v2->fx}, ys[3] = {v0->fy, v1->fy, v2->fy};
            int bx0, by0, bx1, by1;
            tri_pixel_box_raw(xs, ys, bx0, by0, bx1, by1);
            bx0 = (max_(bx0, tx0) - tgx) >> 3; by0 = (max_(by0, ty0) - tgy) >> 3;
            bx1 = (min_(bx1, tx1) - tgx) >> 3; by1 = (min_(by1, ty1) - tgy) >> 3;
            float zm = 0.0f;
            for (int j = by0; j <= by1; ++j)
                for (int i = bx0; i <= bx1; ++i) zm = max_(zm, zmax[j][i]);
            // profundidad mínima del triángulo (persp: 1/q máx; orto: q mín) con margen relativo
            const float zmin = Persp ? 1.0f / max_(v0->q, max_(v1->q, v2->q)) : min_(v0->q, min_(v1->q, v2->q));
            // margen con |zm|: en ortográfica la profundidad puede ser negativa (detrás del ojo)
            if (zmin > zm + std::fabs(zm) * 1e-5f + 1e-6f) { RZ_COUNT(early_out, 1); continue; }
        }
        i64 area = static_cast<i64>(v1->fx - v0->fx) * (v2->fy - v0->fy) - static_cast<i64>(v2->fx - v0->fx) * (v1->fy - v0->fy);
        const bool back = area > 0;
        if (!back) { const VOut* t = v1; v1 = v2; v2 = t; area = -area; }   // orientar: area2 > 0
        TriFx T;
        T.x[0] = v0->fx; T.x[1] = v1->fx; T.x[2] = v2->fx;
        T.y[0] = v0->fy; T.y[1] = v1->fy; T.y[2] = v2->fy;
        T.area2 = area;
        T.inv_area2 = 1.0f / static_cast<float>(area);

        if constexpr (Mode == MeshMode::Overdraw) {
            scan_tri_in(SF, T, tx0, ty0, tx1, ty1, [&](const auto& blk, int bits, __m256, __m256) RZ_LAMBDA_INLINE {
                blk.store_c(_mm256_sub_epi32(blk.load_c(), mask_si(bits)));   // -(-1) = +1
            });
            continue;
        }
        // caras traseras: normal invertida (salvo la normal por defecto sin mesh.nrm, que ya mira a la cámara)
        const float sgn = (back && two_sided && C.has_n) ? -1.0f : 1.0f;
        bool ready = false;
        TriAttr A;
#ifdef RZ_STATS
        if (back) RZ_COUNT(back_setups, 1);
#endif
        scan_tri_in(SF, T, tx0, ty0, tx1, ty1, [&](const auto& blk, int bits, __m256 l1, __m256 l2) RZ_LAMBDA_INLINE {
            // q = 1/Z lineal en pantalla: sólo 3 vértices → se leen directamente (sin TriAttr)
            const float q0 = v0->q;
            const __m256 q = _mm256_fmadd_ps(l1, _mm256_set1_ps(v1->q - q0), _mm256_fmadd_ps(l2, _mm256_set1_ps(v2->q - q0), _mm256_set1_ps(q0)));
            __m256 w;
            if constexpr (Persp) w = simd::rcp_nr(q);
            else w = q;
            const __m256 zb = blk.load_z();
            const int pass = bits & _mm256_movemask_ps(_mm256_cmp_ps(w, zb, _CMP_LT_OQ));
            if (!pass) return;
            RZ_COUNT(passed, 1);
            if (!ready) {
                ready = true;
                tri_attr(A, v0, v1, v2, sgn);
                if constexpr (Wire) {
                    auto len = [](i32 ax, i32 ay, i32 bx, i32 by) { const float dx = float(bx - ax), dy = float(by - ay); return std::sqrt(dx * dx + dy * dy); };
                    const float a2 = static_cast<float>(T.area2) / static_cast<float>(k_sub);
                    A.h[0] = a2 / max_(len(T.x[1], T.y[1], T.x[2], T.y[2]), 1e-3f);
                    A.h[1] = a2 / max_(len(T.x[2], T.y[2], T.x[0], T.y[0]), 1e-3f);
                    A.h[2] = a2 / max_(len(T.x[0], T.y[0], T.x[1], T.y[1]), 1e-3f);
                }
#ifdef RZ_STATS
                RZ_COUNT(tris_any_pass, 1); if (back) RZ_COUNT(back_pass, 1);
#endif
            }
            __m256 p1 = l1, p2 = l2;
            if constexpr (Persp) { p1 = _mm256_mul_ps(l1, _mm256_mul_ps(_mm256_set1_ps(A.q1), w)); p2 = _mm256_mul_ps(l2, _mm256_mul_ps(_mm256_set1_ps(A.q2), w)); }
            auto I = [&](float a0, float d1, float d2) RZ_LAMBDA_INLINE {
                return _mm256_fmadd_ps(p1, _mm256_set1_ps(d1), _mm256_fmadd_ps(p2, _mm256_set1_ps(d2), _mm256_set1_ps(a0)));
            };
            const __m256 nx = I(A.n0[0], A.dn1[0], A.dn2[0]);
            const __m256 ny = I(A.n0[1], A.dn1[1], A.dn2[1]);
            const __m256 nz = I(A.n0[2], A.dn1[2], A.dn2[2]);
            __m256 r = I(A.c0[0], A.dc1[0], A.dc2[0]), g = I(A.c0[1], A.dc1[1], A.dc2[1]), b = I(A.c0[2], A.dc1[2], A.dc2[2]);
            if constexpr (Wire) {   // oscurecer cerca de las aristas (distancia en px = l_i · h_i)
                const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
                const __m256 l0 = _mm256_sub_ps(_mm256_sub_ps(one, l1), l2);
                const __m256 d = _mm256_min_ps(_mm256_mul_ps(l0, _mm256_set1_ps(A.h[0])),
                                               _mm256_min_ps(_mm256_mul_ps(l1, _mm256_set1_ps(A.h[1])), _mm256_mul_ps(l2, _mm256_set1_ps(A.h[2]))));
                const __m256 k = _mm256_fmadd_ps(_mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(one, d), zero), one), _mm256_set1_ps(-0.55f), one);
                r = _mm256_mul_ps(r, k); g = _mm256_mul_ps(g, k); b = _mm256_mul_ps(b, k);
            }
            const __m256i msk = mask_si(pass);
            const __m256i dst = blk.load_c();
            if constexpr (Deferred) {
                // Diferido: color base + profundidad al framebuffer, normal FP16 + cobertura al G-buffer.
                blk.store_c(_mm256_blendv_epi8(dst, pack_rgb(clamp255(r), clamp255(g), clamp255(b)), msk));
                blk.store_z(_mm256_blendv_ps(zb, w, _mm256_castsi256_ps(msk)));
                const int lx = blk.x - tgx, ly = blk.y - tgy;
                const __m128i m16 = _mm_packs_epi32(_mm256_castsi256_si128(msk), _mm256_extracti128_si256(msk, 1));
                const __m128i hn[3] = {_mm256_cvtps_ph(nx, _MM_FROUND_TO_NEAREST_INT), _mm256_cvtps_ph(ny, _MM_FROUND_TO_NEAREST_INT),
                                       _mm256_cvtps_ph(nz, _MM_FROUND_TO_NEAREST_INT)};
                if constexpr (std::is_same_v<std::decay_t<decltype(blk)>, StampBlk>) {
                    for (int k = 0; k < 3; ++k) {
                        u16* p0 = G.n[k] + ly * k_tile + lx;
                        u16* p1 = p0 + k_tile;
                        const __m128i old = _mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p0)), _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1)));
                        const __m128i v = _mm_blendv_epi8(old, hn[k], m16);
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(p0), v);
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(p1), _mm_unpackhi_epi64(v, v));
                    }
                    G.rows[ly] |= static_cast<u64>(pass & 0xF) << lx;
                    G.rows[ly + 1] |= static_cast<u64>(pass >> 4) << lx;
                } else {
                    for (int k = 0; k < 3; ++k) {
                        __m128i* p = reinterpret_cast<__m128i*>(G.n[k] + ly * k_tile + lx);
                        _mm_store_si128(p, _mm_blendv_epi8(_mm_load_si128(p), hn[k], m16));
                    }
                    G.rows[ly] |= static_cast<u64>(pass) << lx;
                }
            } else {
                shade8<Persp>(K, nx, ny, nz, blk.px(), blk.py(), r, g, b);
                __m256i col = pack_rgb(clamp255(r), clamp255(g), clamp255(b));
                if constexpr (Mode == MeshMode::Blend) col = blend_swar(dst, col, a16);
                blk.store_c(_mm256_blendv_epi8(dst, col, msk));
            }
        });
    }
    }
    if constexpr (Deferred) {
        // Pasada de sombreado: cada píxel visible UNA vez, en bloques alineados de 8 (carriles llenos).
        for (int ly = 0; ly < k_tile; ++ly) {
            u64 m = G.rows[ly];
            if (!m) continue;
            G.rows[ly] = 0;
            do {
                const int lx = std::countr_zero(m) & ~7;
                const int bits = static_cast<int>((m >> lx) & 0xFF);
                m &= ~(0xFFull << lx);
                const RowBlk blk(SF, tgx + lx, tgy + ly);
                const __m256i c = blk.load_c();
                __m256 r = chan(c, 16), g = chan(c, 8), b = chan(c, 0);
                const u16* np = G.n[0] + ly * k_tile + lx;
                const __m256 nx = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(np)));
                const __m256 ny = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(np + k_tile * k_tile)));
                const __m256 nz = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(np + 2 * k_tile * k_tile)));
                shade8<Persp>(K, nx, ny, nz, blk.px(), blk.py(), r, g, b);
                blk.store_c(_mm256_blendv_epi8(c, pack_rgb(clamp255(r), clamp255(g), clamp255(b)), mask_si(bits)));
            } while (m);
        }
    }
}

using TileFn = void (*)(Framebuffer&, const MeshCtx&, MeshScratch&, const ShadeConst&, int, bool);
template <bool P, MeshMode M>
static TileFn pick_wire(bool wire) { return wire ? &raster_mesh_tile<P, M, true> : &raster_mesh_tile<P, M, false>; }
template <bool P>
static TileFn pick_mode(MeshMode m, bool wire) {
    switch (m) {
        case MeshMode::Blend: return pick_wire<P, MeshMode::Blend>(wire);
        case MeshMode::Overdraw: return &raster_mesh_tile<P, MeshMode::Overdraw, false>;
        default: return pick_wire<P, MeshMode::Opaque>(wire);
    }
}

} // namespace rz

RasterStats last_mesh_stats() { return rz::mesh_scratch().stats; }

namespace rz {
// Implementación de draw_mesh; record = false no toca last_mesh_stats() (conos de draw_arrow).
void draw_mesh_impl(Framebuffer& fb, const Camera& cam, const Mesh& mesh, const Light& light, const MeshStyle& style, bool record) {
    MeshScratch& S = mesh_scratch();
    RasterStats st;
    const double t0 = now_sec();
    const usize nv = mesh.pos.size(), ntri = mesh.tri.size() / 3;
    st.tris_in = ntri;
    if (record) S.stats = st;
    if (nv == 0 || ntri == 0 || nv >= 0x7FFFFFFFu || ntri >= 0x7FFFFFFFu) return;
    const MeshMode mode = style.debug_overdraw ? MeshMode::Overdraw : (style.alpha < 0.999f ? MeshMode::Blend : MeshMode::Opaque);
    if (mode == MeshMode::Blend && !(style.alpha > 0.0f)) return;
    const View V = make_view(cam, fb);
    if (V.clip.empty() || fb.color.empty() || fb.depth.empty()) return;

    // 1) vértices
    VOut* vo = grow(S.vout, nv);
    if (V.ortho) transform_vertices<true>(V, mesh, style.use_vertex_color, style.base_color, vo);
    else transform_vertices<false>(V, mesh, style.use_vertex_color, style.base_color, vo);
    const double t1 = now_sec();

    // 2) binning
    const int nslots = bin_slots(static_cast<i64>(ntri));
    Bins& B = S.bins;
    if (!B.setup(V.clip, static_cast<i64>(ntri), nslots, 2)) return;
    u32* rects = grow(S.rects, ntri);
    MeshCtx C;
    C.V = &V; C.mesh = &mesh; C.vo = vo; C.idx = mesh.tri.data(); C.nv = static_cast<u32>(nv);
    C.cull_back = !style.two_sided && mode != MeshMode::Overdraw;
    C.has_n = mesh.nrm.size() >= nv;
    C.bins = &B; C.nd = -V.fwd;
    const i64 nt = static_cast<i64>(ntri);
    pool().run_slots([&](int s, int ns) {
        const i64 lo = nt * s / ns, hi = nt * (s + 1) / ns;
        u32* cnt = B.slot_counts(s);
        ClipList& cl = S.clip[s];
        cl.tris.clear(); cl.rects.clear();
        u64 nb = 0;
        i64 t = lo;
        if (!RZ_EXP(3) && C.nv < (1u << 28)) t = bin_pass1_simd(C, lo, hi, rects, cnt, cl, nb);   // offsets de gather i·8 en int32
        for (; t < hi; ++t) bin_tri_scalar(C, t, rects, cnt, cl, nb);   // cola (y variante escalar de referencia)
        S.binned[s].value = nb;
    }, nslots);
    B.finalize();
    pool().run_slots([&](int s, int ns) {
        const i64 lo = nt * s / ns, hi = nt * (s + 1) / ns;
        u32* cur = B.slot_counts(s);
        for (i64 t = lo; t < hi; ++t)
            if (rects[t] != k_no_rect) B.emit(cur, rects[t], static_cast<u32>(t));
        const ClipList& cl = S.clip[s];
        for (usize k = 0; k < cl.rects.size(); ++k) B.emit(cur, cl.rects[k], 0x80000000u | (static_cast<u32>(s) << 24) | static_cast<u32>(k));
    }, nslots);
    const double t2 = now_sec();

    // 3) rasterizado por tiles
    ShadeConst K;
    const Vec3 L = normalize(light.dir);
    K.Lx = L.x; K.Ly = L.y; K.Lz = L.z;
    K.amb = light.ambient; K.kd = light.diffuse; K.ks = light.specular; K.krim = light.rim;
    K.shin8 = max_(light.shininess, 1.0f) * 0.125f;
    K.Rx = V.right.x; K.Ry = V.right.y; K.Rz = V.right.z;
    K.Ux = V.up.x; K.Uy = V.up.y; K.Uz = V.up.z;
    K.Kx = V.fwd.x * V.f - V.right.x * V.cx + V.up.x * V.cy;
    K.Ky = V.fwd.y * V.f - V.right.y * V.cx + V.up.y * V.cy;
    K.Kz = V.fwd.z * V.f - V.right.z * V.cx + V.up.z * V.cy;
    K.Vox = -V.fwd.x; K.Voy = -V.fwd.y; K.Voz = -V.fwd.z;
    K.alpha = style.alpha;
    K.base = style.base_color;
    const usize nthr = static_cast<usize>(max_(pool().size(), 1));
    if (S.gbuf.size() < nthr) S.gbuf.resize(nthr, true);   // máscaras de cobertura a cero
    const TileFn fn = V.ortho ? pick_mode<false>(mode, style.wireframe_overlay) : pick_mode<true>(mode, style.wireframe_overlay);
    const bool two = style.two_sided;
    const u32* ord = B.order.data();
    parallel_for(0, B.busy, 1, [&](i64 lo, i64 hi) {
        for (i64 k = lo; k < hi; ++k) fn(fb, C, S, K, static_cast<int>(ord[k]), two);
    });
    const double t3 = now_sec();

    st.t_vertex = t1 - t0; st.t_bin = t2 - t1; st.t_raster = t3 - t2; st.t_total = t3 - t0;
    u64 nclip = 0, nbin = 0;
    for (int s = 0; s < nslots; ++s) { nclip += S.clip[s].tris.size(); nbin += S.binned[s].value; }
    st.tris_clipped = nclip; st.tris_binned = nbin + nclip;
    st.bin_entries = B.total; st.tiles = B.ntiles; st.tiles_busy = B.busy;
    if (record) S.stats = st;
}
} // namespace rz

void draw_mesh(Framebuffer& fb, const Camera& cam, const Mesh& mesh, const Light& light, const MeshStyle& style) {
    rz::draw_mesh_impl(fb, cam, mesh, light, style, true);
}

} // namespace cfd::render
