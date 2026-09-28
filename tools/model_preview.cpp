// ============================================================================
//  tools/model_preview.cpp — vista previa sin ventana de los modelos del catálogo.
//
//  Para cada modelo:
//    build/models/<id>.png      2×2 vistas (lateral, superior, frontal, 3/4) trazadas
//                               con "sphere tracing" sobre el SDF, color por componente,
//                               sombra suave sobre el suelo y oclusión ambiental.
//    build/models/<id>_vox.png  cortes a resolución de red (dx = 3 cm por defecto):
//                               plano y = 0, plano lateral y = y1 y plano z = z1.
//
//  Uso:  model_preview [id|all] [--dx 0.03] [--w 640] [--h 380] [--novox] [--bench] [--out dir]
//                      [--drs] [--yaw grados] [--ride del tras (mm)] [--aoa grados] [--height mm]
//
//  Trucos: recorte del rayo con la AABB de la escena (slabs) y paralelo por filas con el
//  pool (planificación dinámica: filas con coche cuestan 20× las de fondo). El "sphere
//  tracing" sobre-relajado (Keinert et al. 2014) está implementado pero sólo se usa con
//  --bench: medido, no reduce evaluaciones en este renderizador (ver docs/opt/models.md).
// ============================================================================
#include "core/png.hpp"
#include "core/threadpool.hpp"
#include "core/util.hpp"
#include "models/model.hpp"
#include "core/mem.hpp"
#include "render/colormap.hpp"
#include "render/framebuffer.hpp"
#include "render/font_data.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace cfd;

namespace {

// ---- Imagen ARGB mínima ----------------------------------------------------------------
struct Image {
    int w = 0, h = 0;
    std::vector<u32> px;
    Image(int w_, int h_, u32 c = 0xFFFFFFFFu) : w(w_), h(h_), px(static_cast<usize>(w_) * static_cast<usize>(h_), c) {}
    u32& at(int x, int y) { return px[static_cast<usize>(y) * static_cast<usize>(w) + static_cast<usize>(x)]; }
    void blit(const Image& o, int ox, int oy) {
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x)
                if (ox + x >= 0 && oy + y >= 0 && ox + x < w && oy + y < h) at(ox + x, oy + y) = o.px[static_cast<usize>(y * o.w + x)];
    }
    void fill_rect(int x0, int y0, int x1, int y1, u32 c) {
        for (int y = max_(0, y0); y < min_(h, y1); ++y)
            for (int x = max_(0, x0); x < min_(w, x1); ++x) at(x, y) = c;
    }
};

// UTF-8 → Latin-1 (la fuente Terminus cubre 0x20..0xFF).
std::string to_latin1(const std::string& s) {
    std::string o;
    for (usize i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) o += static_cast<char>(c);
        else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            const unsigned cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
            o += cp < 256 ? static_cast<char>(cp) : '?';
            ++i;
        } else {                               // secuencias de 3-4 bytes (—, ², ...) → sustituto
            usize n = (c & 0xF0) == 0xE0 ? 2 : 3;
            const unsigned cp = n == 2 && i + 2 < s.size()
                                    ? ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) | (static_cast<unsigned char>(s[i + 2]) & 0x3Fu)
                                    : 0;
            o += cp == 0x2014 || cp == 0x2013 ? '-' : (cp == 0xB2 ? '\xB2' : '?');
            i += n;
        }
    }
    return o;
}

void draw_text(Image& im, int x, int y, const std::string& utf8, u32 color, int scale = 2) {
    const std::string s = to_latin1(utf8);
    for (usize k = 0; k < s.size(); ++k) {
        const unsigned c = static_cast<unsigned char>(s[k]);
        if (c < 0x20) continue;
        const uint8_t* g = font::k_small_glyphs[c - 0x20];
        for (int r = 0; r < font::k_small_h; ++r)
            for (int b = 0; b < font::k_small_w; ++b)
                if (g[r] & (0x80 >> b))
                    for (int sy = 0; sy < scale; ++sy)
                        for (int sx = 0; sx < scale; ++sx) {
                            const int px = x + static_cast<int>(k) * font::k_small_w * scale + b * scale + sx, py = y + r * scale + sy;
                            if (px >= 0 && py >= 0 && px < im.w && py < im.h) im.at(px, py) = color;
                        }
    }
}

// ---- Cámara simple --------------------------------------------------------------------
struct View {
    Vec3 eye, fwd, right, up;
    bool ortho = true;
    float scale = 100.0f;    // ortográfica: píxeles por metro; perspectiva: focal en píxeles
    int w = 640, h = 380;
    const char* label = "";
    void ray(float px, float py, Vec3& o, Vec3& d) const {
        const float sx = px - 0.5f * static_cast<float>(w), sy = 0.5f * static_cast<float>(h) - py;
        if (ortho) { o = eye + right * (sx / scale) + up * (sy / scale); d = fwd; }
        else { o = eye; d = normalize(fwd * scale + right * sx + up * sy); }
    }
};
View look(Vec3 eye, Vec3 target, Vec3 up_hint, bool ortho, float scale, int w, int h, const char* label) {
    View v;
    v.eye = eye; v.fwd = normalize(target - eye);
    v.right = normalize(cross(v.fwd, up_hint));
    v.up = cross(v.right, v.fwd);
    v.ortho = ortho; v.scale = scale; v.w = w; v.h = h; v.label = label;
    return v;
}

// Intersección rayo-AABB por slabs: [t0, t1] (t1 < t0 si no corta).
CFD_INLINE void ray_box(const Aabb& b, Vec3 o, Vec3 d, float& t0, float& t1) {
    t0 = 0.0f; t1 = 1e30f;
    for (int a = 0; a < 3; ++a) {
        const float inv = 1.0f / (std::fabs(d[a]) > 1e-12f ? d[a] : 1e-12f);
        float ta = (b.lo[a] - o[a]) * inv, tb = (b.hi[a] - o[a]) * inv;
        if (ta > tb) { const float t = ta; ta = tb; tb = t; }
        t0 = max_(t0, ta); t1 = min_(t1, tb);
    }
}

struct TraceStats { u64 evals = 0; };

// Sphere tracing sobre-relajado (Keinert et al. 2014): paso ω·r con ω = 1.6; si la esfera
// nueva no solapa la anterior (|r| + r_prev < paso) se pudo saltar la superficie → se
// vuelve al paso seguro r_prev y se sigue con ω = 1. eps = eps_abs + t·eps_rel (huella del píxel).
template <bool Relax>
float trace(const sdf::Scene& sc, Vec3 o, Vec3 d, float t0, float t1, float eps_abs, float eps_rel, int& gid, u64& evals) {
    float t = t0, t_prev = t0, r_prev = 0.0f;
    float omega = Relax ? 1.6f : 1.0f;
    for (int i = 0; i < 400; ++i) {
        if (t >= t1) {
            // Un paso relajado puede saltar el final del intervalo sin haberse validado:
            // repetirlo sin relajar antes de rendirse (si no, se pierden piezas finas).
            if (Relax && omega > 1.0f && t > t_prev && t_prev + r_prev < t1) { t = t_prev + r_prev; omega = 1.0f; continue; }
            break;
        }
        int g = 0;
        const float r = sc.eval(o + d * t, &g);
        ++evals;
        if (Relax && omega > 1.0f && t > t_prev && std::fabs(r) + r_prev < t - t_prev) {
            t = t_prev + r_prev;
            omega = 1.0f;
            continue;
        }
        if (r < eps_abs + t * eps_rel) { gid = g; return t; }
        t_prev = t; r_prev = r;
        t += r * omega;
    }
    return -1.0f;
}

float soft_shadow(const sdf::Scene& sc, Vec3 p, Vec3 L, float tmax, u64& evals) {
    float res = 1.0f, t = 0.02f;
    for (int i = 0; i < 64 && t < tmax; ++i) {
        const float h = sc.eval(p + L * t);
        ++evals;
        if (h < 1e-3f) return 0.0f;
        res = min_(res, 10.0f * h / t);
        t += clamp_(h, 0.01f, 0.4f);
    }
    return saturate(res);
}

float ambient_occ(const sdf::Scene& sc, Vec3 p, Vec3 n, u64& evals) {
    float occ = 0.0f, w = 1.0f;
    for (int i = 1; i <= 4; ++i) {
        const float h = 0.03f * static_cast<float>(i);
        occ += (h - sc.eval(p + n * h)) * w;
        w *= 0.6f;
        ++evals;
    }
    return saturate(1.0f - 2.5f * occ);
}

struct RenderOut { u64 evals = 0; double sec = 0; };

template <bool Relax>
RenderOut render_view(const models::Built& b, const View& v, Image& im, bool ground) {
    const sdf::Scene& sc = b.scene;
    // Recorte con la AABB de la GEOMETRÍA (no bounds_m: con spans_domain la geometría sobresale del
    // dominio y un rayo que empezase dentro del sólido sombrearía normales basura).
    const Aabb box = sc.bounds().expanded(0.02f);
    const Vec3 L = normalize(Vec3(-0.35f, -0.55f, 0.85f));
    const Vec3 L2 = normalize(Vec3(0.6f, 0.5f, 0.3f));
    const float eps_abs = v.ortho ? 0.5f / v.scale : 5e-4f, eps_rel = v.ortho ? 0.0f : 0.5f / v.scale;
    std::vector<float> depth(static_cast<usize>(v.w) * static_cast<usize>(v.h), 1e30f);
    std::vector<u8> gids(depth.size(), 0);
    std::vector<Padded<u64>> ev(static_cast<usize>(pool().size()) + 1);
    const double t0 = now_sec();
    parallel_for(0, v.h, 2, [&](i64 lo, i64 hi) {
        const int wi = max_(ThreadPool::worker_index(), 0);
        u64 evals = 0;
        for (i64 y = lo; y < hi; ++y)
            for (int x = 0; x < v.w; ++x) {
                Vec3 o, d;
                v.ray(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, o, d);
                // Fondo: degradado.
                const float gy = static_cast<float>(y) / static_cast<float>(v.h);
                Vec3 col = lerp(Vec3(0.80f, 0.86f, 0.93f), Vec3(0.60f, 0.66f, 0.74f), gy);
                float tg = (ground && d.z < -1e-6f) ? -o.z / d.z : 1e30f;
                if (ground && v.ortho && std::fabs(d.z) < 1e-4f && o.z < 0.0f) {   // vista horizontal: banda de suelo
                    im.at(x, static_cast<int>(y)) = 0xFF6A6E74u;
                    continue;
                }
                float tb0, tb1;
                ray_box(box, o, d, tb0, tb1);
                float t = -1.0f;
                int g = 0;
                if (tb1 >= tb0) t = trace<Relax>(sc, o, d, tb0, min_(tb1, tg), eps_abs, eps_rel, g, evals);
                if (t >= 0.0f) {
                    const Vec3 p = o + d * t;
                    Vec3 n = sc.normal(p, 1.5e-3f);
                    evals += 4;
                    if (dot(n, d) > 0.0f) n = -n;
                    const auto& grp = sc.groups()[static_cast<usize>(g - 1)];
                    const Vec3 base = render::unpack_rgb(render::component_color(grp.component));
                    const float ao = ambient_occ(sc, p, n, evals);
                    const float dif = max_(dot(n, L), 0.0f), dif2 = max_(dot(n, L2), 0.0f);
                    const Vec3 hv = normalize(L - d);
                    const float spec = std::pow(max_(dot(n, hv), 0.0f), 40.0f) * 0.35f;
                    col = base * (0.30f * ao + 0.62f * dif + 0.18f * dif2) + Vec3(spec);
                    depth[static_cast<usize>(y) * static_cast<usize>(v.w) + static_cast<usize>(x)] = t;
                    gids[static_cast<usize>(y) * static_cast<usize>(v.w) + static_cast<usize>(x)] = static_cast<u8>(g);
                } else if (tg < 1e29f) {
                    const Vec3 p = o + d * tg;
                    const int cx = ifloor(p.x * 2.0f), cy = ifloor(p.y * 2.0f);
                    const float chk = ((cx + cy) & 1) ? 0.80f : 0.72f;
                    const float sh = soft_shadow(sc, p + Vec3(0, 0, 1e-3f), L, 8.0f, evals);
                    const float fade = saturate(1.0f - tg / 60.0f);
                    const Vec3 gcol = Vec3(chk, chk, chk * 1.02f) * (0.55f + 0.45f * sh);
                    col = lerp(col, gcol, fade);
                    depth[static_cast<usize>(y) * static_cast<usize>(v.w) + static_cast<usize>(x)] = tg;
                    gids[static_cast<usize>(y) * static_cast<usize>(v.w) + static_cast<usize>(x)] = 255;
                }
                im.at(x, static_cast<int>(y)) = render::rgbf(col.x, col.y, col.z);
            }
        ev[static_cast<usize>(wi)].value += evals;
    });
    RenderOut out;
    out.sec = now_sec() - t0;
    for (auto& e : ev) out.evals += e.value;
    // Contornos: discontinuidad de grupo o de profundidad → oscurecer (lectura de piezas).
    Image cp = im;
    for (int y = 1; y < v.h - 1; ++y)
        for (int x = 1; x < v.w - 1; ++x) {
            const usize i = static_cast<usize>(y) * static_cast<usize>(v.w) + static_cast<usize>(x);
            if (gids[i] == 0 || gids[i] == 255) continue;
            bool edge = false;
            const usize nb[4] = {i - 1, i + 1, i - static_cast<usize>(v.w), i + static_cast<usize>(v.w)};
            for (usize j : nb) edge |= gids[j] != gids[i] || std::fabs(depth[j] - depth[i]) > 0.04f * max_(1.0f, depth[i] * 0.05f);
            if (edge) {
                const Vec3 c = render::unpack_rgb(cp.at(x, y)) * 0.45f;
                im.at(x, y) = render::rgbf(c.x, c.y, c.z);
            }
        }
    return out;
}

// ---- Cortes voxelizados ----------------------------------------------------------------
struct SliceSpec { char axis; float value; const char* label; };

Image voxel_slice(const models::Built& b, char axis, float value, float dx, int cell_px) {
    const sdf::Scene& sc = b.scene;
    const Aabb bb = b.bounds_m.expanded(2.0f * dx);
    const float zlo = b.info.needs_ground ? 0.0f : bb.lo.z;
    // Ejes de la imagen: X horizontal; vertical = Z (cortes y=cte) o Y (corte z=cte).
    const int nx = static_cast<int>(std::ceil((bb.hi.x - bb.lo.x) / dx));
    const float vlo = axis == 'y' ? zlo : bb.lo.y, vhi = axis == 'y' ? bb.hi.z : bb.hi.y;
    const int nv = static_cast<int>(std::ceil((vhi - vlo) / dx)) + (axis == 'y' && b.info.needs_ground ? 1 : 0);
    Image im(nx * cell_px, nv * cell_px, 0xFFFFFFFFu);
    parallel_for(0, nv, 1, [&](i64 lo, i64 hi) {
        for (i64 j = lo; j < hi; ++j) {
            // Fila de celdas (de arriba a abajo en la imagen).
            const int jv = nv - 1 - static_cast<int>(j);
            for (int i = 0; i < nx; ++i) {
                const float x = bb.lo.x + (static_cast<float>(i) + 0.5f) * dx;
                Vec3 p;
                u32 c = 0xFFFFFFFFu;
                if (axis == 'y') {
                    if (b.info.needs_ground && jv == 0) { c = 0xFF505050u; }
                    else {
                        // Con suelo: celda k tiene centro en z = (k - 0.5)·dx (pared en z_celdas = 0.5).
                        const float z = b.info.needs_ground ? (static_cast<float>(jv) - 0.5f) * dx : vlo + (static_cast<float>(jv) + 0.5f) * dx;
                        p = Vec3(x, value, z);
                    }
                } else {
                    p = Vec3(x, vlo + (static_cast<float>(jv) + 0.5f) * dx, value);
                }
                if (c == 0xFFFFFFFFu) {
                    int g = 0;
                    const float d = sc.eval(p, &g);
                    if (d < 0.12f * dx && g > 0) c = render::component_color(sc.groups()[static_cast<usize>(g - 1)].component);
                }
                for (int yy = 0; yy < cell_px; ++yy)
                    for (int xx = 0; xx < cell_px; ++xx) {
                        const bool border = (xx == cell_px - 1 || yy == cell_px - 1) && cell_px >= 4;
                        u32 cc = c;
                        if (border) cc = c == 0xFFFFFFFFu ? 0xFFE8E8E8u : simd::blend_argb(c, 0xFF000000u, 70);
                        im.at(i * cell_px + xx, static_cast<int>(j) * cell_px + yy) = cc;
                    }
            }
        }
    });
    return im;
}

void run_model(int idx, const std::string& outdir, float dx, int W, int H, bool vox, bool bench, const models::Params& params,
               const std::string& suffix) {
    const models::Built b = models::build(idx, params);
    const models::Info& I = b.info;
    const Aabb bb = b.bounds_m;
    const Vec3 c = bb.center(), sz = bb.size();
    const float L = max_comp(sz);
    // Vistas: lateral (desde -Y), superior (desde +Z), frontal (desde -X, morro de frente), 3/4.
    const float s_side = 0.92f * min_(static_cast<float>(W) / sz.x, static_cast<float>(H) / sz.z);
    const float s_top = 0.92f * min_(static_cast<float>(W) / sz.x, static_cast<float>(H) / sz.y);
    const float s_front = 0.88f * min_(static_cast<float>(W) / sz.y, static_cast<float>(H) / sz.z);
    const float s = min_(s_side, s_top);
    View views[4] = {
        look(c - Vec3(0, 10.0f * L, 0), c, Vec3(0, 0, 1), true, s, W, H, "Lateral"),
        look(c + Vec3(0, 0, 10.0f * L), c, Vec3(0, 1, 0), true, s, W, H, "Superior"),
        look(c - Vec3(10.0f * L, 0, 0), c, Vec3(0, 0, 1), true, s_front, W, H, "Frontal"),
        look(c + normalize(Vec3(-1.0f, -0.95f, 0.62f)) * max_(1.35f * L, 0.95f * 0.5f * length(sz) / std::sin(20.0f * k_deg2rad)),
             c + Vec3(0.05f * L, 0, -0.1f * sz.z), Vec3(0, 0, 1), false,
             0.5f * static_cast<float>(H) / std::tan(0.5f * 40.0f * k_deg2rad), W, H, "3/4"),
    };
    const int top = 44;
    Image sheet(2 * W, 2 * H + top, 0xFF20242Au);
    char hdr[256];
    std::snprintf(hdr, sizeof hdr, "%s  |  %.2f x %.2f x %.2f m  |  %zu grupos", I.name.c_str(), sz.x, sz.y, sz.z, b.scene.group_count());
    draw_text(sheet, 10, 6, hdr, 0xFFFFFFFFu, 2);
    u64 evals = 0;
    double sec = 0;
    for (int k = 0; k < 4; ++k) {
        Image im(W, H);
        // Trazado normal por defecto: la sobre-relajación no ahorra evaluaciones aquí (medido con
        // --bench: ±3%), porque dominan normales, AO y sombras, no la marcha del rayo primario.
        const RenderOut r = render_view<false>(b, views[k], im, true);
        evals += r.evals; sec += r.sec;
        draw_text(im, 8, 6, views[k].label, 0xFF101010u, 2);
        sheet.blit(im, (k & 1) * W, top + (k >> 1) * H);
        if (bench) {
            Image im2(W, H);
            double best_relax = 1e9, best_plain = 1e9;
            u64 e_relax = 0, e_plain = 0;
            for (int rep = 0; rep < 3; ++rep) {
                const RenderOut a = render_view<true>(b, views[k], im2, true);
                const RenderOut p = render_view<false>(b, views[k], im2, true);
                best_relax = min_(best_relax, a.sec); best_plain = min_(best_plain, p.sec);
                e_relax = a.evals; e_plain = p.evals;
            }
            std::printf("  bench vista %-8s: relajado %.3f s (%llu evals)  normal %.3f s (%llu evals)\n", views[k].label,
                        best_relax, static_cast<unsigned long long>(e_relax), best_plain, static_cast<unsigned long long>(e_plain));
        }
    }
    const std::string path = outdir + "/" + I.id + suffix + ".png";
    png::write_argb(path, sheet.px.data(), sheet.w, sheet.h, sheet.w);
    std::printf("%-16s %6.2f s  %8.1f Mevals  %.0f ns/eval  -> %s\n", I.id.c_str(), sec, static_cast<double>(evals) * 1e-6,
                sec * 1e9 * static_cast<double>(pool().size()) / static_cast<double>(max_(evals, u64(1))), path.c_str());

    if (vox) {
        // Cortes: y = 0, y lateral (pontones / túneles / endplates), z bajo (fondo).
        const bool car = I.kind == models::Kind::F1Car;
        const float y1 = car ? 0.47f : 0.25f * sz.y;
        const float z1 = car ? 0.10f : c.z;
        const int cp = 5;
        Image s0 = voxel_slice(b, 'y', 0.0f, dx, cp);
        Image s1 = voxel_slice(b, 'y', y1, dx, cp);
        Image s2 = voxel_slice(b, 'z', z1, dx, cp);
        const int lab = 36;
        Image vs(max_(s0.w, s2.w) + 8, s0.h + s1.h + s2.h + 4 * lab, 0xFF20242Au);
        char t0[160], t1[160], t2[160];
        std::snprintf(t0, sizeof t0, "%s  dx = %.0f mm  corte y = 0", I.id.c_str(), dx * 1000.0f);
        std::snprintf(t1, sizeof t1, "corte y = %.2f m", y1);
        std::snprintf(t2, sizeof t2, "corte z = %.2f m (planta)", z1);
        int yy = 4;
        draw_text(vs, 4, yy, t0, 0xFFFFFFFFu, 2); yy += lab; vs.blit(s0, 4, yy); yy += s0.h + 4;
        draw_text(vs, 4, yy, t1, 0xFFFFFFFFu, 2); yy += lab; vs.blit(s1, 4, yy); yy += s1.h + 4;
        draw_text(vs, 4, yy, t2, 0xFFFFFFFFu, 2); yy += lab; vs.blit(s2, 4, yy);
        png::write_argb(outdir + "/" + I.id + suffix + "_vox.png", vs.px.data(), vs.w, vs.h, vs.w);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string which = "all", outdir = "build/models";
    float dx = 0.03f;
    int W = 640, H = 380;
    bool vox = true, bench = false;
    models::Params params;
    std::string suffix;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--drs") { params.drs_open = true; suffix += "_drs"; continue; }
        if (a == "--yaw" && i + 1 < argc) { params.yaw_deg = std::strtof(argv[++i], nullptr); suffix += "_yaw"; continue; }
        if (a == "--ride" && i + 2 < argc) { params.ride_front_mm = std::strtof(argv[++i], nullptr); params.ride_rear_mm = std::strtof(argv[++i], nullptr); suffix += "_ride"; continue; }
        if (a == "--aoa" && i + 1 < argc) { params.aoa_deg = std::strtof(argv[++i], nullptr); suffix += "_aoa"; continue; }
        if (a == "--height" && i + 1 < argc) { params.height_mm = std::strtof(argv[++i], nullptr); suffix += "_h"; continue; }
        if (a == "--dx" && i + 1 < argc) dx = std::strtof(argv[++i], nullptr);
        else if (a == "--w" && i + 1 < argc) W = std::atoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) H = std::atoi(argv[++i]);
        else if (a == "--novox") vox = false;
        else if (a == "--bench") bench = true;
        else if (a == "--out" && i + 1 < argc) outdir = argv[++i];
        else which = a;
    }
    pool().start();
    std::printf("model_preview: %d modelos, %d hilos\n", models::count(), pool().size());
    for (int i = 0; i < models::count(); ++i) {
        if (which != "all" && models::info(i).id != which) continue;
        run_model(i, outdir, dx, W, H, vox, bench, params, suffix);
    }
    pool().stop();
    return 0;
}
