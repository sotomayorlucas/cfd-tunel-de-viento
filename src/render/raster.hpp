// ============================================================================
//  render/raster.hpp — rasterizador 3D por software (contrato público).
//
//  Implementación en render/raster*.cpp (módulo "render core"; ver raster.cpp).
//  Todas las llamadas son paralelas internamente (pool de hilos) y respetan
//  Framebuffer::depth (profundidad de vista lineal). Coordenadas de mundo en
//  celdas. Colores 0xAARRGGBB; el canal A se usa como opacidad donde aplique.
//  Todo se recorta a cam.vp ∩ framebuffer (nunca se escribe fuera).
//  (Bordes: los bloques de 8 píxeles alineados que cruzan el borde de cam.vp se leen y se reescriben
//  con el MISMO valor → no dibujar concurrentemente, desde otro hilo, en los ≤ 7 píxeles de color/depth
//  adyacentes al viewport mientras se rasteriza.)
//  HILOS: las funciones comparten buffers de trabajo persistentes → llamarlas
//  desde un único hilo a la vez (el de render); por dentro usan todo el pool.
//  Entradas degeneradas (NaN, inf, coordenadas enormes, índices fuera de rango,
//  tamaños absurdos) se descartan o se sujetan: nunca cuelgan ni escriben fuera.
// ============================================================================
#pragma once

#include "camera.hpp"
#include "framebuffer.hpp"
#include "mesh.hpp"
#include <span>

namespace cfd::render {

struct Light {
    Vec3 dir = normalize(Vec3(-0.4f, -0.5f, 0.75f));   // HACIA la luz (mundo)
    float ambient = 0.28f;
    float diffuse = 0.72f;
    float specular = 0.35f;
    float shininess = 48.0f;
    float rim = 0.25f;          // realce de contorno (fresnel falso) para leer la silueta
};

struct MeshStyle {
    bool use_vertex_color = true;   // false → base_color
    u32 base_color = 0xFFB8BCC4u;
    bool wireframe_overlay = false;
    float alpha = 1.0f;             // <1 → mezcla (sin escribir depth)
    bool two_sided = true;          // sombrear caras traseras (mallas con huecos)
    // --- Extensión compatible (módulo render core) ---
    bool debug_overdraw = false;    // depuración: cada fragmento cubierto suma 1 al color (sin test
                                    // de profundidad ni sombreado) → mapa de sobre-dibujo / estanqueidad
};

// Estadísticas del último draw_mesh (extensión: para la superposición de rendimiento de la app).
struct RasterStats {
    double t_vertex = 0, t_bin = 0, t_raster = 0, t_total = 0;   // segundos
    u64 tris_in = 0;        // triángulos de la malla
    u64 tris_binned = 0;    // triángulos que sobrevivieron al recorte/culling y se repartieron
    u64 tris_clipped = 0;   // triángulos generados por recorte (plano cercano / banda de guarda)
    u64 bin_entries = 0;    // referencias triángulo→tile
    int tiles = 0, tiles_busy = 0;
};
RasterStats last_mesh_stats();

// --- Primitivas 3D ---------------------------------------------------------------------
// Malla con sombreado por vértice (Gouraud del color × iluminación Blinn-Phong por píxel
// con normales interpoladas). Binning por tiles de 64×64 y rasterizado en paralelo.
// Punto fijo 28.4 + regla top-left (estanca), recorte real en el plano cercano, interpolación
// perspectiva-correcta. Sin mesh.nrm → normal hacia la cámara; sin mesh.color (o
// use_vertex_color=false) → base_color; el alfa del color por vértice se ignora (usar style.alpha).
// Translúcido (alpha < 1): sin escribir depth, cada tile mezcla primero las caras traseras y luego
// las frontales (en una malla cerrada la cara cercana queda encima); dentro de cada grupo, orden de
// envío (sin ordenar por profundidad). Sin mesh.nrm las caras traseras usan la misma normal hacia
// la cámara (no se invierte).
// Rendimiento: two_sided=false es ~20 % más rápido con mallas cerradas (sin caras traseras).
void draw_mesh(Framebuffer& fb, const Camera& cam, const Mesh& mesh, const Light& light, const MeshStyle& style);

// Cuadrilátero texturizado (p.ej. plano de corte). corners en orden: (u0,v0) (u1,v0) (u1,v1) (u0,v1).
// tex: tw×th ARGB (fila 0 = v0); muestreo bilineal "clamp"; alpha global ∈ [0,1] multiplica el
// alfa del texel. Doble cara. depth_write=false para planos translúcidos (con true se escribe
// profundidad donde alfa final ≥ 0.5, así los texels transparentes no tapan nada).
void draw_textured_quad(Framebuffer& fb, const Camera& cam, const Vec3 corners[4], const u32* tex, int tw, int th,
                        float alpha, bool depth_test = true, bool depth_write = true);

// Segmentos 3D: pts en pares (a0,b0,a1,b1,...) con color por segmento o por vértice.
// width en píxeles (≥1), antialias por cobertura. depth_test contra la escena.
// Extremos redondeados; el alfa del color es opacidad; width < 1 → 1 px con alfa atenuado;
// width se sujeta a 256. Si faltan colores se repite el último (o blanco si no hay ninguno).
// No escriben profundidad; el test usa un sesgo pequeño (0.2 % + 0.05) para líneas sobre superficies.
void draw_lines(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                bool per_vertex_color, float width = 1.5f, bool depth_test = true);

// Polilíneas: `pts` concatenadas, `starts[i]` = índice inicial de la i-ésima, `counts[i]` = nº de puntos.
// Color por vértice. Ideal para líneas de corriente. Rangos fuera de `pts` se ignoran.
void draw_polylines(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                    std::span<const u32> starts, std::span<const u32> counts, float width = 1.5f, bool depth_test = true);

// Puntos/partículas: splats circulares suaves de `size` píxeles.
// additive=true → suma saturada (humo luminoso); false → mezcla alfa. No escriben depth.
// Sí se ocultan tras la escena (test con sesgo). Color por punto si colors.size() ≥ pts.size(),
// si no colors[0] (o blanco). size < 1 → 1 px atenuado; size se sujeta a 256.
void draw_points(Framebuffer& fb, const Camera& cam, std::span<const Vec3> pts, std::span<const u32> colors,
                 float size, bool additive);

// Rejilla / plano del suelo en z = z0 con líneas cada `spacing` y animación de cinta (offset_x).
// Intersección rayo/plano por píxel dentro de extent (x,y; si extent está vacío, plano infinito),
// borde con antialias, damero sutil, líneas finas + mayores cada 5 celdas con ancho analítico por
// derivadas, niebla con la distancia (relativa a cam.distance). base.A = opacidad del suelo,
// line.A = opacidad de las líneas. tex (opcional) se mapea sobre extent (huella de Cp) y no se
// mueve con la cinta. Test y escritura de profundidad (escribe donde cobertura > 0.5).
void draw_ground(Framebuffer& fb, const Camera& cam, float z0, Aabb extent, float spacing, float offset_x,
                 u32 base, u32 line, const u32* tex = nullptr, int tw = 0, int th = 0);

// Caja en alambre (túnel de viento).
void draw_box_wire(Framebuffer& fb, const Camera& cam, const Aabb& box, u32 color, float width = 1.0f);

// Flecha 3D (vectores de fuerza): de `from` a `to`, cabeza de tamaño relativo.
// Asta = línea gruesa AA de `width` px; cabeza = cono 3D sombreado (≥ ~2.4·width px de radio).
void draw_arrow(Framebuffer& fb, const Camera& cam, Vec3 from, Vec3 to, u32 color, float width = 2.5f);

// Postproceso: oscurecimiento ambiental en espacio de pantalla barato (bordes por
// discontinuidad de profundidad) para dar volumen. strength ∈ [0,1].
// 12 vecinos por píxel; sólo ocluyen vecinos entre 0.4 % y 30 % más cerca (sin halos en el fondo).
void screen_space_edges(Framebuffer& fb, Rect vp, float strength);

// --- Extensiones compatibles (módulo render core) ------------------------------------------
// Antialiasing FXAA "lite" (Lottes, variante consola) sobre el rectángulo vp. Lee una copia del
// color, detecta bordes por contraste de luma (AVX2, 32 píxeles por instrucción) y sólo filtra
// los píxeles de borde a lo largo de la dirección del borde.
void fxaa(Framebuffer& fb, Rect vp);

} // namespace cfd::render
