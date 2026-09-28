// ============================================================================
//  render/mesh.hpp — malla triangular indexada (producida por geom/, dibujada
//  por render/raster). Coordenadas en unidades de mundo = celdas de la red.
// ============================================================================
#pragma once

#include "../core/mathx.hpp"
#include <vector>

namespace cfd::render {

struct Mesh {
    std::vector<Vec3> pos;     // posición por vértice (celdas)
    std::vector<Vec3> nrm;     // normal unitaria hacia FUERA del sólido
    std::vector<u8> group;     // id de grupo SDF por vértice (1..254; 255 = suelo; 0 = desconocido)
    std::vector<u32> color;    // color por vértice 0xAARRGGBB (lo rellena la visualización cada cuadro)
    std::vector<u32> tri;      // 3 índices por triángulo, orden antihorario visto desde fuera
    Aabb bounds;

    void clear() { pos.clear(); nrm.clear(); group.clear(); color.clear(); tri.clear(); bounds = Aabb{}; }
    usize vertex_count() const { return pos.size(); }
    usize tri_count() const { return tri.size() / 3; }
    void recompute_bounds() { bounds = Aabb{}; for (const Vec3& p : pos) bounds.grow(p); }
};

} // namespace cfd::render
