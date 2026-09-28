// ============================================================================
//  models/registry.hpp — interno: entradas del catálogo (Info + constructor).
// ============================================================================
#pragma once

#include "model.hpp"
#include <vector>

namespace cfd::models::detail {

// Constructor de modelo: recibe Built con info y params YA resueltos; rellena la escena,
// aplica los marcos (set_frames) y, si es coche, los ejes y la referencia de momentos.
using BuildFn = void (*)(Built&);

struct Entry {
    Info info;
    BuildFn build = nullptr;
};

void register_f1(std::vector<Entry>& out);        // models/f1_eras.cpp
void register_objects(std::vector<Entry>& out);   // models/objects.cpp

} // namespace cfd::models::detail
