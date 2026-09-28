// ============================================================================
//  app/main.cpp — punto de entrada del túnel de viento CFD.
//  `cfd --help` muestra las opciones; sin argumentos abre la ventana X11 con
//  el F1 de 2022 a resolución media.
// ============================================================================
#include "app.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    cfd::app::Options o;
    if (!cfd::app::parse_cli(argc, argv, o)) {
        std::fprintf(stderr, "[cfd] error: %s\n[cfd] usa --help para ver las opciones\n", o.error.c_str());
        return 2;
    }
    return cfd::app::run_cli(o);
}
