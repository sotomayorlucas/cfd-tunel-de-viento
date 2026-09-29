# Túnel de viento CFD 3D — Arquitectura y contratos

Simulador CFD interactivo estilo túnel de viento, **C++23 puro**, render por software,
afinado para **Intel Core Ultra 7 155H** (Meteor Lake). Única dependencia del sistema:
Xlib + XShm (ventana). Todo lo demás (solver, geometría, rasterizador, UI, fuentes,
PNG/deflate) está escrito desde cero en este repo.

## CPU objetivo (medido en esta máquina)

| | |
|---|---|
| Núcleos | 6 P (Redwood Cove, HT → CPUs 0-11) + 8 E (Crestmont, CPUs 12-19) + 2 LP-E (tile SoC, CPUs 20-21, sin L3) |
| ISA | AVX2, FMA, F16C, BMI1/2, AVX-VNNI, GFNI, VAES, WAITPKG. **Sin AVX-512** |
| Caché | L1d 48 KB (P) / 32 KB (E); L2 2 MB por P-core y 2 MB por clúster de 4 E; L3 24 MB compartida (P+E) |
| RAM | LPDDR5x, ~14 GB. Ancho de banda de lectura medido con 22 hilos escalar ≈ 30 GB/s (medir mejor con AVX2) |
| Pool de hilos | `detect_topology()` ordena: 1 hilo por P físico → E → hermanos HT → LP-E. Recomendado = 20 (sin LP-E). Dispatch ≈ 4-8 µs |
| Pantalla | 3840×2400 → framebuffer interno 1920×1200 con escalado 2× (AVX2) al presentar |

Hallazgos ya medidos: UMWAIT despierta más lento que PAUSE (no usar); con los LP-E el
dispatch se degrada mucho; la barrera del pool admite hilos rezagados (hay procesos
pesados de fondo en esta máquina: qemu, rustc — los benchmarks deben repetirse y usar la mediana).

## Convenciones globales

* **Espacio modelo** (geometría, metros): suelo en `z = 0`, aire hacia **+X**, el coche mira
  hacia **-X**, `+Y` = **derecha** del piloto (mira hacia -X, Z arriba). Coches: eje delantero en `x = 0`, trasero en `x = +batalla`.
* **Espacio de red / mundo de render** (celdas): centro de la celda `(i,j,k)` en `(i,j,k)`.
  `LatticeMap` convierte (`geom/lattice_map.hpp`). Con suelo, la capa `z = 0` es sólida y la
  pared queda en `z = 0.5` ↔ `z_modelo = 0`.
* Índice lineal `n = x + nx*(y + ny*z)`; **X contiguo**; `nx` múltiplo de 8.
* Unidades de red: `ρ∞ = 1`, `u∞ ≈ 0.05–0.1`, `p = ρ/3`, `Cp = 2(ρ-1)/(3u∞²)`.
* Fuerzas físicas: coeficiente de red × `½ ρ_aire U² A_ref`. Mostrar `SCz = Cl·A` (carga,
  positivo hacia abajo) y `SCx = Cd·A` en m², como en F1.
* Ids de sólido: `0` fluido, `1..254` = 1 + índice de grupo SDF, `255` = suelo.
* Colores `0xAARRGGBB`. Profundidad = distancia de vista lineal (menor = más cerca).
* Textos de UI, comentarios y documentación en **español**.

## Módulos y propiedad de archivos

| Módulo | Archivos (dueño exclusivo) | Contrato |
|---|---|---|
| core | `src/core/*` (ya hecho) | `config.hpp` macros, `mathx.hpp`, `simd.hpp` (f8/i8x AVX2, FP16), `mem.hpp` (Buffer alineado + THP, Arena, Padded), `threadpool.hpp`, `util.hpp` (tiempo, WyRand, bits, PDEP Morton), `png.hpp` |
| geom/sdf | `src/geom/sdf.*`, `lattice_map.hpp` (ya hecho) | escena SDF de grupos + primitivas + CSG + marcos |
| **lbm** | `src/lbm/*` (salvo `field.hpp`), `tests/test_lbm*.cpp`, `tests/test_refine.cpp`, `tools/bench_lbm.cpp`, `tools/calib.cpp` | `lbm/solver.hpp`. Privados: `kernel.hpp` (colisión/momentos genéricos), `solver_impl.hpp` (`Solver::Impl`, una por rejilla), `refine.cpp` (refinamiento local por bloques 2:1: interfaces y paso recursivo; docs/FISICA.md §1.6) |
| **geom (voxel+malla)** | `src/geom/voxelizer.cpp`, `src/geom/mesher.cpp`, `tests/test_geom*.cpp` | `geom/voxelizer.hpp` |
| **models** | `src/models/*.cpp`, `tools/model_preview.cpp`, `tests/test_models*.cpp` | `models/model.hpp` |
| **render core** | `src/render/raster.cpp`, `src/render/colormap.hpp`, `tests/test_raster*.cpp` | `render/raster.hpp` |
| **flowvis** | `src/render/flowvis.hpp`, `src/render/flowvis.cpp`, `tests/test_flowvis*.cpp` | define su propia API |
| **ui + plataforma** | `src/render/draw2d.*`, `src/ui/*`, `src/platform/*.cpp`, `tests/test_ui*.cpp` | `platform/platform.hpp`; define `draw2d.hpp`, `ui.hpp` |
| **app** (integración, al final) | `src/app/*` | une todo |
| **gpu** (fase 3) | `src/gpu/vk.*`, `spirv.*`, `lbm_kernels.*`, `lbm_gpu.*`, `tests/test_gpu*.cpp`, `tools/bench_gpu.cpp` | `gpu/lbm_gpu.hpp` (backend iGPU del solver, Vulkan de cómputo propio; ver `docs/GPU.md`). Se engancha a `lbm::Solver` con la extensión compatible `set_external`/`external_view`/`external_commit`/`set_field_override` |

Contratos compartidos (NO modificar sin coordinar; si hace falta algo, añadirlo de forma
compatible y documentarlo en el informe): `lbm/field.hpp`, `render/framebuffer.hpp`,
`render/camera.hpp`, `render/mesh.hpp`, `render/raster.hpp`, `geom/voxelizer.hpp`,
`models/model.hpp`, `lbm/solver.hpp`, `platform/platform.hpp`.

## Flujo por cuadro (app)

1. `platform->poll(input)` → UI inmediata (panel derecho) + controles de cámara.
2. Si cambió geometría/params: `models::build` → `geom::voxelize` → `solver.set_geometry`
   (el flujo continúa, no se resetea) → `geom::mesh_scene` (malla suave para render).
3. `solver.step(k)` con `k` adaptativo para mantener ~30 FPS (el último paso escribe ρ,u). Con refinamiento local
   (`--refine`, defecto en los F1) cada paso de la red base avanza 2/4 subpasos las cajas finas; la visualización usa
   el campo compuesto `flowvis::MultiField` (`Sim::multi_field()`: la rejilla más fina en cada punto).
4. Fuerzas → media exponencial → Cl, Cd, SCz, SCx, balance, desglose por componente.
5. Render: fondo → suelo (cinta animada / huella de Cp) → malla coloreada (Cp, |u|,
   componente) → planos de corte → líneas de corriente / partículas de humo → volumen de
   vórtices (Q) → flechas de fuerza → bordes SSAO → UI 2D → `present`.

## Reglas de rendimiento (obligatorias en rutas calientes)

* Nada de `std::function`, `virtual`, excepciones, RTTI, asignaciones ni `std::vector` que
  crezca dentro de bucles por celda/píxel. Buffers persistentes (`Buffer<T>`, `Arena`).
* SoA, alineado a 64 B, `CFD_RESTRICT`, `CFD_INLINE`, `[[assume]]`, prefetch donde se mida
  ganancia. AVX2 explícito (`core/simd.hpp`) en kernels; dejar que GCC autovectorice lo trivial.
* Paralelizar con `parallel_for` (grano ajustado: filas/tiles), acumuladores por slot con
  `Padded<T>` + `run_slots` (sin false sharing, sin atómicos en el bucle).
* Todo "truco oscuro" usado debe quedar listado con archivo:línea y ganancia medida en
  `docs/OPTIMIZACIONES.md` (cada módulo añade su sección).

## Compilación para agentes (en paralelo, sin pisarse)

Cada agente compila SUS tests/herramientas con g++ directo a su propio directorio:
`g++ -std=c++23 -O3 -march=native -Isrc -pthread <archivos> -o build/<modulo>/<bin>`
(`-lX11 -lXext` sólo si enlaza plataforma). No usar `make` global hasta la fase de
integración (otros módulos pueden estar a medio escribir). Validar salidas visuales
escribiendo PNG (`core/png.hpp`) en `build/<modulo>/` y mirándolas.
