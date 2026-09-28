# Rasterizador por software ("render core") — optimizaciones y resultados

Módulo: `src/render/raster.hpp` (contrato), `raster_internal.hpp` (piezas compartidas),
`raster_mesh.cpp` (mallas), `raster_prims.cpp` (líneas, puntos, caja, flechas),
`raster_fill.cpp` (cuadrilátero texturizado, suelo), `raster.cpp` (SSAO, FXAA).
Pruebas: `tests/test_raster.cpp`. CPU objetivo: Intel Core Ultra 7 155H (6P+8E+2LPE, AVX2/FMA/F16C, sin AVX-512).

## Tubería de `draw_mesh`

1. **Vértices (AVX2, 8 a la vez)** → registro `VOut` de 32 B por vértice (x,y en punto fijo 28.4,
   q = 1/Z, outcode, normal, color). Una sola carga YMM por vértice en el setup.
2. **Binning por tiles de 64×64** con ordenación por conteo (2 pasadas + prefijos): cada tile recibe una
   lista contigua de triángulos en orden de envío, separada en caras frontales y traseras.
   Los triángulos que cruzan el plano cercano o la banda de guarda se recortan en espacio de vista.
3. **Rasterizado paralelo por tile** (un tile = un hilo → sin carreras ni atómicos): Hi-Z para las caras
   traseras, funciones de arista enteras con regla top-left, sellos 4×2 para triángulos diminutos,
   test de profundidad de 8 píxeles, interpolación perspectiva-correcta y **sombreado diferido**
   (G-buffer FP16 por hilo) con Blinn-Phong + ambiente hemisférico + relleno + borde.

## Resultados (1920×1200, viewport 3D 1500×1150, 20 hilos, medianas de 31 repeticiones)

La máquina tenía carga de fondo intensa (qemu, compilaciones de otros agentes; carga media 4–25):
los tiempos absolutos varían hasta 2× entre ejecuciones. Las comparaciones A/B de abajo se hicieron
**alternando** A y B en el mismo proceso y comparando mínimos y medianas.

| Prueba | Objetivo | Primera versión (mediana) | Final: mediana (mínimo) |
|---|---|---|---|
| `draw_mesh` 298 252 tris (doble cara) | < 8 ms | 4.06 ms | **2.19 ms (2.01)** — una cara: 1.83 (1.73) |
| `draw_mesh` 594 384 tris | — | 4.96 ms | **3.12 ms (2.91)** |
| `draw_mesh` 99 600 tris | — | 2.11 ms | **1.42 ms (1.28)** |
| `draw_polylines` 50 000 segmentos, ancho 1.5, depth | < 3 ms | 1.55 ms | **1.47 ms (1.37)** |
| `draw_points` 200 000 splats 3 px, aditivo / alfa | < 3 ms | 1.64 / 1.81 ms | **1.10 (1.02) / 1.19 (1.05) ms** |
| `draw_ground` con textura de Cp / sin textura | < 2 ms | 1.11 / 0.81 ms | **1.04 (1.01) / 0.72 (0.71) ms** |
| `draw_textured_quad` plano 512×256 bilineal | — | 0.37 ms | 0.34 ms (0.32) |
| `screen_space_edges` (12 muestras) | — | 0.65 ms | 0.68 ms (0.55) |
| `fxaa` (extensión) | — | 0.84 ms | 0.85 ms (0.77) |

(Referencia de ruido: `Framebuffer::gradient` del mismo rectángulo: 0.06–0.07 ms en esta ejecución; con la
máquina saturada llegó a 0.73 ms y todo escala igual.) Desglose final de `draw_mesh` 300 k: vértices 0.06 ms,
binning 0.39 ms, raster 1.73 ms. Los objetivos del test se juzgan con el **mínimo** de 31 repeticiones y, si la
carga media supera nº de CPUs / 2, un objetivo no alcanzado se imprime como AVISO en vez de FAIL (una prueba
de rendimiento no puede depender de lo que hagan otros procesos).

## Trucos aplicados (archivo:línea, efecto medido)

### Rasterizado de triángulos

1. **Punto fijo 28.4 + funciones de arista enteras + regla top-left** —
   `raster_internal.hpp:462` (`scan_tri`), `:525` (`scan_tri_small`). Las aristas "no top-left" restan 1 y
   la cobertura de 8 píxeles se decide con el bit de signo del OR de las 3 aristas y **un solo
   `vmovmskps`**. Estanqueidad demostrada en las pruebas (0 grietas y 0 dobles impactos en 4 cámaras,
   hasta 320 000 triángulos, y también con triángulos recortados).
2. **Banda de guarda ±8192 px con cota de int32** — `raster_internal.hpp:66`. Con 4 bits de subpíxel
   |coordenada| < 2^18; dentro de un tile de 64 px una arista que lo cruza vale |E| ≤ 2^29 → el
   recorrido es int32 exacto. Fuera de la banda, recorte 3D real.
3. **Clasificación trivial de aristas por región** (`raster_internal.hpp:483`): E mín/máx en 64 bits
   sobre la región del tile; arista siempre dentro → se anula (0 coste), siempre fuera → se descarta el
   triángulo para ese tile. Es lo que permite int32 en triángulos enormes.
4. **Top-left sin ramas** — `raster_internal.hpp:532`: `((A>0)|((A==0)&(B>0)))^1` con operadores de bits.
   Micro-benchmark (triángulos de 2.5 px, 1 hilo): **29.0 → 23.5 ns/triángulo (−19 %)**.
5. **Sellos 4×2 para triángulos pequeños** — `raster_internal.hpp:525` (`scan_tri_small<true>`), selección
   en `:592`, bloque `StampBlk` en `:443`. El 90–97 % de los triángulos de una malla densa caben en 8×8 px:
   se recorren con sellos de 4 columnas × 2 filas (dos accesos de 16 B por fila, `_mm256_loadu2_m128`),
   colocados pegados al borde del tile si hace falta para **no solaparse ni salirse del tile** (sin
   carreras con el hilo vecino). Bloques visitados por triángulo 5.5 → 3.95; sombreados 1.42 → 1.15.
   Medido (A/B alternado): raster 1 hilo **26.8 → 22.5 ms (300 k), 37.3 → 31.3 ms (600 k)**; 20 hilos
   mín **3.08 → 2.58 ms (−16 %)**. Micro: 23.3 → 17.5 ns (lado 2 px), 29.0 → 21.5 (3 px), 39.0 → 28.4 (5 px).
6. **Baricéntricas desde las propias funciones de arista** (`cvtdq2ps` + FMA sólo en bloques cubiertos,
   `raster_internal.hpp:558`) en vez de mantener 2 planos float más por bloque.
7. **Máscaras de carriles con LUT `constexpr`** de 256 entradas (bits → máscara YMM, 8 KB) —
   `raster_internal.hpp:162`.

### Tubería de mallas

8. **Transformación de vértices AVX2** — `raster_mesh.cpp:62` (AoS→SoA con 6 cargas de 128 bits +
   5 `shufps`, sin gathers), `:73` (transpuesta 8×8 → registros `VOut` alineados), `:107`.
   Medido: 300 k vértices, 1 hilo **2.39 → 0.59 ms (4.0×)**; 20 hilos 0.22 → 0.135 ms.
9. **Binning por ordenación por conteo sin atómicos** — `raster_internal.hpp:267`. Contadores por slot,
   prefijos, segunda pasada: orden de envío estable (resultado determinista independiente del nº de hilos).
10. **Pasada 1 del binning vectorizada (8 triángulos/iteración)** — `raster_mesh.cpp:257`. Índices AoS→SoA
    con el truco de 3 vías, campos de `VOut` con `vpgatherdd` enmascarado (índices inválidos no se leen),
    área en float **exacta si |Δ| < 2^12** (productos < 2^24; si no, carril escalar exacto de 64 bits),
    caja de píxeles y rect de tiles en SIMD. Imagen idéntica píxel a píxel a la versión escalar (4 cámaras,
    incluida una con recorte cercano). Medido: total 1 hilo **18.21 → 16.94 ms**; 20 hilos mín 2.05 → 1.92 ms.
11. **Prefijos paralelos y contiguos** — `raster_internal.hpp:354`. Recorrer el contador slot-mayor en orden
    lista-mayor saltaba ~4.5 KB por acceso (una línea de caché nueva por contador): **0.13–0.25 → 0.06–0.09 ms**.
12. **Listas frontal/trasera por tile + atributos perezosos** — rect con bit 31 (`raster_internal.hpp:304`),
    bucle `raster_mesh.cpp:423`, atributos calculados sólo si algún bloque pasa la profundidad
    (`raster_mesh.cpp:501`). Bloques traseros sombreados en la malla de 496 k: 39 281 → 2.
    Raster 1 hilo 38.6 → 36.3 ms (−6 %).
13. **Hi-Z para las caras traseras** — `raster_mesh.cpp:416`–`466`. Tras las caras frontales se calcula una vez
    la profundidad máxima por bloque 8×8 del tile (~500 ciclos/tile); una cara trasera cuya profundidad mínima
    queda detrás de todos los bloques que toca se descarta **antes** de orientar/dividir. En la malla de
    300 k se descartan 155 000 de 318 000 referencias. Medido: 1 hilo **27.2 → 22.1 ms (−19 %)**, 20 hilos
    mín 4.43 → 3.80 ms (−14 %).
14. **Sombreado diferido por tile con G-buffer FP16 (F16C)** — `raster_mesh.cpp:41` (G-buffer por hilo, 25 KB),
    `:540` (`vcvtps2ph` al rasterizar), `:569` (pasada de sombreado sobre la máscara de cobertura de 64 bits por
    fila, bloques alineados y carriles llenos). Cada píxel visible se ilumina UNA vez.
    Medido: raster 1 hilo **16.3 → 15.2 ms (−6.5 %)**; 20 hilos mediana 2.58 → 2.39 ms.
    (Sólo en modo opaco; el translúcido sigue en directo porque necesita todas las capas.)
15. **Planificación LPT de tiles** — `raster_internal.hpp:386`: tiles ordenados por carga descendente antes del
    `parallel_for` dinámico (el tile más pesado empieza primero; importa en la CPU híbrida P/E).
    Medido: raster mediana 2.93 → 2.83 ms (300 k) y 4.12 → 4.00 ms (600 k), ~3 %.
16. **Prefetch** del `VOut` de dos triángulos por delante — `raster_mesh.cpp:452` (no medido por separado).
17. **Tiles de 64×64** (color 16 KB + depth 16 KB caben en L1d de 48 KB): comparado con 32×32 en ejecuciones
    emparejadas: 4.73 vs 6.45, 6.56 vs 8.58, 7.53 vs 9.11 ms → 64 px es **20–27 % más rápido**.

### Sombreado

18. **`pow(x, n)` ≈ `max(0, 1 − n(1−x)/8)^8`** (3 cuadrados en vez de exp/log) — `raster_mesh.cpp:386`.
19. **`vrsqrtps` sin Newton** para normalizar N, V y H (12 bits sobran para iluminación) — `raster_mesh.cpp:365`;
    **`vrcpps` + 1 Newton** para Z = 1/q — `raster_mesh.cpp:495`.
20. **Ambiente hemisférico + relleno desde la cámara sólo donde la luz principal no llega** (no sobreexpone)
    — `raster_mesh.cpp:392`: aspecto "estudio" sin coste extra apreciable.

### Mezcla de color y texturas

21. **SWAR en carriles de 16 bits** (`vpmullw`) para mezclar 8 píxeles ARGB: los pesos suman 256 → nunca
    desborda 16 bits — `raster_internal.hpp:194` (`blend_swar`), `:204` (`scale_swar`), `:188` (`alpha16`).
22. **Bilineal SWAR** con 4 `vpgatherdd` y pesos enteros que suman exactamente 256 — `raster_internal.hpp:224`.
23. **Suma saturada por canal** (`vpaddusb`, 32 bytes por instrucción) para partículas aditivas —
    `raster_prims.cpp:336`.

### Líneas y puntos

24. **Binning de segmentos por "cápsula"**: sólo cuentan los tiles cuya distancia al segmento ≤ R + ½ diagonal
    — `raster_prims.cpp:125` (evita repartir líneas largas a toda su caja).
25. **Intervalo exacto de la banda por fila** (`|n·(p−a)| ≤ R`) con 1/nₓ precalculado — `raster_prims.cpp:170`.
26. **Recorte 3D en el plano cercano + Liang–Barsky 2D** contra la tijera ampliada — `raster_prims.cpp:64`:
    coordenadas enormes o detrás de la cámara nunca llegan al recorrido.
27. **Sellos 4×2 para splats pequeños** — `raster_prims.cpp:341`. Medido (A/B): 20 hilos mediana
    **1.88 → 1.52 ms (aditivo), 2.01 → 1.59 ms (alfa)**; 1 hilo 13.0 → 10.25 ms (−21 %).
28. **Núcleo del splat (1 − r²/R²)²** sin sqrt ni exp — `raster_prims.cpp:328`.

### Suelo, cuadriláteros, postprocesos

29. **Suelo por intersección rayo/plano de 8 carriles** con ancho de línea por derivadas analíticas
    (dp/dsx = t·(R − D·R_z/D_z)), filas enteras sobre el horizonte descartadas con 2 comparaciones —
    `raster_fill.cpp:135`.
30. **Detección de bordes del FXAA en u8 saturado, 32 píxeles por instrucción** (`vpmaxub/vpminub/vpsubusb`)
    y filtro escalar sólo en los píxeles marcados, recorridos con `ctz` — `raster.cpp:205`.
31. **SSAO**: 12 cargas desalineadas por bloque de 8 y rango de oclusión relativo sin halos — `raster.cpp:94`.

## Experimentos descartados (medidos, sin ganancia)

| Idea | Resultado | Decisión |
|---|---|---|
| Ventana 8×8 sin ramas (8 sellos siempre, máscara de 64 bits + `ctz`) | raster 1 hilo 26.0 vs 25.6 ms (peor) | descartada |
| Tiles de 32×32 | 20–27 % más lento | 64×64 |
| `d²·rsqrt(d²)` en vez de `vsqrtps` en líneas | 16.4 vs 15.8 ms (sin ganancia) | se deja `sqrt` (`raster_prims.cpp:190`) |
| `vrcpps` en vez de 7 `vdivps` por bloque en el suelo | 3.68 vs 3.68 ms | se deja la división exacta (`raster_fill.cpp:198`) |
| Color constante en segmentos de color casi uniforme | 13.2 vs 13.1 ms | descartada |
| Bloques de línea sin ramas (mezclar y enmascarar siempre) | 20.0 vs 17.4 ms (peor: las ramas ahorran trabajo real) | descartada |

## Correctitud y pruebas (`build/raster/test_raster`)

42 comprobaciones, todas PASS (36 en `--quick`; tras la revisión, `--quick` pasa bajo ASan + UBSan
(+ float-cast-overflow, pointer-overflow) y bajo TSan sin informes; `test_raster_equiv` 30/30, también con ASan):
- cobertura de un cuadrado vs área proyectada: error −0.001 % (persp.) y −0.013 % (orto.), 0 dobles;
  culling de caras traseras correcto (esfera cerrada: todos los píxeles con exactamente 2 capas en doble cara);
- **estanqueidad**: rejillas perturbadas de 28 800 a 320 000 triángulos en 4 cámaras (persp./orto./rasante)
  → 0 grietas, 0 dobles impactos; también con triángulos recortados por el plano cercano;
- orden de profundidad de planos que se cortan en ambos órdenes de envío (18/18 muestras);
- cámara dentro de una esfera: 100 % del viewport cubierto; suelo gigante rasante recortado sin fallos;
- integral de cobertura de líneas AA = ancho (1.0/1.5/3.0/7.0 px → 0.996/1.494/2.996/6.996);
- centroide de un splat = proyección del punto (error < 0.002 px); aditivo exacto (63 → 126);
- UV perspectiva-correctas: frontera rojo/verde en la proyección del punto medio 3D (x = 960 vs 960.0,
  mientras que el punto medio de pantalla está en 2861);
- suelo: profundidad escrita = `Camera::project()` (619.5 vs 619.1), fila del cielo intacta, periodicidad
  exacta de la cinta;
- robustez: NaN, ±inf, 1e30, anchos/tamaños 1e9/NaN/negativos, índices fuera de rango, polilíneas con
  rangos inválidos, viewport mayor que el framebuffer: **0 píxeles escritos fuera del viewport** (color y depth);
- `draw_arrow` no pisa `last_mesh_stats()`.

Integración: `tests/test_flowvis.cpp` (del módulo flowvis) compila y pasa (128/128) contra esta implementación.

PNGs de inspección en `build/raster/`: `mesh_sphere_torus*.png` (con SSAO y FXAA), `mesh_wire.png`,
`mesh_alpha.png`, `mesh_ortho_side.png`, `lines_streamlines.png`, `points_smoke.png`, `textured_quad.png`,
`ground.png`, `bench_*.png`.

## Reproducir

```sh
# pruebas + benchmarks (≈ 10 s)
FLAGS="-std=c++23 -O3 -march=native -mtune=native -fno-plt -fno-semantic-interposition -fomit-frame-pointer \
  -funroll-loops -fno-math-errno -fno-trapping-math -fno-signed-zeros -ffp-contract=fast -fno-rtti \
  -fstrict-aliasing -DNDEBUG -pthread -Isrc"
g++ $FLAGS tests/test_raster.cpp src/render/raster*.cpp src/core/threadpool.cpp src/core/png.cpp -o build/raster/test_raster
./build/raster/test_raster            # --quick omite los benchmarks
# equivalencia rutas rápidas == rutas de referencia (todo el módulo con -DRZ_EXPERIMENTS)
g++ $FLAGS -DRZ_EXPERIMENTS tests/test_raster_equiv.cpp src/render/raster*.cpp src/core/threadpool.cpp -o build/raster/test_raster_equiv
# sanitizadores
g++ -std=c++23 -O1 -g -march=native -fsanitize=address,undefined,float-cast-overflow,pointer-overflow -pthread -Isrc \
  tests/test_raster.cpp src/render/raster*.cpp src/core/threadpool.cpp src/core/png.cpp -o build/raster-asan/test_raster
g++ -std=c++23 -O1 -g -march=native -fsanitize=thread -pthread -Isrc \
  tests/test_raster.cpp src/render/raster*.cpp src/core/threadpool.cpp src/core/png.cpp -o build/raster-tsan/test_raster
```

Instrumentación sin coste en el build normal:
- `-DRZ_STATS`: contadores por hilo (`rz::t_rzc`: setups, bloques, cubiertos, sombreados, histograma de cajas…).
- `-DRZ_EXPERIMENTS`: interruptores A/B en `rz::g_rz_exp` (bit 0 = filas en vez de sellos, 2 = sin Hi-Z,
  3 = binning escalar, 7 = puntos sin sellos, 10 = vértices escalares, 11 = sin orden LPT).
  Sin la macro, `RZ_EXP(b)` es la constante 0 y el compilador elimina las ramas.

## Límites conocidos

- Framebuffers de hasta 8128 px por lado (127 tiles por eje en el rect empaquetado).
- Las funciones comparten buffers persistentes: no llamarlas concurrentemente desde varios hilos.
- El modo translúcido (`alpha < 1`) no ordena por profundidad: por tile, primero las caras traseras y luego
  las frontales (corregido en la revisión; antes era al revés y dominaba la cara lejana), y dentro de cada
  grupo el orden de envío.
- Sin MSAA: las siluetas de malla se suavizan con `fxaa()` (≈ 0.9 ms a 1500×1150).

## Revisión independiente (adversarial)

Se releyó todo el módulo, se recompilaron y ejecutaron las pruebas (normal, ASan+UBSan, TSan) y se
buscaron defectos con sondas propias. Hallazgos (todos corregidos salvo donde se indica):

| # | Gravedad | Defecto | Arreglo |
|---|---|---|---|
| 1 | mayor | **Build unity roto** (`make unity`): `draw2d.cpp` define `k_inf` en un espacio anónimo de `cfd::render` y las funciones públicas con `using namespace rz` lo veían ambiguo (4 errores en `raster.cpp` / `raster_fill.cpp`). | `rz::k_inf` calificado (`raster.cpp:104`, `raster_fill.cpp:205`). La TU unity de todo el proyecto ya no da errores del raster (quedan dos de otros módulos: `models/objects.cpp` `deg`, `flowvis_lines.cpp` `Norm`). |
| 2 | mayor | **Malla translúcida con el orden de capas invertido**: se mezclaban las caras frontales y DESPUÉS las traseras encima → en una malla cerrada dominaba la cara lejana (esfera verde delante / roja detrás: centro R=108 G=54). | En modo Blend cada tile recorre primero la lista trasera y luego la frontal (`raster_mesh.cpp:423`). Ahora R=54 G=108. Prueba nueva. |
| 3 | mayor (latente) | **Binning de segmentos dependiente de redondeos**: el predicado flotante "¿toca la cápsula este tile?" se evaluaba inlineado en DOS sitios (conteo y reparto). Con `-ffp-contract=fast`, LTO o PGO cada copia puede fusionar FMAs/vectorizarse distinto → conteo ≠ reparto → listas de tiles corruptas (ids basura → lectura fuera de rango). No se reprodujo en este build, pero nada lo garantizaba. | `seg_touches_tile` es `noipa` (una sola copia compilada, `raster_prims.cpp:125`) y los rects de un tile no lo evalúan (`Bins::count_if/emit_if`, `raster_internal.hpp:325/341`). |
| 4 | menor | **Normales nulas → píxeles negros**: `rsqrt(0)=inf`, `0·inf=NaN` (malla con normales degeneradas, o normales que se anulan/infradesbordan en FP16): 100 % de los píxeles negros en la sonda. | `+1e-20` antes del `vrsqrtps` (`raster_mesh.cpp:365`). Prueba nueva. |
| 5 | menor | **Sin `mesh.nrm`, las caras traseras (doble cara) invertían la normal por defecto** "hacia la cámara" → quedaban oscuras (0x565D63 frente a 0xBABABA). | No se invierte si la malla no trae normales (`raster_mesh.cpp:484`). Prueba nueva. |
| 6 | menor | Margen del Hi-Z `zm·1.00001` con profundidad negativa (ortográfica, geometría detrás del ojo) quedaba del lado equivocado. | `zm + |zm|·1e-5` (`raster_mesh.cpp:466`). |
| 7 | menor | `raster_seg_tile` calculaba `depth.data() + y·stride` aunque no hubiera plano de profundidad (nullptr + desplazamiento = UB). | Sólo con `DepthTest` (`raster_prims.cpp:181`). |
| 8 | menor (latente) | Recorte cercano/banda de guarda: un vértice que la transformación AVX2 da por dentro (outcode 0) podía quedar "fuera" por 1 ulp con la Z escalar del recorte y sustituirse por un punto re-proyectado → posible grieta de 1 px con el vecino. | En `clip_polygon` los vértices con `orig ≥ 0` cuentan siempre como interiores (`raster_internal.hpp:643`). |

**Pruebas añadidas**: `test_raster.cpp` (orden translúcido, normales nulas/ausentes, FXAA y SSAO funcionales,
determinismo paralelo == serie de TODAS las primitivas —hash de color+depth idéntico—, estanqueidad con 389
triángulos recortados por un `znear` grande) y `tests/test_raster_equiv.cpp` (nuevo): cada ruta rápida frente a
su referencia vía `RZ_EXPERIMENTS` en 5 escenas (persp., una cara, rasante con recorte, orto+alambre,
translúcida): sellos vs filas, Hi-Z sí/no, binning SIMD vs escalar, sellos de puntos, orden LPT → **idénticos bit
a bit** (color y depth). Vértices AVX2 vs escalar no es bit a bit por diseño (orden de FMA; ≤ 43 píxeles de
color distintos y error relativo de depth ≤ 4.3e-4 en siluetas) y se prueba con tolerancia. Las 3 pruebas de
defectos fallan con el código original y pasan con el corregido.

**Rendimiento de los arreglos** (A/B con el código original reconstruido, binarios alternados):
1 hilo fijado a un P-core libre, mínimo de 12: malla 300 k 18.45/18.54 ms (orig.) vs 18.63/18.43 ms; polilíneas
50 k 14.3/14.4 vs 14.7/14.5 ms; 2 000 segmentos largos 22.2/22.9 vs 22.6/22.2 ms → **neutro** (imagen idéntica,
mismo hash). 20 hilos en una ventana tranquila (carga 4.5, `gradient` 0.06 ms): malla 300 k 2.24 ms (orig.) vs
2.18 / 2.32 ms (corregido).

**Afirmaciones del implementador verificadas** (20 hilos, carga ≈ 4.5): malla 300 k 2.18 ms (afirmado 2.19),
una cara 1.84 (1.83), 594 k 3.15 (3.12), 99.6 k 1.44 (1.42), polilíneas 1.45 (1.47), puntos 1.08/1.11
(1.10/1.19), suelo 1.03/0.71 (1.04/0.72), cuadrilátero 0.37 (0.34), SSAO 0.64 (0.68), FXAA 0.89 (0.85);
`test_flowvis` 128/128 contra este módulo. Con la máquina cargada (carga 13–25, otros agentes) los mismos
benchmarks salen 3–5× más lentos: los objetivos del test sólo avisan en ese caso.

**Riesgos que quedan (no corregidos)**:
- SSAO barato sobre superficies planas oblicuas: un suelo PLANO se oscurece 1–2.5 % en primer plano y hasta
  ~10 % cerca del horizonte (la diferencia de profundidad de un plano inclinado supera el umbral del 0.4 %).
  Arreglarlo exige un SSAO consciente del plano (p. ej. comparar con la extrapolación lineal de 1/z desde el
  vecino opuesto); se deja documentado.
- Juntas de polilíneas con extremos redondeados por segmento: el borde AA (y todo el trazo si el color es
  translúcido, p. ej. líneas detrás de un plano de corte translúcido) se mezcla dos veces en cada vértice.
- Los bloques de 8 píxeles que cruzan el borde del viewport se reescriben con el mismo valor (documentado en
  `raster.hpp`): no dibujar concurrentemente en los ≤ 7 píxeles vecinos del viewport.
