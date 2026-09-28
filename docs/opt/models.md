# Optimizaciones del módulo `models`

CPU: Intel Core Ultra 7 155H (Meteor Lake). Todas las medidas con `-O3 -march=native`, con carga
de fondo fuerte en la máquina (qemu + rustc, carga media 11-17 durante las medidas): por eso se
dan **mínimos de N repeticiones** (robustos a interrupciones) y medianas, con el hilo fijado a un
P-core (`taskset -c 2..4`) cuando se comparan variantes, y las variantes se **intercalan**.

La ruta caliente que consume la geometría es `sdf::Scene::eval` (módulo geom, no es de este
módulo): el voxelizador la llama una vez por celda (y 8 veces cerca de la superficie) y el
mallador otras tantas. Lo que este módulo controla es **cómo** se construye la escena (nº y tipo de
primitivas, vértices de los perfiles, compacidad de los grupos) y el coste de `models::build`, que
la app llama en cada cambio de parámetro.

## 1. Perfiles con 14 puntos por cara en vez de 40

`src/models/f1_common.hpp:33-43` (`k_profile_pts`, configurable con `-DCFD_MODELS_PROFILE_PTS`).

La distancia a un perfil (`Wing`) es O(aristas) (`sd_polygon_edges`). `Scene::naca4` usa 40
puntos por cara (81 aristas); a dx = 2-4 cm sobra: con 14 (29 aristas, espaciado coseno) el
polígono se separa del fino ≤ 1.0 mm en una cuerda de 0.35 m (medido; 0.03·dx a 3 cm).

| Medida (`tests/test_models_bench.cpp`, mín. de 5) | 40 puntos | 14 puntos | Efecto |
|---|---|---|---|
| eval en la región de los alerones, f1_2022 | 493 ns | 381 ns | **-23%** |
| eval en la región de los alerones, f1_2011 | 421 ns | 393 ns | -7% |
| eval media de los 18 modelos, puntos uniformes (mín. entre 3 rondas) | 211 ns | 169 ns | -20% |
| eval media, región de alerones | 189 ns | 146 ns | -23% |
| `build` f1_2022 (colocación de ranuras O(n²)) | 766 µs | 141 µs | **5.4×** |

## 2. Colocación de elementos con regula falsi de Illinois

`src/models/f1_common.cpp:83-102` (`place_next`).

La altura de cada flap se resuelve para que la **distancia mínima polígono-polígono** (vértice →
arista en ambos sentidos, exacta para polígonos disjuntos, `elem_elem_dist` en `:71`) sea
exactamente la ranura pedida. Cada evaluación cuesta O(n²). Bisección de 48 pasos → Illinois
(regula falsi modificada, convergencia superlineal, ~6-8 evaluaciones, tolerancia 1 µm):

| `build` (mín. de 9) | bisección | Illinois | Efecto |
|---|---|---|---|
| f1_2022 | 668 µs | 127 µs | **5.3×** |
| f1_2011 | 770 µs | 138 µs | 5.6× |
| f1_1979 | 523 µs | 88 µs | 5.9× |
| f1_wing_ge | 346 µs | 33 µs | **10×** |

Precisión verificada en el test: error de la ranura < 5·10⁻⁵ m en 12 combinaciones gap × ángulo.
Todos los coches se construyen ahora en 0.06-0.25 ms (objetos < 0.1 ms): no hace falta caché de
escenas entre cambios de parámetros.

## 3. Perfiles con espesor mínimo (robustez en red gruesa, no velocidad)

`src/models/f1_common.cpp:10-37` (`naca4_floor`, rampa √x en `:19`), constante en
`f1_common.hpp:52`.

No es un truco de velocidad sino de **corrección a dx = 3 cm**: un flap inclinado θ sólo queda
4-conexo si su espesor vertical ≥ dx·(1 + tan θ); si no, las celdas sólidas se tocan sólo por la
esquina y D3Q19 deja pasar poblaciones por los enlaces diagonales ("fuga" a través del ala). El
test lo mide con un BFS 2D de celdas sólidas del BA al BS (3 secciones × 4 desfases de rejilla por
elemento). Barrido del espesor mínimo (todas las F1):

| espesor mínimo | cortes con fuga |
|---|---|
| 1.5 cm | 50-70% |
| 2.4 cm | ~60% |
| 3.0 cm | 5-30% |
| 3.6 cm | 0 |
| **4.2 cm (elegido)** | **0** (margen) |

## 4. Caché de perfiles por construcción

`src/models/f1_common.cpp:39-47` (`ProfileCache::get`). Un handle por perfil distinto (clave m,
p, t, espesor mínimo, invertido): los elementos iguales comparten los datos del `pool` (menos
memoria y mejor localidad en `sd_polygon_edges`). No medido de forma aislada.

## 5. Geometría "amiga de las AABB"

* `mirror_y` en todas las piezas pares (ruedas, alas, endplates, brazos, bargeboards...):
  la mitad de primitivas y su AABB de primitiva se recorta a y ≥ 0 (`f1_common.cpp:116`, `:158`...).
* Grupos compactos (≤ 13 por coche, 26-50 primitivas): emulando el recorrido de `Scene::eval`
  sobre 10⁵ puntos uniformes se evalúan de media **3.3 grupos por punto** en f1_2022 y 2.9 en
  f1_2011 (el resto se descarta por AABB).
* El primer grupo es siempre el chasis (el más grande): da una cota `best` pequeña pronto.
  **Probado y descartado**: reordenar grupos por volumen de AABB (o al revés) no dio diferencia
  medible por encima del ruido (±10%).
* Cuerpo de Ahmed: la primera primitiva es la caja exacta y las cajas redondeadas prolongadas van
  como `Intersect` → AABB del grupo ajustada (antes 1.44 × 0.89 m; ahora 1.04 × 0.34 m).

## 6. Herramienta de vista previa (`tools/model_preview.cpp`)

* Recorte de cada rayo con la AABB de la escena por slabs (`:117`) y plano del suelo analítico.
* Paralelo por filas con planificación dinámica del pool (`:196`, grano 2): las filas con coche
  cuestan ~20× las de fondo, el reparto dinámico evita que los E-cores retrasen a los P-cores.
* "Sphere tracing" sobre-relajado (Keinert et al. 2014, ω = 1.6 con retroceso, `:133`):
  **medido sin ganancia** (evaluaciones ±3% en las 4 vistas de f1_2022 y f1_2011, porque dominan
  normales, AO y sombras suaves), así que el trazado normal es el de por defecto y el relajado sólo
  se usa con `--bench`. Al medirlo apareció un error clásico que se corrigió: un paso relajado
  que salta el final del intervalo salía del bucle sin validarse y se perdían piezas finas (los
  alerones delanteros del Lotus 79 desaparecían en la vista superior).
* Render completo de los 18 modelos (4 vistas 640×380 + cortes): ~10-15 s con 20 hilos.

## 7. Tests (`tests/test_models.cpp`)

* Voxelización de prueba paralela por filas (`:92`) con la misma regla que el voxelizador
  (`sdf(centro) < 0.12·dx`).
* `group_min_z` (`:47`): punto más bajo de un grupo con "sphere tracing" vertical por columnas
  (no se salta piezas finas, pasos grandes en el vacío), paralelo con acumuladores `Padded<float>`
  por hilo (sin falso compartir).
* Coste de eval: un hilo, mediana de 5 (el objetivo < 2 µs se cumple con holgura: 180-750 ns en
  coches según la carga, 15-600 ns en objetos); y throughput con los 20 hilos (21-160 ns/punto).

## 8. Revisión adversarial (segunda pasada, revisor independiente)

Todo lo anterior se reprodujo: tests 356/356 (ahora 384 + 131, ver abajo), ASan/UBSan y
ThreadSanitizer limpios (sólo fallan los umbrales de tiempo bajo instrumentación), los mismos
tests con los flags del Makefile (`-ffp-contract=fast -fno-signed-zeros -flto ...`) pasan, y el
coste de `Scene::eval` coincide (coches 240-310 ns/punto en un P-core fijado con carga 4-6; 14 vs
40 puntos por perfil: f1_2022 317-330 vs 402-413 ns y `build` 90-145 vs 590-1135 µs, intercalado).

### Defectos encontrados y corregidos

| # | Defecto | Arreglo | Verificación |
|---|---|---|---|
| 1 | **DRS / modo X cerraba ranuras**: el flap se giraba "hasta" -6° (DRS) o -3°/-6° (modo X) aunque ya estuviera más abierto por un incremento de flap negativo → se le AÑADÍA incidencia y su BA caía sobre el elemento anterior. Medido: f1_2026 modo X con flap -20°: flap +6° → -3°, ranura **45 → 12.6 mm** (se cierra a dx = 3 cm); f1_2014 DRS con flap -20°: -4° → -6°, ranura 39 mm | `open_flap()` (`f1_common.hpp:99`): sólo gira si el flap está más cerrado que el ángulo de apertura; usado en `f1_eras.cpp:243,508,509,524` | `test_models`: barrido flaps -20..+20 × DRS en todos los F1: 0 casos de incidencia añadida, 660 pares con ranura exacta 45 mm (cerrado) / ≥ 45 mm (abierto). Control negativo con el código anterior: 8 fallos, mínima 12.6 mm |
| 2 | **Costura en el plano de simetría**: una extrusión espejada que acaba justo en y = 0 (toda ala con raíz y0 = 0: principal delantero, traseros, beam wing, alas libres; láminas de difusor) son dos mitades que sólo se tocan → **sdf = 0 (no < 0) en todo su interior sobre y = 0**. Cualquier prueba `d < 0` (el mallador usa `d < 0.0f`, una voxelización sin engrosamiento, siembra de partículas) ve una lámina "fuera" | `k_sym_overlap` = 1 cm (`f1_common.hpp:58`): la mitad espejada empieza 1 cm por debajo de y = 0 (`add_elements`, `f1_common.cpp:113`; `sym_overlap` para `sheet_xz`/`plate_xz`, `:142`). Misma punta; efecto en la ley de curvatura < 0.01 mm | `test_models`: sdf < 0 en y = 0 donde a ±2 cm es < -1.5 cm (todos los modelos). `test_models_vox`: voxelizador real con `thicken = 0`, sin submuestreo y un plano de celdas en y = 0 exacto: 0 celdas fluidas. Control negativo: 12 modelos fallan; con el voxelizador real 28-38 celdas fluidas en f1_2022/f1_2011 y **63/63 (una rebanada de fluido completa) en naca4412_wing** |
| 3 | **`airfoil_2d` (spans_domain) no llegaba a las paredes**: `bounds_m` era la AABB de la escena, que la primitiva Wing amplía 0.02·cuerda → siguiendo el contrato (ny·dx = extensión Y de bounds_m) el dominio medía 1.04 m para un ala de 1 m: 2 cm de fluido junto a cada pared (≈ 1 celda a 3 cm) → flujo de punta en un caso "2D" | `build_airfoil2d` (`objects.cpp:133`): la geometría sobresale 0.1 m por lado y `bounds_m.y` = ±0.5 m exacto; `registry.cpp:102` respeta un `bounds_m` fijado por el constructor | `test_models_vox`: voxelizador real con ny·dx = bounds_m.y (ny = 32, 33, 50): sección sólida en todas las columnas |
| 4 | `f1_wing_ge`: `moment_ref_m` a nivel del suelo (z = 0) en vez del 25% de cuerda como las demás alas (el momento de cabeceo difería en D·h) | `objects.cpp:173`: 25% de cuerda del principal | test: \|sdf(moment_ref)\| < 2 cm en todas las alas |
| 5 | `resolve_params` dejaba pasar NaN (clamp_ no los filtra) → escena NaN y conversiones float→int indefinidas aguas abajo (voxelizador) | `registry.cpp:56`: no finitos → defecto | test NaN/inf |
| 6 | Vista previa: el rayo se recortaba con `bounds_m`; con `spans_domain` la geometría sobresale y los rayos laterales empezaban dentro del sólido (normales basura) | `model_preview.cpp`: recorte con `scene.bounds()` | inspección de `build/models/airfoil_2d.png` |

Coste de los arreglos: nulo en la ruta caliente. `Scene::eval` media de los 18 modelos (mín. de 9,
P-core fijado, pares intercalados, carga 25): sin solape 162-181 ns, con solape 166-184 ns (ruido).

### Verificación con el voxelizador REAL (`geom::voxelize`, submuestreo 2×2×2)

La primera pasada emulaba la regla del voxelizador; ahora `tests/test_models_vox.cpp` usa el real
(131 comprobaciones, ~1-2 s). Resultados medidos (3-4 desfases de red, 3 secciones por elemento):

| dx | grupos perdidos | cortes de ala con fuga | ranuras cerradas |
|---|---|---|---|
| 2.0 cm | 0 | 0 | 0 |
| **3.0 cm** (diseño) | 0 | 0 | 0 en posición normal, DRS y flaps -20°; **f1_2022 con flaps +20°: 1 de 27** (puente de 1 celda entre el BS romo del principal y el BA del flap a -34°; una ranura de 5.5 cm lo evita — medido — pero se mantuvo 4.5 cm para no agrandar todas) |
| 3.3 cm | 0 | 0 | 0 (antes de los arreglos, f1_2022 1 de 36: depende del desfase de las secciones) |
| 3.6 cm | 0 | f1_1998: 3 de 72 | f1_2022 1/36, f1_2026 2/36 |
| 4.0 cm | Ahmed: patas | 1-6 de 60-72 por coche | 0-3 de 24-36 por coche |

Conclusión: el criterio de diseño (dx = 3 cm) se cumple con el voxelizador real; a partir de
~3.3-3.6 cm empiezan a cerrarse ranuras y a 3.6 cm aparecen las primeras fugas a través de perfiles. `f1_wing_ge` (sin suelo
de espesor, cuerda 0.5 m) está pensado para dx ≈ 4-12 mm: a 6 mm ranura abierta y todos los
grupos presentes; a dx de coche (2-4 cm) su flap fuga (esperado).

## Resultados

`tests/test_models.cpp`: **384 PASS, 0 FAIL** en ~5-13 s; `tests/test_models_vox.cpp` (voxelizador
real): **131 PASS, 0 FAIL** en ~1-2 s. Coste medio de `Scene::eval` por punto
uniforme en la AABB (1 hilo; rango entre la ejecución más tranquila y la más cargada, carga media 1-15):

| modelo | grupos | prims | eval (ns) | build (ms) |
|---|---|---|---|---|
| f1_1967 | 5 | 26 | 156-204 | 0.01 |
| f1_1979 | 9 | 33 | 232-400 | 0.06-0.12 |
| f1_1998 | 12 | 43 | 243-467 | 0.09-0.17 |
| f1_2008 | 12 | 48 | 275-516 | 0.07-0.14 |
| f1_2011 | 11 | 44 | 252-742 | 0.10-0.17 |
| f1_2014 | 11 | 44 | 248-480 | 0.10-0.24 |
| f1_2019 | 13 | 50 | 272-544 | 0.10-0.18 |
| f1_2022 | 13 | 48 | 291-520 | 0.11-0.17 |
| f1_2026 | 11 | 47 | 280-579 | 0.11-0.23 |
| esfera / cilindro / cubo | 1 | 1 | 15-47 | < 0.001 |
| ahmed_25 | 2 | 6 | 45-97 | 0.001 |
| alas NACA / perfil 2D | 1 | 1 | 145-546 | 0.001 |
| f1_wing_ge | 3 | 3 | 193-597 | 0.03-0.08 |
| road_car | 3 | 13 | 93-162 | 0.002 |
