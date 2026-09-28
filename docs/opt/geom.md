# Módulo geom — voxelizador y malladores: optimizaciones, verificación y medidas

Archivos del módulo: `src/geom/voxelizer.hpp` (contrato, ampliado de forma compatible),
`src/geom/voxelizer.cpp` (SDF → `solid_id`), `src/geom/mesher.cpp` (`mesh_scene` con
*Surface Nets* y `mesh_voxels` con *greedy meshing*), `tests/test_geom.cpp` y (revisión)
`tests/test_geom_models.cpp` (catálogo real de `models/`).
CPU objetivo: Intel Core Ultra 7 155H (6P+8E, AVX2/FMA/BMI2, sin AVX-512).

## Resultados frente a los objetivos

Escena de prueba "tipo F1" (`make_f1`, `tests/test_geom.cpp:66`: 10 grupos, 27 primitivas,
~5.6 m: carrocería con taper boxes y cockpit restado, pontones, fondo, difusor con canal
restado, alerones NACA de 2 elementos con endplates, 4 ruedas con llanta restada, halo,
suspensión). Red de 320×128×96, dx = 0.03 m. Pool de 20 hilos.

| Operación | Objetivo | Medido (pared, 20 hilos, mediana) | CPU 1 hilo (mediana) |
|---|---|---|---|
| `voxelize` supersample=on | < 60 ms | **6.1 – 6.4 ms** (5.4 – 6.4 sin la ruta literal de la revisión, §7) | 59 – 60 ms (52 – 57 sin ella) |
| `voxelize` supersample=off | — | **2.0 – 2.1 ms** (1.8 – 2.4) | 19 ms (17 – 21) |
| `mesh_scene` cell_fraction 0.5 (h = 15 mm) | < 300 ms, < ~600k tris | **18.2 – 20.5 ms**, 333 248 tris (+≈1 ms de normales robustas, §7) | 162 – 181 ms (mín.; +7–10 ms de normales) |
| `mesh_voxels` (F1 voxelizado) | — | **0.26 – 0.39 ms**, 8 456 tris | 2.0 – 2.3 ms |

Los rangos son las medianas de distintas ejecuciones con carga de fondo diferente.
Sobre el catálogo **real** de `models/` (18 modelos: 9 épocas de F1, esfera, cilindro, cubo,
Ahmed, alas NACA, ala F1 en efecto suelo, perfil 2D, coche de calle), en red 320×128×96:
voxelización **idéntica celda a celda** a la fuerza bruta en los 18 (opciones por defecto
thicken 0.12 + submuestreo, y thicken 0 sin submuestreo) — **ojo**: esto dejó de cumplirse con los
modelos actuales (f1_1998: 56 celdas, f1_2008: 112) hasta la ruta literal de la revisión (§7); hoy
lo comprueba `tests/test_geom_models.cpp` en cada ejecución. Malla idéntica bit a bit al muestreo
completo, 0 aristas con orientación incoherente, 0 triángulos degenerados, ≤ 0.05 % de triángulos
con normal de vértice opuesta en los coches (0.64 % en el ala de 2 elementos en efecto suelo).
CPU 1 hilo (mín. de 7) en red 256×128×96: `voxelize` 0.07–112 ms por modelo, 940 ms los 18
(835 ms sin la ruta literal).

### Condiciones de medida (léase antes de las tablas)
La máquina tuvo carga de fondo alta y variable durante todo el trabajo (otros agentes
compilando y ejecutando benchmarks del LBM con 16 hilos; carga media 5–25). Un núcleo sintético
puramente de cómputo escaló sólo ×4.9 con 12 hilos en el peor momento y ×7.9 con 20 hilos en
el mejor. Por eso:
1. **Métrica principal: nº de llamadas a `Scene::eval_group`**, contadas con
   `-DCFD_GEOM_COUNTING` (`voxelizer.hpp:127`). Es determinista e independiente de la carga.
2. Tiempo de **1 hilo con `CLOCK_THREAD_CPUTIME_ID`** (CPU del hilo, inmune a desalojos),
   mediana de ≥5; comparaciones A/B **intercaladas** en 3 rondas y se toma el mínimo por variante.
3. Diferencias < 5 % en tiempo se consideran ruido (variantes con código idéntico variaron ±4 %).
4. El tiempo de pared con 20 hilos se da como referencia; escala ×8–9 respecto a 1 hilo, igual
   que el núcleo sintético en el mismo momento, así que el paralelismo no es el cuello.

Reproducir: `g++ -std=c++23 -O3 -march=native <flags del Makefile> -Isrc -pthread
tests/test_geom.cpp src/geom/{voxelizer,mesher,sdf}.cpp src/core/{threadpool,png}.cpp`
y ejecutar con `--benchonly` / `--benchonly-full`; variantes A/B con `-DCFD_GEOM_PRUNE=1|2`,
`-DCFD_GEOM_VOTE_EARLY=0`, `-DCFD_GEOM_SS_NARROW=0`, `-DCFD_GEOM_VERT_SUB=0`,
`-DCFD_GEOM_SIMD=0` (por defecto siempre la variante optimizada).

---

## 1. `RegionEval`: evaluación de la escena restringida a una región, exacta

`voxelizer.hpp:143-238`. Base de todo el módulo: el coste está en el SDF (`sdf.cpp`, que no es
de este módulo), así que la mejor optimización es **no evaluar**. `RegionEval` reproduce
exactamente el pliegue de `Scene::eval` (grupos en orden de índice, salto si
`dist(caja, p) ≥ mejor`, actualización si `d < mejor`, `voxelizer.hpp:196`) sobre una lista
de candidatos podada para una caja. Tres podas, todas demostradas exactas:

* **(a)** `dist(caja_g, región) ≥ min UB_h` (h anteriores conservados, `UB = d_h(c) + L·R`):
  `Scene::eval` salta g siempre (`voxelizer.hpp:185`).
* **(b)** cota inferior **heredada del nodo padre** `LB_g = dc_padre − L·R_padre ≥ min max(UB_h,0)`:
  g se evaluaría pero nunca bajaría el mejor (`voxelizer.hpp:184`). Sin evaluar g.
* **(c)** testigo de **cualquier índice**: `LB_g > max(UB_w, 0)` con w conservado de menor UB
  heredada (`voxelizer.hpp:183`). Demostración por inducción sobre el pliegue: si `d_g > 0`,
  quitar g sólo cambia el resultado si `d_g` fuese el mínimo final, y el final nunca supera
  `max(d_w, 0)`. Es la única regla que puede quitar el **primer** grupo (la carrocería), que
  `Scene::eval` evalúa en todos los puntos.

Además `classify_id` / `classify_sign` (`voxelizer.hpp:214`, `:228`) deciden si una región es
uniforme respetando la semántica real de `Scene::eval`: cuando un punto está dentro (d < 0) de
un grupo, los posteriores se saltan (su distancia de caja 0 ≥ mejor < 0), así que **gana el
primer grupo en orden de índice**, no el más profundo (esto se descubrió porque el fondo plano,
grupo 2, y la rueda trasera, grupo 7, se solapan: la primera versión asignaba el id de la
rueda en 74 celdas; el test de identidad con la fuerza bruta lo detectó).

| Podas activas | voxelize ss=1: evals | CPU 1 hilo | voxelize ss=0: evals | CPU 1 hilo | mesh_scene: evals | CPU 1 hilo |
|---|---|---|---|---|---|---|
| sólo (a) `-DCFD_GEOM_PRUNE=1` | 1 096 641 | 84.6 ms | 346 010 | 27.1 ms | 3 397 613 | 274 ms |
| (a)+(b) `=2` | 1 019 475 | 79.9 ms | 315 850 | 25.0 ms | 3 167 730 | 252 ms |
| **(a)+(b)+(c)** | **712 941** | **54.9 ms** | **214 092** | **18.0 ms** | **2 042 473** | **178 ms** |

(evals = refine + puntos; mesh_scene medido con L = 1.5; con el L = 1.25 final son 1 916 354.)
En hojas 2³ del mallador los candidatos por nodo bajaron de 2.8 a 1.29.

## 2. Voxelizador (`src/geom/voxelizer.cpp`)

| # | Truco | Dónde | Efecto medido |
|---|---|---|---|
| V1 | Recorrer sólo la caja de la escena + margen `max(thr,0)+dx` (fuera: d > margen ⇒ fluido y submuestreos fluidos); el resto a 0 con `memset` por filas, cada celda z ≥ z_min escrita una sola vez | `voxelizer.cpp:281`, `:296` | fuerza bruta en toda la red 1 469–3 569 ms CPU → Scene::eval por celda sólo en la caja 186–228 ms CPU (×8–16) |
| V2 | **Salto jerárquico de bloques** 8³→4³→2³→celda: por nodo, cotas `d_h(c) ± L·R` → todo fluido / todo sólido con id único / subdividir; bloques 8³ alineados (fila de 8 celdas = 1 u64) repartidos dinámicamente (`parallel_for` grano 1) | `voxelizer.cpp:152-215`, `:325` | caja por celda 186 ms → **53.5 ms** CPU (×3.5); pared 26.7 → 6.0 ms. ss=0: 141 → 17 ms CPU (×8) |
| V3 | Submuestreo 2×2×2 sólo donde el voto **puede** diferir del centro: `|d − thr| < L·√3/4·dx` (y `|d| < dx`, definición del contrato) — fuera de esa banda los 8 submuestreos caen del mismo lado (Lipschitz) | `voxelizer.cpp:126`, `:254` | evals 864k → 633k; CPU 65.0 → 54.9 ms (−16 %) |
| V4 | **Voto con salida anticipada**: para en cuanto hay 4 dentro o 5 fuera (resultado idéntico a contar 8) | `voxelizer.cpp:130-134` | evals 784k → 633k; CPU 71.6 → 54.9 ms (−23 %). V3+V4 juntos: 88.6 → 54.9 ms (−38 %) |
| V5 | `RegionEval` (sección 1) por nodo, con cotas heredadas | `voxelizer.cpp:188-190` | ver tabla de la sección 1 (−35 % evals) |
| V6 | Relleno SWAR de filas de bloque (8 bytes = un `u64 = 0x0101…·v`) | `voxelizer.cpp:97` | no aislado (< 1 %: el coste está en el SDF) |
| V7 | Desconmutación del bucle por plantilla (`template <bool SS>`) | `voxelizer.cpp:151` | no aislado |
| V8 | Acumuladores por hilo `Padded<Acc>` indexados por `worker_index()` (sin atómicos ni false sharing) | `voxelizer.cpp:312` | escalado ×9 (53.5 ms CPU → 6.0 ms pared) |
| V9 | (Revisión) **Ruta literal** sólo en las cajas de primitivas restadas mientras `sdf.cpp` tenga el bug de culling (sonda de una vez; cajas ajustadas por mitad con espejo) | `voxelizer.cpp:164-186`, `:224`, `:262` | exactitud recuperada (56/112 celdas → 0 en f1_1998/f1_2008); coste +13 % CPU (escena de prueba), 0 con `sdf.cpp` corregido; cajas por mitad: +30 % → +13 % |

**Constante de Lipschitz.** Las primitivas "cota" de `sdf.cpp` (taper box con estrechamiento,
alas con flecha/estrechamiento, elipsoide de iq) superan |∇d| = 1. Barrido sobre las 19 escenas
(F1 de prueba + 18 modelos reales), CPU 1 hilo sumado:

| L | 1.0 | 1.1 | 1.25 | **1.5 (defecto)** | 2.0 |
|---|---|---|---|---|---|
| CPU 19 escenas | 562 ms | 671 ms | 675 ms | 850 ms | 1 733 ms |
| celdas ≠ fuerza bruta | 0 | 0 | 0 | 0 | 0 |

Se mantiene 1.5 en el voxelizador (determina la física y no tiene autocorrección): +50 % de
coste frente a 1.0 a cambio de margen ante primitivas más agresivas que las de hoy.

## 3. Malla suave `mesh_scene` (`src/geom/mesher.cpp`)

Surface Nets (dual contouring sin QEF): 1 vértice por cubo con cambio de signo, en la media de
los cruces por cero de sus 12 aristas, proyectado sobre la superficie; 1 quad por arista de la
rejilla con cambio de signo, orientado por el signo, partido por la diagonal más corta.

| # | Truco | Dónde | Efecto medido |
|---|---|---|---|
| M1 | **Muestreo en banda estrecha** jerárquico 8³→4³→2³ con `RegionEval` y clasificación de signo | `mesher.cpp:122-159`, `:245-252` | muestreo completo (4.87 M evals `Scene::eval`) 1 141–1 292 ms CPU / 182–242 ms pared → **170–227 ms CPU / 19–27 ms pared** (×6–7 CPU); muestras evaluadas 4.87 M → 0.38 M (7.8 %) |
| M2 | **Sin halo + relleno de esquinas**: las hojas se clasifican sin halo (banda 3× más fina) y una pasada evalúa exactamente las muestras que son esquina de un cubo mixto y faltan: `needed = dilat(M) & ~E` con palabras u64 | `mesher.cpp:326-384` | con halo (1.00 M evals) 222–406 ms CPU → sin halo 0.38 M evals, 170–227 ms CPU (misma malla bit a bit) |
| M3 | **Autocorrección de signos**: si el valor exacto de una esquina contradice el bit clasificado, se corrige y se repite cubos→esquinas hasta el punto fijo | `mesher.cpp:366-384` | coste 0 en el caso común. Con L = 1.0 las 19 mallas pasan a ser idénticas al muestreo completo (antes: 9 de 19 distintas; con L = 1.25: 1 de 19). 3 correcciones en total con L = 1.25 |
| M4 | **Signos en bitboards** (1 bit/muestra, fila X en u64, un byte por bloque 8³ ⇒ cada bloque escribe bytes propios: sin carreras ni atómicos) y **cubos mixtos de 64 en 64** con AND/OR de 4 filas + desplazamiento con acarreo (SWAR) | `mesher.cpp:280-305` | fase "cubos" 0.1–0.3 ms (20 hilos) para 4.87 M muestras |
| M5 | **Índice de vértice por rango**: `P[palabra] + popcount(BZHI(M, i&63))` (BMI2) en lugar de una rejilla densa de índices u32 | `mesher.cpp:171-173` | memoria 19.5 MB → 0.30 MB (+1.8 MB de bitboards S/E/M); con h = 7.5 mm: 132 MB → 2.1 MB |
| M6 | Salida **escrita en su sitio**: conteo por fila → suma prefija → emisión (vértices y triángulos); determinista, sin vectores por hilo ni fusión | `mesher.cpp:306-316`, `:512-584` | fase de quads 0.7–0.9 ms (333k triángulos) |
| M7 | Valores del SDF en **ranuras de 2 KB** (bloque 8³ contiguo) sólo para bloques con superficie (contador atómico, un dueño por bloque) | `mesher.cpp:84-118` | 19.5 MB → 4.9 MB (2 393 ranuras); h = 7.5 mm: 132 MB → 22.7 MB |
| M8 | Vértices por **sub-bloques de 2³ cubos** con poda heredada del bloque 8³ | `mesher.cpp:425-442` | evals 2.53 M → 2.04 M (−19 %); tiempo −2 % (dentro del ruido: el refine extra se come casi toda la ganancia) |
| M9 | **1 paso de proyección** por defecto: el mismo tetraedro (4 evals, Σk = 0, Σkkᵀ = 4I) da d y ∇d → proyección y normal | `mesher.cpp:479-495` | 2 pasos 250–301 ms CPU → 1 paso 170–227 ms; error medio \|d\| en vértices 0.005 → 0.016 mm (h = 15 mm) |
| M10 | Rejilla desplazada **φ−1 muestras**: la caja de la escena toca la superficie en puntos *tangentes*; si un plano de muestras coincide con ellos (desfase 0.007 muestras en el toro) aparecen caras ambiguas (tablero de ajedrez) → aristas no-manifold | `mesher.cpp:199` | toro: 39 aristas no-manifold y 46 triángulos invertidos → 0 |
| M11 | Tablas `constexpr` (12 aristas del cubo, tetraedro) | `mesher.cpp:61-71` | — |
| M12 | Partición del quad por la diagonal más corta | `mesher.cpp:560` | área mínima de triángulo 0.03 celdas² en la esfera, 0 degenerados |
| M13 | (Revisión) **Normales robustas**: normal de caras reunida (gather) desde los bitboards, sin scatter; sustituye al gradiente si cos < 0.5 | `mesher.cpp:586-647` | triángulos con normal de vértice opuesta 1.30 % → 0.14 % (F1); +7.4–8.3 ms CPU 1 hilo, ≈ +1 ms pared |

Escalado con la resolución (pared, 20 hilos): cell_fraction 1.0 → 8.1 ms / 75k tris; 0.5 → 26 ms /
333k; 0.35 → 55 ms / 684k; 0.25 → 106 ms / 1.34 M.

## 4. Malla de vóxeles `mesh_voxels` (`src/geom/mesher.cpp`)

Planos (eje, p) como tareas paralelas; por plano se construyen dos máscaras 2D (normal ±eje, valor
= id de grupo) y se fusionan rectángulos del mismo id (greedy meshing); 4 vértices y 2 triángulos
por rectángulo, orden determinista por suma prefija sobre las tareas.

| # | Truco | Dónde | Efecto medido |
|---|---|---|---|
| G1 | Máscaras de cara con **AVX2** (32 celdas por instrucción: `cmpeq`/`andnot`/`and`), búsqueda de tramos y de la siguiente celda no vacía con `movemask` + `tzcnt`, caja de sólidos con `movemask` + `tzcnt/lzcnt` | `mesher.cpp:670-718`, `:760` | escalar (`-DCFD_GEOM_SIMD=0`) 3.63 ms → **2.09 ms** CPU 1 hilo (×1.74); pared 0.45 → 0.25 ms |
| G2 | Greedy meshing (rectángulos coplanares del mismo grupo) | `mesher.cpp:720` | F1: 86 028 → **8 456** triángulos (×10.2); con el suelo: 250 212 → 8 474 (×29.5) |
| G3 | Sólo la caja de celdas sólidas; espacio de trabajo por llamador persistente (sin reservas tras la 1ª llamada) | `mesher.cpp:779` | — |

## 5. Verificación (`tests/test_geom.cpp`, < 10 s, `PASS/FAIL`, código ≠ 0 si falla)

Voxelizador:
* Esfera r = 10 celdas (centro no alineado), thicken = 0: volumen +0.34 % (ss=0) y +2.11 %
  (ss=1) frente al analítico (±3 %). Con thicken = 0.12 por defecto el volumen crece +5.5 %,
  como se espera (3·0.12/10).
* Identidad salto de bloques == fuerza bruta (`Scene::eval` en **toda** la red, definición
  literal del contrato con submuestreo |d| < dx): escena F1 con restas y thicken 0 (ss 0 y 1),
  escena F1 sin restas y thicken 0.12 (ss 0 y 1) y, desde la revisión, escena F1 **con restas y
  opciones por defecto** (antes sólo INFO): **0 celdas distintas** de 3.9 M.
* (Revisión) Restas que sobresalen de la base con thicken 0.12 y 0.3, ss 0 y 1, 12 escenas con
  desplazamientos aleatorios (48 casos), ruta rápida y ruta de referencia `block_skip=false`:
  0 celdas distintas (307 sin la ruta literal: el test falla con el voxelizador anterior).
* Simetría especular en y: 0 celdas asimétricas. Ids de 2 objetos separados correctos.
  Capa del suelo (z < z_min, 255) intacta; toda la red z ≥ z_min escrita (sin basura).
  `VoxelStats` coherente.

`mesh_scene`:
* Esfera (r = 20 muestras) y toro (R = 0.4, r = 0.15 m): estancas (toda arista en exactamente
  2 triángulos, sin aristas dirigidas repetidas, sin vértices sueltos), normales hacia fuera
  en todos los vértices y triángulos (CCW visto desde fuera), área −0.048 % y −0.030 % de la
  analítica (±2 %), \|d\| máx en vértices 8e-6 m y 1e-4 m, 0 triángulos degenerados.
* F1: < 300 ms y < 600k triángulos, orientación coherente (0 aristas manifold recorridas en el
  mismo sentido), ids de grupo válidos, malla **idéntica bit a bit** al muestreo completo.
* (Revisión) Normales: esfera y toro con 0 normales sustituidas (superficie suave intacta); F1
  con < 0.33 % de triángulos cuya normal de vértice se opone a la geométrica (medido 0.142 %;
  antes 1.30 %).
  Aristas no-manifold: 291 de ~500k (0.06 %), sólo en piezas más finas que el paso (endplates de
  2 cm, placas del fondo): limitación inherente de Surface Nets con un vértice por cubo.

`mesh_voxels`: esfera voxelizada, F1 (con y sin suelo) y una red aleatoria con ids distintos
adyacentes y sólidos tocando los bordes: los rectángulos cubren **exactamente** el conjunto de
caras unitarias ingenuas (misma orientación e id), superficie cerrada (volumen por divergencia =
nº de celdas exacto, Σ n·A = 0), menos triángulos que las caras ingenuas.

`tests/test_geom_models.cpp` (revisión; enlaza `src/models/*.cpp`, ≈ 5 s con 20 hilos): los 18
modelos reales — voxelización == fuerza bruta (por defecto y thicken 0 sin ss), malla ≡ muestreo
completo (posiciones, normales, triángulos), 0 incoherentes, 0 degenerados, ids válidos, sin
NaN, < 1 % de triángulos con normal de vértice opuesta.

Sanitizers: ASan+UBSan sin errores; ThreadSanitizer sin avisos (incluida la ruta de
autocorrección, forzada con L = 1.0 sobre los 18 modelos). Compila como unidad única (unity).
Revisión: repetido tras los cambios (`build/geom-asan/`, `build/geom-tsan/`): ASan+UBSan limpio
en `test_geom` y `test_geom_models`, TSan 0 avisos (sólo fallan, como es de esperar, los umbrales
de tiempo bajo instrumentación). Unity (geom + core + test) compila y pasa.
Imágenes de control con `--png` en `build/geom/*.png` (mini-rasterizador ortográfico del test).

## 6. Problemas encontrados en `sdf.cpp` (no es de este módulo; propuesta al líder)

1. **Resta que sobresale de la pieza base** (`Scene::eval_group`, `Op::Subtract`: `if (bd > -d)
   continue;`). Si el punto está fuera de la base (d > 0) la resta se descarta aunque esté dentro
   de lo restado: el SDF vale `d_base` en vez de `max(d, −e)`. El signo es correcto, pero el campo
   es discontinuo y, con thicken > 0, aparecen **tapas fantasma** de grosor thicken·dx sobre las
   aberturas (cockpit, canales). Ej.: esfera r=1 menos caja que sobresale, `eval(0,0,1.02)` = 0.02
   en vez de 0.30 (lo imprime el test como INFO). Corrección: `bd > max_(-d, 0.0f)` y, en
   `SmoothSubtract`, `bd > max_(pr.k - d, 0.0f)`.
2. **Unión con un punto ya dentro** (`Op::Union: if (bd >= d) continue;`, y lo mismo entre grupos
   en `Scene::eval`): con d < 0 se descartan todas las primitivas/grupos posteriores aunque el
   punto esté más dentro de ellos. El signo es correcto, pero la magnitud interior es la de la
   primera pieza (discontinua; |∇d| ≈ 38 medido en el fondo escalonado de `f1_2019`), lo que
   invalida cotas de Lipschitz y estropea normales justo en las uniones. Corrección exacta:
   saltar sólo si `bd > max(d, 0)`. Si se cambia la regla **entre grupos** en `Scene::eval`, hay
   que cambiar igual `RegionEval::eval` (`voxelizer.hpp:196`) y revisar las podas (b)/(c).

Con ambas correcciones el SDF sería Lipschitz-1 salvo las primitivas "cota", y el voxelizador
podría bajar a L ≈ 1.1–1.25 (−20 a −35 % de tiempo según el barrido de la sección 2).

La corrección 1 es la importante: mientras no se aplique, `voxelize` paga la ruta literal (§7,
+13 % CPU en la escena de prueba, +12.6 % en el catálogo) y las normales del SDF salen invertidas
junto a las restas que sobresalen (difusor de la escena de prueba: 13 % de sus triángulos). Se
comprobó con una copia local corregida de `sdf.cpp` (no se modificó el archivo): la sonda
`sdf_subtract_culling_bug()` pasa a false, la ruta literal se apaga sola (0 celdas) y todos los
tests siguen pasando.

## 7. Revisión independiente (adversarial)

Hallazgos, todos reproducidos antes de corregir y con test de regresión:

1. **Voxelizador distinto de la definición con las opciones por defecto** (mayor). Con thicken > 0
   y restas que sobresalen de la base, el bug de culling de `sdf.cpp` hace el SDF discontinuo
   *en valores positivos* (tapa fantasma ≈ 0 junto a hueco tallado ≫ 0) dentro de la caja de la
   primitiva restada; las cotas de Lipschitz del salto de bloques y el atajo de banda estrecha del
   submuestreo dejaban de valer. Medido con el catálogo actual: f1_1998 56 celdas y f1_2008 112
   celdas distintas de la fuerza bruta (thicken 0.12); escena mínima: 307 celdas en 48 casos.
   Análisis: fuera de esas cajas los valores exteriores de `Scene::eval` son exactos (el culling de
   uniones sólo falsea el interior, d < 0), y con thr ≤ 0 sólo cuenta el signo, que es correcto →
   la clasificación sólo falla con thr > 0 y dentro de las cajas de restas.
   **Corrección — ruta literal** (`voxelizer.cpp:164-186`, `:224-239`, `:262-275`): una sonda de una
   vez detecta el bug (`sdf_subtract_culling_bug()`, esfera menos caja que sobresale, Subtract y
   SmoothSubtract); si está presente y thr > 0, los nodos cuya región toca la caja de una primitiva
   restada (box_frame y su reflejo por separado para piezas con espejo: la caja conjunta de las
   llantas abarcaba todo el ancho del coche, +30 % → +13 %) no se clasifican ni heredan cotas: sus
   hijos parten de la raíz y las hojas aplican la definición literal con `Scene::eval` (voto en
   toda la banda |d| < dx). La ruta `block_skip=false` también pasa a ser literal. Coste medido
   (A/B intercalado): escena de prueba 52–54 → 59–60 ms CPU 1 hilo, 5.4 → 6.1–6.4 ms pared; catálogo
   835 → 940 ms CPU (1 hilo, suma de 18). Desaparece al corregir `sdf.cpp`.
   `VoxelStats::literal_cells` cuenta las celdas por esa ruta.
2. **Normales de vértice invertidas en `mesh_scene`** (mayor). 4 332 triángulos de la escena F1
   (1.30 %) con la normal de vértice opuesta a la geométrica; análisis por vecinos: sólo 69 son
   pliegues reales, los otros 4 263 tienen geometría correcta y **gradiente del SDF erróneo**
   (2 880 en el difusor: resta que sobresale → SDF discontinuo; el resto en piezas más finas que
   h). `flowvis::color_mesh` muestrea Cp en `pos + n·offset`: con la normal invertida muestrea
   dentro del sólido (gris "sin dato" o Cp de otra zona).
   **Corrección — normales robustas** (`mesher.cpp:586-647`): por vértice, suma de las áreas
   vectoriales de sus quads (se reúnen desde los bitboards de signo: 8 bits de esquina → aristas con
   cambio → rangos de los 4 vértices; sin scatter ni atómicos); si el gradiente forma con esa normal
   un coseno < `MeshOptions::normal_min_dot` (0.5), se sustituye. Esfera/toro/cubo/cilindro/Ahmed:
   0 sustituciones. F1 de prueba: 4 332 → 473 triángulos (0.14 %; 2 786 normales sustituidas);
   f1_2019 2 063 → 221; alas NACA 178 → 0. Coste: 7.4–8.3 ms CPU 1 hilo (≈ 5 %), ≈ 1 ms de pared;
   leer los 8 bits de esquina una vez en lugar de 24 consultas bajó la fase de 9.6–10.7 a 7.4–8.3 ms.
3. (menor) La ruta de referencia `block_skip=false` usaba el atajo de banda estrecha del
   submuestreo: ya no era la definición literal en zonas discontinuas. Corregido.
4. (menor, no corregido) `mesh_voxels` produce uniones en T (greedy meshing): cerrada
   geométricamente (verificado) pero no estanca en el sentido "arista = 2 triángulos"; un
   rasterizador sin ajuste exacto de vértices puede mostrar grietas de un píxel.
5. (limitación documentada) Con thicken < 0 la identidad con la fuerza bruta no está garantizada:
   el culling de uniones de `sdf.cpp` hace el SDF interior discontinuo. En la escena de prueba con
   thicken −0.3 coincidió (0 celdas), pero no se garantiza.

Afirmaciones del informe original comprobadas: tiempos (voxelize 5.9 ms / 2.0 ms, mesh_scene
18.2–18.8 ms, mesh_voxels 0.26–0.31 ms, 333 248 y 8 456 triángulos), identidad con la fuerza
bruta en la escena de prueba, malla bit a bit igual al muestreo completo en las 19 escenas,
sanitizers limpios, demostraciones de las podas (a)/(b)/(c) y de `classify_id`/`classify_sign`
(revisadas: correctas bajo sus hipótesis). **No** se sostenía con el catálogo actual: "0 celdas
distintas en los 18 modelos reales" (ver hallazgo 1). Además se probaron casos límite sin fallos:
redes no múltiplo de 8 y escena recortada por la red, z_min ∈ {−5, 0, 1, 3, 7, 8, 9, 15, 29, 40},
escena fuera de la red, red 1×1×2, escena vacía y grupo vacío intercalado, thicken 2.5 y −0.3,
guiñada ±10° y rake 3°, dx 0.015 y 0.08, 200 grupos solapados (voxel y malla), cell_fraction 0,
−1 y 3, max_samples = 100k, halo == sin halo.
