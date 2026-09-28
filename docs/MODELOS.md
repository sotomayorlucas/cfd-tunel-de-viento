# Catálogo de modelos (módulo `models`)

Modelos paramétricos construidos como `sdf::Scene` en **espacio modelo** (metros): suelo en
`z = 0`, el aire va hacia **+X**, el coche mira hacia **-X**, eje delantero en `x = 0` (punto de
contacto) y trasero en `x = batalla`, simétricos respecto a `y = 0`.

* Contrato: `src/models/model.hpp` (`count`, `info`, `find`, `build`, `car_body_frame`,
  `car_wheel_frame`, y las extensiones `resolve_params`, `same_geometry`, `kind_name`).
* Implementación: `registry.cpp` (catálogo, parámetros, marcos), `f1_common.{hpp,cpp}` (piezas
  comunes), `f1_eras.cpp` (coches por época), `objects.cpp` (cuerpos y alas de referencia).
* Vista previa: `tools/model_preview.cpp` → `build/models/<id>.png` (4 vistas) y
  `build/models/<id>_vox.png` (cortes a dx = 3 cm).
* Tests: `tests/test_models.cpp` (384 comprobaciones, ~5-13 s), `tests/test_models_vox.cpp` (131, con el
  voxelizador real `geom::voxelize`; compilar con `src/geom/voxelizer.cpp`), `tests/test_models_bench.cpp`.

## Coches de F1

Dimensiones medidas sobre la geometría construida (AABB con ruedas y alerones; alto = punto más
alto sobre el suelo con la altura de marcha por defecto). Carga (`SCz = Cl·A`, + hacia abajo) y
resistencia (`SCx = Cd·A`) de referencia en m²: **estimaciones de orden de magnitud** (los equipos
no publican datos), útiles sólo para comparar tendencias entre épocas.

| id | Época | Largo × ancho × alto (m) | Batalla | Altura del./tras. (mm) | Rasgos modelados | Parámetros | SCz / SCx ref. |
|---|---|---|---|---|---|---|---|
| `f1_1967` | 1966-67 (Lotus 49) | 4.06 × 1.83 × 0.91 | 2.413 | 100 / 110 | Carrocería "puro" con radiador en el morro, V8 DFV portante expuesto, escapes, ruedas al aire, piloto erguido, arco antivuelco. Sin alerones (ligera sustentación). | altura, guiñada, ruedas | -0.20 / 0.75 |
| `f1_1979` | 1977-82 (Lotus 79) | 4.64 × 2.07 × 1.09 | 2.72 | 50 / 70 | Pontones con intradós de **ala invertida** (Venturi: entrada, garganta, salida delante de las ruedas traseras), **faldones deslizantes** que rozan el suelo (marco de ruedas: siguen al asfalto con cualquier altura), aletas delanteras a los lados del morro, alerón trasero sobre pilar. | altura, flaps, guiñada, ruedas | 2.4 / 0.95 |
| `f1_1998` | 1998-2008 (vía estrecha) | 4.78 × 1.80 × 1.14 | 2.94 | 30 / 60 | 1.8 m, morro alto con alerón delantero colgado de pilones, fondo plano **escalonado** (referencia + escalón de 50 mm) con plank, difusor corto, alerón trasero alto de **3 elementos** + plano inferior, bargeboards simples. | altura, flaps, guiñada, ruedas | 2.9 / 1.05 |
| `f1_2008` | 2006-08 (apéndices) | 5.02 × 1.78 × 1.06 | 3.12 | 25 / 70 | Morro alto en V, alerón delantero de 1.4 m, **bargeboards**, **winglets** sobre pontones, **chimeneas**, **"cuernos"**, **flip-ups**, **aleta de tiburón**, alerón trasero de 1 m + beam wing. | altura, flaps, guiñada, ruedas | 3.6 / 1.25 |
| `f1_2011` | 2009-13 | 5.20 × 1.79 × 1.20 | 3.25 | 25 / 85 | Alerón delantero de **1.8 m** con sección central neutra (flaps sólo exteriores con valla), alerón trasero **estrecho (0.75 m) y alto** con **DRS**, beam wing, difusor simple, morro alto, rake alto. | altura, flaps, DRS, guiñada, ruedas | 3.8 / 1.20 |
| `f1_2014` | 2014-16 (híbridos) | 5.38 × 1.79 × 1.15 | 3.45 | 30 / 75 | **Morro bajo con "dedo"**, alerón delantero de **1.65 m**, alerón trasero menos profundo **sin beam wing**, difusor más corto, DRS. | altura, flaps, DRS, guiñada, ruedas | 3.3 / 1.10 |
| `f1_2019` | 2017-21 (coches anchos) | 5.54 × 1.99 × 1.12 | 3.60 | 30 / 100 | **2 m** de ancho, neumáticos anchos, alerón delantero de 2 m, alerón trasero más bajo y ancho + beam wing, **difusor grande** (0.30 m), bargeboards dobles, **halo**, DRS, rake alto. | altura, flaps, DRS, guiñada, ruedas | 5.0 / 1.35 |
| `f1_2022` | 2022-25 (efecto suelo) | 5.67 × 1.97 × 1.17 | 3.60 | 30 / 80 | **Túneles Venturi** (entradas tras las ruedas delanteras con vallas, garganta, difusor de 0.36 m), borde del fondo, alerón delantero de 3 elementos anclado al morro, **ruedas de 18"** con tapacubos y **deflectores de estela**, alerón trasero **curvado de puntas redondeadas** + beam wing, halo. | altura, flaps, DRS, guiñada, ruedas | 4.4 / 1.15 |
| `f1_2026` | 2026- (aero activa) | 5.37 × 1.90 × 1.13 | 3.40 | 30 / 60 | 1.9 m, batalla 3.4 m, neumáticos más estrechos, **fondo más plano** (túneles y difusor reducidos), **sin beam wing**, bargeboards de in-wash. **Modo Z** (defecto) y **modo X** (`drs_open`): se abren los dos flaps delanteros y el trasero. | altura, flaps, DRS/modo X, guiñada, ruedas | 3.4 / 0.95 |

Notas de fidelidad:
* 2008: el alerón trasero real de 2008 medía 1 m de ancho y era bajo; el estrecho y alto llegó en
  2009 (está en `f1_2011`). Se ha seguido el reglamento real.
* Los alerones traseros de 1998-2019 se estrechan lo justo para dejar ≥ 4.5 cm entre endplate y
  neumático (a dx = 3 cm se fundirían en un puente sólido rueda-alerón).
* Largo: los coches de 2008-2014 salen 0.2-0.4 m más largos que los reales porque el alerón
  trasero se coloca detrás del neumático (misma razón).

Calibración contra el solver (fase 2; detalles y tabla en `docs/FISICA.md`):
* **Alerones traseros 2008-2026 con incidencias mayores que las reales**: plano principal −14° (2008),
  −15° (2011, 2019), −13° (2014), −18° (2022), −16° (2026) y flap 30-36° (antes −6..−9° y 24-30°). En el
  solver el alerón trasero trabaja dentro de la estela del coche (u ≈ 0.5 U) con el flujo subiendo
  15-20°; con los ángulos reales apenas daba carga. Los parámetros `rear_flap` siguen siendo incrementos
  sobre estos valores.
* **Alerones delanteros 1979-2014 más cerrados** (1979 16 → 30°, 1998 18 → 24°, 2008 16 → 32°, 2011/2014
  14/26 → 20/32°): con los reales el eje delantero quedaba en sustentación (balance negativo) y 2008-2014 por
  debajo de 1998.
* **Alerones traseros de 1979 y 1998 más abiertos** (1979 −6°/22° → −2°/10°; 1998 −5°/18°/32° → −3°/10°/22°):
  fuera de la estela del coche esos alerones sí funcionan en el solver y dejaban a 1979 y 1998 por encima de
  2008-2014.
* **2022 y 2026: lomo de la cubierta motor** de la toma de aire a la trasera (antes la toma acababa en
  un escalón de ~30 cm sobre la cubierta y su estela dejaba el alerón trasero sin presión dinámica).
* 2014: el "dedo" del morro se apoya en el plano principal del alerón delantero (antes quedaba a 1-2 cm: una
  rendija de 1 celda a Alta en la que el cálculo divergía).
* Las alturas de marcha de los F1 se simulan subidas según la resolución (hueco mínimo 3.5 celdas; ver
  "altura efectiva" en `docs/FISICA.md`); la geometría del modelo no cambia.

## Objetos de referencia

| id | Tipo | Cotas | Suelo | Parámetros (defecto) | Referencia |
|---|---|---|---|---|---|
| `sphere` | Esfera lisa | D = 1 m | no | altura del punto más bajo (1.5 m) | Cd 0.47 subcrítico |
| `cylinder` | Cilindro finito transversal | D = 0.5 m, L = 2 m (L/D = 4) | no | altura (1.5 m), guiñada | Cd ≈ 0.7 |
| `cube` | Cubo de aristas vivas | 1 m | no | altura (1.5 m), guiñada | Cd ≈ 1.05 |
| `ahmed_25` | Cuerpo de Ahmed, luneta 25° | 1044 × 389 × 288 mm, R = 100 mm, 4 patas Ø30 mm, 50 mm de altura libre | sí | guiñada | Cd = 0.285 (Ahmed 1984) |
| `naca0012_wing` | Ala rectangular NACA 0012 | c = 1 m, b = 3 m | no | AoA (6°), altura (1.5 m) | CL ≈ 0.35 a 6° (Helmbold) |
| `naca4412_wing` | Ala rectangular NACA 4412 | c = 1 m, b = 3 m | no | AoA (4°), altura (1.5 m) | CL ≈ 0.47 a 4° |
| `f1_wing_ge` | Ala F1 invertida de 2 elementos + endplates | c = 0.50 + 0.25 m, b = 1.2 m | sí (cinta) | AoA F1 (4°, + = más carga), altura h (100 mm), ranura (25 mm), flap (+0°, base 20°) | estudio de Zerihan & Zhang: la carga crece al bajar h hasta h/c ≈ 0.1 y luego cae |
| `airfoil_2d` | Perfil NACA 4412 de envergadura completa | c = 1 m, b = 1 m | no | AoA (4°), altura (2 m) | Cl 2D ≈ 0.88 a 4°; **`spans_domain = true`** |
| `road_car` | Turismo fastback tipo DrivAer | 4.67 × 1.76 (2.0 con espejos) × 1.42 m, batalla 2.786 m | sí | altura (150 mm), guiñada, ruedas | Cd ≈ 0.25 |

`ref_length_m` es la mayor dimensión relevante para dimensionar el dominio (largo en los coches,
envergadura en las alas finitas, L en el cilindro); el Reynolds "clásico" de cada caso usa la cuerda
o el diámetro indicados en la tabla. Los objetos de aire libre flotan con su punto más bajo a
`height_mm` (1.5 m por defecto) sobre el suelo potencial z = 0, por si el usuario activa el suelo.

`airfoil_2d` debe atravesar todo el ancho del túnel: la app tiene que hacer `ny·dx` = extensión
Y de `bounds_m` y usar laterales de deslizamiento o periódicos (ver `Info::spans_domain`). En este
modelo `bounds_m.y` es **exactamente** ±0.5 m (no la AABB de la escena, que lleva un margen de
0.02·cuerda) y la geometría sobresale 0.1 m por cada lado: la sección es sólida hasta la pared sea
cual sea el origen de la red (comprobado con el voxelizador real, ny = 32, 33 y 50).

`moment_ref_m`: coches = centro entre ejes a nivel del suelo; alas (también `f1_wing_ge`) = 25% de
la cuerda del plano principal; cuerpos = centro del objeto.

## Parámetros (`models::Params`)

* `ride_front_mm / ride_rear_mm`: altura del plano de referencia (plank) en cada eje. La
  carrocería cabecea (`car_body_frame`: rotación `rot_y(-θ)`, `θ = asin((rr-rf)/batalla)`, el eje
  trasero queda EXACTAMENTE a `ride_rear`); ruedas y faldones de 1979 se quedan en el suelo.
* `yaw_deg`: guiñada alrededor del eje vertical que pasa por el centro entre ejes (coches) o por el
  centro del objeto. Rango ±30°.
* `front_flap_deg / rear_flap_deg`: incremento sobre el ángulo de flap de la época (+ = más carga).
  Los flaps giran manteniendo la **ranura exacta** respecto al elemento anterior.
* `drs_open`: DRS (2011-2025: el flap superior gira alrededor de su borde de salida hasta -6°, la
  ranura se abre ~8 cm) o modo X (2026: flaps delanteros a -3°/-6° y trasero a -3°). **Sólo abre**:
  si un incremento de flap negativo ya dejó el flap más abierto que ese ángulo, no se toca (antes se
  le añadía incidencia y en el modo X con flap -20° la ranura delantera caía a 12.6 mm).
* `wheels_rotating`: movimiento rígido de las ruedas `omega_hat = (0, -1/R, 0)` con centro en el
  eje → la banda de rodadura va a +U∞ como la cinta (lo comprueba el test).
* `aoa_deg`, `height_mm` (punto más bajo), `flap_gap_mm`: alas y objetos.

Todo se sujeta a rangos físicos en `resolve_params`; los parámetros que no están en `param_mask`
se ignoran. `same_geometry(i, a, b)` dice si dos juegos de parámetros dan la misma forma
(`wheels_rotating` sólo cambia el movimiento de pared → no hace falta re-voxelizar).

## Reglas de diseño para redes gruesas (dx ≈ 2-4 cm)

1. **Perfiles con espesor mínimo** (`naca4_floor`): NACA 4 dígitos con un suelo de 4.2 cm desde el
   12% de la cuerda (rampa √x hacia el borde de ataque, borde de salida romo). Un flap inclinado θ
   sólo queda 4-conexo en la red si su espesor vertical ≥ dx·(1 + tan θ); si no, D3Q19 "fuga" por
   los enlaces diagonales entre celdas que sólo se tocan por una esquina. Barrido medido a dx = 3 cm:
   2.4 cm → ~60% de cortes con fuga, 3.0 cm → 5-30%, 3.6 cm → 0, 4.2 cm (elegido) → 0 con margen.
2. **Ranuras de 4.5 cm** entre elementos (las reales son de 1-1.5 cm): con el engrosamiento
   0.12·dx del voxelizador queda al menos 1 celda de fluido. La colocación es exacta (distancia
   polígono-polígono, regula falsi), no a ojo.
3. Placas (endplates, vallas, bargeboards, aletas) de 3.5 cm; suspensión Ø4 cm; fondo de 4 cm.
4. 2-3 elementos por alerón, ≤ 13 grupos por coche (límite del contrato: 254; objetivo: ≤ 60).
5. Nada flota: el test voxeliza a L/320 y exige una sola componente conexa (el suelo cuenta como
   conector: ruedas, patas y faldones apoyan en él).
6. Sin costura en el plano de simetría: las extrusiones espejadas que acaban en y = 0 (alas con raíz
   en el centro, láminas de difusor) se prolongan 1 cm al otro lado (`k_sym_overlap`); si no, el SDF
   vale 0 en todo su interior sobre y = 0 y una prueba `d < 0` (mallador, voxelización sin
   engrosamiento) ve una lámina de fluido dentro del ala.

## Verificación (`tests/test_models.cpp`)

Para cada modelo: construye; cotas plausibles de la época y ancho ≤ reglamento + 2%; ruedas
apoyadas (mínimo en [-10, +2] mm, medido -5 mm); ninguna pieza bajo el suelo; simetría exacta en Y
(20 000 puntos); +30 mm de altura sube el fondo 30 mm ± 4 mm y las ruedas siguen apoyadas; DRS,
modo X, flaps y AoA cambian la geometría; la guiñada gira el modelo; la banda de rodadura va a
+U∞; los objetos de aire libre no tocan el suelo y su punto más bajo = `height_mm`; **todos los elementos de alerón de los coches generan carga** (ángulo negativo y combadura
invertida); **todos los grupos sobreviven a dx = 3 cm**; **todos los elementos de ala son
4-conexos a dx = 3 cm** (3 secciones × 4 desfases de rejilla, BFS 2D) — con un control negativo
verificado (perfiles de 1.5 cm → 50-70% de cortes con fuga); **sin piezas flotantes**; coste medio
de `Scene::eval` < 2 µs por punto. Además: `place_next` deja la ranura con error < 0.05 mm, el DRS
gira alrededor del borde de salida, caché de perfiles, `resolve_params` sujeta rangos.

Además (revisión): barrido de flaps -20..+20 × DRS con ranura exacta y "el DRS sólo abre", sin
costura en y = 0, perfil pseudo-2D sólido hasta las paredes, `moment_ref_m` sobre el ala, parámetros
NaN/inf → defecto.

Resultado: **384 PASS, 0 FAIL** (5-13 s según la carga de la máquina).

`tests/test_models_vox.cpp` repite lo esencial con el **voxelizador real** (submuestreo 2×2×2) a
dx = 3 cm, 3 desfases de red, en posición normal, DRS y flaps ±20°: todos los grupos presentes,
0 cortes de ala con fuga, ranuras abiertas; perfil pseudo-2D; `f1_wing_ge` a dx = 6 mm; sin costura
en y = 0 con `thicken = 0`. **131 PASS, 0 FAIL** (~1-2 s).

## Límites conocidos

* Resolución: todo está diseñado y verificado para **dx ≤ 3 cm**. Con el voxelizador real, a 3.3 cm
  sigue limpio (con otro desfase de secciones se vio 1 ranura cerrada de 36), a 3.6 cm se cierran
  ranuras (f1_2022 1/36, f1_2026 2/36) y aparecen las primeras fugas a través de perfiles (f1_1998:
  3 de 72) y a 4 cm fallan 1-6 de 60-72 cortes por coche.
* Con flaps +20° (extremo), a dx = 3 cm f1_2022 cierra 1 de 27 secciones de la ranura principal-flap
  del alerón delantero (según el desfase de la red); una ranura de 5.5 cm lo evitaría.
* Alturas de marcha: `resolve_params` permite 0-400 mm delante y detrás de forma independiente. Con
  rake extremo la carrocería entra en el suelo (medido: delante 0 / detrás 150 mm → alerón delantero
  de 2011-2019 ~1 cm bajo el suelo; 0 / 400 mm → hasta 10 cm). Con las alturas por defecto nada
  toca el suelo. La UI debería limitar el rake.
* `f1_wing_ge` no lleva perfiles con espesor mínimo: está pensado para dx ≈ 4-12 mm (a dx de coche
  su flap fuga).
