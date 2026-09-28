# flowvis — visualización del flujo: optimizaciones medidas

Módulo: `src/render/flowvis.hpp` (API), `src/render/flowvis.cpp` (magnitudes, cortes,
LIC, huella en el suelo, estela, malla coloreada, sondas), `src/render/flowvis_lines.cpp`
(muestreador FP16, líneas de corriente, humo), `src/render/flowvis_volume.cpp` (volumen de
vórtices), `src/render/flowvis_draw.cpp` (llamadas al rasterizador; separado para que los
cálculos enlacen sin `render/raster*.cpp`). Tests: `tests/test_flowvis.cpp`.

Máquina: Intel Core Ultra 7 155H (6P+8E+2LP-E), 20 hilos del pool, g++ 15.2 `-O3 -march=native`
(y también verificado con las banderas exactas del Makefile, incluida LTO).
**Advertencia sobre las medidas**: durante toda la sesión la máquina tuvo carga de fondo de
otros procesos (carga media 1 min entre 6 y 23: compilaciones y herramientas de los otros
módulos, qemu). Cada cifra es la **mediana** de N repeticiones, y las comparaciones A/B se
hicieron **alternando** binarios A, B, A, B… para que la deriva de carga afecte a ambos. Las
medidas de 1 hilo se fijaron a un núcleo P con `taskset -c 2/3` (mucho más estables). Los
tiempos paralelos de 20 hilos son bimodales cuando hay ~8 núcleos ocupados por otros: un hilo
del pool desplanificado por el SO retiene su trozo y retrasa el final (p. ej. líneas de
corriente 3.8–4.1 ms en ejecuciones "limpias" y 11–13 ms en otras con carga similar).

## 1. Rendimiento frente a objetivos (escena de referencia 256×128×96 = 3.1 M celdas)

Escena: esfera R=14 + par de vórtices de estela contrarrotativos (flujo potencial + Lamb-Oseen).
Medianas de las dos ejecuciones completas más limpias (carga 8–9) del test; rango entre paréntesis
sobre 5 ejecuciones.

| Tarea | Objetivo | Medido | Rango (5 ejec.) |
|---|---|---|---|
| `FlowSampler::update` (empaquetado FP16, 27 MB) | — | **0.8 ms** | 0.78–3.0 |
| Líneas de corriente 2000 semillas × 400 pasos (794 k puntos) | < 10 ms | **3.9 ms** | 3.8–13.2 |
| Humo: paso de 300 k partículas (RK2 + reciclaje + compactación) | < 3 ms | **2.0–2.6 ms** | 1.5–3.3 |
| Volumen Q: `update` (Q + 2³ + ladrillos) | — | **2.3 ms** | 1.8–3.5 |
| Volumen Q: render 1500×1100 media resolución | < 15 ms | **1.9–2.3 ms** | 1.9–7.1 |
| Volumen Q: render 1500×1100 resolución completa | (< 15 ms) | **6.4–7.7 ms** | 6.4–26 |
| Corte Y \|u\| / Y \|ω\| interpolado / Z Cp / X Q | — | 0.03 / 0.04 / 0.03 / 0.14 ms (tras la revisión, §6; antes 0.08 / 0.11 / 0.10 / 0.15) | |
| Corte con LIC 512×384 subtexeles (X, ×4) / 768×288 (Y, ×3) | — | 6–8 ms / 7–9 ms | 4.6–11.5 |
| `color_mesh` 199 k vértices (Cp, trilineal ponderada por fluido) | — | **0.5 ms** | 0.32–1.4 |

Con carga baja (primeras medidas, carga ~6): streamlines 3.87 ms, humo 1.51 ms, volumen 2.6 ms.

## 2. Trucos aplicados (con archivo:línea y efecto medido)

### 2.1 Campo empaquetado AoS FP16 para muestreo aleatorio (F16C) — el truco central
* `flowvis.hpp:204` `FlowSampler::sample`: cada celda son 4×FP16 (ux, uy, uz, ρ-1) = 8 B. Dos
  celdas vecinas en X son 16 B contiguos → la trilineal completa de 4 componentes son
  **4 cargas de 128 bits + 4 `VCVTPH2PS` + 3 FMA en YMM + 1 FMA en XMM** (la interpolación en Y y Z
  se hace sobre las dos celdas a la vez; la de X al final entre las dos mitades del registro).
  Frente a `FieldView::velocity` + `sample(rho)`: 24+8 cargas dispersas de 4 arreglos SoA.
* Medido (1 hilo, 1 M puntos aleatorios, test [9]): **18–27 ns/muestra FP16 frente a
  119–190 ns FP32 SoA → 5.7–7×**. En la integración de líneas de corriente (misma plantilla
  de integrador con los dos muestreadores): **3.8–4.1 ms frente a 8.3–9.1 ms → 2.1–2.3×**.
* Precisión: error relativo máx. 3.7e-4·U∞ frente a FP32 (test [6]); deriva radial de una línea
  de corriente en un vórtice idéntica a la de FP32 (0.043 % vs 0.042 %, test [2]).
* La posición y la velocidad viven en registros XMM `(x,y,z,·)` durante toda la integración
  (`flowvis_lines.cpp:59` `unit` con `RSQRTPS` + 1 Newton), sin ida y vuelta a `Vec3`.

### 2.2 Empaquetado: transposición 4×8 en SSE + stores no temporales
* `flowvis.cpp:448` `FlowSampler::update`: 8 celdas por iteración, `VCVTPS2PH` de cada
  componente y transposición de medias palabras con `unpacklo/hi_epi16` + `unpacklo/hi_epi32`
  (`flowvis.cpp:478`); 64 B completos por iteración escritos con `MOVNTDQ` (`flowvis.cpp:484`).
* Stores no temporales (A/B con `FLOWVIS_NT_PACK`, 20 hilos, 3 rondas alternadas):
  **0.85–0.94 ms frente a 1.14–1.27 ms → ~1.3×** (27 MB que no se releen enseguida: no
  contaminan la caché que usa el solver).
* Celdas sólidas: se conserva la velocidad de pared que escribe el solver (0, cinta o rueda) y
  ρ-1 se sustituye por la media de las vecinas fluidas (el solver escribe ρ=1 en sólidos, lo que
  sesgaría Cp hacia 0 junto a las paredes). Los bloques de 8 celdas sin sólidos se saltan con la
  prueba SWAR `any_bits_u8x8` (8 flags en un `u64`, `flowvis.cpp:510`).

### 2.3 Magnitudes derivadas por filas en AVX2 (+ desconmutación por plantillas)
* `flowvis.cpp:106` `vec_value<Q>` / `flowvis.cpp:147` `row_impl<Q>`: 8 celdas por
  iteración con diferencias centradas (cargas desalineadas a ±1, ±nx, ±nx·ny), FMA para |ω|, Q.
  Los extremos de fila (x = 0 y cola) van por la ruta escalar equivalente.
* Q con la identidad `Q = -½ Σ g_ij g_ji` (`flowvis.cpp:59`): sin construir S ni Ω (menos
  operaciones que la definición de `FieldView::q_criterion`, mismo resultado; test [3]; su efecto
  aislado no se midió).
* Despacho único por fila vía tablas de punteros a instancias de plantilla (`flowvis.cpp:175`
  `k_row_fn`, `k_cell_fn`): el `switch` sobre la magnitud queda fuera de los bucles.
  Lo mismo en líneas/partículas con lambdas plantilla `[&]<Quantity CQ>()` (`flowvis_lines.cpp:282`).
* Medido (dominio completo 3.1 M celdas, 20 hilos, 3 rondas): **|u| 0.61–0.73 ms frente a
  2.3–3.6 ms escalar (~4×); |ω| 0.94–1.24 frente a 5.4–7.5 (~5.7×); Q 1.18–2.0 frente a
  6.5–8.8 (~5×)**. Un corte (≤ 256×128) cuesta 0.08–0.15 ms.

### 2.4 Mapeo de color vectorial y rango robusto
* `flowvis.hpp:107` `map_color8`: normalización con FMA (con pivote en 0 para mapas divergentes:
  el 0 cae en el centro aunque el rango sea asimétrico, p. ej. Cp ∈ [-2.5, 1]) + `VPGATHERDD`
  sobre la LUT constexpr de 256 colores. `max(t,0)` con el orden de operandos que convierte NaN en 0.
  La versión escalar `map_color` (`flowvis.hpp:99`) usa la misma FMA y el mismo redondeo
  (`CVTSS2SI`, al par) → resultados bit a bit idénticos (test [4]).
  Medido (3.1 M valores, 20 hilos): 0.30–0.37 ms frente a 0.40–0.56 ms escalar → ~1.3×
  (limitado por memoria; el beneficio real es no tener ramas por texel).
* `flowvis.cpp` `minmax_finite` / `percentiles` / `robust_range` (reescritos en la revisión, §6):
  mín/máx de los valores FINITOS con `MINPS/MAXPS` + máscara `|x| ≤ FLT_MAX` y `BLENDVPS`
  (ignora NaN = sólidos y ±inf = solver divergido, sin ramas); histograma de 1024 cubetas con los
  índices calculados 8 a la vez en AVX2 → percentiles 1 %–99 %. Sin rango automático sólo se
  hace la pasada de mín/máx (antes siempre había un histograma completo).

### 2.5 Líneas de corriente: K cadenas RK4 entrelazadas (ILP)
* `flowvis_lines.cpp:148` `trace_jobs<CQ, K>`: cada paso RK4 son 4 muestras dependientes
  (latencia pura). Con K líneas independientes avanzando **etapa a etapa** (k2 de todas, k3 de
  todas…, `flowvis_lines.cpp:196`) el núcleo fuera de orden solapa sus cargas y FMA. Un carril
  que termina toma el siguiente trabajo (semilla, sentido) de su rango: no hay carriles ociosos.
* Medido (1 hilo fijo en un núcleo P, 400 semillas × 400 pasos, 3 rondas):

  | K | ns/paso |
  |---|---|
  | 1 | 115–116 |
  | 2 | 62–66 |
  | **4** | **41–42** |
  | 8 | 40–41 |

  **2.8× por hilo con K = 4** (`FLOWVIS_SL_LANES`, `flowvis_lines.cpp:133`). Con 20 hilos
  (carga ~11): 6.9–8.7 ms (K=1) → 5.1–5.8 ms (K=4); con carga menor 3.8–4.1 ms.
* Paso adaptativo: si el giro entre k1 y k4 supera `max_turn_deg` se repite con medio paso;
  en tramos rectos crece ×1.25 hasta `max_step`.

### 2.6 Humo: SoA + anillo + emisión "virtual" + compactación por trozos + prefetch
* `flowvis_lines.cpp:356` `Particles::step`: x/y/z/edad en SoA; reciclaje en anillo
  (sin listas libres); la emisión coloca cada partícula nacida en la fracción `frac` del
  intervalo en `semilla - u·dt·(1-frac)` (`flowvis_lines.cpp:367`) → la advección del propio
  cuadro la deja en `semilla + u·dt·frac`: emisión continua sin pulsos y **sin pasada extra**.
* Advección RK2 + reciclaje + color + escritura de la salida en UNA pasada paralela por trozos
  de 4096; la compactación (`flowvis_lines.cpp:467`) es un prefijo serie de ~75 cuentas + `memcpy`
  paralelos → sin atómicos ni listas.
* Prefetch software de las 4 filas de celdas de la partícula i+8 (`flowvis.hpp:243`,
  `flowvis_lines.cpp:423`; su posición actual ES la primera muestra RK2). Medido (1 hilo fijo):

  | Distancia | ns/partícula |
  |---|---|
  | 0 (sin prefetch) | 50–93 |
  | 4 | 38–49 |
  | **8** | **37–58** |
  | 16 | 37–84 |

  **~1.35× en 1 hilo con carga baja**. Con 20 hilos: neutro dentro del ruido (2.36–2.85 ms frente a
  2.39–2.66 ms): con todos los núcleos el límite pasa a ser el ancho de banda y la cola de trozos.
  *Revisión (§6)*: remedido en 1 hilo fijo (núcleos P 4/8/10, 2 rondas alternadas, 300 k partículas):
  33–43 ns/partícula con prefetch frente a 38–42 sin él (un atípico de 100) → **~1.1–1.15×**, no 1.35×.
* Color: RGB del mapa, canal A = desvanecido × intensidad (el rasterizador multiplica por A tanto en
  aditivo como en mezcla).
* Subpasos RK2 = ⌈dt·1.5·U∞ / max_cells_per_substep⌉ con tope `max_substeps` (4): el coste por
  cuadro queda acotado aunque el solver avance muchos pasos por cuadro en mallas pequeñas.
* Precisión: el campo FP16 introduce un sesgo relativo ≤ 2^-11 en u (U = 0.08 → 0.0800171): el
  test comprueba el avance medio en flujo uniforme con tolerancia relativa 4e-4.

### 2.7 Volumen de vórtices
* **Ladrillos 8³ con máximo (+1 vóxel de margen)** (`flowvis_volume.cpp:190`, salto en el bucle
  `flowvis_volume.cpp:306`). A/B (`FLOWVIS_VOL_BRICKS`, 20 hilos): media resolución 8.6–10.3 ms
  frente a 20–30 ms sin ladrillos (**~2.5×**); completa 27–32 frente a 73–110 ms.
* **Recorte de rayos a la caja de ladrillos ocupados + rectángulo de pantalla** de su proyección
  (`flowvis_volume.cpp:441`, `:449`): sólo se trazan los píxeles que pueden ver vórtices y cada
  rayo empieza en la caja ocupada. A/B (20 hilos, 3 rondas): media resolución **9.7–10.4 → 2.5–3.0 ms
  (3.5×)**; completa **29–30 → 8.3–8.5 ms (3.5×)**: la resolución completa también cumple el objetivo.
* Umbral aplicado en el render (cuantización `255·Q/full`, isovalor = índice de LUT): mover el
  deslizador del umbral no exige recalcular el volumen.
* Prueba de vacío de las 8 esquinas con SWAR "¿algún byte > n?" (bit hack *hasmore*,
  `flowvis_volume.cpp:48`) sobre un `u64` formado con 4 cargas de 16 bits (`flowvis_volume.cpp:234`).
* Estilo `Surface`: cruces del isovalor refinados por interpolación lineal; gradiente (6 muestras)
  **sólo en los cruces**; Blinn-Phong de dos caras + contorno; varias capas semitransparentes.
  Estilo `Cloud`: LUT de opacidad `1-exp(-σ·d·Δs)` precalculada por cuadro (256 `exp`).
* Terminación temprana (A > 0.985), parada en la profundidad opaca de la escena (profundidad de
  vista → t del rayo dividiendo por cos θ), ruido IGN (Jimenez) para romper el bandeado
  (`flowvis_volume.cpp:39`), y al saltar ladrillos se reanuda en la rejilla de pasos del rayo
  (sin costuras entre ladrillos).
* **Media resolución + reescalado 2× bilineal SWAR**: pesos fijos 3:1 → `(9a+3b+3c+d)/16` con 4
  canales × 16 bits en un `u64`: expandir con `PDEP`, dos `3x+y`, `>>4`, comprimir con `PEXT`
  (BMI2, `flowvis_volume.cpp:589`, `:603`) + composición `over` premultiplicada con suma saturada
  `PADDUSB` (`flowvis_volume.cpp:390`). Rechazo por profundidad a resolución completa con la
  profundidad del primer aporte de cada rayo (`flowvis_volume.cpp:601`): los bordes de la malla
  no se "manchan" de vórtice.
* Planos translúcidos: el tramo del rayo detrás de cada corte translúcido pesa ×(1-opacidad)
  (`flowvis_volume.cpp:320`); exacto si el color del corte ≈ el de la escena, buena aproximación
  en general (test [8]: plano opaco → 0 exacto; al 50 % → 0.4–0.6 de la energía).
* Máximo global lock-free en `update` comparando los bits de floats ≥ 0 como `u32`
  (`flowvis_volume.cpp:160`).
* Supresión de Q/|ω| junto a paredes (ruido de escalera del vóxel): OR SWAR de 7 palabras de 8 flags
  (`flowvis_volume.cpp:58`). Coste medido: 0–15 % del `update` (3.1 vs 2.9–3.3 ms, dentro del ruido).

### 2.8 LIC en los cortes (opcional)
* `flowvis.cpp:710` `build_lic`: dirección unitaria por celda precalculada (sin sqrt/div por paso),
  ruido blanco fijo, núcleo de Hann, normalización por z-score. **AVX2 de 8 subtexeles consecutivos**
  (`flowvis.cpp:758`): bilineal de la dirección con 8 `VPGATHERDPS`, sólido y ruido con 2 gathers
  más (el de bytes lee 32 bits con escala 1 sobre un búfer con 8 bytes de relleno), carriles muertos
  enmascarados. Medido (512×384 subtexeles, ±20 pasos): **1 hilo 123 → 39 ms (3.1×); 20 hilos
  12 → 4.9 ms (2.5×)**.

### 2.9 Malla coloreada, sondas, estela
* `flowvis.cpp:1019` `color_mesh`: trilineal **ponderada por fluido** (las esquinas sólidas pesan 0,
  `flowvis.cpp:191`): Cp de superficie sin extrapolaciones, sin sesgo por el ρ=1 de los sólidos.
  Error máx. frente a flujo potencial analítico en una esfera: 0.017 en Cp (test [6]).
  199 k vértices en 0.3–0.7 ms (paralelo por bloques de 2048).
* `flowvis.cpp:981` `wake_survey`: ∫(1-Cp0) dA y ∫(v²+w²)/U∞² dA; test con estela gaussiana
  analítica: 56.55 frente a 56.55 celdas².

## 3. Alternativas medidas y descartadas

| Alternativa | Resultado | Decisión |
|---|---|---|
| Advección de 8 partículas por registro AVX2 con `VPGATHERDPS` desde FieldView FP32 (32 gathers por muestra) | Aleatorio: 110 ns/muestra frente a 22–47 ns del FP16 empaquetado por partícula; coherente (en caché): 4.9 frente a 4.6 ns | No adoptado: nunca gana |
| LIC escalar con 4 líneas entrelazadas (ILP) | 1 hilo: 807–847 frente a 683–826 ns/subtexel (sin ganancia: limitado por nº de instrucciones, no por latencia) | Sustituido por la versión AVX2 |
| LIC: quitar 2 divisiones enteras por paso | 1 hilo: 685–985 frente a 699–895 ns/subtexel (ruido) | Se mantiene por claridad, sin efecto medible |
| Prefetch de partículas con 20 hilos | 2.36–2.85 frente a 2.39–2.66 ms | Se mantiene (gana en 1 hilo, neutro en paralelo) |
| K = 8 carriles en líneas de corriente | Igual que K = 4 en 1 hilo; más desequilibrio de cola en paralelo | K = 4 |
| Volumen: sombreado desactivado | Diferencia dentro del ruido (el gradiente sólo se evalúa en los cruces) | Sombreado activado por defecto |

## 4. Tests (`tests/test_flowvis.cpp`, 128 comprobaciones con rasterizador / 125 sin él, ~2–4 s; más `tests/test_flowvis_review.cpp`, 75, §6)

Campos analíticos construidos en el test. Resultado: **PASA (128/128; 125/125 con `-DFLOWVIS_NO_RASTER`)**, también con las banderas
exactas del Makefile (`-flto`, `-fno-rtti`, `-ffp-contract=fast`…).

1. Flujo uniforme: línea recta (desviación < 1e-4 celdas), longitud exacta 9.5 con paso fijo, fin en
   el borde, ambos sentidos ordenados aguas arriba → abajo, semillas inválidas descartadas.
2. Vórtice de Lamb-Oseen: 3.5–10.6 vueltas con deriva radial máx. 0.043 % (FP16) / 0.042 % (FP32).
3. Q y |ω|: sólido rígido (Q = Ω², |ω| = 2Ω exactos), deformación pura (Q = -a² < 0, ω = 0),
   Lamb-Oseen (Q > 0 en el núcleo, < 0 a 2.5 rc), AVX2 = escalar en todas las magnitudes con sólidos.
4. Cortes: orientación y valores exactos en los 3 ejes (campo lineal, con interpolación entre capas),
   colores = `map_color`, esquinas, `value_at`, `pick` por rayo de cámara, sólidos en gris,
   rango automático = percentiles exactos, pivote divergente, `map_color8` = `map_color`, LIC
   alineado con el flujo (variación transversal 14× la longitudinal).
5. Humo: conservación `emitidas = vivas + muertas + sobrescritas` en cada cuadro, tasa exacta,
   población estacionaria (±3 %), muertes por sólido/salida/edad, anillo, avance medio = U·dt
   (también con 150 pasos en un cuadro y subpasos topados).
6. Malla: Cp de superficie frente a flujo potencial analítico (err. 0.017), remanso rojo /
   ecuador azul, sondas (|u|, Cp0 ≈ 1 en flujo potencial, sólido), FP16 frente a FP32 (3.7e-4).
7. Estela gaussiana (integral exacta), huella en el suelo (capa automática, extensión, succión).
8. Volumen: imágenes PNG (media/completa/nube/magnitud), oclusión por profundidad sintética
   (0 píxeles del oclusor modificados), umbral en render, planos translúcidos.
9. Rendimiento (medianas; los objetivos sólo cuentan como fallo con carga < 6 o `FLOWVIS_STRICT_PERF=1`).
10. Imágenes compuestas con el rasterizador real (`build/flowvis/composed.png`, `smoke.png`,
    `ground_effect.png`, `lic_x150.png`, `lic_y64.png`, `wake_slice.png`) — revisadas visualmente.

## 5. Compilación

```
# sin rasterizador (tests numéricos + PNG del volumen):
g++ -std=c++23 -O3 -march=native -Isrc -pthread -DFLOWVIS_NO_RASTER tests/test_flowvis.cpp \
    src/render/flowvis.cpp src/render/flowvis_lines.cpp src/render/flowvis_volume.cpp \
    src/core/threadpool.cpp src/core/png.cpp -o build/flowvis/test_flowvis
# regresiones de la revisión (sin rasterizador; también con -O1 -g -fsanitize=address,undefined,float-cast-overflow):
g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_flowvis_review.cpp src/render/flowvis.cpp \
    src/render/flowvis_lines.cpp src/render/flowvis_volume.cpp src/core/threadpool.cpp -o build/flowvis/test_flowvis_review
# completo (imágenes compuestas):
g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_flowvis.cpp src/render/flowvis*.cpp \
    src/render/raster_*.cpp src/core/threadpool.cpp src/core/png.cpp -o build/flowvis/test_flowvis_full
```

## 6. Revisión adversarial (segundo ingeniero)

Todo lo anterior se reprodujo recompilando desde cero (`-O3 -march=native`, las banderas exactas
del Makefile con LTO, y `-O1 -g -fsanitize=address,undefined,float-cast-overflow`). Los tests del
implementador pasan (128/128 con rasterizador) y ASan/UBSan no encontró nada **en esos tests**; los
defectos siguientes sólo aparecen con entradas que no cubrían. Nuevo test de regresión:
`tests/test_flowvis_review.cpp` (R1–R11, 75 comprobaciones; no necesita el rasterizador).

### 6.1 Defectos encontrados y corregidos

| # | Gravedad | Defecto | Reproducción | Corrección |
|---|---|---|---|---|
| 1 | crítico | `robust_range`: un solo valor ±inf (o un rango > FLT_MAX) en el corte → `(x-vmin)·0 = NaN` → `(int)NaN` = INT_MIN como índice del histograma en la pila → **SEGV** en `SliceView::update` aunque `auto_range` esté apagado (siempre se calculaba mín/máx con histograma). Un solver que diverge produce inf (FP16 satura). | `probe_inf`: exit 139 en `-O3`; ASan: `index -2147483648 out of bounds` | Sólo cuentan valores finitos (máscara `|x| ≤ FLT_MAX`), índices sobre mitades (`x/2 - vmin/2`, nunca desborda), sujeción del índice, rango final con `hi > lo` garantizado. Tests R1, R9. |
| 2 | mayor | `FlowSampler` y la trilineal ponderada (`color_mesh`, `probe`, `sample_quantity`) sujetaban a `n-1-1e-4`, que **redondea a n-1 cuando n-1 ≥ 2048** (espaciado de float 2.4e-4) → la celda `x0+1` se leía fuera del búfer (16 B en `sample`). Un túnel de 2056×… celdas es plausible con 14 GB. | ASan: `heap-buffer-overflow ... READ of size 16` con nx = 2056 | `below_last(n)`: mayor float < n-1 restando 1 ulp a los bits → x0 ≤ n-2 para cualquier n, coste nulo. Test R11. (`FieldView::sample/velocity` del núcleo tiene el mismo problema: ver informe.) |
| 3 | mayor | `VortexVolume::render` recortaba el viewport sólo por la derecha/abajo: con `cam.vp.x/y < 0` escribía/leía antes de la fila (fuera del búfer en la fila 0). | ASan: SEGV en `flowvis_volume.cpp` con vp = (-40,-30,…) en primer plano | vp ∩ framebuffer. Test R4 (además: nada fuera de `cam.vp` cambia y la media resolución es invariante al desplazar el viewport). |
| 4 | menor | `SliceView`: `corners()`, `pick()`, `value_at_world()` y `translucent_plane()` leían `params.axis` actual, no el del último `update()` → si la UI cambia el eje con la simulación en pausa, la textura (p. ej. nx×nz) se dibuja con las esquinas del otro eje y la sonda lee con ejes cruzados. Idem `fade_below` para decidir si escribe profundidad. | test R5 falla con el código original | Se guardan `axis_` y `fade_` en `update()`; nuevo `SliceView::axis()`. |
| 5 | menor | `VortexVolume`: `color_by` y `full` se hornean en `update()` pero `render()` usaba los `params` actuales (LUT de color y umbral) → colores/umbral incoherentes hasta el siguiente `update()`. | test R10 | Se guardan `col_ux_`, `col_map_`, `full_` en `update()`; documentado qué parámetros son de update y cuáles de render. |
| 6 | menor | UB por conversiones float→entero con NaN/negativos: `(u8)saturate(NaN)` en el color del volumen (campo divergido), `(u32)` de alfa negativo si `Particles::params.intensity < 0`, `(int)NaN` de `SliceParams::pos = NaN`. | UBSan `float-cast-overflow` | NaN → 0, intensidad sujeta a [0,1], pos NaN → 0. |
| 7 | menor (test) | Con ASan y carga < 6 los objetivos de tiempo pasaban a ser obligatorios → fallo espurio del test instrumentado. | reproducido | En compilaciones instrumentadas o sin optimizar los objetivos son informativos (salvo `FLOWVIS_STRICT_PERF=1`). |

### 6.2 Efecto medido de las correcciones de rendimiento

* `robust_range` (plano 256×96 = 24.5 k valores, 3 % NaN, 1 hilo fijo en núcleo P, 2 núcleos ×
  2 rondas alternadas, medianas de 401): **mín/máx 23.3–24.0 → 3.45–3.7 µs (6.7×)**;
  **percentiles 1–99 % 23.2–23.9 → 15.5–16.0 µs (1.5×)** (índices de cubeta en AVX2:
  22 → 12 µs aislado; 2/4/8 sub-histogramas intercalados medidos y descartados: 12.7/14.8/13.6 µs).
* `SliceView::update` completo (20 hilos, escena 256×128×96, 4 rondas alternadas A/B):

  | Corte | Antes (ms) | Después (ms) |
  |---|---|---|
  | Y \|u\| | 0.045–0.060 | 0.020–0.037 |
  | Y \|u\| + rango automático | 0.098–0.138 | 0.036–0.072 |
  | Z Cp | 0.070–0.094 | 0.024–0.040 |
  | Z Cp + rango automático | 0.123–0.163 | 0.045–0.078 |

  (≈ 2–2.8×; la huella en el suelo es un corte Z y se beneficia igual.)

### 6.3 Afirmaciones verificadas de forma independiente

* Líneas de corriente K = 4 carriles entrelazados (1 hilo fijo, 400 semillas × 400 pasos, 3 núcleos
  × 2 rondas): K=1 105–121 ns/paso (un atípico de 212), K=4 37–44 ns/paso → **~2.7×** (se afirmaba 2.8×). ✔
* FP16 empaquetado frente a FieldView FP32 (1 hilo, aleatorio): 18.7–27 frente a 131–139 ns/muestra. ✔
* Objetivos (20 hilos, carga 4.4): líneas 2000×400 3.70 ms (< 10), humo 300 k 1.40 ms (< 3),
  volumen media resolución 1500×1100 1.16 ms (< 15), completa 4.24 ms; con carga 21 siguen
  cumpliéndose (4.78 / 2.37 / 2.18 ms). ✔
* Prefetch de partículas: sólo ~1.1–1.15× en 1 hilo (se afirmaba ~1.35×). ✘ (sobrestimado)
* Imágenes revisadas de nuevo: `composed.png`, `volume_half.png`, `ground_effect.png`, `smoke.png`,
  `lic_x150.png` — correctas.
