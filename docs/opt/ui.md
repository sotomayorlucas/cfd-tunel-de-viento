# Módulo `ui` + plataforma — optimizaciones y resultados

Archivos: `src/render/draw2d.{hpp,cpp}`, `src/ui/{ui,demo}.{hpp,cpp}`,
`src/platform/{platform.hpp,x11.cpp,headless.cpp,upscale.cpp}`, `tests/test_ui.cpp`,
`tests/test_ui_bench.cpp`, `tools/ui_demo.cpp`.

CPU: Intel Core Ultra 7 155H, pool de 20 hilos. **La máquina tenía carga de fondo pesada**
(carga media 11–21: qemu, compilaciones de otros agentes), así que todas las cifras son
**medianas** de 101 repeticiones (micro-benchmarks) o 300 cuadros, y se dan rangos de 3
corridas cuando variaban. Las razones antes/después son más estables que los tiempos absolutos.

## Compilar y ejecutar

```sh
S="src/ui/ui.cpp src/ui/demo.cpp src/render/draw2d.cpp src/platform/headless.cpp src/platform/upscale.cpp src/core/png.cpp src/core/threadpool.cpp"
g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_ui.cpp       $S -o build/ui/test_ui        && build/ui/test_ui
g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_ui_bench.cpp $S -o build/ui/test_ui_bench  && build/ui/test_ui_bench
g++ -std=c++23 -O3 -march=native -Isrc -pthread tools/ui_demo.cpp $S src/platform/x11.cpp -o build/ui/ui_demo -lX11 -lXext
build/ui/ui_demo --frames 60        # X11; --headless, --scale N, --size WxH
```
Sanitizers: el mismo test con `-O1 -g -fsanitize=address,undefined` y `--rapido` (omite rendimiento y capturas).

## Trucos usados (archivo:línea → efecto medido)

| # | Truco | Dónde | Antes → después (mediana) |
|---|---|---|---|
| 1 | **Relleno opaco AVX2**: `vmovdqa` de 8 píxeles + **pelado de cabeza a 32 B** con `VPMASKMOVD` y cola con `VPMASKMOVD` (sin bucle escalar) | `draw2d.cpp:199-218` (`peel_px`, `store_span2`/`store_span`) | 420×1200: escalar 0.143–0.186 ms → **0.073–0.103 ms (×1.8–2.0)**. Frente a `std::fill_n` autovectorizado por GCC: **empate** (×0.93–1.04). Sin el pelado, el AVX2 perdía contra `fill_n` (0.077 vs 0.071 ms): los stores de 32 B desalineados (panel en x=1500 → desfase de 16 B) cruzaban línea de caché la mitad de las veces. |
| 2 | **Mezcla alfa AVX2 a 16 bits/canal** con `src·a` precalculado (1 `vpmullw` + 1 `vpaddw` + 1 `vpsrlw` por mitad) y alineado igual que (1); misma fórmula que el SWAR escalar → **bit-exacto** (test) | `draw2d.cpp:162-180` (`ConstBlend`), `219-243` (`blend_span`) | 420×1200 al 50 %: float por canal 0.47–0.64 ms → **0.135–0.22 ms (×2.2–3.8)**; frente al SWAR escalar (que GCC autovectoriza) **×1.2–1.5** |
| 3 | **SWAR escalar** `blend_px`: R y B en una multiplicación de 32 bits, G en otra; `alpha256(a) = a + (a>>7)` (255 ↦ 256 exacto, sin división por 255) | `draw2d.hpp:70-77` | base del punto (2); usado en bordes antialias |
| 4 | **Texto: LUT constexpr byte→máscara de 8 carriles** (8 KiB, en L1) → **1 `VPMASKMOVD` por fila de glifo 8×16** (2 en 16×32) | `draw2d.cpp:141-148` (LUT), `990-997` | 3000 glifos: bucle por bit 0.164–0.232 ms → **0.037–0.059 ms (×3.9–4.5)** |
| 5 | **Rango de filas no vacías por glifo precalculado en compilación** (tablas combinadas Latin-1 + 31 glifos extra, `constexpr`) → sin ramas por filas vacías | `draw2d.cpp:103-138` | incluido en (4) |
| 6 | **UTF-8 → índice de glifo con ruta SWAR**: 8 bytes ASCII por iteración (detección "haszero" de bytes < 0x20 y ≥ 0x80, resta 0x20 sin préstamos) | `draw2d.cpp:328-340` | ASCII: 1 comparación por 8 bytes (no medido aislado) |
| 7 | **Conteo de glifos SWAR + `popcount`**: bytes de continuación `10xxxxxx` = `w & ~(w<<1) & 0x80…80` | `draw2d.cpp:370-382` | 45 KB: 18.5–53.6 µs → **3.0–8.8 µs (×5.4–6.2)**. Medir texto = contar glifos (fuente de ancho fijo) |
| 8 | **DrawList + rasterizado paralelo por franjas**: comandos POD grabados (vectores reutilizados, 0 asignaciones en régimen estacionario); cada hilo reproduce todo recortado a su franja → mismo resultado bit a bit (test) | `draw2d.cpp:1302-1326` | render del panel: 1 franja 0.163–0.283 ms → **0.038–0.088 ms (×3.2–4.6)** |
| 9 | **Degradado vertical con tramado Bayer 4×4** en punto fijo 8.4 → patrón de 4 píxeles por fila en un registro: sin bandas en fondos oscuros a coste de un relleno | `draw2d.cpp:151`, `635-668` | 1500×1200: 0.19–0.45 ms (1 hilo), igual que un relleno plano |
| 10 | **Antialias por distancia**: sólo las celdas de esquina (r×r) calculan `sqrt`; tramos centrales con (1). Sombras: interior saltado/rellenado de golpe, coste ∝ perímetro×blur | `draw2d.cpp:487-533`, `591-633` | panel de 420×1200 redondeado = coste de un rectángulo |
| 11 | **Polilínea con cobertura MÁXIMA por fila** (uniones sin doble mezcla) + **salto de 8 en 8 de tramos vacíos** con `VCMPPS`+`VMOVMSKPS` | `draw2d.cpp:720-779` | polilínea 160 pts a lo ancho de 1400 px: 0.123 → **0.076 ms (×1.6)** con el salto (misma corrida relativa) |
| 12 | **Gráficas: diezmado máx/mín por columna** (≤ 2 puntos por píxel, conserva picos) y área bajo la curva con degradado | `ui.cpp:1225-1246`, `draw2d.cpp:867-902` | coste acotado por el ancho en píxeles, no por nº de muestras |
| 13 | **Escalado 2× AVX2**: 8 → 16 píxeles con 2 `VPERMD`; la fila fuente se lee 1 vez y las 2 filas destino se escriben desde los mismos registros (sin `memcpy` de la primera) | `upscale.cpp:31-50` | 1 hilo, 1920×1200→3840×2400: escalar+memcpy 2.29–3.39 ms → AVX2 **1.78–2.88 ms (×1.1–1.3)** |
| 14 | **Stores no temporales** (`VMOVNTDQ` + `SFENCE` por trozo) hacia la imagen MIT-SHM (la lee el servidor X, no nosotros) | `upscale.cpp:23-27`, `90`, `103` | 1 hilo: **×1.6–2.1**. Con 20 hilos el micro-benchmark es **inconcluso** (×0.67–1.40). Extremo a extremo en X11 NT gana: espera al servidor 3.8/3.98 ms vs 4.23/4.25 ms sin NT (2 pares de corridas). Se deja NT por defecto; `CFD_NO_NT=1` lo desactiva |
| 15 | **Escalado en paralelo** por filas (grano 8 filas fuente ≈ 245 KB) | `upscale.cpp:111` | 1 hilo NT 0.86–1.79 ms → pool **0.63–1.01 ms (×1.4–2.7)**, 36–59 GB/s escritos (limitado por memoria) |
| 16 | **MIT-SHM de doble búfer** + espera de `ShmCompletion` con `XCheckTypedEvent` (no consume la entrada) + `poll()` del socket; `IPC_RMID` inmediato (sin segmentos huérfanos) | `x11.cpp:242-294`, `336-348`, `350-376` | ver "present()" abajo |
| 17 | **Ids FNV-1a** + tabla de estado de direccionamiento abierto (2048 ranuras, sin asignaciones) | `ui.cpp:76-110` | construcción del panel completo (43 widgets, 2 gráficas, 193 comandos): **0.014–0.045 ms** |
| 18 | **Recorte en 64 bits** (sujetar antes de volver a `int`) y saturación de rectángulos/coordenadas a ±2^23/2^30 | `draw2d.cpp:277-285`, `417-425` | corrección (ver "fallos encontrados") |

Trucos descartados o sin efecto (honestidad):
* AVX2 manual para rellenos opacos **no** mejora a `std::fill_n` autovectorizado (empate): se mantiene por el control de alineación y la cola sin bucle.
* `_mm256_unpacklo/hi_epi32 + permute2x128` frente a `VPERMD`: la tarea es de ancho de banda; no se midió diferencia (se usa `VPERMD`: 2 µops en vez de 4).
* Sin `perf` (sin permisos en esta máquina): los perfiles se hicieron con cronómetros por primitiva (`tests/test_ui_bench.cpp`).

## Rendimiento (objetivo: panel completo < 1.5 ms)

* Panel de demostración completo a 1920×1200 (43 widgets, 2 gráficas, gráfico de barras,
  barra de colores, métricas; 193 comandos): **construcción 0.014–0.045 ms + render
  0.033–0.069 ms = 0.066–0.115 ms** (medianas de 300 cuadros en varias corridas). ~15–20×
  por debajo del objetivo.
* Dentro del bucle real de `ui_demo` (los hilos del pool duermen entre despachos): UI
  0.08–0.47 ms.
* Escala 2 (framebuffer nativo 3840×2400, fuente 16×32): UI + visor ficticio 1.2–3.7 ms.

### present() 1920×1200 → ventana 3840×2400 (X11/XWayland, MIT-SHM, 20 hilos)

(El gestor de ventanas ajustó la ventana a 3840×2262 → framebuffer 1920×1131; el camino de
redimensionado funcionó.) Medianas de 60–90 cuadros, varias corridas:

| Fase | p50 |
|---|---|
| Escalado AVX2 NT paralelo (37 MB) | **0.61–0.83 ms** (p90 hasta 1.6–4 ms con la carga) |
| `XShmPutImage` + `XFlush` | **0.01–0.03 ms** (asíncrono) |
| Espera a `ShmCompletion` | 0–5.6 ms: sólo aparece cuando la demo va a > ~150 FPS; XWayland tarda ~9–10 ms en consumir cada imagen de 37 MB. A los ~30 FPS de la app la espera es 0 (doble búfer) |

Verificación de la presentación: se leyó la ventana del servidor con `XGetImage`: 0 bloques
2×2 no uniformes de 2 171 520 (el escalado 2× llega intacto a la pantalla) y el contenido
coincide con el panel renderizado.

## Pruebas (`tests/test_ui.cpp`) — PASS, 227 comprobaciones (204 originales + 23 del revisor)

* UTF-8: acentos españoles, inválidos → un `?` por secuencia, extras (Δ ρ → ≈ ✓), μ → µ,
  ruta SWAR == escalar.
* Glifos `áéíóúñÑ¿¡º` comparados **píxel a píxel** con `font_data.hpp` (8×16 y 16×32) y
  texto translúcido.
* Mezcla AVX2 == SWAR escalar (bit-exacto).
* Recorte: 1500 primitivas aleatorias × 5 recortes × {Painter, DrawList paralela} con
  coordenadas enormes (±1e9, INT_MAX/INT_MIN), negativas y NaN: **0 píxeles escritos
  fuera** del clip ni en el relleno del stride; ASan+UBSan limpios.
* DrawList paralela == ejecución en serie (bit a bit, 2 números de franjas).
* Widgets: botón true exactamente 1 vez (y no si se pulsa fuera y se suelta dentro, ni al
  revés); slider monótono, sujeto a [lo,hi], captura del ratón fuera del panel, ajuste fino
  con Mayús (10 %), `slider_int`; checkbox/radio/interruptor/botón conmutable; combo (abre,
  elige, cierra; clic fuera cierra **sin** activar lo de debajo; Esc); rueda (paso exacto,
  sujeción arriba/abajo, fuera del panel no desplaza); `wants_mouse` (panel sí, visor no,
  arrastre iniciado en el visor no toca la UI); teclado (Tab, Espacio, ←/→, Intro para
  escribir, **coma decimal**, Ctrl+clic, doble clic que restaura el valor, Esc); tooltip a
  los 0.5 s; avisos que expiran.
* Escalado AVX2 == referencia para escala 1–3, ventana mayor/menor (márgenes negros),
  NT sí/no, paralelo sí/no; ventana headless.
* Capturas revisadas a ojo: `build/ui/ui_demo.png`, `ui_demo_bottom.png`,
  `ui_demo_popup.png`, recortes ×2 y `ui_escala2_top.png`.

## Fallos reales encontrados por las pruebas (y corregidos)

1. **Desbordamiento en la intersección de recortes**: `x + w` se calculaba en 64 bits pero se
   truncaba a `int` antes de sujetar → con `x ≈ INT_MIN` el borde derecho salía enorme y
   positivo → escritura fuera del framebuffer (ASan). `draw2d.cpp:417`.
2. **Bucle infinito en `shadow`** cuando el tramo interior quedaba vacío pero con
   `sx0 == sx1`. `draw2d.cpp:609-614`.
3. **Colisión de ids**: la sección "Modelo" y el combo "Modelo" tenían el mismo id (la
   cabecera robaba el clic). Ahora las cabeceras usan id "salado" (`ui.cpp:465`) y, sin
   `NDEBUG`, un detector avisa de ids duplicados por cuadro (`ui.cpp:268`).
4. UBSan: `x + w` con signo en `DrawList::text` con x ≈ INT_MAX → saturación (`sat_i`).

## Revisión adversarial independiente (revisor del módulo)

Se recompilaron y ejecutaron desde cero todas las pruebas y herramientas: `test_ui` (O3),
`test_ui` con **ASan+UBSan en modo completo** (no sólo `--rapido`), **ThreadSanitizer**
(`--rapido`: render paralelo por franjas y escalado paralelo, 0 avisos), `test_ui_bench` (×5),
compilación con las banderas exactas del Makefile (LTO, `-fno-rtti`, `-DNDEBUG`, sin avisos) y
`ui_demo` en X11 (MIT-SHM escala 2, `CFD_NO_SHM=1` → XPutImage, y escala 1).

### Afirmaciones del implementador comprobadas

* 204/204 comprobaciones PASS; sanitizers limpios; micro-benchmarks reproducidos (mismas
  razones: relleno ×1.8 vs escalar y empate con `fill_n`, mezcla ×1.2–1.5 vs SWAR, texto
  ×4.4, `utf8_count` ×6.2, render por franjas ×4.4–6.2, escalado pool+NT 0.66–0.73 ms,
  NT vs normal con 20 hilos inconcluso ×0.74–1.20).
* Panel completo: 0.061–0.089 ms (construcción + render), objetivo < 1.5 ms: se cumple.
* X11: MIT-SHM a escala 2, ventana 3840×2262 (el gestor la ajusta), escalado p50 0.63 ms,
  `XShmPutImage` 0.006 ms, espera a `ShmCompletion` ≈ 4 ms sólo porque la demo va sin límite
  de FPS. Fallback XPutImage: `put` ≈ 9.8 ms/cuadro (funciona, lento como se esperaba).

### Defectos encontrados y corregidos (todos con prueba de regresión que fallaba antes)

| # | Defecto | Corrección |
|---|---|---|
| 1 | `Painter::round_rect` con grosor ≥ medio lado: las celdas de esquina medían `ri = max(r, t) > w` y **pintaban fuera del rectángulo** (con `t` enorme, un cuadrado de 2^20 px; con `t = 3` en una caja de 4 px, 1-2 px fuera). El test de recorte no lo veía (seguía dentro del clip); lo destapó la nueva prueba Painter == DrawList (la DrawList lo descartaba por su caja) | contorno que lo cubre todo → `fill_round_rect`; celdas de esquina limitadas a su mitad del rectángulo (sin solapes = sin doble mezcla) — `draw2d.cpp:545-566` |
| 2 | `slider_int` + Mayús (ajuste fino) **no se movía nunca** con arrastres normales: el incremento sub-unidad (≈0.05/px) se redondeaba cada cuadro | acumulador continuo por slider en una ranura salada — `ui.cpp:860-876` |
| 3 | Tras un Tab (foco visible), un clic en el visor dejaba `wants_keyboard()` a **true para siempre** → la app no recibía teclas | clic que va a la app abandona el foco visible — `ui.cpp:188` |
| 4 | Esc que cerraba un popup / cancelaba la edición / quitaba el foco no se marcaba como consumido: la app lo veía también (`ui_demo` **salía** al cerrar un desplegable con Esc) | `esc_consumed_` → `wants_keyboard()` true ese cuadro — `ui.cpp:199-205`, `229` |
| 5 | `PlotSeries` con `offset` negativo → **lectura fuera del búfer** (ASan: stack-buffer-overflow en `plot_lines`) | offset normalizado a `[0, count)`, series nulas/`count<0` ignoradas — `ui.cpp:1136-1155` (máx. 8 series) |
| 6 | UTF-8: un byte de continuación huérfano daba `?` al decodificar pero no contaba en `utf8_count` → `text_width` no medía lo dibujado y `DrawList::text` **recortaba la cola** (Painter y DrawList dibujaban distinto) | los huérfanos no producen glifo → recuento == decodificación para CUALQUIER entrada (fuzz de 20 000 cadenas) — `draw2d.cpp:349` |
| 7 | Degradado vertical: tras el pelado a 32 B el cuerpo reutilizaba la fase Bayer de `x0` → periodo 4 roto en la unión (costura vertical en fondos) | patrón indexado por columna absoluta + `store_span2` con fase de cabeza y de cuerpo — `draw2d.cpp:205-218`, `653-666` |
| 8 | Mezcla AVX2 (`ConstBlend`) dejaba el alfa del destino mezclado; el SWAR lo fuerza a 0xFF → no era bit-exacta con destino no opaco (framebuffer recién creado = alfa 0) | `OR 0xFF000000` (1 instrucción por 8 px; mezcla 0.12–0.15 ms, sin cambio medible) — `draw2d.cpp:162-180` |
| 9 | UB: `rect(…, INT_MAX)` → `t*2` desborda (UBSan); en DrawList `float(INT_MAX) = 2^31 → int` al reproducir | grosor sujeto a 2^24 — `draw2d.cpp:470`, `1112`, `1124` |
| 10 | `item_hovered()` tras un combo con el popup abierto devolvía el hover del último elemento del popup | se restaura — `ui.cpp:1025` |
| 11 | `--focus_n_` tras segmentos/elementos del popup: con la lista de Tab llena (512) borraba la parada de otro widget | `press_behavior(…, focusable=false)` — `ui.cpp:296`, `736`, `1005` |

Pruebas nuevas en `tests/test_ui.cpp`: Painter inmediato == DrawList grabada (3 semillas ×
1500 primitivas con coordenadas absurdas), fuzz UTF-8 (recuento coherente), fase del tramado
Bayer para 8 alineaciones, contorno nunca fuera de su rectángulo (3000 casos), y
regresiones de 2, 3, 4, 5, 8 y 10. `random_ops` usa ahora a veces grosor `INT_MAX` (UBSan).

### Sin corregir (conscientemente)

* Escalas 3-4 (`CFD_SCALE`) usan un bucle escalar con división por píxel en el escalado; no
  es la ruta de esta pantalla (escala 2 AVX2).
* `blit` opaco (`alpha=false`) copia el alfa de la fuente tal cual (memcpy); X11 y el PNG sin
  alfa lo ignoran.
* Composición de teclas muertas: si la tecla siguiente no compone, se descarta la tilde.

## Cambios de la fase 2 (integración de la app, ingeniero A)

Arreglos pequeños hechos al integrar el panel del túnel (no cambian el rendimiento medido arriba):

* **Asa del slider sobre el valor** (`ui.cpp`, `slider_core`): la barra vertical del asa tachaba
  los dígitos cuando el valor quedaba bajo el texto ("250 km/h" con el asa sobre la "h", "+0|0°").
  Ahora, si el asa cae sobre el texto, se dibuja como dos marcas cortas arriba y abajo del texto.
* **`Context::text_wrapped_colored(color, fmt, ...)`** (extensión compatible de `ui.hpp`): texto
  ajustado al ancho con color propio (avisos en ámbar, fuentes de datos en gris). `text_wrapped`
  usa la misma rutina (`wrapped_core`) con `text_dim`. Antes los avisos largos se truncaban con "…".
