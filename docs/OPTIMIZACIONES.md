# Optimizaciones — índice

Resumen de los trucos de rendimiento más importantes del proyecto, módulo a módulo, con la
ganancia **medida** en el Intel Core Ultra 7 155H (6 P + 8 E + 2 LP-E, AVX2/FMA/F16C/BMI2, sin
AVX-512, ~82 GB/s de ancho de banda RMW). El detalle completo (archivo:línea, metodología, A/B
intercalados, experimentos descartados) está en cada documento de módulo:

| Módulo | Documento | Resultado clave |
|---|---|---|
| Solver LBM | [`docs/opt/lbm.md`](opt/lbm.md), [`docs/BENCHMARKS_LBM.md`](BENCHMARKS_LBM.md) | ~1000 MLUPS FP16S (94 % del techo de memoria), ~505 MLUPS FP32; ~800 MLUPS con un F1 y la pasada de pared de la fase 2; refinamiento local: taps en el kernel, escrituras sólo de las direcciones leídas y fantasmas AVX2 (§6: interfaz 11.7 → 7 ms/paso a Media) |
| Voxelizador y malladores | [`docs/opt/geom.md`](opt/geom.md) | coche F1 voxelizado en ~6 ms y mallado (333 k triángulos) en ~20 ms con 20 hilos |
| Modelos | [`docs/opt/models.md`](opt/models.md) | `build` de un coche en 0.06-0.25 ms; eval del SDF −20 % |
| Rasterizador | [`docs/opt/raster.md`](opt/raster.md) | 300 k triángulos en 2.2 ms a 1500×1150 |
| Visualización del flujo | [`docs/opt/flowvis.md`](opt/flowvis.md) | 2000 líneas × 400 pasos en 3.9 ms; 300 k partículas en ~2 ms |
| UI + plataforma | [`docs/opt/ui.md`](opt/ui.md) | panel completo en ~0.1 ms; escalado 2× a 4K en ~0.7 ms |
| Aplicación | este documento, §7 | cuadro completo sin asignaciones; actualizaciones sólo cuando cambian el campo o los ajustes |

> **Ruido de medida.** Toda la sesión de desarrollo hubo carga ajena en la máquina (qemu, rustc,
> otros agentes: carga media 2-27). Las cifras son medianas (o mínimos, cuando se indica) de
> comparaciones A/B **intercaladas**; algunas decisiones de cómputo se validaron además con
> `llvm-mca` (determinista). Donde el ruido impidió concluir, los documentos lo dicen.

---

## 1. Solver LBM (`src/lbm/solver.cpp`)

El kernel está **limitado por memoria** con el chip completo: el objetivo fue mover los mínimos
bytes por celda y paso y que 6 P-cores ya saturen el bus.

| Truco | Efecto medido |
|---|---|
| **Esoteric-Pull in-place** (Lehmann 2022): una sola copia de las poblaciones, 19 punteros por paso sin ramas por paridad | la mitad de memoria y de tráfico que A-B: 187 MB para 3.1 M celdas FP16S |
| **FP16S**: poblaciones desplazadas `f − w`, escaladas ×2¹⁵ y guardadas en half con F16C (`vcvtph2ps`/`vcvtps2ph`); aritmética en FP32 | 76 B/celda·paso frente a 152: **1.8-2× más MLUPS**; Cd de la esfera a 0.43 % de FP32 |
| **Sesgo de zancada** k·4 KiB + 3 líneas entre las 19 corrientes (evita el aliasing de conjuntos de L1/L2) | **+10…+17 %** FP16S, +2.5 % FP32 |
| **Relleno** delante/detrás de cada dirección en vez de índices envueltos | cero comprobaciones de límites en el kernel |
| **Clases de bloque SWAR** (8 flags en un `u64`): Skip / Pure / Masked / Ground / Scalar | los bloques sólidos no se tocan; sólo los vecinos de ruedas van por la ruta escalar |
| **Carriles sólidos por mezcla** con el valor recargado (`vpblendvb`, máscara 32→16 bits con `vpackssdw`) | vectoriza bloques con sólidos y fronteras |
| **Salida vectorizada** (u(x−1) con `vpermps`) y **equilibrio exacto sin ramas** en fronteras | entrada/salida/campo lejano en la ruta AVX2 |
| **Momentos por pares** (s = fᵢ+fᵢ₊₁, d = fᵢ−fᵢ₊₁) en dos cadenas independientes; **escala FP16 plegada** en los pesos | instrucciones del kernel FP16-Reg 1403 → 694; +5-11 % por núcleo |
| **Ruta crítica corta** (árbol para \|Π\|², (1−ω) con una FMA final) | llvm-mca: 143 → 129 ciclos por bloque de 8 (FP16-Reg) |
| **Cinta móvil vectorizada** (corrección de Ladd constante en z = 1) | +4…+18 % por núcleo en el caso coche |
| **Plantillas** `<Precisión, Colisión, Macro, Masked, Ground>` (loop unswitching) | sin ramas por celda salvo la clase del bloque |
| **ρ,u fusionados** en el último paso del `step(n)` | sin pasada macro extra |
| **Divergencia a coste ~0** (`_CMP_NGT_UQ`, un `movemask` por fila) | detecta NaN/ρ fuera de rango sin coste medible |
| **Grano dinámico ~4096 celdas** para CPU híbrida P/E | 1 fila −23 %, 4 filas −7…−12 % frente a 16 filas |
| **Fuerzas** por lista compacta ordenada por id (ordenación por conteo), trozos fijos → suma determinista | 0.14 ms/paso (3-4 % del paso); fuerzas idénticas bit a bit con 1 y 20 hilos. **Fase 2**: con el rebote interpolado (Bouzidi) y la ley de pared, la pasada de pared/fuerzas lee las 19 poblaciones de cada nodo de pared: 1.0-1.2 ms/paso a media (~17 % del núcleo; medido 12:15 y 13:17 con la máquina casi en reposo) |

Probado y **descartado** con medida: prefetch software (±3 %), stores no temporales para ρ,u
(−2.5 %), pares de bloques entrelazados (−1 %), recorrido en z descendente (−15…−20 %), fijar
afinidad con carga ajena (−10…−30 %), usar los 2 LP-E (−5…−15 %).
Recomendación para este equipo: pool por defecto de 20 hilos sin fijar, FP16S + Regularizado + LES.

## 2. Geometría (`src/geom/voxelizer.cpp`, `mesher.cpp`)

El coste está en el SDF: la mejor optimización es **no evaluar**.

| Truco | Efecto medido |
|---|---|
| **`RegionEval`**: evaluación de la escena restringida a una caja con tres podas demostradas exactas (cota de caja, cota heredada del padre, testigo de cualquier índice) | −35 % de evaluaciones del SDF |
| **Salto jerárquico de bloques 8³→4³→2³** con cotas de Lipschitz (todo fluido / todo sólido / subdividir) | voxelizado 186 → 53.5 ms CPU (×3.5); 6 ms de pared con 20 hilos |
| Submuestreo 2×2×2 sólo donde el voto **puede** cambiar + **voto con salida anticipada** | −38 % de tiempo de voxelizado |
| Mallado *Surface Nets* en **banda estrecha**, signos en **bitboards** (1 bit/muestra) y cubos mixtos de 64 en 64 (SWAR) | 1.1-1.3 s → 170-227 ms CPU (×6-7); 19-27 ms de pared |
| Índice de vértice por rango con **`popcount(BZHI(...))`** (BMI2) en vez de una rejilla densa | memoria 19.5 MB → 0.3 MB |
| Malla de vóxeles con máscaras **AVX2** (32 celdas por instrucción) + *greedy meshing* | F1: 86 k → 8.5 k triángulos; 0.25 ms |

## 3. Modelos (`src/models/`)

| Truco | Efecto medido |
|---|---|
| Perfiles NACA con 14 puntos por cara (espaciado coseno) en vez de 40 | eval del SDF −20…−23 %; `build` 5.4× más rápido |
| Colocación de flaps con **regula falsi de Illinois** (ranura exacta, tolerancia 1 µm) | `build` 5.3-10× más rápido (0.03-0.13 ms) |
| Geometría "amiga de las AABB" (`mirror_y`, ≤ 13 grupos compactos) | 3.3 grupos evaluados por punto de media en el F1 2022 |
| Perfiles con **espesor mínimo** de 4.2 cm (no es velocidad: robustez a dx = 3 cm) | 0 cortes de ala con fuga D3Q19 (frente a 50-70 % con 1.5 cm) |

## 4. Rasterizador (`src/render/raster*.cpp`)

| Truco | Efecto medido |
|---|---|
| Punto fijo 28.4 + funciones de arista enteras + regla top-left **sin ramas** | −19 % por triángulo pequeño; estanco (0 grietas) |
| **Sellos 4×2** para triángulos pequeños (90-97 % de una malla densa) | raster −16 % (20 hilos) |
| Binning por tiles 64×64 con **ordenación por conteo sin atómicos**, listas frontal/trasera, **Hi-Z** para caras traseras, planificación **LPT** | `draw_mesh` 300 k triángulos: 4.06 → **2.19 ms** |
| **Sombreado diferido** por tile con G-buffer FP16 (F16C) de 25 KB por hilo | sólo se sombrea el fragmento visible |
| `pow(x,n) ≈ max(0, 1 − n(1−x)/8)⁸`, `vrsqrtps` sin Newton, mezcla SWAR en carriles de 16 bits, suma saturada `vpaddusb` para partículas | 200 k splats en 1.1 ms |
| FXAA-lite con detección de bordes en u8 saturado (32 píxeles por instrucción) | 0.85 ms a 1500×1150 |

## 5. Visualización del flujo (`src/render/flowvis*.cpp`)

| Truco | Efecto medido |
|---|---|
| **Campo empaquetado AoS FP16** (ux, uy, uz, ρ−1 = 8 B/celda): la trilineal de 4 componentes son 4 cargas de 128 bits + 4 `VCVTPH2PS` | muestreo aleatorio 5.7-7× más rápido que FP32 SoA; líneas 2.1-2.3× |
| Empaquetado con transposición SSE + **stores no temporales** | 0.8 ms para 3.1 M celdas (~1.3× frente a stores normales) |
| **K = 4 cadenas RK4 entrelazadas** por hilo (ILP sobre la latencia de las muestras) | 2.8× por hilo |
| Humo: SoA + anillo + emisión "virtual" sin pulsos + compactación por trozos + prefetch | 300 k partículas en 1.4-2.6 ms |
| Magnitudes derivadas por filas en AVX2 (diferencias centradas, Q = −½ Σ gᵢⱼgⱼᵢ) | \|ω\| ~5.7×, Q ~5× frente a escalar |
| Volumen de vórtices: **ladrillos 8³** + recorte de rayos a la caja ocupada + media resolución con reescalado bilineal SWAR vía **PDEP/PEXT** | render 1500×1100 en ~2 ms (3.5× por el recorte, 2.5× por los ladrillos) |
| LIC con 8 subtexeles por registro (`VPGATHERDPS`) | 1 hilo 123 → 39 ms |

## 6. UI y plataforma (`src/render/draw2d.cpp`, `src/ui/`, `src/platform/`)

| Truco | Efecto medido |
|---|---|
| Texto con **LUT constexpr byte → máscara de 8 carriles**: 1 `VPMASKMOVD` por fila de glifo | ×3.9-4.5 |
| Conteo de glifos UTF-8 por **SWAR + popcount** | ×5.4-6.2 |
| **DrawList** grabada (sin asignaciones) y reproducida en paralelo por franjas, bit a bit idéntica a la serie | render del panel ×3.2-4.6 |
| Mezcla alfa AVX2 a 16 bits por canal, bit-exacta con el SWAR escalar | ×2.2-3.8 frente a float |
| Escalado 2× AVX2 (`VPERMD`) + **stores no temporales** hacia la imagen MIT-SHM, en paralelo | 1920×1200 → 3840×2400 en 0.6-1.0 ms |
| MIT-SHM con doble búfer y espera de `ShmCompletion` sin consumir la entrada | presentación asíncrona (XShmPutImage ~0.02 ms) |

## 7. Aplicación (`src/app/`)

Cifras extremo a extremo (cuadro completo, FPS, MLUPS por fase) en el [README](../README.md#rendimiento).

| Técnica | Dónde | Efecto medido |
|---|---|---|
| **Cero asignaciones por cuadro**: búferes persistentes en todos los módulos (malla, flowvis, raster, DrawList, UI); textos formateados en `char[]` de pila (`SiStr fmt_si`, HUD con líneas en la pila); historia en anillos fijos | `app.cpp`, `panel.cpp`, `view.cpp` | **0 llamadas a malloc/calloc/realloc/aligned_alloc en 30 cuadros** de régimen (coche con humo + líneas + huella; ala con corte Cp + vórtices; esfera con LIC + nube Q), verificado en `tests/test_app.cpp` [7] **interceptando malloc** en todo el proceso (hilos del pool incluidos). Con ventana X11 (contador `LD_PRELOAD`): las únicas asignaciones en régimen son 2 por cuadro **dentro de libxcb** (`XFlush` → registro de petición "checked" que Xlib pide para cada petición), fuera de nuestro código |
| **Visualización limitada en frecuencia**: un campo nuevo se refleja en muestreador FP16, líneas, corte, huella, volumen Q y color de la superficie como mucho cada 50/83/167 ms (fluidez/equilibrado/máx. simulación); el humo avanza cada cuadro con el último muestreador | `view.cpp` (`View::update`, `field_interval`) | a 30 FPS la actualización (2-4 ms a media, 6-10 ms a alta) se paga en 1 de cada 2-3 cuadros → ese tiempo va al solver |
| **Pasos de red adaptativos con parte mínima garantizada**: k = presupuesto / t_paso con presupuesto = máx(T_objetivo − t_resto, t_resto·s/(1−s)); s = 25/50/80 % según la prioridad; suavizado ×1.5 | `app.cpp` (`adaptive_steps`) | con mallas grandes el FPS baja en vez de congelar la simulación (antes: 1 paso/cuadro y 10 pasos/s con render lento) |
| **Redibujo en reposo a ~10 Hz**: en pausa y sin entrada no se re-renderiza ni presenta cada cuadro (se cede la CPU con `usleep`) | `app.cpp` (`App::frame`, `idle_throttle`) | CPU ~0 en pausa sin perder la respuesta a la entrada |
| **Reconstrucción con antirrebote** al arrastrar deslizadores (≤ 8/s, malla gruesa 1·dx mientras se arrastra, fina 0.5·dx al soltar) y `same_geometry` para no re-voxelizar sin necesidad; el flujo NO se reinicia | `app.cpp` (`apply_ui_params`), `sim.cpp` | reconstrucción completa (modelo + vóxeles + solver + malla) ≈ 70 ms a media |
| **Movimientos de pared incrementales**: sólo se llama a `set_wall_motion` cuando cambia el estado de un id (cada cambio reconstruye listas del solver) | `sim.cpp` (`apply_wall_motions`) | 1 reconstrucción en vez de 3 por cambio de parámetros |
| **Sonda** por la profundidad del framebuffer (sin trazar rayos contra la malla) | `view.cpp` (`unproject`) | O(1) por cuadro |
| **Suelo del pabellón**: plano infinito con el mismo rasterizador de suelo por píxel, 2 cm bajo el del túnel (gana el test de profundidad sin sesgo) | `view.cpp` (`View::render`) | el borde del dominio ya no corta la escena en vistas laterales |
| **LTO + `-fipa-pta` + `-fdevirtualize-at-ltrans`**; **PGO** con contadores `-fprofile-update=single` + `-fprofile-correction` (con `=atomic` los 20 hilos se pelean por las mismas líneas de caché: el primer caso de la carga de perfilado tardó 376 s frente a 111 s con `single`, ambos con carga ajena ~30; `-fprofile-correction` es imprescindible con `single`, si no GCC rechaza el perfil por recuentos inconsistentes) | `Makefile` | medido (4 rondas intercaladas, carga 13-31, mejor de 4): LTO 543 MLUPS / 54.7 ms por cuadro, PGO 542 / 61.0 ms → **sin ganancia medible** (núcleo limitado por memoria); binario 2.0 → 1.5 MB |
| **`make unity`** (una sola TU): `x11.cpp` al final (las macros de Xlib como `None` rompían el resto) | `Makefile` | 575 MLUPS / 61.8 ms en la misma medida → **sin ganancia medible** frente a LTO; compila en ~35 s (LTO paralelo ~15 s). Repetido con el solver de la fase 2 (13:20): la carga ajena (18-37) impidió concluir; sigue compilando sin avisos |
| **Flecha de carga en el punto del balance** (x = eje del. + (1 − balance)·batalla) y flechas paralelas a la vista omitidas | `view.cpp` | la flecha no se sale del coche con poca carga ni aparece como un disco en vistas frontal/trasera/superior |
| **Umbral de vórtices adimensional** Q·L²/U² (convertido a Q·dx²/U² de flowvis con L/dx) | `view.cpp`, `panel.cpp` | el mismo umbral muestra los mismos vórtices en cualquier resolución |
| **Escala de cuantización del volumen Q que sigue al umbral** (`VolumeParams::full` ≥ 4·umbral, en escalones ×2): el volumen se guarda en 8 bits hasta `full`; con el umbral por encima todo saturaba (isosuperficies en bloques e idénticas para cualquier umbral). Los escalones ×2 evitan recalcular el volumen (~2-10 ms) en cada movimiento del deslizador | `view.cpp` (`View::update`) | corrección visual; recálculo sólo al cruzar un escalón (comprobado en `tests/test_app.cpp` [9]) |

## Backend iGPU (fase 3, `src/gpu/`)

Detalle y medidas: [`docs/opt/gpu.md`](opt/gpu.md); diseño: [`docs/GPU.md`](GPU.md). Resumen de trucos
(F1 2022, Media, kernel de celdas FP16S en la Xe-LPG):

| Truco | Archivo | Efecto medido |
|---|---|---|
| Vulkan sin cabeceras (ABI a mano + `dlopen`) y SPIR-V generado por el propio programa, especializado por paridad/precisión/dominio | `src/gpu/vk.*`, `spirv.*`, `lbm_kernels.cpp` | — |
| Subgrupo 16 obligatorio | `vk.cpp create_pipeline` | ×1.9 frente a 8; ×4.3 frente a 32 (derrames) |
| 2 celdas por hilo con palabras f16×2 | `lbm_kernels.cpp gen_step_pair` | 645 → 812 MLUPS |
| Desplazamiento de cada dirección en la base del descriptor (2 juegos por paridad) | `lbm_gpu.cpp make_set` | +33 % (1 celda/hilo) |
| Cargas sin dependencia del flag; escrituras en rama | `gen_step_*` | 369/434 frente a derrames; E2 709 vs 1127 |
| τ0(x) en el shader | `tau_at` | +16 % |
| Ladd de paredes móviles en el kernel de nodos (código del kernel de celdas más corto) | `gen_boundary`, `ladd_fixup` | +13 % |
| Una sola copia de la colisión | `gen_step_pair` | +5.6 % |
| Lotes de n pasos, búfer de comandos reutilizado, semáforo de línea temporal | `lbm_gpu.cpp` | 1 paso/envío 444 → 100 pasos 744 MLUPS |
| `cycle()`: encolar antes de publicar; campos en doble búfer HOST_CACHED | `lbm_gpu.cpp`, `app/sim.cpp` | CPU dibuja en paralelo: +36–97 % FPS a Media |
