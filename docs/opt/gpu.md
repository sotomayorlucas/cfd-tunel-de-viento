# Backend iGPU — optimizaciones y medidas (Xe-LPG, Meteor Lake)

Diseño y uso: [`docs/GPU.md`](../GPU.md). Aquí: qué limita a esta iGPU, cada truco con su efecto
**medido**, y lo que no funcionó.

> **Condiciones.** Misma máquina (Core Ultra 7 155H, 128 EU Xe-LPG a 2250 MHz, LPDDR5x compartida
> ~82 GB/s), con carga de fondo variable de otros procesos (carga media 1–17 durante el trabajo).
> Tiempos de GPU con marcas de tiempo (`vkCmdWriteTimestamp`, 52.08 ns/tic), medianas de 3–5
> rondas tras un lote de calentamiento (la iGPU baja de frecuencia a los ~50 ms de reposo). La
> frecuencia medida durante los lotes es 2250 MHz (`gt/gt0/rps_act_freq_mhz`). Reproducir:
> `build/tools/bench_gpu --bw | --cpu-read | --lbm [--res media] [--single] [--sg N] [--wg N]
> [--coherent] [--rerecord] [--fp32]` y las variables `CFD_GPU_EXP`, `CFD_GPU_3COLL`,
> `CFD_GPU_CLASSIFY`, `CFD_GPU_PX` (experimentos de §3).

## 1. Resultado

| F1 2022, FP16S, paso completo (celdas + nodos + reducción) | CPU 20 hilos AVX2 | iGPU |
|---|---|---|
| Media, 6.11 M celdas | **869 MLUPS** | **744 MLUPS** (kernel de celdas 814 MLUPS, 61.9 GB/s) |
| Media, FP32 | 451 MLUPS | **498 MLUPS** (kernel de celdas 536 MLUPS, **81.5 GB/s**) |
| Rápida, 2.57 M celdas | 865–888 MLUPS (sin carga) | 734 MLUPS (kernel 824, 62.6 GB/s) |
| Rápida, FP32 | 348 MLUPS (con carga) | 511 MLUPS (kernel 554, 84.1 GB/s) |

* En **FP32** la iGPU llega al **techo de memoria** (81–84 GB/s, lo mismo que mide la CPU) y supera
  a la CPU. En FP16S se queda en 62 GB/s: la limita el número de mensajes de memoria por byte
  (§3), no el ancho de banda.
* La ganancia real no es de MLUPS sino de **CPU libre**: con `--gpu` la CPU dibuja el cuadro
  anterior mientras la iGPU calcula el siguiente (§5).

## 2. Límites medidos del hardware

`bench_gpu --bw` (6.3 M celdas / búferes de 256 MB):

| Patrón | memoria de la GPU | subgrupo 8 | 16 | 32 |
|---|---|---|---|---|
| copia u32 (L+E) | COHERENT / CACHED | 50 GB/s | 66 | **71** |
| 19 corrientes f16 in-place (patrón LBM, 1 celda/hilo), WG 256 | COHERENT | 38 | **76** | 73 |
| ídem | CACHED | 38 | **78** | 74 |
| Esoteric-Pull real, 1 celda/hilo, sin física (`CFD_GPU_EXP=1`) | CACHED | | **85 GB/s = 1127 MLUPS** | |

* **SIMD8 no sirve** para memoria (mensajes de 8 carriles); SIMD16 es el óptimo; **SIMD32 no
  existe en el LSC**: el compilador parte cada acceso en dos `send(16)` (y además derrama).
* Los dos tipos de memoria de la GPU (`0x7` coherente y `0xb` cacheado) dan **el mismo** ancho de
  banda a la GPU. Para la CPU no: `bench_gpu --cpu-read` (96 MB escritos por la GPU):
  **WC (coherente) 0.25 GB/s** frente a **CACHED 20.6 GB/s + invalidate 1.9 ms** → todo lo que
  la CPU lee (campos ρ,u, fuerzas) y también las poblaciones (bajadas) va en CACHED.
* Envío + espera ≥ 50–100 µs (también medido por la ruta DRM, `docs/DRM.md`) → lotes de muchos pasos.

## 3. Kernel de celdas: historia de las medidas

Rápida (2.57 M celdas), sólo el kernel de celdas, subgrupo 16, WG 256. Cada fila es la mediana
medida en su momento con `bench_gpu --lbm` (las variantes ya retiradas se citan por lo que medían).

| # | Variante | MLUPS | Nota |
|---|---|---|---|
| 0 | port directo: 1 celda/hilo, cargas dentro de `if (!sólido)`, u de la salida por `subgroupShuffleUp`, tabla τ0(x) en memoria | 541 | SIMD16, 0 derrames; FP32 da lo mismo (420 MLUPS a 64 GB/s) → no limita la memoria |
| E1 | patrón de memoria Esoteric-Pull puro | 1127 | cota superior (85 GB/s) |
| E2 | E1 + las cargas dentro de un `if` sobre el flag | 709 | las cargas esperan al byte de flags: latencia de DRAM en serie |
| 1 | cargas incondicionales, sólidos reescriben su valor con `select` | 369 | las 19 cargas vivas toda la colisión: **7:25 derrames** |
| 2 | ídem con las escrituras dentro de `if (!sólido)` | 434 | 0 derrames |
| 3 | τ0(x) como constantes (sin tabla) | 505 | **+16 %**: una segunda ida y vuelta dependiente por hilo → τ0(x) se calcula en el shader (§4) |
| E3 | E1 + k multiplicaciones por población | 1121 (k=0) / ~825 (k=1…16) | la ALU no limita; lo caro es que cada valor viaja en un mensaje de 16 bits |
| — | modos de coma flotante (RTE16, preservar subnormales) | sin efecto | se dejan (paridad bit a bit con F16C) |
| E4 | **2 celdas/hilo con palabras u32 (f16×2)**, alineadas, + FMAs | **1102** | las FMAs vuelven a ser gratis: el límite era el nº de mensajes (el LSC mueve 16 carriles por mensaje; 16×2 B = 32 B) |
| 4 | modo par (2 celdas/hilo) sobre el kernel real | 666 | pero **20:24 derrames** (2 colisiones a la vez) |
| 5 | **descriptores de ubicación** (el desplazamiento de cada dirección en la base del descriptor, §4) | par 724 · 1 celda 679 (+33 %) | un solo registro de dirección para las 38 cargas/escrituras |
| — | colisión de A y luego de B (guardar A en f16) | 409 | el compilador las entrelaza igual: 119:159 derrames. Descartado |
| — | recalcular los productos por peso en cada par | 685 | sin efecto. Descartado |
| 6 | Ladd devuelve correcciones de momentos (no 18 poblaciones) | 707 | 0 derrames, pero sin ganancia: los derrames no eran el cuello |
| E5 | patrón real del modo par (impares con `d32 V2` + 2 escrituras de 16 bits), sin física | 1018–1028 | cota del modo par |
| — | bisección (`CFD_GPU_PX`): flags constantes 0 → 827; todo quitado → 873 | | el coste está en el TAMAÑO del código (rutas raras), no en la carga de flags |
| — | grupos "puros" (86 % de los grupos: flags deducidos de las coordenadas, sin cargarlos) | 666 vs 702 | **peor**: dos cuerpos → el doble de código. Queda como experimento (`CFD_GPU_CLASSIFY`) |
| 7 | **Ladd al kernel de nodos** (el de celdas pierde la ruta de paredes móviles) | 794 | **+13 %** |
| 8 | **una sola copia de la colisión** (pares mixtos sólido/fluido con escrituras de media palabra) | **835–852** | +5.6 % frente a 3 copias (`CFD_GPU_3COLL=1`: 791) |
| — | parámetros como literales (sin cargas escalares) | 826 vs 852 | sin efecto. Descartado |

Lección principal en Xe-LPG: el kernel NO está limitado por la ALU (4.6 TFLOPS) ni, en FP16S,
por el ancho de banda, sino por (a) **mensajes de memoria por byte** (datos de 16 bits → mitad de
bytes por mensaje), (b) **cargas dependientes** (cada ida y vuelta en serie cuesta ~15 %) y (c)
**tamaño del binario**: el caché de instrucciones es pequeño y cada ruta rara (paredes móviles,
pares mixtos) intercalada en el binario resta un 5–15 % aunque nunca se ejecute. En FP32 los
mensajes ya llevan 64 B y el mismo kernel llega al techo de memoria.

Barrido final a Media (6.11 M): subgrupo 16 + WG 256 **812**, WG 128 804, WG 512 827 (dentro
del ruido), subgrupo 32 **187** (derrames masivos), subgrupo 8 428, memoria coherente 813 (=),
1 celda/hilo (`--single`) 645.

## 4. Trucos que quedaron (con archivo)

| Truco | Dónde | Efecto medido |
|---|---|---|
| SPIR-V propio especializado al generar: paridad, macro, precisión, colisión, tamaño del dominio (divisiones por constantes → multiplicaciones mágicas), sin ramas por configuración | `lbm_kernels.cpp` | — (base) |
| Tamaño de subgrupo obligatorio 16 (`VK_EXT_subgroup_size_control`, `REQUIRE_FULL_SUBGROUPS`) | `vk.cpp create_pipeline` | SIMD16 vs 8: ×1.9; vs 32: ×4.3 |
| **2 celdas por hilo, palabras f16×2**; direcciones impares con 2 palabras contiguas (el compilador las fusiona en `d32 V2`) + 2 escrituras de 16 bits | `gen_step_pair` | 645 → 812 MLUPS (Media) |
| **Descriptores de ubicación**: 19 vistas por paridad con `ranura·S + P + desplazamiento` en la base (alineación mínima 4 B; impares con base par e índice +1); 2 conjuntos de descriptores alternos | `make_set`, `location()` | +33 % (1 celda), +9 % (par) |
| Cargas incondicionales (sin dependencia del flag) y escrituras en rama | `gen_step_*` | E2 709 vs E1 1127; 434 vs 369 |
| τ0(x) de la esponja calculado en el shader (mismas operaciones que `build_tau`) | `tau_at` | +16 % |
| u de la salida: en modo par, la celda A del mismo hilo; en modo 1 celda, `subgroupShuffleUp` | `gen_step_*` | sin memoria compartida ni recarga |
| Ladd de paredes móviles implícitas escrito por el kernel de nodos (+ corrección en los relevos) | `gen_boundary`, `ladd_fixup` | +13 % |
| Una sola copia de la colisión | `gen_step_pair` | +5.6 % |
| Fuerzas: registros (nodo, id) ordenados por id, trozos de ≤ 256 de un id, `subgroupAdd` + memoria compartida, suma final en doble en la CPU (determinista, sin atómicos de coma flotante) | `gen_reduce`, `publish` | 0.06 ms/paso a Media |
| Divergencia: `subgroupAny` + `subgroupElect` → un atómico por subgrupo como mucho | `gen_step_*` | — |
| Lotes de n pasos en un búfer de comandos; parámetros por paso en una tabla indexada por la constante de empuje → **grabado una vez y reenviado** | `record`, `command_buffer` | grabar cuesta 31–168 µs (1–100 pasos); con reutilización 0 |
| Semáforo de línea temporal (sin vallas), `synchronization2` para las barreras | `vk.cpp` | — |
| La reducción del paso i sin barrera con el kernel de celdas del paso i+1 | `record` | se solapan |
| `cycle()`: el lote siguiente se encola ANTES de publicar el anterior (campos en mitades alternas, fuerzas en regiones alternas) | `lbm_gpu.cpp` | la GPU no espera los ~4 ms de publicación por cuadro (Media) |
| Memoria CACHED para todo (UMA, sin copias) | `upload` | CPU lee campos a 20.6 GB/s (WC: 0.25) |

Lotes (Media, tiempo de GPU): 1 paso por envío **444 MLUPS**, 4 → 707, 16 → 735, 100 → 744 (con 1
paso la GPU se enfría durante cada hueco de publicación). Relevo en caliente CPU↔GPU (cambio de
geometría): bajada 20 ms + subida 30 ms; primer enganche 120 ms (+ compilación de los 7 pipelines la
primera vez; luego caché de sombreadores de Mesa).

## 5. Solapamiento CPU/GPU en la app

`build/cfd --res media --frames 300 --spf N [--gpu]` (ventana sin servidor X: bucle interactivo
completo —simulación, visualización, render, UI— sin presentar), F1 2022, Media:

Dos rondas intercaladas CPU/iGPU (carga de fondo de otros procesos 5–15 durante la medida: perjudica
más al solver de la CPU, que usa los 20 hilos, que a la iGPU). Medianas de 300 cuadros:

| Pasos por cuadro / visualización | CPU sólo: cuadro (FPS) | iGPU + CPU dibujando: cuadro (FPS) | Ganancia |
|---|---|---|---|
| 4 fijos, visualización por defecto | 49.5 / 52.4 ms (20.2 / 19.1) | 45.5 / 38.3 ms (22.0 / 26.1) | +9 … +37 % |
| 8 fijos | 100.5 / 100.7 ms (9.9 / 9.9) | 71.7 / 74.3 ms (14.0 / 13.5) | **+36 … +41 %** |
| 4 fijos + humo + volumen de vórtices + líneas de corriente | 86.5 / 90.0 ms (11.6 / 11.1) | 43.9 / 53.7 ms (22.8 / 18.6) | **+68 … +97 %** |
| adaptativo (defecto de la app) | 1 paso/cuadro, 35.0 / 32.5 FPS → 35 / 33 pasos/s | 1–3 pasos/cuadro, 45.9 / 37.3 FPS → 46 / 112 pasos/s | más FPS y hasta 3× pasos/s |

En modo iGPU la columna "sim" del cuadro es sólo la espera al lote en vuelo (p.ej. 8 pasos: 52–59 ms
de espera frente a 84 ms de simulación en la CPU): el dibujo (8–25 ms) queda escondido detrás de la
GPU. Con la visualización pesada la CPU ya no tiene que repartir sus núcleos entre el solver y el
dibujo, y ahí la ganancia es máxima. Sin carga de fondo y con la visualización por defecto la
ventaja se reduce (la CPU sola es ~17 % más rápida en MLUPS FP16S que la iGPU).

## 6. Lo que no se hizo / límites

* Kernel de nodos: SIMD8 con derrames (45:27), 0.67 ms/paso a Media (8 %). Precalcular en la CPU
  los datos de los vecinos (máscaras, ranura de id por enlace) lo empeoró (0.77–0.83 ms: más
  derrames y más palabras por nodo). Lo siguiente sería un bucle real sobre los enlaces (código
  18× más corto) con una vista única de las poblaciones.
* FP16S a 62 GB/s: para llegar al techo harían falta 4 celdas por hilo (mensajes de 16 B por
  carril), con el doble de registros → derrames con 128 GRF. No probado.
* Sin reparto del dominio CPU + GPU a la vez (mismo ancho de banda: poca ganancia esperable).
