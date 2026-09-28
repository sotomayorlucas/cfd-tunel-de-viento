# Backend iGPU del solver LBM (fase 3) — Vulkan de cómputo propio, sin bibliotecas

El solver D3Q19 puede avanzar en la gráfica integrada (Intel Arc de Meteor Lake, **Xe-LPG**, PCI
`8086:7d55`, 128 EU, controlador `i915` + Mesa ANV) con **código propio de principio a fin**:

* **sin cabeceras de Vulkan** (no están instaladas): `src/gpu/vk.hpp` declara a mano sólo los
  tipos, enumeraciones, estructuras (ABI x86-64 comprobada con `static_assert` de tamaños y
  desplazamientos) y punteros a función que se usan; `libvulkan.so.1` se carga con `dlopen`
  y todo lo demás con `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr` (las funciones de
  dispositivo van directas al controlador, sin el trampolín del cargador). Sin capas de validación.
* **sin compilador de shaders** (no hay glslang/shaderc/spirv-tools): `src/gpu/spirv.hpp` es un
  emisor de SPIR-V 1.5 escrito aquí — escribe las palabras binarias por secciones — con un DSL
  tipado (`F`, `U`, `B`, búferes f32/f16/u32/u8, memoria compartida, if/else estructurados,
  operaciones de subgrupo, GLSL.std.450) en el que los kernels del LBM se escriben como C++.
  Los bucles se desenrollan al generar: **el generador es el preprocesador**; cada pipeline se
  especializa (precisión, colisión, paridad, macro, tamaño del dominio) con constantes literales.
* La ruta DRM cruda (`src/gpu/drm_i915.*`, `docs/DRM.md`, de otro ingeniero) sirvió para medir
  latencias de envío y la frecuencia; la ruta de cómputo real es Vulkan (compilador de Mesa
  para el ISA de Xe; escribir ISA de Xe a mano no es razonable).

Optimizaciones y medidas detalladas: [`docs/opt/gpu.md`](opt/gpu.md).

## Archivos

| Archivo | Contenido |
|---|---|
| `src/gpu/vk.hpp/.cpp` | binding mínimo de Vulkan + `vk::Context`: instancia (sin capas), elección de la iGPU (se descarta llvmpipe), características activadas por cadena `Vulkan11/12/13Features` (16/8 bits en búferes, `shaderFloat16`, `shaderInt8/16`, semáforos de línea temporal, `synchronization2`, `subgroupSizeControl`, `computeFullSubgroups`, `hostQueryReset`), búferes mapeados persistentes, pipelines con tamaño de subgrupo obligatorio, conjuntos de descriptores, barreras, semáforo de línea temporal, marcas de tiempo |
| `src/gpu/spirv.hpp/.cpp` | emisor SPIR-V (`Module`) + DSL (`Kernel`) |
| `src/gpu/lbm_kernels.hpp/.cpp` | generación de los shaders del LBM: celdas (1 o 2 celdas por hilo), nodos de pared, reducción de fuerzas; experimentos de medida |
| `src/gpu/lbm_gpu.hpp/.cpp` | `gpu::LbmGpu`: memoria, subida/bajada de estado, lotes de pasos, publicación de fuerzas y campos |
| `src/lbm/solver.hpp/.cpp` | extensión compatible (fase 3): `WallNode` público, `ExternalView`, `set_external`, `revision`, `ramp_at`, `external_commit`, `set_field_override` |
| `tests/test_gpu_spirv.cpp` | el emisor contra la GPU real (6 grupos de pruebas) |
| `tests/test_gpu.cpp` | paridad GPU ↔ CPU (9 casos) + estabilidad y fuerzas del F1 2022 |
| `tools/bench_gpu.cpp` | `--bw` (anchos de banda), `--cpu-read` (lectura de la CPU), `--lbm` (MLUPS, perfil por kernel) |

## Uso

```sh
build/cfd --gpu                      # ventana: la GPU simula mientras la CPU dibuja
build/cfd --headless --gpu --steps 1500
build/cfd --gpu --spf 8 --frames 300 # medidas con pasos por cuadro fijos
build/tools/bench_gpu --lbm --res media
```

En el panel: **Configuración → Solver: CPU / iGPU** (se puede cambiar en caliente: el estado
pasa de un lado a otro sin reiniciar el flujo). Si no hay dispositivo Vulkan utilizable, el
conmutador avisa y el solver sigue en la CPU; los tests pasan con una nota.

## Diseño

### El solver de la CPU sigue siendo el dueño del estado

`lbm::Solver` conserva la configuración, la geometría (flags, ids, nodos de pared con su `q`
de Bouzidi, normal, distancia y factor de arista calculados con la SDF), la rampa, el tiempo y
las fuerzas publicadas. `LbmGpu` se engancha (`Solver::set_external`) y avanza los pasos con
**su propia copia** de las poblaciones en memoria de la iGPU (UMA: la misma LPDDR5x):

* Antes de que el solver lea o modifique su estado (`set_geometry`, `set_wall_motion`,
  `set_ground`, `set_viscosity`, `set_wall_model`, `reset_flow`, `init`, `step` en la CPU,
  `total_mass`) llama al **gancho**: el backend termina el lote en vuelo, lo publica y baja
  las poblaciones a la CPU (20 ms a Media). La operación de la CPU cambia `revision()` y el
  siguiente lote vuelve a subir todo (30 ms a Media).
* `Solver::external_commit` publica pasos, fuerzas (último paso y media del lote),
  divergencia y MLUPS; `set_field_override` hace que `field()` devuelva los campos macro de la
  GPU (sin copia).
* Así la app no cambia: toda la lógica de modelos, barridos, fuerzas y visualización sigue
  hablando con `lbm::Solver`.

### Memoria (UMA, todo mapeado)

| Búfer | Tipo | Nota |
|---|---|---|
| poblaciones (19·S elementos, misma disposición que la CPU) | `DEVICE_LOCAL\|HOST_VISIBLE\|HOST_CACHED` | la GPU las lee igual de rápido que de la memoria coherente; la CPU las baja 80× más rápido |
| campos macro ρ, u (2 mitades) | CACHED | la CPU los lee para dibujar: **WC = 0.25 GB/s, CACHED = 20.6 GB/s** + invalidate 1.9 ms/96 MB |
| flags, ids, nodos de pared, tabla de movimientos, parámetros, fuerzas por trozo | CACHED | escritos por la CPU con `vkFlushMappedMemoryRanges` |

No hay copias por PCIe ni búferes de subida: la CPU escribe/lee directamente la memoria que
usa la GPU (con flush/invalidate porque el tipo cacheado no es coherente en MTL).

### Tres kernels por paso

1. **Celdas** (el 90 % del tiempo): Esoteric-Pull in-place, idéntico a `rows_kernel`:
   momentos en pares, entrada/campo lejano/salida de equilibrio (la salida copia u de x−1),
   Regularizada o BGK + Smagorinsky con τ0(x) de la esponja, ley de pared/amortiguamiento
   (`kNearWall`), macro ρ,u en el último paso del lote, detección de divergencia (un atómico
   por subgrupo como mucho). **Dos celdas contiguas en x por hilo** con palabras f16×2 (ver
   `docs/opt/gpu.md`). Rebote implícito: las celdas sólidas no escriben.
2. **Nodos de pared** (un hilo por `WallNode`): Bouzidi lineal, modelo Slip (imposición de la
   tensión de Werner-Wengle), ruedas interpoladas con término de pared móvil y corrección de masa,
   **término de Ladd** de las paredes móviles implícitas (cinta; ruedas sin rebote interpolado)
   escrito ya en la población entrante, y fuerza/momento por (nodo, id) con referencia manométrica
   y corrección galileana (Wen et al.). Hasta 4 ids por nodo (en el F1 2022 ningún nodo pasa de 4;
   el backend lo cuenta y avisa).
3. **Reducción**: registros (nodo, id) ordenados por id en trozos de ≤ 256 de un mismo id; un
   grupo por trozo, suma por subgrupo + memoria compartida → 6 floats por trozo y paso. La CPU
   suma los trozos en doble precisión (como la CPU suma sus acumuladores por hilo).

Entre kernels hay una barrera global de cómputo (`vkCmdPipelineBarrier2`). La reducción del paso
*i* no tiene barrera con el kernel de celdas del paso *i+1* (no comparten datos): se solapan.

### Lotes, comandos y solapamiento con el dibujo

* Un lote = n pasos (hasta 2048) en **un** búfer de comandos. Los parámetros de cada paso
  (u∞ de la rampa, escala de las paredes móviles, velocidad de la cinta, región de fuerzas) van
  en una tabla indexada por la constante de empuje del paso, así el búfer de comandos depende sólo
  de (n, paridad inicial): se **graba una vez y se reenvía** (caché de 16).
* Sincronización: un **semáforo de línea temporal** (sin vallas); `submit` devuelve el valor a
  esperar. Marcas de tiempo al principio y al final de cada lote (y por kernel con `profile`).
* `LbmGpu::cycle(n)`: espera el lote anterior, **encola el siguiente enseguida** y sólo entonces
  publica el terminado (campos macro en mitades alternas, fuerzas en regiones alternas): la GPU no
  espera a la publicación. En la app (`Sim::gpu_async`, bucle interactivo) cada cuadro hace
  `cycle(k)` y dibuja el campo del lote anterior mientras la GPU calcula el siguiente; los
  resultados llevan un lote de retraso. Fuera del bucle interactivo `Sim::step` es síncrono.
* Pasos por cuadro adaptativos con la iGPU: el lote se dimensiona con el **tiempo de GPU por
  paso** para durar lo que el resto del cuadro (o el objetivo de FPS), no con el tiempo de espera.

### Convención de Ladd en los relevos CPU ↔ GPU

La CPU suma el término de pared móvil implícita al **cargar** la población que llega del sólido;
la GPU lo deja **escrito** por el kernel de nodos (así el kernel de celdas no lleva la ruta de
paredes móviles: código más corto, +13 %). En cada relevo, `LbmGpu` corrige esas posiciones en
la copia (subida: −Sc·l(t); bajada: +Sc·l(t)) con las mismas fórmulas: en FP32 el relevo es
exacto (test [7] y [9]); en FP16S añade un redondeo a 16 bits en esas posiciones.

## Paridad (tests/test_gpu.cpp)

Dos `lbm::Solver` idénticos, uno en la CPU y otro en la GPU, dominio 96×48×32, 120–240 pasos:

| Caso | |Δu|/u∞ | |Δρ| | |ΔF|/|F|máx |
|---|---|---|---|
| [1] FP32 Regularizada, sin suelo, rebote implícito | 3.4e-6 | 1.2e-7 | 1e-6 |
| [2] FP32 BGK | 7.1e-6 | 3.0e-7 | 6e-6 |
| [3] FP32 escena completa: cinta, Bouzidi + Slip, ruedas girando impermeables con huella, 2 cuerpos que se tocan (76 nodos multi-id), rampa | 9.9e-7 | 1.2e-7 | 1e-7 |
| [4] FP16S escena completa | 1.2e-3 | 9e-5 | 2.5e-4 |
| [5] FP32 LogLaw + implícito + ruedas con Ladd | 3.2e-6 | 2.4e-7 | 2.4e-5 |
| [6] FP16S suelo fijo + LogLaw + Bouzidi (120 pasos) | 4.6e-3 | 1.2e-4 | 5e-4 |
| [7] set_geometry, set_viscosity, set_smagorinsky, reset_flow con un lote en vuelo, detach → CPU | ≤ 1.7e-6 | 1.2e-7 | — |
| [9] 8 lotes solapados (`cycle`) con cambio de geometría en vuelo y rampa | 5e-7 | 1.2e-7 | 6e-8 |

Diferencias de FP32 = orden de las operaciones y contracción en FMA del compilador de la GPU. En
FP16S la diferencia GPU–CPU es **del mismo orden que la de la CPU consigo misma** con u∞·(1+10⁻⁶)
(caso [4]: 1.25e-3 frente a 1.05e-3): el redondeo a 16 bits amplifica cualquier diferencia de
1 ulp. El test imprime esa referencia en cada caso. El caso [6] es físicamente sensible (una
perturbación de 10⁻⁶ crece ×100 en 240 pasos junto al suelo fijo), por eso se compara a 120 pasos.
La conversión f32→f16 de la GPU es **bit a bit igual a F16C** (RTE, subnormales incluidos:
test_gpu_spirv [2], 65 536 valores con empates).

F1 2022 a resolución rápida, configuración de la app (FP16S, Bouzidi + Slip, cinta, ruedas),
600 pasos: fuerzas medias del último lote Fx 5.6726 / 5.6722, Fz −2.4008 / −2.3899 (CPU/GPU),
sin divergencia. En la app completa (`--headless --steps 1500`): SCz 1.2492 / 1.2504 m²,
SCx 2.766 / 2.773 m².

## Limitaciones

* **Tiempo**: a Media la iGPU sola da ~740 MLUPS por paso completo frente a ~870 de la CPU sin
  carga (ver `docs/opt/gpu.md`): no es más rápida que los 22 hilos AVX2 con el mismo ancho de
  banda compartido. La ganancia real es **liberar la CPU** para dibujar (solapamiento).
* Resultados con un lote de retraso en el bucle interactivo (invisible a 20–40 FPS).
* Un cambio de geometría con la GPU activa cuesta una bajada + subida de las poblaciones
  (≈ 50 ms a Media, además de la voxelización).
* Nodos de pared con más de 4 ids distintos: la fuerza de los ids sobrantes se pierde (se cuenta
  en `LbmGpuStats::overflow_nodes`; 0 en todos los modelos probados).
* Memoria: la GPU tiene su propia copia de las poblaciones y 2 mitades de campos macro (Ultra
  FP16S ≈ +1.9 GB).
* Descriptores: un solo búfer por ranura de dirección (`maxStorageBufferRange` = 4 GB aquí): FP32
  cabe hasta ~1000 M celdas; el backend lo comprueba.
