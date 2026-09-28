# Ruta DRM cruda (i915) — prueba de concepto "bare metal"

Ruta **experimental y de diagnóstico**: hablar con la iGPU (Intel Arc de Meteor Lake, Xe-LPG,
PCI `8086:7d55`, 128 EU) sólo con `ioctl` sobre `/dev/dri/renderD128`, sin libdrm, sin Mesa y
sin ninguna otra biblioteca. **No** es la ruta de cómputo del simulador (esa es Vulkan compute,
`src/gpu/vk.*`); sirve para medir lo que la GPU puede dar y qué cuesta hablar con ella.

| Archivo | Contenido |
|---|---|
| `src/gpu/drm_i915.hpp/.cpp` | `Device` (VERSION, GETPARAM, QUERY, REG_READ, VM, contextos), `Bo` (GEM_CREATE_EXT con PAT, MMAP_OFFSET WC/WB/UC), `execbuf` (softpin, `batch_start_offset`), `gem_wait`, `Batch` (codificador de comandos MI_* y XY_FAST_COPY_BLT) |
| `tools/drm_probe.cpp` | informe completo (identificación, envío, latencias, anchos de banda, coherencia) |
| `tests/test_drm.cpp` | codificación de comandos (siempre) + lotes reales en rcs/bcs/ccs y blit verificado (si hay GPU; si no, `SKIP`) |

Compilación (el `Makefile` aún no incluye `src/gpu/*.cpp`; se compila a mano):

```sh
g++ -std=c++23 -O3 -march=native -flto=auto -Wall -Wextra -Isrc -pthread \
    tools/drm_probe.cpp src/gpu/drm_i915.cpp -o build/drm/drm_probe
g++ -std=c++23 -O3 -march=native -flto=auto -Wall -Wextra -Isrc -pthread \
    tests/test_drm.cpp  src/gpu/drm_i915.cpp -o build/drm/test_drm
build/drm/drm_probe            # informe completo (~15 s)
build/drm/drm_probe --rapido   # versión corta
```

Compila sin avisos con las opciones del `Makefile` (C++23, `-O3 -march=native`, LTO, `-Wall
-Wextra`) y el test pasa también con ASan/UBSan. Si algún día se añade `src/gpu/*.cpp` a
`LIB_SRC`, `make test` enlazará `test_drm` sin más cambios.

## Estructuras del kernel

`/usr/include/drm/` trae las cabeceras **UAPI** del kernel (`drm.h`, `i915_drm.h`; no son
libdrm), así que se usan directamente. `drm_i915.cpp` fija con `static_assert` el tamaño de
las 23 estructuras que pasan al kernel (y dos `offsetof`) para no compilar nunca contra un ABI
inesperado. La macro C `I915_DEFINE_CONTEXT_PARAM_ENGINES` se sustituye por un bloque de bytes
construido a mano.

## Qué funciona (todo, sin privilegios: sólo la ACL de usuario sobre `renderD128`)

1. `DRM_IOCTL_VERSION` → `i915 1.6.0`. (También funciona sobre `card1`, el nodo primario.)
2. `I915_GETPARAM`: chipset `0x7d55`, revisión 8, **128 EU**, 8 subslices, frecuencia del
   timestamp del CS **19.2 MHz** (52.08 ns por tick), OA 38.4 MHz, `HAS_LLC = 0` (la GPU no
   comparte la LLC de la CPU), softpin y `NO_RELOC` sí, parser de comandos 0 (no hay
   validación de lotes: el hardware + PPGTT aíslan). `SLICE_MASK`/`SUBSLICE_MASK` → `EINVAL`
   (el kernel ya no las da en Xe-HP+; la topología va por QUERY).
3. `I915_QUERY`:
   * motores: `rcs0`, `bcs0`, `vcs0`, `vcs1`, `vecs0`, **`ccs0`** (un motor de sólo cómputo);
   * memoria: una región de sistema de 14.70 GiB (iGPU: no hay VRAM);
   * topología: 1 slice × 8 subslices (Xe-cores) × 16 EU = 128 EU;
   * HWCONFIG del GuC (56 claves): coherentes con lo anterior `MAX_DUAL_SUBSLICES=8`,
     `MAX_NUM_EU_PER_DSS=16`, `NUM_THREADS_PER_EU=8` (→ 1024 hilos HW), `MAX_RCS/CCS/VCS/
     VECS/COPY = 1/1/2/1/1`. (Los nombres salen del enum `intel_hwconfig` de IGT; sólo esos
     están comprobados contra otra fuente.)
   * `GUC_SUBMISSION_VERSION` → `ENODATA` en este kernel.
4. `I915_REG_READ` de `RING_TIMESTAMP` del rcs (`0x2358`, con y sin `I915_REG_READ_8B_WA`):
   funciona; **6-12 µs** por lectura (el kernel toma *forcewake* en cada una).
5. `I915_GEM_VM_CREATE` + `I915_GEM_CONTEXT_CREATE_EXT` con extensiones `ENGINES`
   (`[0]=rcs0 [1]=bcs0 [2]=ccs0`) y `VM`: PPGTT de 48 bits (256 TiB).
6. `I915_GEM_CREATE_EXT` (con y sin `SET_PAT` 0/2/3/4) y `I915_GEM_MMAP_OFFSET` WC/WB/UC:
   todas las combinaciones se aceptan.
7. `I915_GEM_EXECBUFFER2` con **softpin** (`EXEC_OBJECT_PINNED | SUPPORTS_48B_ADDRESS`,
   `I915_EXEC_NO_RELOC`; en Xe-HPG/Xe-LPG el kernel ya no acepta relocalizaciones) en rcs0,
   bcs0 y ccs0, con lotes escritos a mano: `MI_STORE_DATA_IMM`, `MI_STORE_REGISTER_MEM`,
   `MI_COPY_MEM_MEM`, `MI_FLUSH_DW`, `XY_FAST_COPY_BLT`, `MI_BATCH_BUFFER_END`. Resultados
   verificados desde la CPU; 16 `MI_STORE_DATA_IMM` seguidos tardan 5 ticks (≈ 260 ns).
8. `I915_GEM_WAIT` con timeout de 2 s en todas las esperas (nunca se colgó la GPU).

## Números medidos

Medianas; la máquina tenía carga ajena variable (carga media 0.4-6). Salida completa de una
ejecución: `build/drm/probe_run1.txt`.

### Latencia de envío (lote trivial: SRM del timestamp + SDI + BBE, 400 envíos)

| µs (mediana [p10, p90]) | rcs0 | bcs0 | ccs0 |
|---|---|---|---|
| sólo la ioctl `EXECBUFFER2` | 6.0 [2.5, 7.5] | 6.0 [2.8, 6.7] | 4.4 [2.5, 5.2] |
| envío → la CPU **ve el dato** (sondeo del mapeo WC) | 72-93 | 58-70 | **45** |
| envío → vuelve `I915_GEM_WAIT` | 95-113 | 73-79 | 69-73 |
| encadenados sin esperar: CPU por envío | 18 | 6.2 | 2.3 |
| encadenados: separación **en la GPU** entre lotes consecutivos | 18.3 | 8.2 | 14.3 |
| en frío (GT en RC6 tras 50 ms ociosa), `GEM_WAIT` | 0.8-1.2 ms | — | — |

Lectura:
* La ioctl cuesta 3-7 µs; el resto del viaje (≈ 40-85 µs) es kernel → GuC → hardware →
  vuelta. Es el suelo de latencia de cualquier API encima de i915 (Vulkan/ANV usa esta misma
  ioctl), así que un `vkQueueSubmit + vkWaitForFences` no puede bajar de ~50-100 µs aquí.
* Sondear un valor escrito por la GPU en memoria mapeada ahorra ~20-30 µs frente a esperar
  al *fence* (`GEM_WAIT`). Se comprobó que el `MI_STORE_DATA_IMM` es visible para la CPU en
  cuanto se ejecuta (añadir ~350 µs de trabajo detrás en el mismo lote apenas la retrasa:
  +4 µs en bcs, +28-38 µs en rcs/ccs, frente a +350 µs del final del lote).
* Aunque se encadenen, cada petición cuesta 8-18 µs de GPU (breadcrumbs y flushes que el
  kernel añade por petición). En rcs la GPU espera a la CPU (18 µs/envío por ioctl).
* Tras ~50 ms ociosa la GT entra en RC6 (0 MHz): el primer envío paga ~1 ms.
* **No** se parte la latencia en "envío→inicio GPU" y "GPU→CPU" con la correlación de
  relojes: cada motor sólo puede leer su propio `RING_TIMESTAMP` (el de otro motor lee 0 por
  `MI_STORE_REGISTER_MEM`), `I915_REG_READ` sólo expone el del rcs, y los contadores de
  motores distintos resultaron no ser comparables (desfases constantes de decenas de µs).
  La deriva frente a `CLOCK_MONOTONIC` sí es pequeña (±10-20 ppm, dentro del ruido de la
  correlación).

### Ancho de banda de copia de la GPU (motor de copia, `XY_FAST_COPY_BLT` lineal 32 bpp)

Tiempo medido **en la GPU** con `RING_TIMESTAMP` del bcs antes de la copia y tras
`MI_FLUSH_DW`; resultado verificado por muestreo.

| tamaño | copia GB/s | lectura+escritura GB/s |
|---|---|---|
| 8 MiB | 28.7 | 57 |
| 16 MiB | 29.3 | 58 |
| 32 MiB | 28.7 | 57 |
| 64 MiB | 13.2-16.1 | 26-32 |
| 256 MiB (PAT por defecto / PAT0 WB / PAT3 WB coherente) | 12.2-13.2 | **24.5-26.4** |

* Con conjuntos de trabajo grandes (lo que es un campo LBM: cientos de MiB) el blitter se
  queda en **~25 GB/s de tráfico**, ≈ 30 % de los ~82 GB/s que mueve la CPU. No depende del
  PAT ni de la frecuencia de la GT (se muestreó 1050-2250 MHz a mitad de copia con el mismo
  resultado), ni del tamaño de cada blit (8/32/64 MiB por comando dan lo mismo).
* Repetir la copia de los **mismos** 32 MiB dentro de un lote sostiene ~27-30 GB/s y 64 MiB
  ~22 GB/s: hay una caché o un efecto de TLB en el camino de la GPU a memoria que cubre
  ~32 MiB. No se ha averiguado cuál.
* **Esto no mide el ancho de banda de los EU.** El blitter es una unidad fija con su propio
  límite; un kernel de cómputo con muchos hilos (la ruta Vulkan) puede sacar bastante más.
  Sirve como cota inferior de lo que la GPU consigue de la memoria.
* `MI_COPY_MEM_MEM` (el propio command streamer copiando dwords): ~300 ns por comando →
  0.013 GB/s. Inútil para mover datos; sólo vale para parches pequeños dentro de un lote.

### CPU ↔ buffers GEM: coherencia y ancho de banda por el mapeo (**lo más útil para Vulkan**)

Coherencia (blit de 4 MiB, dos rondas, sin `clflush` explícito):

| PAT del objeto + mmap | GPU ve lo que escribe la CPU | CPU ve lo que escribe la GPU |
|---|---|---|
| por defecto + WC | sí | sí |
| por defecto + WB | sí (el kernel hace `clflush` en execbuf) | sí |
| PAT0 (WB no coherente) + WB | **no** | **no** |
| PAT0 + WC | sí | sí |
| PAT3 (WB, coherente 1 vía) + WB | sí | sí |
| PAT4 ("coherente 2 vías") + WB | **no** | **no** (en MTL se comporta como no coherente) |

Ancho de banda de la CPU a través del mapeo (64 MiB, 1 hilo):

| mapeo | escritura `memcpy` | lectura `memcpy` | lectura `MOVNTDQA` |
|---|---|---|---|
| WC (PAT por defecto) | 18.3 GB/s | **0.16 GB/s** | 4.3 GB/s |
| WB, PAT por defecto | 18.2 GB/s | 20.9 GB/s | 19.2 GB/s |
| WB, PAT3 coherente | 18.3 GB/s | 20.9 GB/s | 19.3 GB/s |

Leer desde la CPU memoria mapeada WC (la memoria "host visible" sin caché) es **130× más
lento** que desde WB: leer así un campo de 100 MB costaría ~0.6 s por cuadro.

## Recomendaciones para el proyecto (ruta Vulkan)

1. **Memoria de lectura CPU**: los campos que la CPU tenga que leer (ρ, u para el render, las
   fuerzas) deben ir en un tipo de memoria `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`
   (en MTL equivale a WB + PAT coherente 1 vía, ~21 GB/s por hilo). Nunca leer desde un tipo
   sin `HOST_CACHED` (WC): 0.16 GB/s con `memcpy`. Si no queda otra, `MOVNTDQA` en varios hilos.
   Para subir datos (geometría, parámetros) WC va bien: 18 GB/s de escritura.
2. **Envíos**: cada `vkQueueSubmit` + espera cuesta ≥ 50-100 µs de ida y vuelta y 8-18 µs de
   GPU aunque se encadenen. Grabar **muchos pasos LBM en un solo command buffer** (o
   reutilizarlo) y esperar una vez por cuadro. A 30 FPS esto es < 0.3 % del cuadro.
3. **Cola de cómputo**: `ccs0` fue el motor con menor latencia (45 µs hasta ver el dato frente a
   72-93 µs en rcs). Si ANV expone una familia de colas sólo-cómputo, usarla.
4. **RC6**: si la GPU queda ociosa > ~50 ms, el siguiente envío paga ~1 ms extra y la
   frecuencia arranca baja (se vio 1050-2250 MHz): calentar antes de medir y no dejarla ociosa
   entre pasos si importa la latencia.
5. **Timestamps**: `timestampPeriod` debería ser 52.08 ns (19.2 MHz). Sólo comparar
   timestamps del mismo motor/cola.
6. **Techo esperable**: 128 EU × 8 carriles FP32 × 2 (FMA) × 2.25 GHz ≈ 4.6 TFLOPS FP32 de pico
   (derivado de GETPARAM/HWCONFIG y sysfs). LBM está limitado por memoria: con el mismo bus que
   la CPU (82 GB/s medidos) la GPU sólo ganaría si sus EU sacan más ancho de banda que los hilos de
   la CPU. El blitter no pasa de ~26 GB/s con conjuntos grandes; la cifra que decide
   es la de un kernel Vulkan de streaming (medirla en la ruta Vulkan).

## ¿Kernels en los EU por DRM crudo? — análisis de viabilidad (no se intentó)

Qué haría falta para lanzar un kernel de cómputo propio en Xe-LPG sólo con i915:

1. **Estado del pipeline en el lote** (volúmenes "Command Reference" y "GPGPU" de los PRM
   públicos de Intel para Alchemist/Meteor Lake): `PIPELINE_SELECT` (GPGPU) en rcs o nada en
   ccs, `STATE_BASE_ADDRESS` (bases general/superficie/dinámica/instrucciones/bindless + MOCS),
   `STATE_COMPUTE_MODE`, `CFE_STATE` (hilos máximos, *scratch*), `COMPUTE_WALKER` con su
   `INTERFACE_DESCRIPTOR_DATA` embebido (puntero al kernel, SLM, barreras, hilos por grupo),
   datos *inline*/indirectos para los argumentos, y `PIPE_CONTROL` para invalidar/vaciar cachés.
   Unos 100-200 dwords. IGT (`lib/gpgpu_fill.c`, `lib/gpu_cmds.c`, variantes `xehp_*`) tiene una
   secuencia funcional de referencia para Xe-HP+ que acota mucho el trabajo.
2. **El binario del kernel en ISA Xe-LPG**: instrucciones de 128 bits (sin compactar), 128 GRF de
   32 B por hilo, SIMD8/16, accesos a memoria por mensajes `send` a la LSC con descriptores de
   mensaje (cargas/almacenes A64 *stateless*), fin de hilo con EOT. No hay ensamblador público
   independiente de Mesa (IGT lleva los kernels ya ensamblados como arrays hexadecimales): habría
   que escribir un mini-ensamblador (~20 opcodes: `mov`, `add`, `mul`, `mad`, `math`, `cmp`,
   `sel`, `shl`, `and`, `or`, `send`…) con las tablas de codificación del PRM.
3. **El kernel LBM D3Q19** (FP16 en memoria): 19 cargas + 19 almacenes por celda, ~200-300 FLOP de
   colisión, máscara de sólidos y contornos: cientos de instrucciones con asignación de registros
   y planificación a mano.

Esfuerzo estimado: 1-2 semanas para un "hola mundo" (cada hilo escribe su id) siguiendo IGT;
3-6 semanas más para un kernel LBM correcto y rápido, más el mini-ensamblador.

Riesgos:
* Depuración casi a ciegas: una codificación errónea da basura o **cuelga la GPU**; el GuC
  reinicia el motor, el kernel veta el contexto y, tras cuelgues repetidos, puede vetar al
  cliente. No hay
  `printf` ni depurador de shaders.
* *Workarounds* de hardware (volumen "Workarounds" del PRM) que ANV/i915 aplican para MTL y que
  habría que replicar sin saber cuáles afectan a cómputo.
* Portabilidad nula: Meteor Lake puede pasar al driver `xe` (uAPI distinta, `xe_drm.h`); todo lo
  de i915 dejaría de servir, y la ISA cambia en cada generación (Xe2/Lunar Lake ya es distinta).
* Ganancia de rendimiento esperable: **ninguna** para LBM. El kernel está limitado por ancho de
  banda y el compilador de Mesa genera buen código de streaming; lo único que ahorra la vía
  cruda es sobrecoste de envío (decenas de µs), irrelevante cuando un envío cubre varios pasos
  de milisegundos.

**Conclusión**: no compensa. El cómputo en la iGPU debe ir por Vulkan compute. La ruta DRM cruda
queda como herramienta de diagnóstico (latencias de envío, ancho de banda del blitter,
coherencia de cada PAT/mapeo, frecuencias) y como referencia de lo que Vulkan hace por debajo.
