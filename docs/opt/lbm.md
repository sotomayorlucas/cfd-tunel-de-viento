# Módulo LBM — optimizaciones, verificación y medidas

Solver D3Q19 con streaming in-place **Esoteric-Pull** (Lehmann 2022), almacenamiento FP32 o
**FP16S** (F16C), colisión BGK o **Regularizada** + Smagorinsky, kernel AVX2 de 8 celdas.
Archivos: `src/lbm/lattice.hpp`, `src/lbm/solver.hpp` (contrato, ampliado de forma compatible),
`src/lbm/solver.cpp`, `tests/test_lbm.cpp`, `tools/bench_lbm.cpp`.
Tablas completas de rendimiento: [`docs/BENCHMARKS_LBM.md`](../BENCHMARKS_LBM.md).

## Resumen

* **256×128×96 (rejilla por defecto), FP16S + Regularizado + LES: ≈ 870 MLUPS** (paso completo, con fuerzas por
  id cada paso) con el pool por defecto; kernel ≈ 910–1060 MLUPS = **91–100 % del techo de memoria** medido
  (RMW in-place 81 GB/s). Objetivo ≥ 300 MLUPS: ×2.9. FP32 ≈ 500 MLUPS (78–80 GB/s, 97–98 % del techo).
* Por núcleo: P-core ≈ 200 MLUPS, E-core ≈ 82 (tiempo de CPU): con el chip completo manda la memoria; los trucos
  de cómputo aseguran que 6 P-cores ya den ~80 % (FP16S) / ~100 % (FP32) del máximo y que el solver aguante la carga
  de fondo que tiene esta máquina.
* Correctitud: Esoteric-Pull = referencia A-B independiente a 1e-7 (FP32) con entrada, salida, campo lejano,
  obstáculo, cilindro giratorio, cinta móvil, esponja y rampa; masa 2.5e-9; Couette 0.21 % (perfil) y tensión de pared
  0.3 % (fuerza); 22/22 tests tras la revisión adversarial (§4: 1 defecto de física, 1 carrera de datos, corregidos).

> **Condiciones de medida.** La máquina tuvo carga de fondo muy alta y variable durante todo el
> trabajo (otros agentes compilando y ejecutando tests: carga media 4–27). Por eso:
> (1) las comparaciones A/B se hicieron **intercaladas** (ronda a ronda, mismas condiciones);
> (2) las decisiones de cómputo se validaron además con **llvm-mca 21** (modelo Golden Cove
> `-mcpu=alderlake`, determinista, sin ruido); (3) todas las cifras son medianas de ≥5.
> Donde el ruido impidió una conclusión, se dice.

---

## 1. Algoritmo y por qué es correcto

### Esoteric-Pull (una sola copia de las poblaciones)
Para cada par opuesto (i, i+1) y paridad p = t&1 (`solver.cpp:13-18`, punteros en `step_ptrs`,
`solver.cpp:617`):

| p | carga f_i | carga f_{i+1} | guarda f_i | guarda f_{i+1} |
|---|---|---|---|---|
| 0 | [i+1][n] | [i][n+c_i] | [i][n+c_i] | [i+1][n] |
| 1 | [i][n] | [i+1][n+c_i] | [i+1][n+c_i] | [i][n] |

Cada celda escribe exactamente las posiciones que lee → sin carreras, 1 copia (mitad de memoria y
de tráfico que A-B). El puntero de escritura de f_i **es** el de lectura de f_{i+1}: el kernel no
tiene ninguna rama por paridad (19 punteros precalculados por paso).

### Rebote implícito = rebote *full-way* (resultado propio, verificado)
Se demuestra fácilmente que **ningún** esquema in-place sin carreras puede dar rebote half-way
implícito: la posición que n lee como población entrante desde un sólido tendría que escribirla n
(caso sólido) y el vecino (caso fluido) en el mismo paso. Lo que realmente ocurre al no procesar
los sólidos es que la población saliente vuelve **dos pasos después** (el nodo sólido la "guarda"
un paso y la devuelve invertida): rebote full-way, que en estado estacionario coloca la pared en
el mismo sitio (mitad de enlace). La referencia A-B del test lo modela con nodos sólidos de
"colisión de inversión" y coincide a 1e-7; una referencia con rebote half-way **difiere en 3.6e-2**
(control negativo 2e) → la prueba discrimina.

### Fuerzas (intercambio de momento)
Tras el paso τ, las dos posiciones de cada enlace fluido→sólido guardan g_k(τ) y g_k(τ-1)
(independiente de la paridad). F = Σ (g_k(τ) + g_k(τ-1) + 2w_k − 6w_k c_k·u_w) c_k sobre la lista
compacta de sólidos frontera; momento (s − c_k/2)×F_k = s×F_k porque F_k ∥ c_k (`solver.cpp:911`).

### Fronteras (todas vectorizadas salvo paredes móviles no-suelo)
Entrada/campo lejano: equilibrio (ρ=1, u=u∞(t)); salida: equilibrio con ρ=1 y u de x−1;
suelo None/Static/Moving; paredes móviles con corrección de Ladd; esponja con τ0(x) por plano;
rampa smoothstep de u∞ (también escala las velocidades de pared).

---

## 2. Trucos aplicados (archivo:línea → efecto medido)

| # | Truco | Dónde | Efecto medido |
|---|---|---|---|
| 1 | **Esoteric-Pull in-place**, 1 copia SoA por dirección, 64 B alineado, THP | `solver.cpp:13-18, 617`; THP en `core/mem.hpp` | Memoria 256×128×96 FP16S: **187 MB** (A-B necesitaría ~2×). THP efectivo: 170 de 188 MB RSS en páginas de 2 MiB (medido en `/proc/self/smaps_rollup`). |
| 2 | **FP16S desplazado** f̃=f−w, ×2^15 exacto, vcvtph2ps/vcvtps2ph (F16C) | `solver.cpp:51, 301-311, 376-388` | 76 B/celda·paso vs 152 en FP32. FP16S vs FP32: Cd esfera 0.43 % de diferencia; deriva de masa 4e-7 (test 3); error vs referencia A-B 5e-5 (2d). |
| 3 | **Relleno (padding) en vez de envolver índices**: P ≥ nx·ny+nx+64 delante y detrás de cada dirección | `solver.cpp:1024` | Cero comprobaciones de límites y cero ramas en el kernel; sólo las celdas de frontera de equilibrio (que ignoran lo que cargan) tocan el relleno. Demostrado por el test 2 (caras incluidas). |
| 4 | **Sesgo anti-aliasing de caché**: zancada = k·4 KiB + 3 líneas | `solver.cpp` (`init`) | Sin sesgo, N·2 B es múltiplo de 4 KiB y las 19 corrientes caen en los mismos conjuntos de L1/L2. Medido (`CFD_LBM_SKEW` 0 vs 3, 3 rondas alternas, 14 hilos): FP16S **+10..+17 %**, FP32 +2.5 %. Cualquier sesgo ≥ 1 línea sirve. |
| 5 | **Clases de bloque por SWAR** sobre 8 bytes de flags (u64): Skip / Pure / Masked / Ground / Scalar | `solver.cpp:56-66, 794-808` | Bloques 100 % sólidos no se tocan; la ruta escalar queda sólo para vecinos de ruedas. |
| 6 | **Carriles sólidos por mezcla con el valor recargado** (sin vmaskmov, sin carreras: en Esoteric-Pull la posición sólo la toca esa celda en este paso); FP16: máscara 32→16 bits con `vpackssdw` + `vpblendvb` | `solver.cpp:369-389` | Permite vectorizar bloques con sólidos y fronteras. |
| 7 | **Salida vectorizada**: u(x−1) por desplazamiento de carril con `vpermps` (la salida siempre es el carril 7) | `solver.cpp:336-343` | Antes: 2 de cada 32 bloques por fila (entrada+salida) irían a la ruta escalar (~+40 % de tiempo estimado). |
| 8 | **Equilibrio exacto sin ramas**: (1−ω)=0 en carriles de frontera → f̃* = f̃eq exacto | `solver.cpp:344-351` | Fronteras de equilibrio en la ruta AVX2 sin divergencia de flujo de control. |
| 9 | **Momentos "en flujo" por pares** (s = f_i+f_{i+1}, d = f_i−f_{i+1}) con dos cadenas independientes (ejes / diagonales) | `solver.cpp:110-136` | Instrucciones vectoriales del kernel FP16-Reg: **1403 → 694**, accesos a pila **434 → 152**; +5..11 % MLUPS/núcleo (tiempo de CPU, 1 hilo). Cadenas separadas: ver #11. |
| 10 | **Escala FP16 plegada** en pesos (W·2^15), en 1/ρ y en FMAs de Π^neq (nunca se reescalan las 19 poblaciones) | `solver.cpp:143-164, 194-196, 317-323` | −28 multiplicaciones por bloque de 8 celdas. |
| 11 | **Ruta crítica corta** (por núcleo, el kernel está limitado por LATENCIA, no por puertos: el ROB de 512 entradas sólo contiene ~1 bloque de ~410 instr.): árbol para \|Π\|², (1−ω) aplicado al final con una FMA por dirección, ρ en 2 cadenas | `solver.cpp:169-177, 197-222` | llvm-mca (ciclos simulados por bloque de 8): FP16-Reg **143 → 129**, FP16-BGK **134 → 123**, FP32-Reg **111 → 101** (−10 %). |
| 12 | **Equilibrio con W plegado** (estilo FluidX3D): f̃eq± = W(ρ−1) + (Wρ/2)(a²−3u²) ± (Wρ)a | `solver.cpp:188-196` | 5 op/par en vez de 7 (el efecto neto en ciclos quedó dentro del #11). |
| 13 | **Cinta móvil vectorizada** (clase Ground): la corrección de Ladd sobre z=1 es constante (sólo dirs 9 y 16, ±6w·u_g) → bit interno `kGroundOnly` | `solver.cpp:63-66, 292-310` | La capa z=1 (1/nz de las celdas) salía por la ruta escalar (~5-10× más lenta por celda). Tiempo de CPU/núcleo: +4..+18 % en el caso "coche" (ruido alto); caso vacío idéntico (control). Validado vs referencia (2a). |
| 14 | **Plantillas** `<Precision, Collision, Macro, Masked, Ground>` → todas las decisiones en compilación (loop unswitching); 1 puntero a función por paso | `solver.cpp:291, 515-555` | Sin ramas por celda salvo la clase del bloque. |
| 15 | **ρ,u fusionados en el kernel** en el último paso (sin pasada extra). Stores no temporales + `sfence` disponibles (`Tuning::nt_macro`) | `solver.cpp` (`store_macro`) | NT medido −2.5 % (3/3 rondas, 1 paso macro de 5): el `sfence` por trozo y la relectura inmediata por la visualización pesan más que el RFO ahorrado → **por defecto stores normales**. |
| 16 | **Divergencia a coste ~0**: comparaciones no ordenadas (`_CMP_NGT_UQ`) detectan NaN y ρ∉(0.2,5) con 3 op/bloque; un único `movemask` por fila y un atómico sólo si hay error | `solver.cpp:353-359, 545-547` | Detectó correctamente la divergencia de BGK a τ≈0.5006 en el benchmark. |
| 17 | **Grano dinámico ~4096 celdas** (16 filas en nx=256) para CPU híbrida P/E | `solver.cpp` (`run_kernel`) | FP16S-Reg, intercalado, 2 campañas: 1 fila −23 %, 4 filas −7..−12 %, 32 filas −6 %, 64 filas −9 % frente a 16 filas. |
| 18 | **Tope de hilos del kernel** (`Tuning::max_threads`) con reparto dinámico propio sobre `run_slots` | `solver.cpp:892` | Medido: con un pool de 20, limitar a 14 es 5-15 % PEOR que un pool de 14 (no controla qué núcleos entran; pueden ser hermanos HT). Queda como opción; **no recomendado**. |
| 19 | **Pasada de fuerzas**: lista compacta ordenada por id (ordenación por conteo, O(L)), trozos fijos de ~256 entradas (reducción **determinista**), punteros por dirección sin ramas, tablas `float` constexpr | `solver.cpp:564-574, 843-851, 911-1002` | 1.1 ms/paso en 1 hilo (≈4 ns/enlace); 0.14-0.2 ms/paso con 20 hilos ≈ 3-4 % del paso. Fuerzas bit a bit idénticas con 1 y 20 hilos (test 9). |
| 20 | **Retículo constexpr** con identidades verificadas por `static_assert` sobre pesos enteros (w·36): Σw, Σwc, Σwcc = δ/3, 3er y 4º orden, tabla de opuestos | `lattice.hpp` | Errores de tabla imposibles de compilar. |
| 21 | **Ruta sin LES** (C_s = 0): 1−ω₀(x) de tabla por plano; rama uniforme por paso (predicción perfecta) en vez de 2 sqrt + 1 div en la ruta crítica | `solver.cpp` (`block_vec`, `omc0`) | llvm-mca (ciclos/bloque): FP32-BGK **106 → 72**, FP32-Reg 106 → 87, FP16-BGK 129 → 113, FP16-Reg 131 → 123. Validado por 2a/2c/3/4 (usan C_s = 0). |

### Probado y descartado (medido)
| Idea | Resultado |
|---|---|
| Recorrer filas en z **descendente** (para que el suelo esté en L3 al calcular fuerzas) | **−15..−20 %** MLUPS y fuerzas más lentas: rompe los flujos ascendentes del prefetcher L2 entre filas. Revertido (`solver.cpp:811`). |
| Prefetch software a 1/2/4/8 bloques | Sin efecto significativo (±3 %, dentro del ruido) en FP16 y FP32 → desactivado (`Tuning::prefetch`). |
| Pares de bloques entrelazados (`f8x2`, segmentación software de 16 celdas) | llvm-mca: FP32-Reg −10 % ciclos, pero FP16-Reg +3 %, FP16-BGK +10 %, FP32-BGK +26 % (derrames). Medido a chip completo (grano 16): 792 vs 798 MLUPS (−1 %) → opcional (`Tuning::pair_blocks=1`), apagado. |
| rcp/rsqrt + Newton en lugar de div/sqrt | Las div/sqrt ocupan 1 µop del puerto 0 y no son cuello de botella; la aproximación añade 3 µops FMA → no se aplicó. |
| Fusión de la pasada de fuerzas en el kernel | Estimado: +2.5 % de cómputo en bloques junto a sólidos vs 3-4 % que cuesta la pasada → ganancia neta ≤ 1-3 % con mucha complejidad; no se implementó (ver known issues). |

### Lección de medida: el pool recién creado penaliza la primera medición
El benchmark recrea el pool de hilos para cada configuración (nº de hilos, afinidad). Los hilos nuevos
nacen junto al hilo padre y el planificador (CFS) tarda decenas de ms en repartirlos por los núcleos,
así que la PRIMERA medición corta tras cada reinicio salía penalizada un 20-60 %. Como el orden de los
casos era fijo, la víctima era siempre FP16S-Regularizado, lo que parecía indicar que "Regularizado es
un 25-60 % más lento que BGK con el chip completo", aunque por núcleo sólo lo es un 5 %. Pistas que lo
delataron: con un programa que no reinicia el pool, Regularizado iba a ~900 MLUPS en 384×160×128, igual
que BGK; y la diferencia no dependía de ν, de FTZ/DAZ, de subnormales (0 halves subnormales contados en
el buffer), de la ruta escalar (ruedas quietas: igual) ni del sesgo de zancada. Corregido en
`tools/bench_lbm.cpp`: 150 ms de calentamiento tras cada reinicio y orden de casos rotado por ronda.
Las comparaciones A/B *dentro* de una misma configuración (grano, prefetch, pares, sesgo, NT) no
quedan invalidadas: todas las variantes sufrían la misma penalización.

### ¿Memoria o cómputo?
* Por núcleo (1 hilo, tiempo de CPU): P-core ≈ 200 MLUPS FP16S, E-core ≈ 82 → 6P+8E ≈ 1.9 G celdas/s de
  cómputo, frente a un techo de memoria de ≈ 1066 MLUPS FP16S / 533 FP32 (RMW in-place a 81 GB/s).
* A chip completo (bench corregido): FP16S 970–1075 MLUPS (91–100 % del techo), FP32 515–523 (97–98 %).
  → **Con todo el chip el kernel está limitado por memoria en ambas precisiones.** Los trucos de cómputo
  (9-13, 21) suben el rendimiento POR NÚCLEO (validados con tiempo de CPU de 1 hilo y llvm-mca), lo que importa
  con pocos hilos, en los E-cores y cuando la máquina está compartida con otros procesos (el caso de este usuario);
  6 hilos en los P-cores ya dan ~80 % (FP16S) y ~100 % (FP32) del máximo.
* Experimento "colisión nula" (mismas lecturas/escrituras sin aritmética): FP16S ≥ 885, FP32 ≥ 480 MLUPS
  (cota inferior: medido con el bench que aún penalizaba la primera medición, ver abajo).

---

## 3. Resultados de los tests (`build/lbm/test_lbm`, 22/22 PASS, ~62 s con 20 hilos)

Compilar y ejecutar: `g++ -std=c++23 -O3 -march=native -Isrc -pthread tests/test_lbm.cpp src/lbm/solver.cpp src/core/threadpool.cpp -o build/lbm/test_lbm && build/lbm/test_lbm [filtro]`

| Test | Resultado |
|---|---|
| 1 Identidades D3Q19 | error máx. de momentos 1.5e-8 (float), opuestos correctos (+ static_assert exactos) |
| 2a Esoteric-Pull vs A-B (BGK, entrada, salida, campo lejano, obstáculo, cilindro giratorio, cinta móvil, esponja, arranque impulsivo), 250 pasos FP32 | max\|dρ\| **1.2e-7**, max\|du\| **1.8e-7** (tol 1e-5); fuerzas por id err. rel. 1.3e-6 |
| 2b ídem Regularizado + Smagorinsky, suelo fijo, rampa | 8.1e-8 / 8.0e-8 |
| 2c ídem Regularizado sin suelo | 8.7e-8 / 1.2e-7 |
| 2d FP16S vs A-B (info) | 5.4e-5 / 6.6e-5 |
| 2e Control negativo (referencia con rebote half-way) | difiere 3.6e-2 → la prueba discrimina |
| 2f *(revisión)* id 255 fuera de la capa z=0 con cinta móvil | max\|du\| **1.8e-7** (antes de la corrección: **0.11** → FALLABA, ver §4) |
| 2g *(revisión)* FP16S + Regularizado + Smagorinsky vs A-B (info) | 3.9e-5 / 4.4e-5 (única cobertura de `noneq<Sc=2^15>`) |
| 3 Masa, cavidad cerrada con tapa móvil, 1000 pasos | FP32 deriva **2.5e-9**; FP16S 3.7e-7 |
| 4 Couette antisimétrico (placas ±U/2), BGK y Regularizado | error máx. **0.21 %** de U |
| 4b *(revisión)* Tensión de pared en parches 16×16 de las placas (ids propios) vs τ = ρνU/h | error máx. 1.14 % (gradiente de presión leve de las caras abiertas), parte antisimétrica **0.29 %**. Valida la fórmula de fuerzas (incl. −6w c·u_w) contra un valor ANALÍTICO, no contra la referencia (que usa la misma fórmula) |
| 5 Esfera Re=100, D=16, 208×112×112 (λ=d/H=0.145, bloqueo 1.7 %) | Cd = 1.214 vs 1.092 (+11.2 %, tol 15 %). Estudio del implementador: D=16/24/32 con λ=0.2 → 1.28/1.20/1.197. *Corrección de la revisión:* ese mismo estudio muestra un efecto de **resolución** de ≈ −6 % de D=16 a D=24; y medido aparte (FP16S, D=16) λ=0.145 → 1.226, λ=0.101 (256×160×160) → 1.214: el confinamiento sólo explica ~1 % entre esos λ. El exceso de D=16 es sobre todo resolución (esfera escalonada de 16 celdas) + confinamiento/distancia a la entrada; la fórmula de fuerzas está validada aparte (4b) |
| 6 Esfera FP16S vs FP32 | Cd 1.2762 vs 1.2817: **0.43 %** |
| 7 Estabilidad ν=1e-5, u=0.1, Cs=0.16, Regularizado, cinta + ruedas girando, 3000 pasos | sin divergencia en FP16S y FP32, \|u\|max 0.22 |
| 8 Dos esferas simétricas (ids 1 y 2) | asimetría relativa **8e-7** |
| 9 Determinismo 1 vs 20 hilos | campos FP32 y fuerzas **bit a bit idénticos** |
| 10 set_geometry en marcha | el flujo se conserva (⟨u_x⟩ 0.0625 → 0.0625), sin divergencia |
| 11 Cambios de estado en marcha | rampa smoothstep (u(50)=u∞/2 exacto), set_inflow desde el valor actual, suelo None→Moving→Static→None, u de la cinta en el campo macro, clear_wall_motions, reset_flow (reposo, steps()=0) |
| 12 *(revisión)* Variantes de `Tuning` (pares `block_vec2`, prefetch, stores NT, `max_threads`, FTZ/DAZ, grano 1 y 7) | campos ρ,u y fuerzas **bit a bit idénticos** a la ruta por defecto (FP32-Reg-LES, FP16S-BGK, FP16S-Reg-LES) |

---

## 4. Revisión adversarial independiente (revisor del módulo `lbm`)

Todo lo anterior se recompiló y re-ejecutó desde cero; además se compiló el módulo con
**ASan+UBSan** (`build/lbm-asan/`) y **TSan** (`build/lbm-tsan/`), y se añadieron 5 pruebas.

```
g++ -std=c++23 -O1 -g -march=native -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc -pthread \
    tests/test_lbm.cpp src/lbm/solver.cpp src/core/threadpool.cpp -o build/lbm-asan/test_lbm
g++ -std=c++23 -O1 -g -march=native -fsanitize=thread -Isrc -pthread \
    tests/test_lbm.cpp src/lbm/solver.cpp src/core/threadpool.cpp -o build/lbm-tsan/test_lbm
```

### Defectos encontrados y corregidos

| # | Gravedad | Defecto | Corrección | Verificación |
|---|---|---|---|---|
| R1 | mayor | **Carrera de datos** en `rebuild()` paso 2 (marcado de `kNearMoving`): `parallel_for` por planos z; el hilo del plano z hacía `F[n] \|= kNearMoving` mientras el del plano z±1 leía ese mismo byte como vecino `F[m]`. UB según el modelo de memoria de C++ (en x86 benigno en la práctica: sólo se mira `kMoving`, que no cambia). TSan: 3-5 avisos por prueba (`solver.cpp:754/756` originales). | Dos pasadas por **paridad de z** (`solver.cpp:743-772`): cada pasada escribe planos de una paridad y sólo lee los de la otra, que nadie escribe en esa pasada. Sin atómicos. | TSan: **0 avisos** en 2, 9, 11, 12 (antes 3-5 por prueba). `set_geometry` 256×128×96 con coche + 4 ruedas: 6.1 → 6.3 ms mediana (dentro del ruido; no es ruta caliente). |
| R2 | mayor | **Física errónea con id 255 fuera del suelo**: con cinta móvil, cualquier sólido id 255 era "suelo" para el atajo vectorial `kGroundOnly`, que aplica la corrección de Ladd constante SÓLO en las dirs 9/16 (válida únicamente si el sólido móvil está en z−1). Un sólido 255 lateral/encima recibía la corrección en la dirección equivocada. El contrato reserva 255 al suelo, pero nada lo impedía (un voxelizador con 255 grupos lo produciría). | `kGroundOnly` exige además que el vecino móvil esté en la capa z=0 (`m < nx·ny`, `solver.cpp:764`); si no, la celda va por la ruta escalar general (Ladd exacto por dirección). | Nueva prueba **2f** vs referencia A-B: antes **max\|du\| = 0.11** (FALLA), después **1.8e-7**. Resto de la suite idéntica bit a bit (la ruta de la cinta real no cambia). |
| R3 | menor | `tools/bench_lbm.cpp`: `Case::m/mk/fm[64]` se desbordaban con > 64 combinaciones de `--threads × --pin × --grain × --pf × --pair × --maxt`. | `CFD_CHECK(tcfg.size() <= 64, …)`. | — |

### Pruebas añadidas (cobertura que faltaba)

* **2f** — id 255 fuera de la capa z=0 con cinta móvil, contra la referencia A-B (detectó R2).
* **2g** — FP16S + Regularizado + Smagorinsky contra la referencia (única cobertura de `noneq<Sc = 2^15>`, la escala plegada en Π): 4.4e-5, nivel FP16.
* **4b** — **tensión de pared analítica** en parches 16×16 de las placas de Couette (ids propios): valida la fórmula de fuerzas por intercambio de momento, incluido el término −6w c·u_w, contra τ = ρνU/h y no contra la referencia (que usa la misma fórmula). Parte antisimétrica 0.29 %; además una referencia 1D periódica independiente en double da error **0.000 %** con rebote full-way y half-way (programa de la revisión, no incluido).
  *Trampa detectada al escribirla:* `forces_mean()` promedia TODOS los pasos del último `step(n)`; desde un arranque impulsivo el transitorio sesga la media +4..7 % → promediar sólo en estado estacionario (`step(n-100,false); step(100,true)`).
* **12** — todas las variantes de `Tuning` (pares `block_vec2`, prefetch, stores NT, `max_threads`, FTZ/DAZ, grano 1/7) dan campos y fuerzas **bit a bit idénticos** a la ruta por defecto (ninguna estaba cubierta).

### Afirmaciones del implementador: reproducidas / no sostenidas

| Afirmación | Resultado de la revisión |
|---|---|
| 17/17 tests PASS, < 3 min | **Sí** (58.9 s). Tras la revisión: 22/22 en 62 s. ASan+UBSan: limpio (1, 2, 3, 4, 8, 9, 10, 11, 12). |
| EP ≡ A-B a ~1e-7, control half-way 3.6e-2 | **Sí**, idéntico. (El "err rel F 3.65e-4" de 2b, no citado en el informe, es un artefacto: fuerzas ~5e-5 en el paso 20 frente al suelo de escala 1e-3.) |
| Determinismo 1 vs 20 hilos | **Sí** (y TSan confirma que kernel y fuerzas no tienen carreras). |
| ≥ 300 MLUPS FP16S en 256×128×96; ~870 MLUPS | **Sí, y más**: con carga 3-4, FP16S-Reg **998–1010** MLUPS (kernel 1039–1049, 79 GB/s), BGK 1014–1031; FP32 503–509 (78–79 GB/s). |
| Techo RMW ~81.7 GB/s; kernel al 91-100 % | **Sí**: RMW medido 83.5–84.9 GB/s (14-20 hilos); kernel FP16S 79 GB/s = **94 %**, FP32 79 GB/s = 93 %. |
| Por núcleo P ≈ 200, E ≈ 82 MLUPS (tiempo de CPU) | **Sí**: CPU 0 FP16S-Reg 213, FP32-Reg 208; CPU 12 FP16S-Reg 81. |
| Cd esfera +11 % "por confinamiento, no por resolución" | **Parcialmente falso** (corregido en §3): el propio estudio D=16→24 muestra −6 % por resolución, y reducir λ de 0.145 a 0.101 sólo baja Cd 1.226 → 1.214. El test pasa (tolerancia 15 %). |

### Riesgos que quedan (no corregidos)
* Bit interno `kGroundOnly = 1<<5` visible en `FieldView::flags`: los consumidores deben usar máscaras (hoy todos lo hacen). Pendiente reservarlo en `lbm/field.hpp` (archivo compartido, no editado).
* Rebote implícito full-way: transitorios con 1 paso de retardo frente a half-way (estacionario idéntico). Es lo que pide la especificación (sólidos sin procesar).
* La fuerza del id 255 (suelo) incluye la presión sobre TODO el plano z=0 (~−N_xy/3 en z): no usarla como carga aerodinámica.
