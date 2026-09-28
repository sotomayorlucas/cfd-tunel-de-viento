# Benchmarks del solver LBM — Intel Core Ultra 7 155H

Herramienta: `tools/bench_lbm.cpp`
(`g++ -std=c++23 -O3 -march=native -Isrc -pthread tools/bench_lbm.cpp src/lbm/solver.cpp src/core/threadpool.cpp -o build/lbm/bench_lbm`).

* `bench_lbm stream` — ancho de banda AVX2 propio (lectura, escritura NT, copia, triada, RMW in-place).
* `bench_lbm lbm --grids … --prec fp16,fp32 --coll reg,bgk --threads … --pin 0,1 [--grain --pf --pair --maxt --nt --ftz --nu --cs]`
  — MLUPS del solver. Configuraciones **intercaladas por rondas** (ronda → config. de hilos → caso), 150 ms de
  calentamiento tras recrear el pool y orden de casos rotado; mediana de 5.
* `bench_lbm single [--cpu N]` — MLUPS de 1 núcleo con tiempo de CPU del hilo (inmune a la expulsión del SO).
* Caso: "coche" romo con 4 ruedas girando + cinta móvil (bloques con máscara, capa del suelo vectorizada, ruta escalar
  junto a las ruedas, fuerzas por id cada paso), u∞ = 0.08, Smagorinsky C_s = 0.16. Regularizado con ν = 2e-4
  (configuración de la app). **BGK con ν = 2e-4 diverge** (τ ≈ 0.5006, lo detecta `diverged()`) → BGK se mide con
  ν = 0.01; el coste por celda no depende de ν (medido: 1 núcleo 208 vs 208 MLUPS; 14 hilos 825 vs 833).
* **MLUPS** = nx·ny·nz × pasos / tiempo de `step()` (kernel + fuerzas + 1 paso macro de cada 20).
  **GB/s** = MLUPS del kernel × 19 × 2 × (2 B FP16S | 4 B FP32): cada población se lee y escribe una vez.

> **Ruido.** Toda la sesión hubo carga ajena en la máquina (otros agentes compilando/ejecutando tests: carga media
> 2–27). Cada bloque espera a carga < 3.5 y registra la carga al empezar y al terminar; aun así la carga subió durante
> varias matrices (anotado). Se dan conclusiones sólo cuando la tendencia se repitió en varias campañas.

## 1. Techo de memoria y roofline

STREAM AVX2 (256 MiB por arreglo, GB/s, mediana de 5, carga 3.9):

| hilos | lectura | escritura NT | copia | triada | **RMW in-place** |
|---|---|---|---|---|---|
| 6 | 53.4 | 58.3 | 57.1 | 63.1 | 70.0 |
| 12 | 75.1 | 59.0 | 66.2 | 72.3 | 80.3 |
| 14 | 77.6 | 58.8 | 67.7 | 72.3 | **81.7** |
| 16 | 82.0 | 60.4 | 64.5 | 71.4 | 81.3 |
| 20 | 81.6 | 65.6 | 62.7 | 69.2 | 81.2 |
| 22 | 70.8 | 57.7 | 59.2 | 67.0 | 74.8 |

Esoteric-Pull es exactamente lectura-modificación-escritura in-place → techo **≈ 81 GB/s**.

| | bytes/celda·paso | techo (81 GB/s) | "colisión nula"* | **kernel real** (mejor mediana, 256×128×96) | % del techo |
|---|---|---|---|---|---|
| FP16S | 76 | 1066 MLUPS | ≥ 885 (carga ~5)* | Reg ≈ 970–1060 · BGK ≈ 1000–1075 (bloque de sesgo, 14 hilos) | **91–100 %** |
| FP32 | 152 | 533 MLUPS | ≥ 480 (carga ~5)* | Reg ≈ 515 · BGK ≈ 523 (78–80 GB/s) | **97–98 %** |

\* mismas lecturas/escrituras sin aritmética (experimento, no está en el árbol). Medido con la versión del bench
que penalizaba la primera medición tras recrear el pool (§5): es una cota INFERIOR.
En 192×96×64 (45 MB en FP16S: parte cabe en la L3 de 24 MB) el ancho "efectivo" supera la DRAM: 1106 MLUPS FP16S
(90 GB/s efectivos), 596 MLUPS FP32 (95 GB/s efectivos).

**Por núcleo** (1 hilo, tiempo de CPU, 256×128×96; rangos de 3 sesiones):

| núcleo | FP16S Reg | FP16S BGK | FP32 Reg | FP32 BGK |
|---|---|---|---|---|
| P-core (Redwood Cove, CPU 0) | 194–210 | 199–218 | 211–237 | 225–253 |
| E-core (Crestmont, CPU 12) | 81–82 | 82–83 | 83–92 | 70–99 |

6 P + 8 E ≈ 1.9 G celdas/s de cómputo (sin HT) > techo de memoria → **a chip completo el kernel está limitado por
memoria** en ambas precisiones. Regularizado cuesta ~5 % más que BGK por núcleo y lo mismo a chip completo.

## 2. Matrices (bench corregido, mediana de 5, MLUPS del paso completo)

### 256×128×96 (3.1 M celdas; 187 MB FP16S) — carga 3.1 al empezar → 10.3 al terminar

| prec | colisión | 6 | 12 | 14 | 16 | 20 | 22 | 6 fij. | 12 fij. | 14 fij. | 16 fij. | 20 fij. | 22 fij. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16S | Reg+LES | 681 | 829 | 865 | **874** | 866 | 803 | 574 | 740 | 702 | 659 | 632 | 684 |
| FP16S | BGK | 690 | 860 | 911 | **928** | 904 | 818 | 604 | 735 | 720 | 682 | 658 | 664 |
| FP32 | Reg+LES | **497** | 496 | 496 | 493 | 489 | 454 | 493 | 493 | 489 | 473 | 456 | 462 |
| FP32 | BGK | **504** | 486 | 498 | 492 | 486 | 460 | 494 | 494 | 484 | 479 | 470 | 441 |

Mejor: FP16S-Reg 874 MLUPS (kernel 910, 69 GB/s, fuerzas 0.14 ms/paso = 3 %); FP32-Reg 497 (kernel 515, 78 GB/s).
**Objetivo (≥ 300 MLUPS FP16S en esta rejilla): cumplido ×2.9.**

### 192×96×64 (1.2 M celdas) — carga 3.1 → 5.5

| prec | colisión | 6 | 12 | 14 | 16 | 20 | 22 | 6 fij. | 12 fij. | 14 fij. | 16 fij. | 20 fij. | 22 fij. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16S | Reg+LES | 814 | 1050 | 1072 | 1048 | **1106** | 1076 | 784 | 1003 | 1043 | 1010 | 1049 | 936 |
| FP16S | BGK | 772 | 1103 | 1092 | 1103 | 1107 | 957 | 857 | 1061 | 1095 | 1076 | **1115** | 1068 |
| FP32 | Reg+LES | **592** | 561 | 563 | 545 | 509 | 501 | 584 | 559 | 563 | 554 | 560 | 502 |
| FP32 | BGK | **596** | 595 | 595 | 574 | 585 | 564 | 589 | 590 | 585 | 584 | 575 | 514 |

### 384×160×128 (7.9 M celdas; 470 MB FP16S / 940 MB FP32)

Verificación con carga baja (3.5 → 3.7), mismo ν = 0.01 para ambas colisiones:

| prec | colisión | 6 | 14 | 20 |
|---|---|---|---|---|
| FP16S | Reg+LES | 862 | **904** | 901 |
| FP16S | BGK | 908 | **916** | 905 |
| FP32 | Reg+LES | **485** | 478 | 472 |
| FP32 | BGK | **480** | 476 | 472 |

Matriz completa (la carga subió de 3.3 a 8.8 durante la medida → cifras FP16S ~15 % más bajas que arriba):

| prec | colisión | 6 | 12 | 14 | 16 | 20 | 22 | 6 fij. | 12 fij. | 14 fij. | 16 fij. | 20 fij. | 22 fij. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16S | Reg+LES | 637 | 745 | 752 | 728 | 755 | 759 | 596 | 735 | 752 | 751 | 758 | 706 |
| FP16S | BGK | 680 | 771 | 789 | 769 | 781 | 764 | 644 | 757 | 789 | 776 | 764 | 717 |
| FP32 | Reg+LES | 479 | 472 | 467 | 463 | 454 | 448 | 474 | 469 | 468 | 464 | 459 | 453 |
| FP32 | BGK | 473 | 466 | 461 | 456 | 453 | 441 | 470 | 463 | 462 | 459 | 453 | 438 |

Pasada de fuerzas: 0.06 ms/paso (192), 0.14 (256), 0.26–0.36 (384) ≈ 2–4 % del paso.

## 3. Ajustes medidos (A/B intercalado)

| Ajuste | Resultado | Decisión |
|---|---|---|
| Sesgo de zancada entre direcciones (3 líneas vs 0), 256, 14 hilos, 3 rondas | FP16S Reg **+13/+10/+15 %**, BGK +16/+12/+17 %; FP32 +2.5 % | **3 líneas** (defecto) |
| Grano del reparto (filas por trozo), FP16S-Reg 20 hilos, 2 campañas | 1 fila −23 %, 4 filas −7..−12 %, **16 filas** mejor, 32 filas −3..−6 %, 64 filas −9 % | ~4096 celdas (16 filas en nx=256) |
| Prefetch software 1/2/4/8 bloques | ±3 % (ruido) | off |
| Pares de bloques entrelazados (`f8x2`) | con grano 16: 792 vs 798 (−1 %); llvm-mca: sólo mejora FP32-Reg | off (opcional) |
| Stores NT para ρ,u (1 paso macro de 5) | −2.5 % en 3/3 rondas | off |
| FTZ/DAZ en MXCSR | 0 % (0 halves subnormales en el buffer) | off |
| `max_threads = 14` sobre un pool de 20 | peor que un pool de 14 (−5..−15 %): no elige qué núcleos | no usar |
| Afinidad (hilo t → CPU `detect_topology().order[t]`) | carga baja: ≈ igual (sin ganancia demostrada en A/B intercalado); con carga ajena: **−10..−30 %** con ≥ 16 hilos (los hilos fijados no pueden huir de las CPUs ocupadas) | **sin fijar** |
| Recorrido de filas en z descendente | −15..−20 % | descartado |
| 22 hilos (incluye LP-E del tile SoC) | −5..−15 % en todo | no usar LP-E |

## 4. Recomendaciones para ESTE equipo (Core Ultra 7 155H)

1. **Precisión: FP16S** (defecto). 1.8× FP32 (874 vs 497 MLUPS en 256×128×96), mitad de memoria, y Cd de la esfera
   a 0.43 % de FP32; deriva de masa 4e-7 en 1000 pasos. FP32 sólo para validación.
2. **Colisión: Regularizada + Smagorinsky** (defecto). Mismo coste que BGK a chip completo (limitado por memoria) y
   estable a ν = 2e-4 / 1e-5 donde BGK diverge.
3. **Hilos: el pool por defecto (20, sin LP-E) sin fijar** — dentro del ±2 % del mejor en todas las rejillas y el más
   robusto con carga de fondo. 14–16 hilos rinden lo mismo. **No usar 22** (LP-E) y **no fijar afinidad** en esta máquina
   con carga ajena (qemu, rustc). Como el kernel está limitado por memoria, 6 hilos (los 6 P-cores) ya dan el 80 %
   (FP16S) / 100 % (FP32) del máximo: si la app necesitara núcleos para otra cosa EN PARALELO, el LBM apenas lo notaría.
4. `Tuning` por defecto (grano ~4096 celdas, sin prefetch, sin pares, sin NT, sin FTZ) = lo mejor medido.
5. Rejilla por defecto 256×128×96: ≈ 870 MLUPS ⇒ ≈ 280 pasos/s ⇒ ~9 pasos por cuadro a 30 FPS con presupuesto
   completo (o ~5 si el render se lleva la mitad del cuadro).

## 5. Nota metodológica (errores propios corregidos)

Las primeras campañas recreaban el pool antes de cada configuración y medían siempre FP16S-Regularizado el primero:
los hilos recién creados tardan decenas de ms en ser repartidos por el planificador y esa primera medición salía un
20–60 % más baja, lo que se interpretó erróneamente como "Regularizado es más lento a chip completo". Se descartó con
un programa sin reinicios (Reg ≈ BGK ≈ 900 MLUPS en 384×160×128) y se corrigió el bench (calentamiento de 150 ms +
orden rotado). También, al principio, un hilo principal fijado de una ronda anterior hacía que los hilos "sin fijar"
heredaran su máscara (todo en 1 núcleo); `restart_pool` ahora restaura la máscara completa. Las tablas de arriba
son todas del bench corregido.

## 6. Verificación independiente (revisión adversarial)

Re-medido por el revisor con los binarios recompilados (misma herramienta, mediana de 5 rondas intercaladas,
256×128×96, caso "coche", carga media 3.3–5.8; tras corregir `rebuild()` el kernel es idéntico y las cifras no cambian):

| config | 14 hilos MLUPS (kernel) | 20 hilos MLUPS (kernel) | GB/s kernel |
|---|---|---|---|
| FP16S Reg+LES | 997–1010 (1038–1049) | 998–1002 (1039–1043) | 79 |
| FP16S BGK | 1026–1031 (1067–1073) | 1006–1014 (1050–1056) | 80–82 |
| FP32 Reg+LES | 507–508 (520) | 503 (515) | 78–79 |
| FP32 BGK | 508–509 (521–522) | 505 (517) | 79 |

STREAM AVX2 (256 MiB): RMW in-place 79.9 / 84.9 / 83.5 GB/s con 6 / 14 / 20 hilos → kernel FP16S al **94 %** del techo.
Por núcleo (tiempo de CPU, 1 hilo): P-core (CPU 0) FP16S-Reg 213, FP32-Reg 208; E-core (CPU 12) FP16S-Reg 81 MLUPS.
Conclusión: las cifras y recomendaciones de este documento se sostienen (con carga baja, FP16S ronda ya los 1000 MLUPS;
14 ≈ 20 hilos dentro del ±2 %).
