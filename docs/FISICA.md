# Física del túnel de viento: método, fuerzas, límites y calibración

Este documento explica **qué calcula** el solver, **cómo se miden las fuerzas** (y las correcciones que
hacen falta para que sean correctas), **qué se ha verificado** con pruebas automáticas, la **altura de
marcha efectiva** que impone la resolución, la **tabla de calibración** medida con el solver real y,
sobre todo, **qué se puede y qué no se puede predecir** con él. Todo lo que aquí aparece como número
está medido con el propio programa (se indica el caso y la resolución).

Resumen honesto en tres líneas:

* Es un LBM D3Q19 con LES y ley de pared a **resoluciones de centímetros** (dx = 21-48 mm en un F1). Las
  **tendencias** (efecto suelo, DRS, guiñada, alturas, flaps, épocas con/sin alerones) son útiles; los
  **valores absolutos** de carga de un F1 moderno salen **muy por debajo** de los reales (0.3-0.7 m² de
  SCz a Media/Alta en vez de 3-5 m²) y la resistencia **por encima** (SCx ≈ 1.5-2.1 m² en vez de 1.0-1.4).
* Las pruebas de física básicas pasan: presión de referencia (manométrica) exacta, invariancia galileana,
  efecto Magnus del orden de la literatura, rebote interpolado independiente de la posición en la red,
  fricción turbulenta de placa plana con la ley de pared, simetría y signo de la fuerza lateral.
* Donde falla es en lo que la red no resuelve: alerones de 7-15 celdas de cuerda, capas límite de una
  fracción de celda, ranuras de 1-2 celdas, desprendimientos inducidos por la escalera de vóxeles.

---------------------------------------------------------------------------------------------------

## 1. Método numérico

### 1.1 Retículo, almacenamiento y colisión

* **D3Q19** (19 velocidades), unidades de red: celdas y pasos; ρ∞ = 1, p = ρ/3 (c_s² = 1/3).
* **Esoteric-Pull** (Lehmann 2022): streaming *in situ* con una sola copia de las poblaciones (la mitad de
  memoria y de tráfico que A-B). Verificado bit a bit contra una referencia A-B independiente
  (tests/test_lbm.cpp, pruebas 2a-2j).
* **FP16S**: se guardan las poblaciones *desplazadas* f̃ = f − w escaladas por 2¹⁵ en IEEE half (F16C); la
  aritmética es FP32. Diferencia con FP32 en el Cd de una esfera a Re = 100: 0.27 % (prueba 6 de test_lbm).
* **Colisión regularizada RECURSIVA** (`lbm::Collision::Recursive`, Malaspinas 2015; Coreixas et al., PRE 96,
  033306, 2017): el no equilibrio se reconstruye con su parte de 2.º orden (Π^neq, proyección de Hermite) **y** la de
  3er orden obtenida recursivamente de Π^neq (a3neq_αβγ = u_α Π_βγ + u_β Π_αγ + u_γ Π_αβ, en las 6 combinaciones
  de 3er orden que soporta D3Q19). Con sólo la proyección de 2.º orden (`Regularized`, el esquema anterior) el
  solver era **linealmente inestable** a ν = 1·10⁻⁴ con u∞ = 0.09: ver §1.5. Más **viscosidad de volumen** propia
  (`Config::bulk_omega` = 1: la traza de Π^neq se relaja con ω_b = 1, ν_b = 1/9 en red), que amortigua el sonido y
  el modo par/impar sin tocar la cortante (sólo actúa sobre ∇·u). En una **capa de 8 celdas junto a los cuerpos**
  se mantiene la proyección de 2.º orden (con la que se calibró la física de pared, §1.5). + **Smagorinsky** (LES) con C_s = **0.10** (el
  valor clásico para flujos de cizalla; con 0.16 la capa límite engordaba y se despegaba antes). La viscosidad
  turbulenta sale del tensor de no equilibrio local (sin diferencias finitas).
* Viscosidad molecular de red por defecto **ν = 1·10⁻⁴** (τ = 0.5003). Da las mismas fuerzas que 1·10⁻⁵
  (NACA 0012 a 6°: CL 0.274 frente a 0.277) con 10× más margen de estabilidad. Con 3·10⁻⁴ la resistencia
  ya sube un 3 %.
* **u∞ = 0.09** en unidades de red (Mach de red 0.156: compresibilidad despreciable, Cp de estancamiento
  1 + Ma²/4 = 1.006).

### 1.2 Contornos del túnel

* Entrada (x = 0), laterales y techo: **frontera de equilibrio de campo lejano** (ρ = 1, u = u∞).
* Salida (x = nx−1): equilibrio con ρ = 1 y u extrapolada; **esponja** en el 12 % final del dominio
  (viscosidad creciente hasta ν = 0.12) para que la estela no se refleje.
* Suelo: ninguno (campo lejano), **fijo** (no deslizante: túnel antiguo, crece capa límite) o **cinta
  móvil** a u∞ (Ladd): el suelo real visto desde el coche.
* Dominio de un coche: 0.5 L aguas arriba, 1.6 L aguas abajo y sección con **bloqueo ≤ 10 %**. Con el
  bloqueo y la estela, la presión estática aguas arriba del coche queda en Cp ≈ +0.1-0.2 respecto a la
  de la entrada (balance de cantidad de movimiento de un túnel cerrado). Para mostrar Cp como en un
  túnel real, `Sim::rho_ref` da la "toma estática" (ρ medio de un plano a mitad de camino entre la
  entrada y el objeto).
* **Arranque impulsivo** (u = u∞ en todo el fluido desde el paso 0). Genera un pulso acústico que se va
  en ~0.4 pasos de flujo (RMS de Cp en todo el fluido del F1 2022 a Rápida, `tools/noise_probe`: 0.26 a 0.12 PF,
  0.16 a 0.25 PF, 0.14 a 0.37 PF y 0.13 estable —el campo de presión del propio coche—; antes de la corrección de
  §1.5 bajaba a 0.14 y volvía a SUBIR a 0.18-0.19 por el ruido). Una rampa desde el reposo sigue siendo mucho peor
  también con el esquema nuevo (el túnel tiene que "llenarse" de forma compresible): con 0.1 o 0.3 PF de rampa, a
  los 2.5 PF el Cp medio 40 celdas aguas arriba del F1 2022 sigue en +0.55-0.59 (impulsivo: +0.21 estable desde
  0.4 PF) y en la esfera en +1.9. Las fuerzas del primer ½ paso de flujo no se promedian.
* **Recuperación automática**: si el solver detecta NaN o ρ fuera de (0.2, 5), `Sim::step` triplica la
  viscosidad y reinicia el flujo (la app lo avisa).

### 1.3 Paredes sólidas

* **Rebote interpolado** (Bouzidi, Firdaouss y Lallemand 2001) con la distancia a la **superficie real**
  (la función de distancia con signo del modelo): la pared deja de ser una escalera. Esfera a Re = 100 con
  D = 10 celdas: Cd 1.202/1.198 al desplazar la esfera ¼ y ½ celda (0.3 % de diferencia) frente a
  1.299/1.293 con la escalera (Schiller-Naumann: 1.092) — prueba [4] de tests/test_aero.cpp.
* **Paredes móviles**: las **ruedas** (y cualquier pared móvil que no sea la cinta) usan el mismo rebote
  interpolado más el término de pared móvil g_q·6w_k(c_k·u_w); en una escalera de vóxeles el rebote
  implícito de Ladd "soplaba" en las caras (la velocidad tangente a la rueda tiene componente normal a
  las caras de los vóxeles): esfera giratoria a Re = 100, α = 0.5 → CL 0.62 / CD 1.46 con Ladd,
  **0.51 / 1.31** con el rebote interpolado (CD sin giro 1.20). Las paredes impermeables (ruedas) restan
  además la masa neta que la pared inyecta en cada nodo. La **cinta** sigue con rebote implícito + Ladd.
* **Huella de contacto**: las 1-2 capas de fluido encajonadas entre el neumático y la cinta se rellenan y
  se mueven con la cinta (un neumático real se aplana en una huella de ~15-20 cm). Sin ellas la cuña
  rueda-suelo de 1-2 celdas "bombea" (Cp ±13) y da sustentación espuria.

### 1.4 Ley de pared (modelo `Slip`, por defecto)

Con dx de centímetros, la capa límite real de un coche (y⁺ de la primera celda ~ 100-1000) cabe en una
fracción de celda. Un rebote no deslizante la convierte en una capa artificial de 3-8 celdas con fricción
×5-10. El modelo por defecto hace tres cosas en cada nodo de fluido junto a una pared fija:

1. **Impone exactamente la tensión de pared** de Werner-Wengle (ley potencial 1/7 con subcapa lineal),
   calculada con la velocidad tangente del nodo, su distancia real a la pared y la **viscosidad del aire
   real** a la velocidad elegida (`wall_nu`: por eso la velocidad en km/h cambia —poco— la fricción).
   Para ello la pared se "mueve" tangencialmente a una velocidad s·t̂: la fuerza tangencial que el nodo
   transmite es lineal en s, F_t(s) = F_ns − G·s (F_ns: la del rebote no deslizante con las poblaciones del
   paso; G: la conductancia de los enlaces, descontando la corrección de masa), y se elige s para que
   F_t = τ_w·A_nodo, con s ∈ [0, |u_t|] (la pared sólo reduce la fricción del rebote; nunca empuja).
2. Mantiene el **no deslizamiento en aristas vivas** (bordes de salida, esquinas: residuo de planitud de
   la distancia ≥ 0.4 celdas) → condición de Kutta y desprendimiento fijo en aristas; y cuando la capa
   límite está resuelta (y⁺ < 5).
3. **Amortigua la viscosidad de Smagorinsky** en esa primera celda (τ = max(τ₀, ½ + ¼(τ_LES − ½)), en la
   línea del amortiguamiento de van Driest): sin ello la viscosidad turbulenta, alimentada por la cizalla
   de la propia pared, engordaba la capa límite y la despegaba (NACA 0012 a 6°: CL 0.15 → 0.24; con
   C_s = 0.10, 0.30).

Medidas (tests/test_aero.cpp [6] y ensayos): placa plana a Re_x ~ 10⁶-10⁷ con la superficie real fuera
del centro del enlace: **cf = 1.3× la turbulenta** en la prueba (0.0045 frente a 0.0034; 0.5-1.1× en placas
más largas), velocidad de la primera celda 0.70 U; con rebote no deslizante cf ×2.9 y 0.28 U (×5-10 con
Smagorinsky sin amortiguar). Límite conocido: en superficies **inclinadas respecto
a la red** la escalera de vóxeles sigue añadiendo resistencia (placa a 1:10: cf ≈ 4× la turbulenta) y,
a sotavento de cada escalón descendente, la primera celda queda casi parada: con gradiente adverso esto
**adelanta el desprendimiento** (es la causa principal de la poca sustentación de los perfiles finos).

Otros modelos disponibles (`--wall`): `none` (rebote no deslizante con LES) y `log` (ley de pared por
viscosidad en la primera celda).

### 1.5 Ruido del campo lejano: diagnóstico y corrección

**Síntoma** (revisión de la fase 2): con los defectos (ν = 1·10⁻⁴, τ = 0.5003, u∞ = 0.09, C_s = 0.10) el campo
lejano estaba lleno de ruido de escala de red: dameros en los cortes de Cp, bandas en |u|, criterio Q positivo en
todo el dominio (el umbral de vórtices tuvo que subirse a 600 en los coches).

**Medida** (`tools/noise_probe`, ver su cabecera): regiones UPin (x = 2-6), UP40 (~40 celdas aguas arriba del
objeto), SIDE y TOP (franjas del 10 % junto a los laterales y el techo, antes de la esponja); RMS espacial de u_x/U − 1
y de Cp, cociente de paso alto hp = RMS(φ − media de los 6 vecinos)/σ(φ) (≈ 0 suave, ≈ 1 ruido blanco, 2 damero),
σ temporal por celda en los últimos 0.5 PF y la segunda diferencia temporal d2t = RMS(φ_{t+1} − ½(φ_t + φ_{t+2}))
(grande si hay una oscilación de periodo 2 pasos, la firma de ω ≈ 2).

**Causa.** No eran (principalmente) reflexiones en las caras: el **túnel VACÍO** con cinta, sin ningún cuerpo, se
llenaba solo. El RMS de u_y,u_z crecía ×10 cada 250 pasos desde la entrada y saturaba en **±38 % de u∞**
(sólo lo contenía la viscosidad de Smagorinsky que el propio ruido activa; con C_s = 0 divergía). Sin suelo quedaba
limpio... hasta que se le daba una semilla (un cubo de 6 celdas durante 100 pasos): entonces se llenaba igual.
Es la **inestabilidad lineal de la regularización de 2.º orden** (PR) a ν → 0 con Ma ≈ 0.16: el no equilibrio
de 3er orden se pone a cero en cada paso y los modos que no ve la proyección no se amortiguan. Mapa de estabilidad
del túnel vacío (376×100×68, 3000 pasos, sembrado con el cubo o con la cinta, que siembra sola):

| Esquema | u∞ = 0.07 | 0.09 | 0.11 | 0.13 | 0.16 | 0.20 |
|---|---|---|---|---|---|---|
| PR (anterior) | ruido 30 % | **38 %** | | | | |
| PR + ν_b (ω_b = 1) | | 36 % (tarda más en crecer) | | | | |
| PR con ν = 3·10⁻⁴ / 1·10⁻³ | | estable | | | | |
| RR (no equilibrio de 3er orden) | | estable | 1 % | 1 % | | |
| sólo el equilibrio de 3er orden (ρuuu) + ν_b, sin el no equilibrio | | **23 %** | | 30 % | | |
| **RR + ν_b** | | estable | | estable | estable | **estable** (también sin LES) |

(Ruido = RMS de u_⊥/u∞ tras 3000 pasos; "estable" = la semilla se va aguas abajo y el campo vuelve a ~10⁻⁴.) La
parte que estabiliza es el **no equilibrio** de 3er orden; su equilibrio (ρuuu) no aporta nada y no se usa. La
viscosidad de volumen añade margen para las zonas rápidas junto a los cuerpos (u ≈ 1.5-2 u∞ en las succiones) y
amortigua el sonido: sin ella queda un modo acústico de periodo 2 (σ_t(Cp) = 0.05 aguas arriba del F1 2026).
Otras hipótesis, medidas: FP16S (en FP32 igual), la esponja (sin esponja igual), reflexiones de las caras de
equilibrio (el túnel vacío sin semilla ni cinta está limpio: no generan ruido; tras la corrección los modos acústicos
del túnel que quedan son de ~10⁻³ en Cp → no se añadieron capas absorbentes: no queda nada que absorber) y el
arranque (una rampa sigue siendo peor, §1.2).

**Antes / después** (Rápida, 2.5 PF desde el arranque impulsivo, FP16S; CPU y GPU dan lo mismo a 3 cifras):

| Caso · región | RMS(u_x/U − 1) | σ espacial Cp · hp | σ_t(u_x/U) | σ_t(Cp) | d2t(Cp) |
|---|---|---|---|---|---|
| túnel vacío (dominio del F1 2026) · UP40 | 5.3 % → **0.05 %** | 0.160 · 0.98 → 0.0002 | 0.039 → 0.0000 | 0.137 → 0.0001 | 0.24 → 0.0001 |
| túnel vacío · SIDE | 4.9 % → 0.05 % | 0.114 → 0.0003 | 0.032 → 0.0000 | 0.082 → 0.0001 | 0.19 → 0.0002 |
| F1 2026 · UPin | 2.5 % → 0.9 % | 0.122 · 0.91 → 0.016 · 0.03 | 0.020 → 0.0001 | 0.100 → **0.0011** | 0.18 → 0.0005 |
| F1 2026 · UP40 | 5.7 % → 1.4 % | 0.154 · 1.04 → 0.017 · 0.04 | 0.048 → 0.0003 | 0.139 → **0.0013** | 0.25 → 0.0008 |
| F1 2026 · SIDE / TOP | 10.3 / 9.4 % → 7.8 / 6.5 % | 0.19 / 0.18 · 0.94 → 0.10 / 0.09 · 0.01 | 0.057 → 0.007 | 0.145 → 0.004 | 0.27 → 0.0010 |
| F1 2022 · UP40 | 6.1 % → 1.5 % | 0.153 · 1.03 → 0.017 · 0.04 | 0.046 → 0.0003 | 0.139 → 0.0013 | 0.25 → 0.0008 |
| NACA 0012 6° · UPin | 3.2 % → 0.15 % | 0.167 · 0.89 → 0.022 · 0.02 | 0.023 → 0.0001 | 0.141 → 0.0012 | 0.25 → 0.0005 |
| esfera · UP40 | 0.8 % → 0.2 % | 0.032 · 0.71 → 0.010 · 0.07 | 0.007 → 0.002 | 0.064 → 0.020 | 0.034 → 0.0008 |

Lo que queda en UP40/SIDE/TOP con un cuerpo es **estacionario y suave** (hp ≈ 0.01-0.04): el Cp de bloqueo aguas
arriba (+0.2 en los coches, ver §1.2) y la aceleración del flujo a los lados del coche (u ≈ 1.07 U junto a las
paredes laterales). En la esfera queda una oscilación global de Cp aguas arriba (σ_t 0.02, periodos de 74 y 148
pasos): un modo acústico del túnel excitado por la estela (las caras de equilibrio reflejan), 3× menor que antes.
Espectro de u_x a lo largo de y 40 celdas aguas arriba del F1 2026: energía en k > ½k_Nyquist 9 % → 3 %; sonda de
Cp: de periodos de 21-42 pasos (ruido) a 650-680 (el tránsito acústico del túnel, nx/c_s = 651).

("Después" = esquema final, con la capa de abajo; sin capa el campo lejano da lo mismo a 2 cifras. En todo el
fluido del NACA el cociente de paso alto de Cp queda en 0.39 (antes 0.92; sin capa 0.09): algo de ruido de escala
de red sobrevive dentro de la capa de 2.º orden, junto al ala y en los torbellinos de punta.)

**La RR también cambia la física de pared → capa de 2.º orden junto a los cuerpos.** Con la RR en TODO el dominio
el campo lejano quedaba igual de limpio, pero cambiaba la calibración mucho más que el ruido de una tanda
(Media, 4-5 PF, media de los 2 últimos): NACA 0012 a 6° CL 0.30 → **0.13** (la sustentación cae de forma continua
de 0.29 a 0.10 en 6 PF: desprendimiento desde el borde de ataque en la cara de succión), ala en efecto suelo a 60 mm
1.05 → 0.70, Ahmed CL −0.29 → +0.82, F1 2022 SCz 0.59 → 1.06. Descartado que fuera el campo lejano limpio: con la
proyección de 2.º orden a ν = 3·10⁻⁴ (estable, sin ruido) el NACA mantiene CL 0.29 y con la RR a esa misma ν cae a
0.09; la viscosidad de volumen no influye (0.22 con y sin ella). La causa es la RR dentro de la capa límite y las
capas de cortadura a estas resoluciones (menos disipación de los momentos de 3er orden en zonas mal resueltas; la ley
de pared y la calibración se hicieron con la proyección de 2.º orden). Solución: `Config::rr_wall_layer` = **8**
celdas: en una capa de Chebyshev de 8 celdas alrededor de los sólidos que no son el suelo (extendida a bloques de 8
en x; bit interno `kLayer` de los flags, también en la iGPU) la colisión no añade el término de 3er orden. Medido
(NACA 0012 a 6°, Media, CL a 6 PF): sólo la 1.ª celda 0.22, capa de 3 celdas 0.17 y bajando, 6 celdas 0.31 estable
(= 2.º orden). F1 2022 a Rápida, SCz (2.º orden con ruido: 0.50): capa 6 → 0.73, **8 → 0.51**, 12 → 0.46, 24 → 0.35.
Ruido del F1 2026 con la capa: 6-8 celdas igual que sin capa; 12 empieza a reaparecer junto al coche (d2t(Cp) 2.5e-3
en SIDE); 24 vuelve (d2t 0.043): la inestabilidad crece dentro de una capa de 2.º orden gruesa.

**Calibración antes / después** (Media, `tools/noise_probe --calib 2`, 4 PF coches / 5 objetos, media de los 2
últimos; "antes" = árbol anterior medido igual, reproduce la tabla de §5.1 dentro del ruido: F1 2022 0.59 / 2.04
frente a 0.62 / 2.01):

| Caso | antes | después (RR + ν_b + capa 8) | RR sin capa |
|---|---|---|---|
| F1 2022 SCz / SCx (m²) | 0.59 / 2.04 (bal. 22 %) | **0.82 / 2.15** (bal. 35 %) | 1.06 / 2.15 |
| F1 1967 SCz / SCx | −0.22 / 0.95 | −0.19 / 1.02 | −0.17 / 1.00 |
| ala en efecto suelo, 60 mm, CL / CD | 1.05 / 0.355 | 1.21 / 0.384 | 0.70 / 0.306 |
| ala en efecto suelo, 300 mm, CL / CD | 0.86 / 0.255 | 0.92 / 0.278 | 0.81 / 0.249 |
| NACA 0012 6°, CL / CD | 0.295 / 0.082 | **0.301 / 0.083** | 0.132 / 0.080 |
| esfera CD | 0.349 | 0.367 | 0.257 |
| Ahmed 25°, CD / CL | 0.659 / −0.29 | 0.619 / +0.10 | 0.710 / +0.82 |

Lectura honesta: el NACA 0012 (ala libre, el caso más limpio) no cambia. La esfera y los coches suben la resistencia
un 5-8 % (no baja: el ruido NO era la causa del exceso de resistencia de §5); el Ahmed la baja un 6 %. Las cargas que
dependen de aire limpio delante (alerón delantero del F1 2022 0.50 → 0.63 m², fondo −0.06 → +0.11; ala en efecto
suelo +7..16 %) suben más que el ruido de una tanda (±0.05-0.1 m²): antes trabajaban en un "flujo libre" con ±38 % de
fluctuación espuria. La tabla de §5.1 es la del esquema anterior; con el nuevo, F1 2022 a Media = 0.82 / 2.15 m².

Capturas antes/después (sin ventana, Rápida, 2.5 PF): `build/app/shots/noise/` (cortes de Cp del ala en efecto suelo
y del F1 2022, |u| del perfil pseudo-2D, Cp en superficie del NACA 0012, vórtices Q del F1 2026 con umbral 150 y del
F1 2022 con el nuevo umbral por defecto). Umbral de vórtices por defecto (Q·L²/U²): coches 600 → **300**, cuerpos 800 →
**400**; alas libres se quedan en 30 porque con menos aparece el ruido que queda dentro de la capa junto al ala.

Coste: ver docs/opt/lbm.md §5 y docs/GPU.md (CPU: por núcleo llvm-mca 129 → 131 ciclos por bloque de 8 en FP16S;
iGPU: el kernel de celdas pasa de ~7 a ~9 ms/paso a Media, +20 % de instrucciones y algunos derrames de registros en
el modo de 2 celdas por hilo).

---------------------------------------------------------------------------------------------------

## 2. Fuerzas

### 2.1 Intercambio de cantidad de movimiento por grupo

La fuerza sobre cada sólido (por id de grupo: alerón, fondo, ruedas…; id 255 = suelo) se obtiene por
intercambio de cantidad de movimiento en cada enlace fluido → sólido, en la pasada de contorno que sigue
a cada paso: F_k = (f_out + f_in)·c_k, sumando momentos respecto a la referencia del modelo. Se promedian
todos los pasos de cada llamada (`forces_mean`) y la app filtra con una constante de ½ paso de flujo.

### 2.2 Corrección manométrica (presión de referencia)

**Problema.** La presión absoluta de red en reposo es p∞ = ρ/3 = 0.333, unas **80 veces** la presión
dinámica ½ρu² = 0.004. En un sólido cerrado se cancela, pero un grupo que **no es una superficie cerrada
para el fluido** —una rueda apoyada en la cinta, dos grupos que se tocan, el plank— recibe p∞·A en la
cara tapada: una sustentación espuria enorme (era parte de la "sustentación de las ruedas" de la primera
integración).

**Solución (en el solver, `lbm::Config::force_gauge`, por defecto activada).** Se integra p − p∞: como las
poblaciones ya se guardan desplazadas (f̃ = f − w), basta con no sumar el término 2w_k del estado en
reposo. Es exacta (no una estimación), gratuita y afecta también al suelo (id 255: ahora es la fuerza
aerodinámica real sobre la cinta).

**Prueba** (test_aero [1]): fluido en reposo, caja apoyada en el suelo y un segundo grupo pegado a ella:
con la corrección |F| = 0 exacto (también con el rebote implícito); sin ella F = −p∞·A de las caras
tapadas, coincidiendo con la suma enlace a enlace esperada (base de 80 celdas² → F_z = −27.1).

### 2.3 Paredes móviles: intercambio galileanamente invariante

En paredes que se mueven (ruedas, cinta) el intercambio clásico no es invariante galileano. Se usa la
forma de Wen et al. (J. Comput. Phys. 266, 2014): F = Σ c_k(f_out + f_in) − u_w Σ(f_out − f_in)
(`force_galilean`, por defecto activada).

**Pruebas.** (test_aero [2]) Couette plano con las placas a U_m ∓ U/2 en dos sistemas de referencia
(U_m = 0 y U_m = U): la tensión de los parches centrales coincide con la analítica (∓0.1067) y entre
sistemas (diferencia < 2 %). (test_aero [3]) **Efecto Magnus**, esfera giratoria a Re = 100 con D = 10
celdas: α = ωD/2U = 0.5 → CL +0.56, CD 1.34; α = 1 → CL +0.84, CD 1.53 (sin giro CD 1.19); con la
corrección de masa de pared impermeable (la que usan las ruedas) α = 0.5 → CL +0.51, CD 1.31. Sustentación
hacia el lado que se mueve con el flujo y creciente con α. Literatura: la correlación de Oesterlé & Bui Dinh
(1998) da 0.58 (α = 0.5) y 0.69 (α = 1); las DNS de esferas giratorias a Re = 100 dan algo menos (≈ 0.35-0.45).
El resultado está en ese intervalo para α = 0.5 (algo alto para α = 1) y baja con la resolución (D = 16: 0.49).

### 2.4 Coeficientes y balance

* CL (+ = carga), CD, CS respecto a ½ρ∞u∞²·A_ref; SCz = CL·A, SCx = CD·A (m²); newtons y kgf a la
  velocidad elegida con el aire real (ρ = 1.225 kg/m³). Sólo los ids 1-254 (el suelo no cuenta).
* Desglose por **componente** (alerones, fondo, difusor, ruedas…): suma exacta = total (test_aero [5]).
* **Balance** (% de la carga en el eje delantero) a partir de la fuerza total y el momento de cabeceo
  respecto a la referencia: se resuelve el reparto de la carga entre los dos puntos de contacto. Sólo se
  muestra con |CL| > 0.05. Con cargas totales pequeñas (0.2-0.6 m²) el balance es muy sensible: un
  alerón delantero que funciona y uno trasero en la estela dan balances > 70 % o incluso fuera de
  [0, 100] %.

### 2.5 Otras pruebas

* Simetría: Ahmed sin guiñada, fuerza lateral |CS| < 0.02 + 5 % CD. Guiñada +10° (el morro gira hacia
  −y): CS < 0 (test_aero [5]).
* Esfera a Re = 100: Cd 1.21 (Schiller-Naumann 1.09, +11 % con D = 16 y bloqueo 1.7 %) (test_lbm 5).
* Masa conservada en cavidad cerrada, Couette analítico (0.2 %), determinismo 1 vs N hilos (test_lbm).

---------------------------------------------------------------------------------------------------

## 3. Altura de marcha efectiva (resolución)

Un F1 lleva el plank a 20-80 mm del suelo; a Media dx = 36 mm, así que 30 mm es **menos de una celda**:
el fondo queda sellado contra la cinta, el flujo bajo el coche no pasa y el fondo da **sustentación**.
Aun con 2.5 celdas de hueco el fondo seguía en sustentación (F1 2019 a Media: SCz del fondo −0.40 m²;
con 3.5 celdas, −0.14; F1 2022: −0.16 → −0.01).

Por eso el solver ve el coche **subido** (los dos ejes lo mismo: el rake se conserva):

  h_eff = (h⁴ + g⁴)^¼   con h la menor de las dos alturas pedidas y **g = 3.5·dx**

= g cuando h → 0, 1.19·g cuando h = g y ≈ h en cuanto h ≳ 1.5·g. Es monótona y suave, así que un barrido
de altura conserva la tendencia (comprimida por debajo de ~g). A Media (dx 35.6 mm) un F1 2022 pedido a
30/80 mm se simula a 125/175 mm; a Alta (27.5 mm) a 97/147; a Ultra (21 mm) a 76/126.

`Sim::params` es lo pedido (deslizadores), `Sim::params_eff` lo que se construye;
`ride_eff_front_mm/rear_mm`, `ride_gap_min_mm` (= g) y `ride_limited` se muestran en el panel/HUD como
"altura efectiva (resolución)". Los barridos (`--sweep ride`) usan la altura PEDIDA en el eje x.

Consecuencia: **el efecto suelo de los coches de túneles (1979, 2022+) está muy atenuado** a Media: la
carga del fondo que se obtiene es la de un coche a 10-15 cm del suelo. Para estudiar alturas reales hace
falta Ultra (o más) — y aun así las ranuras laterales del fondo quedan en 1-2 celdas.

---------------------------------------------------------------------------------------------------

## 4. Reynolds y resolución: qué se resuelve y qué no

* **Reynolds real** de un F1 a 250 km/h: ~2.5·10⁷ (longitud del coche). El flujo resuelto es LES con
  viscosidad de red mínima: el "Reynolds de red" U·L/ν ≈ 2·10⁵ no significa nada físico; lo que fija la
  física cerca de la pared es la ley de pared con la ν del aire real (`wall_nu`).
* **Celdas por cuerda**: alerón delantero de un F1 a Media, 7-10 celdas por elemento; alerón trasero,
  7-9; ranuras de 45 mm = 1.3 celdas (cerradas en la práctica). Un perfil necesita ≳ 40-60 celdas de
  cuerda para que la sustentación se acerque a la real con este método.
* **Espesores**: placas y endplates de 1 celda; perfiles finos de 1-3 celdas de espesor en la parte
  trasera: la escalera de vóxeles adelanta el desprendimiento.
* **Estela**: con capas límite gruesas y desprendimientos, la estela del coche es más ancha y lenta que
  la real: a Media el alerón trasero del 2022 recibe u ≈ 0.5 U con el flujo subiendo 15-20° (difusor +
  beam wing + estela). Por eso su carga es tan baja y por eso se han **subido las incidencias** de los
  alerones traseros modernos en la calibración (sección 5).
* **Ruedas**: con giro, la resistencia de las ruedas sale MAYOR que paradas (F1 2022 a Media: SCx
  ruedas 0.97 con giro, 0.79 paradas); en túnel real el giro la reduce ~20-25 %. Es sobre todo presión
  (estela), no fricción: el rebote interpolado de las paredes móviles la bajó sólo un 4 % (1.06 → 1.02) y
  quitar la huella de contacto la sube (1.29). No se ha resuelto.

Tamaños de celda de los presets (F1 2022): Rápida 47.7 mm (2.5 M celdas), Media 35.6 mm (6.1 M), Alta
27.5 mm (13 M), Ultra 21.3 mm (28 M).

---------------------------------------------------------------------------------------------------

## 5. Calibración de los modelos contra este solver

Criterio: mantener cada época reconocible y las reglas de diseño para redes gruesas
(docs/dev/api_summaries.md), y mover sólo parámetros que el solver no puede reproducir con los valores
reales:

* **Alerones traseros 2008-2026: incidencias mayores que las reales** (plano principal −13..−18° en vez
  de −6..−9°; flap 30-36° en vez de 24-30°): trabajan en la estela con el flujo subiendo 15-20°; con los
  ángulos reales daban SCz 0.04-0.17 m². Ganancia medida: +0.07 m² en el 2022 (0.10 → 0.17).
* **Lomo de la cubierta motor** (2022/2026) de la toma de aire a la trasera: antes la toma (un
  elipsoide) acababa en un escalón de 30 cm sobre la cubierta.
* **Hueco mínimo bajo el fondo g = 3.5 celdas** (sección 3).
* **Alerones delanteros 1979-2014 más cerrados** (1979 16 → 30°, 1998 18 → 24°, 2008 16 → 32°,
  2011/2014 14/26 → 20/32°): con los reales el eje delantero de esos coches quedaba en sustentación
  (balance negativo) y 2008-2014 por debajo de 1998. El alerón delantero responde como cabe esperar (flap
  +10° en el 2022: carga del alerón 0.62 → 0.83 m²).
* **Alerones traseros de 1979 y 1998 más abiertos** (1979 −6°/22° → −2°/10°; 1998 −5°/18°/32° →
  −3°/10°/22°): fuera de la estela (sobre la caja de cambios, o alto) esos alerones sí funcionan y dejaban a
  1979 y 1998 por encima de 2008-2014.
* No se ha tocado la geometría de los fondos/difusores.
* Los ángulos "de la época" que muestra el panel (`Info::default_front_flap_deg/rear`) son ya los calibrados.

Lo que NO se ha podido conseguir (y por qué): las cargas totales de los F1 modernos siguen siendo
~5-8 veces menores que las reales y la resistencia ~1.5 veces mayor; ninguna combinación razonable de
ángulos, cuerdas, ranuras (hasta 75 mm) o alturas del alerón trasero lo levanta de SCz ≈ 0.3 aislado
(L/D ≈ 1 a Media; 0.36 a Ultra): el límite es la resolución, no la geometría. Agrandar los alerones
×3 lo "arreglaría" pero dejaría coches irreconocibles.

### 5.1 Tabla (sim vs referencia)

Medido con `calib` (el mismo `Sim` que la app) el 28-09 con el árbol final: preset **Media**, 4 pasos de flujo
(coches) o 5 (objetos) desde el arranque impulsivo y **media de los 2 últimos**; cinta móvil y ruedas
girando en los F1; alturas de marcha por defecto (se simulan subidas: sección 3); velocidades por defecto.
Ruido de una tanda a otra con la misma geometría: ±0.05-0.1 m² en SCz de un F1 (estelas no estacionarias;
el solver es determinista, así que repetir da lo mismo: hay que cambiar la duración para verlo).

**Coches de F1 (m²; carga +).** Referencias = estimaciones de orden de magnitud (`Info::ref_ClA/ref_CdA`).

| Modelo | dx (mm) | SCz sim | SCx sim | balance | Al. del. | Al. tras. (+beam) | Fondo+difusor | Ruedas SCz / SCx | SCz / SCx ref. |
|---|---|---|---|---|---|---|---|---|---|
| f1_1967 | 28.9 | **−0.22** | 0.95 | — | — | — | — | −0.13 / 0.52 | −0.20 / 0.75 |
| f1_1979 | 31.8 | **+0.42** | 1.55 | 9 % | +0.30 | +0.21 | +0.12 (pontones-ala) | −0.27 / 0.71 | 2.4 / 0.95 |
| f1_1998 | 32.9 | **+0.58** | 1.64 | −6 % | +0.25 | +0.28 | +0.45 | −0.21 / 0.78 | 2.9 / 1.05 |
| f1_2008 | 33.7 | **+0.33** | 1.62 | 29 % | +0.40 | +0.17 | +0.19 | −0.21 / 0.78 | 3.6 / 1.25 |
| f1_2011 | 34.1 | **+0.38** | 1.76 | 58 % | +0.52 | +0.16 | +0.12 | −0.27 / 0.71 | 3.8 / 1.20 |
| f1_2014 | 34.5 | **+0.32** | 1.69 | 72 % | +0.48 | +0.09 | +0.15 | −0.24 / 0.73 | 3.3 / 1.10 |
| f1_2019 | 35.8 | **+0.58** | 2.07 | 68 % | +0.73 | +0.12 | +0.16 | −0.31 / 0.87 | 5.0 / 1.35 |
| f1_2022 | 35.6 | **+0.62** | 2.01 | 24 % | +0.49 | +0.15 | +0.34 | −0.26 / 0.97 | 4.4 / 1.15 |
| f1_2026 (Z) | 34.2 | **+0.39** | 1.85 | 68 % | +0.72 | +0.18 | −0.01 | −0.38 / 0.84 | 3.4 / 0.95 |

A **Alta** (4 PF): f1_2022 SCz **+0.66** / SCx 1.95 (dx 27.5 mm; al. del. +0.69, tras. +0.10, fondo+difusor
+0.22); f1_1979 **+0.66** / 1.45 (dx 24.6; pontones-ala +0.35); f1_2011 **+0.61** / 1.65 (dx 26.4; al. del. +0.80).

Lectura honesta:
* Orden obtenido: 1967 (sustentación, bien) ≪ 2014 ≈ 2008 ≈ 2011 ≈ 2026 ≈ 1979 (0.31-0.42) < 1998 ≈ 2019 (0.58)
  < 2022 (0.62). Los extremos están bien (1967 con ligera sustentación; 2022 el que más carga a Media y a
  Alta; 2026 por debajo de 2022); el orden intermedio está dentro del ruido y 1998 sale alto.
* SCz absolutas: **5-10 veces menores** que las reales; SCx **1.3-1.8 veces mayores** (las ruedas solas ya
  dan 0.5-1.0 m²). L/D 0.2-0.45 (real 2.5-4).
* Reparto: el alerón delantero (en aire limpio) es el que mejor funciona; el trasero da 0.1-0.3 (real
  0.8-1.5); el fondo+difusor de los coches de efecto suelo es positivo pero pequeño (a Media el coche va a
  12-15 cm del suelo); ruedas con |SCz| < 0.4 (objetivo < 0.3: 1979-2022 cumplen salvo 2019/2026 ≈ 0.3-0.4).
* Balance: muy disperso (−6 % a 70 %), porque la carga total es pequeña y la trasera casi no carga.

**Efectos de configuración (Media, mismas condiciones):**

| Caso | SCz | SCx | Comentario |
|---|---|---|---|
| f1_2022 base | +0.62 | 2.01 | |
| f1_2022 DRS abierto | +0.16 | 1.91 | ΔSCx **−0.10 (−5 %)**, pierde el 75 % de la carga |
| f1_2011 DRS | +0.34 (0.38) | 1.69 (1.76) | ΔSCx −0.07 (−4 %) |
| f1_2019 DRS | +0.40 (0.58) | 1.97 (2.07) | ΔSCx −0.11 (−5 %) |
| f1_2026 modo X | −0.12 (0.39) | 1.75 (1.85) | ΔSCx −0.11 (−6 %); sin carga |
| f1_2022 guiñada +5° | +0.63 | 2.11 | SCy −0.14 m² (hacia −y, signo correcto) |
| f1_2022 suelo fijo (ruedas paradas) | +0.58 | 1.76 | |
| f1_2022 cinta, ruedas paradas | +0.39 | 1.89 | las ruedas giratorias dan más carga y +0.12 de SCx |
| f1_2022 altura pedida 10/60 → ef. 125/175 | +0.57 | 2.04 | |
| f1_2022 altura pedida 30/80 → ef. 125/175 | +0.62 | 2.01 | (base) |
| f1_2022 altura pedida 60/110 → ef. 128/178 | +0.57 | 2.04 | |
| f1_2022 altura pedida 120/170 → ef. 155/205 | +0.50 | 2.05 | la carga baja al subir el coche |

Nota de la revisión: las filas 10/60 y 30/80 simulan prácticamente la MISMA geometría (125.0 / 125.1 mm
efectivos), así que su diferencia (0.57 frente a 0.62) mide el ruido de la tanda, no un efecto de la altura.
A Media sólo el paso a 120/170 mm (155 mm efectivos) cambia de verdad la geometría; por debajo de ~g la
tendencia de altura de un F1 no es observable a esta resolución (usar `f1_wing_ge`).

El DRS reduce la resistencia en la dirección correcta pero **mucho menos** que en la realidad (−10..−25 %):
el alerón trasero trabaja en la estela y aporta poca resistencia que quitar (SCx 0.16-0.22 m²).

**Objetos de referencia (Media):**

| Modelo | dx | Resultado | Referencia | Valoración |
|---|---|---|---|---|
| sphere (D = 1 m, Re real 4.6·10⁶) | 27.6 mm | CD 0.34 | 0.47 subcrítico, 0.1-0.2 supercrítico | entre ambos regímenes (la transición no se resuelve) |
| cylinder (L/D = 4) | 33.7 mm | CD 0.97 | ≈ 0.7 | alto |
| cube (1 m) | 33.2 mm | CD 0.95 | ≈ 1.05 | bien (aristas vivas) |
| ahmed_25 | 8.6 mm | CD 0.65 | 0.285 | **2.3× alto**: burbuja tras los radios delanteros y estela |
| road_car | 38.0 mm | CD 0.62 | ≈ 0.25 | **2.5× alto** |
| naca0012_wing (AR 3) | 23.6 mm | CL −0.28 / −0.06 / +0.10 / +0.30 a −6/0/3/6°; CD 0.064 (0°), 0.082 (6°) | CL 0.39 a 6° (Helmbold), CD₀ ≈ 0.012 | pendiente 0.048 /° (objetivo 0.06-0.08, teoría 0.066); asimetría de 0.06 a 0° por la escalera; CD₀ 5× alto |
| naca4412_wing (AR 3) | 23.7 mm | CL 0.11 / 0.33 / 0.45 / 0.61 / **0.79** / 0.56 a 0/4/8/12/**16**/20° | CL 0.47 a 4°, pérdida 12-18° | **máximo a 16° y caída a 20°**; niveles ~70 % |
| f1_wing_ge (c = 0.75 m) | 11.4 mm | CL 0.56 / **1.06** / 0.89 / 0.96 / 0.88 a h = 40/**60**/100/150/300 mm | Zerihan & Zhang: sube al bajar h hasta h/c ≈ 0.1 y cae | **máximo a h/c ≈ 0.08 y caída a 40 mm**: comportamiento clásico; niveles ~40 % de los reales |
| airfoil_2d (4412 pseudo-2D) | 15.2 mm | Cl 0.41 a 4° | 0.88 | **la mitad**: las caras laterales del túnel son de campo lejano (u = u∞), no periódicas: matan la circulación cerca de los extremos |

Pruebas de resolución: alerón trasero AISLADO del f1_2022 (sólo ese grupo en el túnel), SCz / SCx:
Media 0.27 / 0.30, Alta 0.31 / 0.29, Ultra 0.36 / 0.26; NACA 0012 a 6°: Rápida 0.27 / 0.092, Media 0.30 /
0.081, Alta 0.29 / 0.073 (la resistencia converge despacio; la sustentación no mejora).

---------------------------------------------------------------------------------------------------

## 6. Guía honesta: qué predice y qué no

**Útil (tendencias y comparaciones):**
* Diferencias grandes de configuración: con/sin alerones, suelo fijo vs cinta, ruedas paradas vs
  girando, DRS/modo X abierto vs cerrado (la resistencia baja claramente), guiñada (signo y orden de la
  fuerza lateral), efecto de la altura del ala en efecto suelo, flaps más/menos cerrados.
* Visualización cualitativa: dónde se acelera o se frena el aire, zonas de succión, estelas, torbellinos
  de punta, huellas de presión en el suelo.
* Objetos romos de aristas vivas (cubo, cilindro): el desprendimiento lo fijan las aristas y el orden de
  Cd sale bien.

**Poco fiable:**
* Valores absolutos de carga de un F1 (muy bajos) y de resistencia (altos). No usar las cifras en N/kgf
  como si fueran las de un coche real.
* Balance aerodinámico de los F1 (la carga total es pequeña y el alerón trasero trabaja en una estela
  demasiado lenta).
* Pérdida de perfiles, polares finas, máximos de CL, efecto de ranuras de alerones multi-elemento.
* Alturas de marcha reales de F1 (se simulan subidas, sección 3) y todo lo que dependa de sellar el
  fondo contra el suelo.
* Cuerpos redondeados a alto Reynolds (esfera, Ahmed): la transición y el desprendimiento sobre
  superficies lisas no están resueltos; el Ahmed sale con 2-3 veces la resistencia experimental porque el
  flujo se despega tras los radios delanteros (burbuja de 4-5 celdas en los laterales, medido a Rápida).

---------------------------------------------------------------------------------------------------

## 7. Parámetros numéricos por defecto y opciones

| Parámetro | Defecto | Opción | Notas |
|---|---|---|---|
| u∞ de red | 0.09 | — | Mach de red 0.156 |
| ν de red | 1·10⁻⁴ | `--nu` | tras una divergencia `Sim` la triplica y reinicia el flujo |
| C_s (Smagorinsky) | 0.10 | `--cs` | 0.16 engorda la capa límite (sección 1.1) |
| Modelo de pared | Slip | `--wall none\|log\|slip` | sección 1.4 |
| Rebote | interpolado (Bouzidi) | `--bb interp\|implicit` | el modelo Slip requiere el interpolado |
| Colisión | regularizada recursiva (RR) + ν_b (ω_b = 1), capa de 2.º orden de 8 celdas junto a los cuerpos | — (`lbm::Config::collision/bulk_omega/rr_wall_layer`) | §1.1 y §1.5 |
| Arranque | impulsivo | `--ramp PF` | una rampa desde el reposo tarda > 5 PF en asentarse |
| Esponja | 12 % final en x, ν → 0.12 | — | |
| Hueco mínimo bajo el fondo | 3.5 celdas | — | `Sim::k_gap_cells` (sección 3) |
| Precisión | FP16S | `--fp32` | diferencia de Cd 0.3 % |

Estabilidad: `cfd --stability` recorre los 18 modelos (con `--res` un preset). Con los defectos de arriba:
18/18 estables en Rápida (1.5 PF), Media (1 PF) y Alta (1 PF); en Ultra se han comprobado f1_2022,
f1_1979, f1_2011, f1_2014 y f1_wing_ge (0.7 PF). La comprobación encontró una divergencia a Alta (f1_2014 a
0.4 PF): el "dedo" del morro dejaba una rendija de 1 celda sobre el plano principal del alerón delantero en la
que la velocidad se disparaba; se ha apoyado el dedo en el alerón. Las rendijas de 1 celda son el caso típico
de inestabilidad: si aparece una al cambiar parámetros, la recuperación automática de `Sim` sube ν y reinicia.

Coste: la pasada de contorno (rebote interpolado + ley de pared + fuerzas) cuesta ~1.05 ms por paso a Media
frente a ~6.3 ms del kernel (≈ 17 %): necesita las 19 poblaciones de cada nodo de pared.

**Cómo mejorar un resultado:** subir de preset (la tendencia con la resolución del alerón trasero
aislado del 2022: SCz 0.27 → 0.31 → 0.36 de Media a Ultra) y comparar configuraciones siempre a la
misma resolución.
