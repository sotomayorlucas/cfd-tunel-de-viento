# Fase 2 — coordinación entre A (integración/UX) y B (física/calibración)

Reglas: cada uno AÑADE entradas al final de su sección (con `>>` o ediciones pequeñas; nunca
reescribir el archivo entero). Fecha/hora aproximada al principio de cada entrada.

## A → B (peticiones y avisos de integración)

- [A 06:30] `make unity` ya compila: los renombrados ODR (`obj_deg` en models/objects.cpp y
  `LineNorm` en flowvis_lines.cpp) ya estaban hechos. El último choque era la macro `None` de
  Xlib (x11.cpp entraba antes que src/app/*): el Makefile ahora pone x11.cpp al final de la
  unidad única y añade `-Wno-subobject-linkage` (avisos inocuos de Solver::Impl). Sólo toqué la
  regla `unity` del Makefile.
- [A 06:30] Petición: cuando conviertas los ganchos TEMPORAL (CFD_CS, CFD_RAMP, CFD_NU0F,
  CFD_NOGI, CFD_BGK, CFD_SPONGE) en campos de SimConfig, anótalo aquí (nombre del campo, rango,
  defecto) y yo lo cableo en cli.cpp (`--cs`, `--nu`, `--ramp`...) y en el panel.
- [A 06:30] Petición: si añades a Sim una "altura de marcha efectiva" (resuelta por la red), un
  indicador de calidad de la resolución (p. ej. celdas bajo el fondo, ranuras cerradas) o notas de
  calibración, anota aquí el nombre del campo/función y su significado: lo muestro en HUD/panel/README.
- [A 06:30] Yo NO toco sim.cpp. Si necesito algo de Sim lo pido aquí; mientras tanto lo resuelvo
  en app/view/panel/cli.

- [A 06:50] Observación física (para B, no la toco): el túnel tiene un gradiente medio de presión
  a lo largo de X por el bloqueo. f1_1967, rápida, cinta, 1.5 PF: Cp medio del plano (fluido) = +0.11
  en x=2…42, ≈0 en la salida; en la capa z=1 (junto a la cinta) +0.16…+0.27 aguas arriba. Cuadra con
  ΔCp ≈ CD·A/A_túnel (1.34 × 10 % de bloqueo). Consecuencia visual: la huella de Cp en el suelo y la
  superficie salen "rosadas" aguas arriba. En túneles reales Cp se refiere a la presión estática
  aguas arriba; si expones un `Sim::rho_ref` (ρ medio de un plano ~0.3 L aguas arriba) o algo
  equivalente, lo uso para la sonda/leyendas (flowvis fija ρ∞ = 1 y el pivote del mapa en 0).
  También se ven bandas finas paralelas a X en la huella del suelo (ondas acústicas transversales?)
  con la cinta: visibles en build/app/shots/qa/f1_1967_34.png.
- [A 06:50] tests/test_app.cpp: el caso [6] ya NO exige CL > 0 al F1 2022 (eso es calibración: va en
  tu tests/test_aero.cpp); sólo fuerzas finitas, CD > 0, simetría y balance coherente. El ala en
  efecto suelo sí exige carga > 0. El caso [5] usa Sim::gauge_ref_f/gauge_ref_s/mref_cells/
  gauge_forces: si cambias o quitas esa API, adapta o borra el bloque [5] (tienes permiso) y avísame.

- [A 07:05] ¡Importante (usabilidad)! Con tu arranque desde el reposo (cfg.ramp_ft = 0.3) el túnel
  tarda muchísimo en establecerse: cube, rápida, sin suelo (nx = 248, 1 PF = 250 pasos), ρ medio de
  plano → Cp y ux/U (entrada | centro | salida): 1 PF: Cp +3.6|+1.8|−0.1, ux/U 0.83|0.22|0.03;
  2 PF: +2.5|+2.0|0.0, ux 0.88|0.39|0.11; 4 PF: +1.3|+1.1|0.0, ux 0.94|0.68|0.56; 8 PF: +0.4|+0.3|0.0,
  ux 0.98|0.91|0.88. Todas las superficies/huellas salen ROJAS (Cp ≫ 1) durante los primeros ~5-10 PF,
  y las capturas sin ventana de 1.5-2 PF quedan inservibles (build/app/shots/qa/cube_def.png,
  f1_wing_ge_def.png, naca0012_wing_def.png). Ojo: el "paso de flujo" usa la longitud del OBJETO, pero
  llenar el dominio necesita ~nx/u∞ pasos (≈ 11 PF de cubo). Antes (arranque impulsivo con u = u∞ en
  todo el fluido) no pasaba. ¿Puedes arrancar con u = u∞ en el fluido (impulsivo) + rampa sólo en las
  paredes móviles, o inicializar con el flujo potencial/uniforme? Mientras tanto yo no cambio nada.

- [A 07:15] Con el árbol actual (tu sim.cpp/solver.cpp a medias, supongo) `build/tests/test_app` falla:
  f1_2022 con 1.2 M celdas (304×79×52, dx 61 mm, cinta) DIVERGE en [6] y en [7] ("divergencia: nu ->
  9e-5"); en X11 a media también vi una divergencia al arrancar. El ala en efecto suelo va bien.

- [A 07:50] He cableado tus campos actuales de SimConfig en la UI y la CLI (si los renombras, dímelo o
  corrige tú la línea que rompa en app.cpp/cli.cpp/panel.cpp): `--wall none|log|slip` → cfg.wall
  (índices 0/1/2 del enum WallModel {None, LogLaw, Slip}; el panel usa ese orden), `--bb interp|implicit`
  → cfg.interp_bb, `--ramp PF` → cfg.ramp_ft, `--cs` → cfg.cs. En el panel (Túnel → "Física del
  solver") los tres se aplican con un reinicio completo (Sim::init). Si alguno se puede cambiar en
  caliente, expón una función en Sim y lo cambio para no reiniciar.
- [A 07:50] La velocidad (km/h): el tooltip y la ayuda aún dicen "sólo reescala N/kgf". Con tu ley
  de pared (wall_nu depende de speed_kmh) eso deja de ser cierto; confírmame la semántica final y la
  redacto (o escribe tú la frase para docs/FISICA.md y la copio).

- [A 08:00] Para las cifras finales de rendimiento (README) necesito ~5-8 min con la máquina en
  reposo (tus `calib` mantienen la carga en 20-30 y las medidas salen ×2-3 peor). Cuando termines
  una tanda, si puedes, anota aquí "B: pausa HH:MM" y no lances nada durante unos minutos; yo mido
  entonces. También me vendrá bien saber cuándo el F1 deja de diverger (ahora f1_2022 rápida diverge
  5 veces en 2 PF con el árbol actual) para hacer las capturas finales.

- [A 08:05] Ruido de escala de red en el campo (para B): con el árbol actual, naca0012_wing rápida
  4 PF (sin divergencias, CL 0.378 sust., CD 0.070), el criterio Q sale positivo en MOTAS por todo el
  dominio, también lejos del ala (build/app/shots/qa/naca_vort.png con umbral Q·L²/U² = 20 y
  naca_vort5.png con 5: el dominio entero se llena). Los torbellinos de punta se ven, pero rodeados
  de ruido. En airfoil_2d (corte Y de |u|) ya vi antes un damero claro (build/app/shots/qa/
  airfoil_2d_def.png). ¿Ondas acústicas / modo damero del arranque o de la esponja? El umbral de Q
  del panel ahora es adimensional con la longitud del objeto (Q·L²/U², defecto 20) para que no
  dependa de la resolución; lo ajustaré por tipo de modelo cuando el campo esté limpio.

- [A 08:20] Rendimiento (para B): `cfd --bench --res media` (f1_2022, 6.08 M celdas). A las 06:55 y
  07:19 (carga 4-8): kernel 6.3-6.4 ms/paso, fuerzas 0.33-0.45 ms/paso (920 MLUPS). Con tu solver
  actual (08:14-08:16, carga 8-30): kernel 8.2-14 ms/paso pero fuerzas 1.8-4.5 ms/paso: la pasada de
  fuerzas ha pasado de ~5 % a ~20 % del paso (¿la corrección galileana/Bouzidi por enlace, o el
  gauge?). Con tanta carga no es concluyente, pero la proporción fuerzas/kernel es la que cambió.

- [A 08:55] Gracias: cableado todo lo de 08:40. Panel Configuración: "Altura efectiva (resolución):
  94 / 144 mm (pedida 30 / 80; hueco mínimo 89 mm = 2.5 celdas)" (ámbar si ride_limited) + tooltip; HUD:
  una línea ámbar con la altura efectiva si está limitada; CLI (salida sin ventana) y CSV de fuerzas: altura
  pedida → efectiva. Velocidad: tooltip/ayuda/--help dicen ahora que fija el Reynolds real de la ley de
  pared. `--bb` acepta también 0|1. test_app ya pasa entero (1833 comprobaciones) con tu árbol de las 08:30.
- [A 08:55] Aviso: f1_2022 con `--cells 350000` (dx ≈ 88 mm, 1/3 del preset Rápida) diverge 5 veces
  seguidas (ν hasta 7e-3) en 1-2 PF; con 800 k (dx ≈ 64 mm) y con los presets va bien. En test_app [9]
  uso 800 k para el coche. Si quieres que --cells muy bajos sean estables (o un mínimo razonable por
  modelo), dímelo y lo limito en la CLI.

- [A 09:00] `make test` con tu árbol de las 08:30: test_app, flowvis, geom, geom_models PASAN; test_lbm
  FALLA 8/22 (2a-2d, 2f, 2g "EP vs A-B": error relativo de fuerzas ~1.0 → la referencia A-B no incluye
  la corrección manométrica/galileana ni el rebote interpolado; 7 "estabilidad nu=1e-5": FP16S |u|max
  0.52, FP32 Fx = NaN). Es tu módulo; `make test` para en el primer fallo, así que los tests posteriores
  (models, raster, ui) no llegan a ejecutarse hasta que test_lbm pase.

- [A 09:10] QA con tu árbol de las 08:30, rápida 3 PF (capturas en build/app/shots/*.png): DIVERGEN
  road_car (122 divergencias, ν hasta 2e-2) y f1_1998 (4); el resto estable. Cargas (SCz m², carga +):
  1967 −0.16, 1979 +0.74, 2008 +0.31, 2011 +0.49, 2014 +0.28, 2019 +0.62, 2022 +0.36, 2026 +0.32
  (modo X −0.15), 1979 suelo fijo −1.31; ala en efecto suelo CL +1.52 (SCz 1.36, ref 2.4); NACA 0012
  6°: CL 0.42 (teoría AR 3 ≈ 0.39); esfera CD 0.24; cilindro 0.79; cubo 0.75; Ahmed CD 0.70 (ref 0.285).
  En los cortes (airfoil_2d_lic.png, f1_2022_lateral_cp.png) el damero de escala de red sigue ahí.

- [A 11:45] REANUDADO (tras la parada de ~10:22). Con el árbol actual (solver.cpp 10:11): `make test` ENTERO en
  verde en build/ (test_lbm 26/26, test_aero 25/25, test_app 1833/1833, resto OK) y `cfd --stability --res
  rapida --ft 1`: 18/18 estables (f1_2022, road_car y f1_1998 ya no divergen). Hago ahora QA visual (rápida,
  build/app/shots/qa2/), X11 y medidas. Recordatorio: el README enlaza `docs/FISICA.md` (tuyo): ¿lo escribes?
  Si no existe al final, quito el enlace. Sigo necesitando una "B: pausa HH:MM" de 5-8 min para medir
  rendimiento con la máquina tranquila (ahora test_aero de build-phys ocupa 10 núcleos).

- [A 11:58] QA con el árbol actual (rápida, 2.5 PF; build/app/shots/qa2/): (1) El arranque impulsivo deja un
  PULSO ACÚSTICO enorme: en X11 a media a 0.07 PF la superficie y la huella del suelo son bandas rojas/azules
  de Cp ±1 (build/app/shots/qa2/x11_media.png). Se va en ~1 PF, pero es lo primero que ve el usuario. ¿Una
  rampa CORTA (0.1-0.2 PF, sólo para suavizar el pulso; no la de 0.3 PF desde el reposo) o amortiguar las
  fronteras? (2) La regla de altura efectiva sube también coches que YA tienen hueco suficiente: f1_1967 pedido
  100/110 mm con g = 97 mm → 139/149 mm (√(h²+g²) siempre suma). ¿Intencionado? Si la red resuelve 100 mm
  (2.6 celdas) yo esperaría h_eff ≈ h. (3) Balance de coches a rápida/2.5 PF fuera de la batalla: 1979 −170 %,
  2008 −35 %, 1998 −5 % (carga trasera > total: el eje delantero recibe sustentación). Lo marco en rojo/ámbar
  en panel/HUD ("fuera de la batalla"), el cálculo no lo toco. (4) Sigue la huella del suelo "rosada" aguas
  arriba (Cp +0.2-0.3 en todo el suelo) y las rayas finas paralelas a X; si expones `Sim::rho_ref` lo uso.

- [A 12:25] Más QA (rápida, 2.5-4 PF): (5) presión de referencia: el corte X de Cp0 detrás del f1_2022 da en
  corriente libre Cp0 hasta 1.38 (debería ser 1.0) → hay un desplazamiento global de ρ de +0.2…+0.35 en Cp,
  coherente con la huella rosada; afecta a Cp/Cp0 de TODO lo que se visualiza (no a las fuerzas, que ya son
  manométricas). (6) Damero de escala de red muy visible en cortes de Cp de f1_wing_ge (qa2/wing_ge_slice.png)
  y rayas en |u| de airfoil_2d (qa2/airfoil.png); en las alas NACA la Cp de la superficie sale a rayas en
  la envergadura (qa2/naca0012.png, naca4412.png). (7) Cifras (SCz/SCx m²): 1967 −0.26/1.29, 1979 +0.28/2.09
  (suelo fijo +0.01/1.55), 1998 +0.71/2.16, 2008 +0.34/1.94, 2011 +0.46/1.97 (DRS +0.33/1.93), 2014
  +0.34/2.02, 2019 +0.53/2.39, 2022 +0.32/2.35, 2026 +0.53/2.37 (modo X −0.04/2.28); ala GE +0.84/0.34; NACA
  0012 8° CL 0.26 (teoría AR 3 ≈ 0.5); NACA 4412 4° CL 0.24; airfoil_2d CL 0.46 CD 0.094; esfera CD 0.37;
  cilindro 0.91; cubo 0.97; Ahmed CD 0.92 (ref 0.285, antes 0.70); turismo CD 0.83 (ref 0.25). La resistencia
  de TODOS los coches ~2× la real y Ahmed empeoró respecto a las 09:10. (8) Arreglado en mi lado: el volumen de
  vórtices saturaba la cuantización (umbral > `VolumeParams::full`) → isosuperficies en bloques; ahora `full`
  sigue al umbral (view.cpp) y el umbral por defecto del coche es Q·L²/U² = 600.

- [A 12:25] Rendimiento con la máquina en reposo (carga 1.8 al empezar, 12:15, `--bench --res media`):
  790 MLUPS, núcleo 6.53 ms/paso, **fuerzas 1.17 ms/paso = 18 % del núcleo** (a las 07:19: 0.33-0.45 ms,
  5-7 %). La pasada de fuerzas sigue siendo 3× más cara que antes de Bouzidi/galileana: ¿se puede hacer sólo
  cada N pasos (la media exponencial no lo necesita en cada paso) o vectorizar la lista de enlaces?
  set_geometry en la reconstrucción: 60 ms (media). Tengo un script que mide solo cuando la carga baja de 3
  (y no hay make): si puedes, deja la máquina libre ~6 min y anota "B: pausa HH:MM".

- [A 13:40] Gracias por `Sim::rho_ref` y la nueva regla (h⁴+g⁴)^¼: cableado. Cp y Cp0 de TODO lo visual
  (superficie, huella, líneas por Cp, corte Cp/Cp0 incl. rango automático) y de la sonda se refieren ahora a
  rho_ref. Para eso añadí a flowvis un campo `ColorScale::offset` (se resta al valor antes de mapear; defecto 0,
  test_flowvis sigue en verde): cambio pequeño en render/flowvis.hpp (map_color, map_color8, normalize) y
  flowvis.cpp (scale_t255 y el rango automático del corte). La huella del suelo ya no sale rosada
  (qa2/f1_2022_rhoref.png). Panel Visualización: "Cp referido a la toma estática (Δ ±x frente a ρ = 1)".
  Pulso acústico: sigo tu propuesta (HUD "ondas de presión del arranque (~1 PF)"), no toco el solver.

- [A 14:05] Estado de A al cerrar esta tanda: `make -j16` y `make unity` sin avisos; `make test` entero en
  verde (13:05, con tu árbol de entonces); test_app 1837/1837 (14:00). Nuevo en la app: leyenda de colores por
  componente, etiquetas DRS/modo Z-X explícitas, balance fuera de 0-100 % en rojo/ámbar, umbral Q del coche 600
  con cuantización corregida, Cp/Cp0 referidos a rho_ref (ColorScale::offset en flowvis), textos de la nueva
  regla de altura efectiva. README: tabla de calibración marcada como instantánea de las 11:50 (cámbiala o
  pídeme que la actualice cuando calibres), y el enlace a `docs/FISICA.md` dice que lo redactas tú (aún no
  existe). Rendimiento: sólo dos muestras en reposo (12:15 y 13:20; 790-824 MLUPS a media); PGO/unity con el
  solver nuevo sin conclusión por la carga (README lo dice).

## B → A (campos nuevos de Sim, cambios de comportamiento)


- [B 08:40] **API de Sim cambiada (física)** — ya compila en `build-phys` y tests/test_app.cpp [5] adaptado
  (permiso de A, bloque [5] reescrito: ahora comprueba fuerza nula en reposo con `Sim::id_forces`):
  * ELIMINADOS: `Sim::gauge_forces`, `gauge_ref_f/gauge_ref_s`, `gi_f/gi_s`, `gi_enabled`, `nu_start_factor`,
    `nu_sched_ft`, `nu_at()` y TODOS los ganchos de entorno TEMPORAL (CFD_CS, CFD_RAMP, CFD_NU0F, CFD_NOGI,
    CFD_BGK, CFD_SPONGE). La corrección manométrica y la galileana viven ahora en el SOLVER
    (`lbm::Config::force_gauge`, `force_galilean`, por defecto true): `solver.forces()`/`forces_mean()` ya son
    fuerzas p − p∞ (el id 255 = suelo también: ahora es la fuerza aerodinámica real sobre la cinta).
  * NUEVO `void Sim::id_forces(const lbm::ForceSample&, Vec3* f, Vec3* m) const` (copia por id; úsalo si
    necesitas fuerzas por id de una muestra del solver).
  * NUEVOS en `SimConfig` (para cablear en CLI/panel si quieres):
      - `lbm::WallModel wall = lbm::WallModel::Slip` — modelo de pared: None (rebote no deslizante),
        LogLaw (ley de pared por viscosidad), Slip (defecto: pared deslizante parcial + tensión de pared).
        CLI sugerida: `--wall none|log|slip`.
      - `bool interp_bb = true` — rebote interpolado (Bouzidi) con la superficie real. CLI: `--bb 0|1`.
      - `float ramp_ft = 0` — arranque: 0 = impulsivo (defecto); > 0 = rampa desde el reposo en ese nº de
        pasos de flujo (NO recomendado: el túnel tarda > 5 PF en asentarse). CLI: `--ramp F`.
      - `cs` (Smagorinsky, ya existía, defecto 0.16) y `nu` (ya existía). CLI: `--cs 0.16`, `--nu`.
  * NUEVOS en `Sim` (para mostrar):
      - `float wall_nu` y `float wall_nu_lat() const`: ν (red) de la ley de pared = aire real a `speed_kmh`
        → la velocidad YA NO es sólo un reescalado: cambia (poco) la fricción de pared. Se sincroniza sola
        en `step()/update_results()` cuando cambia `speed_kmh` (no hace falta reiniciar).
      - Altura de marcha efectiva (ver entrada siguiente).
- [B 08:40] **Altura de marcha efectiva (resolución)** — para mostrar en panel/HUD ("altura efectiva
  (resolución)"):
  * `Sim::params` = lo pedido (saneado; lo que muestran los deslizadores). `Sim::params_eff` = lo que se
    construye y ve el solver (`built.params` == `params_eff`).
  * `float Sim::ride_eff_front_mm, ride_eff_rear_mm` — alturas efectivas (mm).
  * `float Sim::ride_gap_min_mm` — hueco mínimo resoluble g = 2.5·dx (mm) (`Sim::k_gap_cells` = 2.5).
  * `bool Sim::ride_limited` — true si la red obliga a subir el coche. Regla: la menor altura h pasa a
    √(h² + g²) y la otra sube lo mismo (rake conservado). Ej. Media (dx 35.6 mm): 30/80 → 94/144 mm.
    Texto sugerido: "Altura efectiva (resolución): 94 / 144 mm (pedida 30 / 80; hueco mínimo 89 mm = 2.5 celdas)".
  * Los barridos (`--sweep ride`) siguen usando la altura PEDIDA en el eje x.
- [B 08:40] Otros cambios de comportamiento: ruedas con "huella de contacto" (las 1-2 capas bajo el
  neumático se rellenan y se mueven con la cinta; `solid_cells` las incluye); voxelización sin voto de
  mayoría (coherente con el rebote interpolado). El arranque ya NO parte del reposo: `solver.config().ramp_steps`
  = 0 por defecto y el primer ½ paso de flujo no entra en la media de fuerzas.
- [B 08:40] Sobre tu observación del gradiente de presión en el túnel: era sobre todo el ARRANQUE DESDE EL
  REPOSO (rampa), no el bloqueo: en el túnel VACÍO con rampa quedaba Cp +0.5 y u = 0.76 u∞ a la salida tras
  2.3 PF (la sobrepresión del arranque sale muy despacio por las fronteras de equilibrio). Con arranque
  impulsivo el Cp medio del plano vuelve a ≈ 0 (queda el +0.1-0.2 local por bloqueo delante del coche).
  `Sim::rho_ref` lo miro después (anotaré aquí si lo añado).
- [B 13:25] REANUDADO (B). Estado y respuestas a tus puntos (árbol de build-phys de las 13:20):
  * **Estabilidad**: `--stability --res rapida --ft 2` 18/18 estables (11:41, antes de los cambios de abajo; lo
    repito al final con todos los presets).
  * **Viscosidad por defecto 1e-5 → 1e-4** (`Sim::nu`, k_default_nu): mismas fuerzas (NACA 0012 6°: CL 0.274 vs
    0.277, CD igual; Ahmed igual), 10× más margen de estabilidad y menos ruido de escala de red. El `--nu` de la
    CLI sigue igual. Con 3e-4 la resistencia sube ~3 % (no lo uso).
  * **Ley de pared (modelo Slip) v2**: ahora IMPONE EXACTAMENTE la tensión de pared de Werner-Wengle en cada nodo
    (antes la subestimaba: fricción ×4-10 la turbulenta en una placa plana). Placa plana: cf 0.5-1.1× la
    turbulenta. En superficies inclinadas respecto a la red (escalera de vóxeles) sigue habiendo resistencia de
    más (placa 1:10: ×4). Sin cambios de API.
  * **Tu punto (2) altura efectiva**: tienes razón, cambiada la regla: h_eff = (h⁴ + g⁴)^¼ (= g con h → 0, ≈ h en
    cuanto h ≳ 1.5 g; 1.19 g con h = g). f1_1967 100/110 con g = 97 → 117/127 (antes 139/149). Mismos campos
    (`ride_eff_*`, `ride_limited` sigue siendo Δ > 0.5 mm). F1 2022 a Media: 30/80 → 89/139 (antes 94/144).
  * **Tu punto (4) `Sim::rho_ref`**: AÑADIDO (`float Sim::rho_ref`, se actualiza en cada `Sim::step`): ρ medio
    del plano x a mitad de camino entre la entrada y el objeto (la "toma estática" de un túnel real). Úsalo como
    Cp = 2(ρ − rho_ref)/(3u²) en sondas/leyendas. Las fuerzas no dependen de él. El Cp +0.1..+0.2 "aguas arriba"
    que ves es real en este túnel (bloqueo 10 % + la estela: balance de cantidad de movimiento).
  * **Tu punto (1) pulso acústico del arranque impulsivo**: medido (f1_2022 rápida, RMS de Cp en todo el fluido):
    0.30 a 0.12 PF, 0.18 a 0.25 PF, 0.14 a 0.38 PF, estable 0.13 desde 0.5 PF. Una rampa corta lo EMPEORA (0.1 PF:
    RMS 1.95 → 0.23 a 2 PF: el túnel tiene que "llenarse") y subir ν al arranque no lo amortigua (probado 3e-3 y
    1e-2 durante 0.25 PF: idéntico). Propuesta: no tocar el solver; en la UI, durante t < 0.3-0.5 PF, mostrar
    "arranque (transitorio acústico)" y/o atenuar la huella/superficie Cp (las fuerzas ya excluyen el 1.er ½ PF).
  * **Coste de la pasada de fuerzas**: `--bench --res media` 13:17 (carga 13-16): kernel 6.3-6.5 ms, fuerzas
    1.05 ms (17 %). Es el rebote interpolado + la ley de pared: necesitan las 19 poblaciones de cada nodo de pared
    (antes 2 por enlace). La acumulación ya es por nodo (probado: sin ganancia medible). Lo dejo así.
  * **Tu punto (3) balance fuera de la batalla** y las cargas: es la calibración (siguiente paso; toco models/).
- [B 15:00] **Cambios de física (build-phys 14:40)** — sin cambios de API salvo el defecto de `cs`:
  * **`SimConfig::cs` 0.16 → 0.10** (valor clásico de flujos de cizalla). Con 0.16 la capa límite engordaba y se
    despegaba antes (NACA 0012 AR 3 a 6° con 42 celdas de cuerda: CL 0.15 → 0.30). La ayuda de `--cs` en cli.cpp
    dice "defecto 0.16": cámbiala a 0.10 cuando puedas (tuyo). `--stability --res media --ft 1.5`: 18/18 estables.
  * Modelo Slip: además de imponer τ_w, AMORTIGUA la viscosidad de Smagorinsky en la 1.ª celda junto a paredes
    fijas (τ = max(τ0, ½ + ¼(τ_LES − ½)), tipo van Driest). Ya no hay "fricción ×4-10".
  * **Ruedas (paredes móviles ≠ cinta) con rebote interpolado** (Bouzidi + término de pared móvil, escrito en la
    pasada de contorno; el kernel ya no les suma Ladd). Esfera giratoria Re 100 α 0.5: CL 0.62 → 0.51, CD 1.46 →
    1.31 (sin giro 1.20). La cinta sigue con rebote implícito + Ladd.
  * `models/f1_eras.cpp`: quitados mis ganchos de depuración CFD_NOAIRBOX/NOCOCKPIT/NOMIRRORS/NOHALO; 2022/2026
    tienen ahora un lomo de cubierta motor (toma de aire → trasera) en vez del escalón tras la toma.
  * Aviso honesto sobre cargas (lo detallo en docs/FISICA.md): a Media los alerones tienen 7-10 celdas de cuerda y
    el solver les da ~¼-⅓ de la carga real (alerón trasero AISLADO del 2022: SCz 0.27, L/D 0.9 a Media; 0.36 a
    Ultra). La calibración mejorará el orden entre épocas, pero las SCz absolutas de los F1 van a quedar por
    debajo de las reales (~0.5-1.5 m² en vez de 3-5). Los barridos/tendencias sí son útiles.
- [B 15:40] **Hueco mínimo bajo el fondo: `Sim::k_gap_cells` 2.5 → 3.5** (con 2.5 el fondo seguía en
  sustentación: F1 2019 a Media SCz del fondo −0.40 → −0.14; F1 2022 −0.16 → −0.01). F1 2022 a Media: pedida
  30/80 → efectiva **125/175 mm** (g = 125 mm). Tu texto del panel usa `Sim::k_gap_cells`, así que ya dirá
  "3.5 celdas". test_aero lo usa simbólicamente.
  * Calibración de modelos: alerones traseros 2008-2026 con más incidencia (trabajan en la estela con el flujo
    subiendo 15-20°); lomo de cubierta motor en 2022/2026. docs/FISICA.md ya existe (borrador; relleno la tabla
    de calibración al terminar la tanda final, ~16:40). Frase para la velocidad (km/h), por si la quieres:
    "La velocidad fija las fuerzas en N/kgf y el Reynolds real de la ley de pared (fricción); el flujo
    resuelto es el mismo."
  * Tanda final de calibración en marcha (15:35 → ~16:45, satura la CPU). Pausa para tus medidas: te la
    anoto aquí en cuanto termine.
- [B 17:20] **Cierre de B (física/calibración)** — árbol de build-phys de las 17:05; `make BUILD=build-phys test`
  ENTERO en verde (test_aero 33/33, test_app 1837/1837, test_lbm 26/26, models 384, models_vox 131, geom,
  geom_models, flowvis, raster, ui). Cambios desde la entrada de las 15:40:
  * **docs/FISICA.md escrito** (método, fuerzas y correcciones con sus pruebas, altura efectiva, Reynolds y
    resolución, TABLA DE CALIBRACIÓN a Media y Alta, guía honesta). El enlace del README ya vale.
  * Calibración en models/f1_eras.cpp: flaps delanteros 1979-2014 más cerrados, traseros 2008-2026 con más
    incidencia, 1979/1998 traseros más abiertos (detalle en docs/MODELOS.md). `Info::default_*_flap_deg` (lo
    que muestra tu panel como "Flaps de la época") ya son los calibrados.
  * Resultado a Media (SCz/SCx m²): 1967 −0.22/0.95, 1979 +0.42/1.55, 1998 +0.58/1.64, 2008 +0.33/1.62,
    2011 +0.38/1.76, 2014 +0.31/1.70, 2019 +0.58/2.07, 2022 +0.62/2.01, 2026 +0.39/1.85 (modo X −0.12/1.75).
    DRS: −4..−5 % de SCx. Son valores MUY por debajo de los reales (docs/FISICA.md §5-6): si muestras
    "ref. real aprox." junto al resultado, conviene una nota tipo "orden de magnitud; ver docs/FISICA.md".
  * Solver: las escrituras de las paredes móviles interpoladas usan la velocidad de pared del paso siguiente
    (rampas) y `set_geometry`/`reset_flow` actualizan la tabla de movimiento antes de escribir (1.er paso).
  * test_aero: nuevas [6] ley de pared (placa plana), [7] balance (casos analíticos), [8] DRS.
  * Comprobación de estabilidad final en marcha (rápida/media/alta/ultra parcial, termina ~17:50). **Pausa de
    B desde que termine** (lo anoto aquí): puedes medir rendimiento con la máquina tranquila.
- [B 17:55] Estabilidad final (build-phys, defectos): Rápida 18/18 (1.5 PF), Media 18/18 (1 PF), Alta 18/18 (1 PF;
  f1_2014 divergía a 0.4 PF por una rendija de 1 celda entre el "dedo" del morro y el alerón: corregido en
  models, docs/MODELOS.md), Ultra comprobados f1_2022/1979/2011/2014/f1_wing_ge (0.7 PF). Tests de build-phys
  en verde. **B: pausa desde las 17:55** — no lanzo nada más; la máquina queda libre para tus medidas.

## Revisión independiente (18:30, tras el cierre de A y B)

- Verificado: `make -j16` y `make unity` sin avisos; `make test` entero en verde; ganchos TEMPORAL eliminados (sólo
  quedan CFD_LBM_SKEW y los de x11: CFD_SCALE/CFD_NO_SHM/CFD_NO_NT). Corrección manométrica (término 2w_k) y
  galileana (−6w_k(c_k·u_w)u_w, Wen 2014) revisadas en solver.cpp: coinciden con la fórmula de solver.hpp.
- Re-medido con la CLI (Media, 4 PF coches / 5 PF ala): f1_2022 +0.55/2.02 (B: +0.62/2.01), DRS +0.19/1.88 (−7 % SCx;
  B −5 %), f1_1967 −0.21/0.94 (B −0.22/0.95), f1_wing_ge CL 1.02 a 60 mm y 0.87 a 300 mm (B 1.06/0.88). Dentro del ruido.
- Arreglos: ayuda de `--cs` (defecto 0.10); README (hueco 3.5 celdas, 125/175 mm, enlace a FISICA.md, tabla de
  calibración final a Media); tooltip "orden de magnitud" junto a "Real aprox."; aviso en el panel Barrido cuando
  todo un barrido de altura queda bajo el hueco mínimo; umbral Q por defecto de cuerpos 60 → 800 (con 60 el ruido
  llenaba todo el dominio del Ahmed); ui::plot_lines ya no pisa el título con la leyenda en paneles estrechos;
  nota en FISICA.md §5.1: las filas 10/60 y 30/80 son la misma geometría efectiva (su diferencia es ruido).
- Pendiente (física): ruido de escala de red / acústico en el campo lejano con ν = 1e-4 (RMS de u_x/U 6-9 % aguas
  arriba del coche a 2.5 PF en rápida; 2 % con ν = 1e-3) que ensucia cortes y criterio Q; la prueba [2] de
  test_aero sólo compara la parte antisimétrica: en el sistema móvil los dos parches reciben además −0.012 (11 %
  de la tensión) en modo común.
