# Resúmenes de API por módulo (fase 1, tras revisión)

Generado a partir de los informes de los implementadores y revisores. La fuente de verdad son las cabeceras;
esto resume semántica, invariantes y trampas para el integrador.

## lbm

### API (revisor)
cfd::lbm::Solver (pimpl, /home/lucas/cfd/src/lbm/solver.hpp). The API is unchanged by the review; I only added comments.

Core API:
- init(Config): allocates storage (SoA per direction with padding, THP), builds flags and calls reset_flow(). nx must be a multiple of 8 and at least 16; ny and nz at least 4.
- config().
- set_geometry(solid_id[N]): keeps the flow. Cells that become fluid are set to equilibrium (rho=1, u=0). Rebuilds flags, kNearMoving, block classes, active rows and the compact boundary list. It is now race-free.
- set_wall_motion(id, WallMotion{v, omega, center}) in lattice units.
- clear_wall_motions(), set_ground(GroundMode), set_viscosity(nu), set_smagorinsky(cs), set_moment_reference(cells).
- set_inflow(u): smoothstep ramp from the current value.
- reset_flow(): rho=1. With ramp_steps>0 it starts from rest and u_inf (inlet, far field, belt and wheels) ramps up; otherwise it is an impulsive start. Resets steps() to 0.
- step(n, update_macro=true): forces per id every step; the last step writes rho and u fused into the kernel.
- field(): FieldView. u_inf is the configured target. Solid cells report rho=1 and u = wall velocity or 0.
- forces(): last step only. Force of the fluid ON each solid id 0..255 (drag positive along +x), moments about the reference, total over ids 1..254.
- steps(), last_mlups(), memory_bytes(), current_u_inf().
- diverged(): sticky NaN or rho outside (0.2, 5) until reset_flow.

Compatible extensions:
- forces_mean(): mean over ALL n steps of the last step(n); exclude transients when measuring a steady state.
- total_mass(): diagnostic, O(N), single-threaded.
- Tuning {row_grain, nt_macro, prefetch, pair_blocks, max_threads, ftz}, set_tuning(), tuning(): the defaults are the best measured, and every variant is verified bitwise identical to the default.
- last_force_seconds(), last_kernel_seconds().

Semantics other modules must know:
- The 6 domain faces are always boundaries and override solid ids: kInlet at x=0, y=0, y=ny-1, z=nz-1, and z=0 without ground; kOutlet at x=nx-1.
- With a ground, z=0 is solid id 255, plus kMoving when the belt moves.
- A solid with id 255 outside z=0 (out of contract) is treated as ground: it moves with the belt through the exact scalar path and its force adds to id 255.
- flags may carry the internal bit 1<<5: always test with masks.
- Wall velocities scale with u(t)/u_inf during ramps.
- Everything is in lattice units.
- Call from a single thread; the solver is not reentrant.

Recommended on the Core Ultra 7 155H: the default pool (20 threads, no LP-E cores) without pinning; FP16S + Regularized + LES (the Config defaults).

Measured on 256x128x96: about 1000 MLUPS FP16S and 505 MLUPS FP32, around 94% of the 84 GB/s memory ceiling.

### API (implementador)
cfd::lbm::Solver (pimpl, src/lbm/solver.hpp) implementa exactamente el contrato original: init(Config) reserva memoria (SoA con relleno, THP), construye flags y hace reset_flow(); set_geometry(solid_id[N]) conserva el flujo (celdas sólido→fluido se inicializan en equilibrio ρ=1,u=0) y reconstruye flags/kNearMoving/clases de bloque/lista de fronteras; set_wall_motion/clear_wall_motions/set_ground/set_inflow (rampa smoothstep desde el valor actual)/set_viscosity/set_smagorinsky/set_moment_reference/reset_flow; step(n, update_macro) avanza n pasos (fuerzas por id CADA paso; el último paso escribe ρ,u fusionado en el kernel); field() devuelve FieldView (u_inf = u∞ configurada; sólidos: ρ=1, u = velocidad de pared o 0); forces() = último paso (fluido SOBRE el cuerpo, arrastre +x); steps(), last_mlups(), memory_bytes(), diverged() (pegajoso hasta reset_flow), current_u_inf().
Extensiones compatibles: forces_mean() (media sobre los n pasos del último step), total_mass() (diagnóstico O(N), incluye poblaciones en vuelo en sólidos), struct Tuning {row_grain, nt_macro, prefetch, pair_blocks, max_threads, ftz} + set_tuning()/tuning() (los valores por defecto ya son los mejores medidos), last_force_seconds(), last_kernel_seconds().
Semántica que el resto debe conocer: (1) las 6 caras son SIEMPRE fronteras (kInlet en x=0,y=0,y=ny-1,z=nz-1 y z=0 sin suelo; kOutlet en x=nx-1): los ids de sólido en caras se ignoran; con suelo, z=0 es sólido id 255 (+kMoving si Moving). (2) flags puede llevar el bit interno 1<<5 (vecino móvil = sólo la cinta): usar máscaras, no igualdades. (3) reset_flow con ramp_steps>0 arranca del REPOSO y u∞ (entrada, campo lejano, cinta y ruedas) sube con smoothstep; ramp_steps=0 → arranque impulsivo con u∞; reinicia steps() a 0. (4) velocidades de pared se escalan con u(t)/u∞ durante las rampas. (5) nx múltiplo de 8 y ≥16 (CFD_CHECK). (6) Unidades de red; forces en unidades de red por paso; momento M=Σ s×F respecto a la referencia.
Hilos: usa cfd::pool() (parallel_for dinámico, grano ~4096 celdas; run_slots con trozos fijos para fuerzas → resultados bit a bit deterministas con cualquier nº de hilos). Llamar desde un solo hilo (el principal); no es reentrante. Recomendación para el 155H: pool por defecto (20 hilos, sin LP-E) SIN fijar afinidad; FP16S + Regularizado + LES (defectos de Config).

### Cambios de contrato
src/lbm/solver.hpp (propio) ampliado de forma COMPATIBLE: forces_mean(), total_mass(), struct Tuning + set_tuning()/tuning(), last_force_seconds(), last_kernel_seconds(); comentarios de cabecera actualizados con la semántica exacta (reset_flow parte del reposo con rampa si ramp_steps>0 —el comentario original decía 'u=u∞ en todo el fluido'—, las caras son siempre fronteras, bit interno 1<<5, velocidades de pared escaladas por la rampa, rebote full-way).
Cambio sugerido (NO hecho) en lbm/field.hpp (compartido): reservar el bit 1<<5 (p.ej. 'kInternal5 = 1u<<5 // uso interno del solver: vecino móvil = sólo la cinta') para que ningún otro módulo lo reutilice; los consumidores deben usar máscaras (flags & kSolid), nunca igualdades sobre flags.
Nota para app/voxelizador: los ids de sólido que caigan en las 6 caras se ignoran (las caras son entrada/salida/campo lejano); la capa z=0 la gestiona el solver (id 255). Para el pool: dejar el defecto (20 hilos, sin set_affinity). ARCHITECTURE.md pide listar trucos en docs/OPTIMIZACIONES.md: los del LBM están en docs/opt/lbm.md (según la tarea); el lead puede enlazarlos.

### Problemas conocidos
- El rebote implícito de Esoteric-Pull es full-way (la población vuelve 2 pasos después): estacionario idéntico al half-way, pero los transitorios en paredes difieren en orden temporal; verificado contra una referencia full-way; una referencia half-way difiere 3.6e-2 en transitorios fuertes
- Esfera Re=100: Cd +11.2 % sobre Schiller-Naumann con el dominio del test (pasa la tolerancia del 15 %). Por el estudio D=16/24/32 el exceso restante viene del confinamiento por las fronteras de campo lejano en equilibrio (Dirichlet en u y ρ) y de la distancia a la entrada, no de la resolución ni del solver
- La pasada de fuerzas va separada del kernel (2–4 % del paso y una barrera extra); fusionarla en el kernel se estimó ≤1–3 % de ganancia y no se implementó
- Tuning::max_threads no controla qué núcleos participan (limitación de la API del pool): medido peor que un pool pequeño, no recomendado
- Todas las mediciones de rendimiento tienen ruido alto por la carga ajena (carga media 2–27); dos configuraciones idénticas llegaron a diferir 10–18 % entre rondas. Las primeras campañas multihilo tenían un artefacto propio (primera medición tras recrear el pool penalizada 20–60 %, siempre FP16-Reg) que se detectó y corrigió; las tablas finales son del bench corregido
- El techo 'colisión nula' (≥885 MLUPS FP16S, ≥480 FP32) se midió con el bench antes de la corrección: es una cota inferior
- BGK con ν=2e-4 (τ≈0.5006) diverge en el caso coche aunque haya LES (lo detecta diverged()); para Re altos usar Regularizado (defecto)
- FP16S: |f-w| debe mantenerse < 2 (rango de half con escala 2^15); flujos patológicos (Mach alto) desbordarían a Inf, lo que diverged() detecta
- Las caras del dominio anulan ids de sólido; con suelo la capa z=0 entera es sólido 255, así que la fuerza del id 255 incluye todo el suelo
- total_mass() es O(N) y de un solo hilo (sólo diagnóstico/tests)
- Los números de línea de docs/opt/lbm.md corresponden al estado final de solver.cpp; si otro agente lo edita, pueden desplazarse

### Riesgos restantes (revisor)
- The internal flag bit 1<<5 (kGroundOnly) appears in FieldView::flags. lbm/field.hpp (shared) should reserve it, and consumers must keep using masks, never equality.
- Implicit Esoteric-Pull bounce-back is full-way: transients near walls lag one step behind half-way. The steady state is identical. This is what the spec asks for (solid cells are not processed).
- Sphere Cd at D=16 is about 11% above Schiller-Naumann, mostly from resolution. Absolute Cd values at coarse resolution are biased high by a few to about 10%; relative comparisons are reliable.
- The id-255 force includes the pressure on the whole z=0 plane. The app must not use it as an aerodynamic load.
- BGK at nu=2e-4 (tau about 0.5006) diverges in the car case. diverged() detects it, but the app should force Regularized at high Re, which is the default.
- Performance numbers are noisy because of heavy background load on the machine (load 3-9 during my runs). Pinning and 22-thread conclusions were not re-measured by me.
- Tuning::max_threads does not choose which cores participate (a limitation of the pool API). It measured worse than a smaller pool and is not recommended.
- Only the ASan-affordable subset was run under sanitizers. Tests 5, 6 and 7 (big domains) ran only in the optimised build.
- Line references in docs/opt/lbm.md match the current solver.cpp. They will shift if the file is edited again.

## geom

### API (revisor)
Header: src/geom/voxelizer.hpp, namespace cfd::geom. Every change is a backward-compatible addition; the original signatures are unchanged.

**1) voxelize**
`VoxelStats voxelize(const sdf::Scene&, const LatticeMap&, int nx, int ny, int nz, u8* solid_id, const VoxelOptions& = {})`
- Writes every cell with z ≥ z_min: 0 = fluid, 1 + group index = solid. Cells with z < z_min are never touched.
- For thicken ≥ 0 the result is identical cell by cell to the literal definition: Scene::eval(centre) < thicken·dx, with a 2×2×2 majority vote (≥ 4/8) when supersample is on and |d| < dx.
- VoxelOptions: thicken 0.12, supersample true, z_min 1, block_skip true, lipschitz 1.5. block_skip=false is now a truly literal reference path.
- VoxelStats: solid_cells, lo/hi, seconds, and the NEW literal_cells (cells evaluated by the literal path inside subtracted-prim boxes while the sdf.cpp bug is present).
- NEW `bool sdf_subtract_culling_bug()`: one-time probe of the linked sdf.cpp.
- Uses cfd::pool() (call pool().start() first). Can be called nested from a pool job, where it runs serially.

**2) mesh_scene**
`void mesh_scene(scene, map, cell_fraction, render::Mesh& out)` and `mesh_scene(scene, map, cell_fraction, out, const MeshOptions&, MeshStats* = nullptr)`
- Surface Nets with sample spacing h = cell_fraction·dx.
- pos: lattice cell coordinates. tri: CCW seen from outside, deterministic. group: 1-based per vertex. color: component_color when fill_color is set. bounds is filled.
- nrm: outward unit SDF gradient, or the face-average normal where the gradient disagrees by more than acos(normal_min_dot). The latter is the NEW robust-normals pass.
- MeshOptions: projection_steps 1, block_skip true, halo false, lipschitz 1.25, fill_color true, max_samples 96M, and the NEW normal_min_dot = 0.5 (≤ -1 disables the pass).
- MeshStats: grid dims, h, sampled, corner_fill, sign_fixes, vertices, quads, mixed_blocks, per-phase timings, and the NEW normal_fixes and t_normals.
- Keeps a thread_local workspace per calling thread.

**3) mesh_voxels**
`void mesh_voxels(const u8* solid_id, nx, ny, nz, render::Mesh& out, bool skip_ground = true)`
- Greedy rectangles between solid (1..254, plus 255 when skip_ground is false) and non-solid cells. Outside the grid counts as air.
- Faces sit at cell centre ± 0.5, with 4 vertices and 2 triangles each.
- Flat outward normals, per-vertex group, grey color.
- Geometrically closed but has T-junctions.

**4) RegionEval and aabb_gap**
- RegionEval: an exact per-region evaluator equal to Scene::eval (init_all / refine / eval / classify_id / classify_sign). Unchanged by this review.
- aabb_gap: unchanged.

**Files changed or added in this review**
- /home/lucas/cfd/src/geom/voxelizer.cpp
- /home/lucas/cfd/src/geom/mesher.cpp
- /home/lucas/cfd/src/geom/voxelizer.hpp
- /home/lucas/cfd/tests/test_geom.cpp
- NEW /home/lucas/cfd/tests/test_geom_models.cpp
- /home/lucas/cfd/docs/opt/geom.md (new §7 review section, updated numbers and file:line references)

**Build commands**
- Main test: `g++ <Makefile flags> -Isrc tests/test_geom.cpp src/geom/{voxelizer,mesher,sdf}.cpp src/core/{threadpool,png}.cpp`
- Catalogue test: `g++ <flags> -Isrc tests/test_geom_models.cpp src/geom/{voxelizer,mesher,sdf}.cpp src/core/threadpool.cpp src/models/*.cpp`

**Contract changes needed outside this module (for the lead):** apply the sdf.cpp Subtract/SmoothSubtract culling fix given in remaining_risks.

### API (implementador)
Header: src/geom/voxelizer.hpp (namespace cfd::geom). Original contract unchanged; only compatible additions.

1) VoxelStats voxelize(const sdf::Scene&, const LatticeMap&, int nx, int ny, int nz, u8* solid_id, const VoxelOptions& = {})
   - Writes every cell with z >= opt.z_min (0 = fluid, 1 + group index = solid). Cells with z < z_min are never touched (the ground layer belongs to the solver).
   - A cell is solid if Scene::eval(to_model(i,j,k)) < thicken*dx. With supersample, cells with |d| < dx use a 2x2x2 majority vote (>= 4/8) and keep the centre's group id.
   - The result is identical cell by cell to evaluating Scene::eval on every cell, including group ids with Scene::eval's "first group that contains p wins" semantics.
   - Parallel over the thread pool. Not re-entrant on the same output buffer.
   - Options: thicken=0.12, supersample=true, z_min=1, plus the new fields block_skip=true and lipschitz=1.5.

2) void mesh_scene(scene, map, cell_fraction, render::Mesh& out)
   New overload: mesh_scene(scene, map, cell_fraction, out, const MeshOptions&, MeshStats* = nullptr).
   - Algorithm: Surface Nets with sample spacing h = cell_fraction*dx.
   - pos: lattice cell coordinates.
   - nrm: outward unit normal (SDF gradient from a tetrahedron).
   - group: 1-based group id per vertex.
   - color: filled with render::component_color of the group when MeshOptions::fill_color is set (default).
   - tri: CCW seen from outside. Deterministic output. bounds is filled.
   - Uses a thread_local workspace owned by the calling thread (no allocations after the first call). Call it from the app or main thread.
   - MeshOptions: projection_steps=1, block_skip=true, halo=false, lipschitz=1.25, fill_color=true, max_samples=96M.
   - MeshStats: grid dims, h, eval counts, sign_fixes, per-phase timings.

3) void mesh_voxels(const u8* solid_id, nx, ny, nz, render::Mesh& out, bool skip_ground=true)
   - Greedy-merged faces between solid (1..254, and 255 when skip_ground is false) and non-solid cells. Outside the grid counts as air.
   - 4 vertices and 2 triangles per rectangle. Faces sit at cell centre ± 0.5.
   - Per-vertex group id and flat outward normals. color is set to grey 0xFFB8BCC4.

4) New public utility geom::RegionEval: init_all(scene, group_boxes), refine(parent, region, centre, L*R + margin), eval(p, &gid), classify_id(LR, thr), classify_sign(LR).
   - An evaluator restricted to an AABB that returns exactly what Scene::eval returns for points inside the region, while pruning groups that cannot matter.
   - Other modules (for example flowvis) can use it for fast repeated SDF queries in a small region.

Threading: all functions use cfd::pool() internally (call pool().start() first). They are safe to call nested from a pool job, which then runs serially.

### Cambios de contrato
These are compatible additions to my own header, src/geom/voxelizer.hpp. Existing callers compile unchanged:
- New fields VoxelOptions::block_skip (bool, true) and VoxelOptions::lipschitz (float, 1.5), added at the end of the struct.
- New struct MeshOptions: projection_steps, block_skip, halo, lipschitz, fill_color, max_samples.
- New struct MeshStats.
- New overload mesh_scene(scene, map, cell_fraction, out, const MeshOptions&, MeshStats* = nullptr). The original 4-argument signature is kept.
- New public utility struct geom::RegionEval and the helper geom::aabb_gap.
- Optional instrumentation macros CFD_GEOM_COUNTING and CFD_GEOM_PRUNE (no-op by default).
- mesh_scene now also fills Mesh::color with render::component_color of each vertex's group, and mesh_voxels fills it with grey. The contract says the visualization fills colour each frame; this is only a sensible default.

Needed changes in src/geom/sdf.cpp, which I did not modify (for the lead):
1) BUG in Scene::eval_group, Op::Subtract: `if (bd > -d) continue;` drops the subtraction whenever the point is outside the base shape (d>0), even when it is inside the subtracted box.
   - Result: a discontinuous SDF. The sign stays correct, but with thicken > 0 the voxelizer produces phantom solid lids of thickness thicken·dx over openings (cockpit, channels).
   - Fix: `if (bd > max_(-d, 0.0f)) continue;` and, for SmoothSubtract, `if (bd > max_(pr.k - d, 0.0f)) continue;`.
   - Reproduction: sphere r=1 minus a box that protrudes from it; eval(0,0,1.02) returns 0.020 instead of 0.300. test_geom prints this as INFO.
2) Union culling once inside: `Op::Union: if (bd >= d) continue;`, and the same rule between groups in Scene::eval. With d<0 every later prim or group is skipped even when the point is deeper inside it.
   - Result: interior magnitudes are wrong and discontinuous (|∇d|≈38 measured on the f1_2019 stepped floor). Normals at unions are slightly affected and Lipschitz bounds break.
   - Exact fix: skip only if `bd > max(d, 0)`.
   - If the GROUP-level rule in Scene::eval changes, RegionEval::eval (voxelizer.hpp:173) must mirror it and pruning rules (b) and (c) must be re-checked. The group id semantics would also change from 'first group containing p' to 'deepest group'.
   - With both fixes, VoxelOptions::lipschitz could safely drop to about 1.1-1.25, which is 20-35% faster.

### Problemas conocidos
- Correctness of the block skip (and exact equality with brute force) assumes |∇d| ≤ L near the surface and SDF ≥ box distance outside each group's box. Neither is guaranteed by sdf.cpp, which has bound primitives and the two culling issues above. Measured: voxelizer exact on all 19 scenes for L in [1.0, 2.0]; the default is 1.5 for margin. More aggressive primitives in future models could produce a few differing cells. There is no self-correction in the voxelizer.
- With the sdf.cpp Subtract bug still present and thicken > 0, voxelize matched brute force on every tested scene (0 cells), but this is not guaranteed around protruding subtractions. Either result contains the phantom lids caused by the bug. The test therefore checks identity on the subtracting scene with thicken=0 (exact by sign) and with thicken=0.12 on a scene without subtractions.
- mesh_scene: Surface Nets with one vertex per cube produces a few non-manifold (pinched) edges on features thinner than the sample spacing h: 0.06% of edges on the F1 stand-in, 0-192 edges on the real models (f1_wing_ge has the most). These are not cracks and orientation stays consistent, but the 'every edge shared by exactly 2 triangles' property only holds for well-resolved shapes (verified on the sphere and torus).
- mesh_scene with projection_steps=1 (default): about 1.3% of F1 triangles have a geometric normal disagreeing with the averaged vertex normals, only at sharp or thin features. With projection_steps=2 it drops to about 0.6% at +30-40% cost.
- mesh_voxels uses greedy rectangles, which create T-junctions. The surface is geometrically closed (exact face coverage and divergence volume verified), but not topologically watertight in the 'edge shared by 2 triangles' sense, so a rasterizer without exact vertex snapping may show rare single-pixel cracks.
- mesh_scene and mesh_voxels keep a thread_local workspace per calling thread (up to tens of MB for very fine meshes; max_samples caps it). The memory is not released until that thread exits.
- Wall-clock timings are noisy because other agents' benchmarks and compiles ran concurrently. Ranges are reported, and the before/after claims rely on load-independent eval counts and per-thread CPU time. Some micro-optimizations (SWAR row fill, template loop unswitching) are not isolated or measured.
- The integration check against the real models/ catalogue used a scratch harness, not a committed test, because tests/test_geom.cpp must not depend on the models module that is still being developed.

### Riesgos restantes (revisor)
- sdf.cpp (not this module's file) still has both culling bugs. The lead should apply the one-line fixes: Subtract `if (bd > max_(-d, 0.0f)) continue;`, SmoothSubtract `if (bd > max_(pr.k - d, 0.0f)) continue;`. Verified with a local copy: the probe then disables the literal path automatically, voxelize gets about 12% faster, the discontinuity-driven gradients go away, and every geom test passes. The Union interior culling bug (`bd >= d` when d < 0) is still worth fixing too. If the GROUP-level rule in Scene::eval changes, RegionEval::eval and pruning rules (b)/(c) must be re-checked.
- The block-skip exactness still relies on two hypotheses: |∇d| ≤ lipschitz (1.5) outside subtracted-prim boxes, and each group's SDF being ≥ the distance to its box outside that box. sdf.cpp does not guarantee either. Now checked on all 18 current models by tests/test_geom_models.cpp, but future models with more aggressive bound primitives could break it; rerun that test whenever models change.
- thicken < 0 is not guaranteed identical to brute force because of the interior union culling in sdf.cpp.
- mesh_voxels T-junctions: geometrically closed but not edge-watertight, so a rasterizer may show single-pixel cracks.
- mesh_scene: Surface Nets with one vertex per cube leaves about 0.06% non-manifold edges and about 0.14% of triangles with vertex normals opposing their geometric normal on features thinner than h (endplates, trailing edges). The f1_wing_ge model reaches 0.64%. Normals there are bisectors, not inward, after the robust-normal pass.
- The literal-path cost (+13% CPU on the stand-in, up to +28% on f1_2008 on 1 thread) is paid while the sdf.cpp bug exists. That is still about 10-12 ms wall per F1 car at 256×128×96 with 20 threads, far below the 60 ms target.
- Timings are noisy: load average ranged 6-20 from other agents. Reported numbers are medians or minima of interleaved A/B runs.
- The models module has a unity-build ODR clash of its own (models/objects.cpp: `deg()` redefined in an anonymous namespace), found while testing `make unity`. Not a geom issue; flagging it for the models owner and the lead.

## models

### API (revisor)
Namespace cfd::models; contract in src/models/model.hpp. The catalogue has 18 entries in stable order: f1_1967, f1_1979, f1_1998, f1_2008, f1_2011, f1_2014, f1_2019, f1_2022, f1_2026, sphere, cylinder, cube, ahmed_25, naca0012_wing, naca4412_wing, f1_wing_ge, airfoil_2d, road_car.

Functions:
- int count(); const Info& info(int); int find(const std::string&) (returns −1 if missing); Built build(int, const Params&). build() is reentrant, takes 0.07-0.11 ms per car, and Scene::eval on the result is const and thread-safe (checked with TSan).
- Params resolve_params(int, const Params&): fills defaults (< 0 / < −900 sentinels). NEW: non-finite values (NaN/inf) also fall back to defaults. Clamps ride 0..400 mm, yaw ±30°, flaps ±20°, AoA ±30°, height 2..5000 mm, gap 5..150 mm, and zeroes params not in param_mask.
- bool same_geometry(int, const Params&, const Params&): wheels_rotating only changes wall motion.
- const char* kind_name(Kind).
- Xform car_body_frame(ride_front_m, ride_rear_m, wheelbase_m, yaw_rad): pitch rot_y(−asin((rr−rf)/wb)) about the front reference point, lift by rf, then yaw about (wb/2, 0, 0). Xform car_wheel_frame(wb, yaw): yaw only.

Built fields:
- scene: frames already applied.
- info.
- params: resolved.
- bounds_m = scene.bounds(). EXCEPTION, CHANGED: for Info::spans_domain models (airfoil_2d), bounds_m.y is exactly the domain width (±0.5 m) and the geometry overhangs it by 0.1 m per side. The app must set ny·dx to that Y extent and use slip or periodic sides.
- front_axle_m / rear_axle_m: ground contact points, yawed.
- moment_ref_m: cars = mid-wheelbase at ground level; wings, now including f1_wing_ge, = quarter chord of the main element; bodies = centre.

Info extensions: reg_width_m, default_front_flap_deg, default_rear_flap_deg, spans_domain, ref_source.

Semantics:
- front/rear_flap_deg is an increment over the era default (+ = more downforce); slot gaps stay exactly 45 mm at every setting.
- drs_open = DRS on 2011-2025 cars (top rear flap to −6°) or X-mode on 2026 (front flaps to −3°/−6°, rear to −3°). CHANGED: it only ever OPENS a flap. A flap already flatter than the target is left alone.
- f1_wing_ge: aoa_deg in F1 convention (+ = LE down), front_flap_deg = flap increment, flap_gap_mm = slot, height_mm = lowest wing point.
- Free-air objects float with their lowest point at height_mm (default 1.5 m).

Groups: id = 1 + index; ≤ 13 groups and ≤ 50 primitives per car. Wheels are one Frame::Wheels group per axle with RigidMotion omega_hat = (0, −1/R, 0) about the axle (only when wheels_rotating) and 5 mm tyre penetration. The 1979 skirts and 2022 wheel-wake deflectors are Frame::Wheels groups with NO motion, so the app must read wall motion from Group::motion.

Geometry guarantees (tested): sdf < 0 in the interior everywhere, including the y = 0 plane (new k_sym_overlap = 1 cm for mirrored extrusions ending at y = 0). At dx ≤ 3 cm, with the real voxelizer, all groups survive, wing elements are leak-tight and slots stay open (default and DRS).

Tests:
- tests/test_models.cpp (384 checks): g++ -std=c++23 -O3 -march=native -Isrc -pthread src/core/threadpool.cpp src/core/png.cpp src/geom/sdf.cpp src/models/*.cpp tests/test_models.cpp
- NEW tests/test_models_vox.cpp (131 checks): the same command plus src/geom/voxelizer.cpp.
- tests/test_models_bench.cpp.

Tool: tools/model_preview.cpp writes build/models/<id>.png and <id>_vox.png.

Docs: /home/lucas/cfd/docs/MODELOS.md and /home/lucas/cfd/docs/opt/models.md (new section 8: review findings, fixes and the real-voxelizer dx table).

### API (implementador)
Namespace cfd::models (contract src/models/model.hpp). Catalogue (stable order: F1 chronological then reference objects), 18 entries: f1_1967, f1_1979, f1_1998, f1_2008, f1_2011, f1_2014, f1_2019, f1_2022, f1_2026, sphere, cylinder, cube, ahmed_25, naca0012_wing, naca4412_wing, f1_wing_ge, airfoil_2d, road_car.
- int count(); const Info& info(int); int find(const std::string&) (-1 if missing); Built build(int index, const Params&).
- Params resolve_params(int, const Params&) [NEW]: fills defaults (<0 / <-900 sentinels), clamps (ride 0..400 mm, yaw ±30, flaps ±20, aoa ±30, height 2..5000 mm, gap 5..150 mm), zeroes params not in Info::param_mask.
- bool same_geometry(int, const Params&, const Params&) [NEW]: true if the solid shape is identical (wheels_rotating only changes wall motion -> no re-voxelization needed).
- const char* kind_name(Kind) [NEW].
- Xform car_body_frame(ride_front_m, ride_rear_m, wheelbase_m, yaw_rad): pitch rot_y(-asin((rr-rf)/wb)) about the front-axle reference point then +rf, then yaw rot_z about (wb/2,0,0); body point (0,0,0)->z=rf, (wb,0,0)->z=rr exactly. car_wheel_frame(wb, yaw): yaw only.
- Built: scene (frames already set), info, params (resolved), bounds_m (= scene.bounds()), front_axle_m/rear_axle_m (ground contact points, yawed), moment_ref_m (cars: mid-wheelbase at ground; wings: quarter chord; bodies: centre).
Group conventions: group id = 1 + index (<= 13 groups per car, <= 50 prims). Wheels: one group per axle ('Ruedas delanteras'/'Ruedas traseras', Component FrontWheels/RearWheels, Frame::Wheels, mirror_y), RigidMotion omega_hat=(0,-1/R,0), center=(x_axle,0,R-0.005) only if wheels_rotating (contact patch velocity = +1·U∞, tested); tyres penetrate 5 mm. Other Frame::Wheels groups WITHOUT motion: 1979 'Faldones deslizantes' (Sidepods) and 2022 'Deflectores de estela' (Suspension) — the app must take wall motion from Group::motion (zero for these). Sidepod appendages/bargeboards use Component::Sidepods; the f1_wing_ge object uses WingMain/WingFlap/Endplate; Ahmed stilts use Suspension.
Semantics: front/rear_flap_deg = increment over era default (+ = more downforce, slot gap preserved); drs_open = DRS (2011-2025: top rear flap rotated about its trailing edge to -6°) or X-mode for 2026 (front flaps to -3°/-6°, rear flap to -3°). For f1_wing_ge: aoa_deg in F1 convention (+ = LE down = more downforce), front_flap_deg = flap increment, flap_gap_mm = slot gap, height_mm = lowest point of the wing. Free-air objects (needs_ground=false) float with their lowest point at height_mm (default 1500 mm) above the potential ground z=0. Info extensions [NEW]: reg_width_m, default_front_flap_deg, default_rear_flap_deg, spans_domain (airfoil_2d: app must set domain width = Y extent of bounds_m with slip/periodic sides), ref_source. ref_length_m = largest dimension relevant for domain sizing (car length, wing span, cylinder L).
Threading: catalogue is a function-local static (thread-safe init, immutable afterwards); build() is reentrant (no globals), costs 0.06-0.25 ms per car; Scene::eval on the result is const/thread-safe. All wing elements use thickness-floored inverted NACA profiles (>= 4.2 cm) and 4.5 cm slot gaps so they survive dx = 3 cm without diagonal leaks.

### Cambios de contrato
Backward-compatible extensions to src/models/model.hpp (my module's contract):
(1) New Info fields: reg_width_m (regulation max width, 0 = free), default_front_flap_deg and default_rear_flap_deg (absolute era flap angle, informational), spans_domain (bool, pseudo-2D: domain width must equal the model's Y extent with slip/periodic sides; set for airfoil_2d), ref_source (const char*, where ref_ClA/ref_CdA come from).
(2) New functions: Params resolve_params(int, const Params&), bool same_geometry(int, const Params&, const Params&), const char* kind_name(Kind).
(3) Params comments now state that front/rear_flap_deg positive = more downforce, and that for f1_wing_ge front_flap_deg is the flap increment.

Needed from other modules / the lead, NOT made by me:
(a) docs/ARCHITECTURE.md and geom/sdf.hpp say '+Y = izquierda del piloto'. With the car facing -X in a right-handed Z-up frame, +Y is the driver's RIGHT. Harmless because all models are symmetric, but the text should be corrected.
(b) The app must honour Info::spans_domain for airfoil_2d (ny·dx = Y extent of bounds_m, slip or periodic laterals).
(c) The app must take wheel wall motion from Group::motion: some Frame::Wheels groups have no motion (1979 skirts, 2022 wheel-wake deflectors).
(d) Optional geom speed-up suggestion: in Scene::eval / eval_group, compare squared AABB distances (Aabb::distance computes a sqrt per group and per primitive), or evaluate groups in ascending box-distance order.

### Problemas conocidos
- Reference SCz/SCx (ref_ClA/ref_CdA) for the F1 cars are order-of-magnitude estimates, not published data (flagged in Info::ref_source). Sphere, cube and Ahmed Cd are classic literature values; Ahmed Cl ≈ 0.3 is approximate.
- Aerodynamics are verified only geometrically (downforce orientation, continuity, connectivity). No LBM run was done: the solver belongs to another module and was not available to me.
- Coarse-grid concessions: slot gaps of 4.5 cm (real 1-1.5 cm) and elements ≥ 4.2 cm thick with a blunt trailing edge. On fine grids these add base drag and a less realistic slot flow.
- 2008 rear wing follows the real 2008 regulations (1.0 m wide, low), not the brief's 'narrow tall' (that arrived in 2009 and is in f1_2011).
- Cars 2008-2014 come out 0.2-0.4 m longer than the real ones, and rear-wing spans for 1998-2019 were narrowed slightly, to keep ≥ 4.5 cm between endplate and tyre (they would fuse into a solid wheel-wing bridge at dx = 3 cm).
- Voxel checks (group survival, leaks, connectivity) use my own emulation of the voxelizer rule (sdf(centre) < 0.12·dx). The real geom::voxelize with 2×2×2 supersampling was not available to verify against.
- 1979 sliding skirts are in the wheel frame and overlap the sidepods only while front ride height is below ~335 mm; resolve_params allows up to 400 mm, so they would detach above ~335 mm.
- Suspension arms are placed in the body frame for the given ride height (rebuilt each time); cosmetic only.
- Timing figures are noisy because of heavy background load; ranges and min-of-N are reported. Group-reordering and tracer over-relaxation were inconclusive or no-gain and were not adopted.
- Shapes are deliberately simplified: box-like sidepods, driver as sphere + capsule, rectangular endplates, 2-3 elements per wing. The road car is 'DrivAer-like', not the exact DrivAer geometry.
- tests/test_models_bench.cpp matches the Makefile 'make test' pattern (tests/test_*.cpp). It always exits 0 and takes ~6 s.

### Riesgos restantes (revisor)
- Resolution: the design and the tests hold for dx ≤ 3 cm. With the real voxelizer, 3.3 cm is clean in my sampling, although an earlier section offset showed 1 of 36 slots closed. At 3.6 cm: f1_1998 leaks in 3/72 sections, f1_2022 closes 1/36 slots and f1_2026 2/36. At 4 cm: 1-6 of 60-72 sections leak per car and 0-3 of 24-36 slots close. If the app picks dx from about 140 cells per car length (≈ 3.9 cm), multi-element wings will partly leak or clog. The app should keep dx ≤ 3 cm for the F1 cars.
- f1_2022 with flaps +20 at dx = 3 cm: 1 of 27 sampled front-wing slot sections closes (a one-cell bridge, grid-offset dependent). A 5.5 cm k_min_gap removes it but was not adopted.
- Extreme rake (independent 0-400 mm ride heights) pushes front wings and bodywork into the ground; the UI should limit rake.
- f1_wing_ge has no thickness-floored profiles. It is meant for dx ≈ 4-12 mm; at car-scale dx its flap leaks. The app should size its domain from ref_length_m (1.2 m).
- For spans_domain models bounds_m is deliberately NOT scene.bounds() in Y. Consumers must size the domain from bounds_m and use scene.bounds() only for geometric clipping (the voxelizer already does).
- The leak and slot checks are 2D in-plane (xz) connectivity at 3 spanwise sections × 3 grid offsets; full 3D flow paths through D3Q19 diagonal links across y were not analysed. No LBM run was done: aerodynamic behaviour, including the ref_ClA/ref_CdA order-of-magnitude estimates, is unverified.
- The geom voxelizer and mesher prune regions assuming SDF Lipschitz ≤ 1.5. I did not verify this for every model primitive; TaperBox nose slopes are estimated around 1.16.
- k_sym_overlap slightly stretches the zcurve/xcurve law at the centre of the 2022/2026 wings (the offset is < 0.01 mm): negligible, but the shape differs by that amount from before.
- Timing thresholds in test_models (2 µs) fail under sanitizer instrumentation and could flake under very heavy load (they measured up to about 1 µs at load 16).

## raster

### API (revisor)
Namespace cfd::render (src/render/raster.hpp). The rules below apply to every function.
- Each call is internally parallel on cfd::pool(). The functions share persistent scratch buffers, so call them from ONE thread at a time (the render thread).
- Output is clipped to cam.vp ∩ framebuffer and nothing outside is modified. The 8-px aligned blocks straddling the vp edge are re-written with unchanged values, so do not draw concurrently into the ≤7 px next to the viewport.
- Degenerate input (NaN/inf, 1e30, out-of-range or non-multiple-of-3 indices, absurd widths or sizes, bad polyline ranges) is discarded or clamped. Checked under ASan, UBSan and TSan.
- Colours are 0xAARRGGBB. Depth is linear view depth, smaller = nearer.

- draw_mesh(fb, cam, mesh, light, style)
  - Requires fb.depth.
  - 28.4 fixed point with the top-left rule (watertight, including near-plane and ±8192 px guard-band clipping). Perspective-correct normals and colours.
  - Per-pixel Blinn-Phong: hemispheric ambient, camera fill, specular, rim. Zero-length normals no longer give NaN or black pixels.
  - two_sided=false culls back faces (~20% faster on closed meshes).
  - two_sided=true shades back faces with the normal flipped, except when mesh.nrm is missing: then all faces use the normal facing the camera.
  - style.alpha<1 blends without writing depth. Per tile it draws back faces first, then front faces (near side on top), in submission order within each group, with no depth sort.
  - Missing mesh.color, or use_vertex_color=false, uses base_color. Vertex-colour alpha is ignored.
  - wireframe_overlay darkens pixels near triangle edges.
- draw_textured_quad(fb, cam, corners[4], tex, tw, th, alpha, depth_test=true, depth_write=true)
  - Corners (0,0) (1,0) (1,1) (0,1). Texel centres at (i+½)/tw, row 0 = v0.
  - Clamped bilinear, perspective-correct, double-sided.
  - depth_write writes depth only where the final alpha ≥ 0.5.
- draw_lines(fb, cam, pts in pairs, colors, per_vertex_color, width=1.5, depth_test=true) and draw_polylines(fb, cam, pts, colors per vertex, starts, counts, width=1.5, depth_test=true)
  - Anti-aliased by coverage with round caps. Colour A is opacity.
  - width<1 draws 1 px with attenuated alpha. Width is clamped to 256.
  - Missing colours repeat the last one (or white).
  - No depth write. Depth test bias is 0.2% + 0.05.
  - Invalid polyline ranges are ignored.
- draw_points(fb, cam, pts, colors, size, additive)
  - Soft (1−r²/R²)² splats. additive uses saturating add; otherwise alpha blend.
  - Depth-tested with bias, no depth write.
  - Per-point colour if colors.size() ≥ pts.size(), else colors[0] (or white). size is clamped to [1,256]; below 1 it is attenuated.
- draw_ground(fb, cam, z0, extent, spacing, offset_x, base, line, tex=nullptr, tw=0, th=0)
  - Per-pixel ray/plane intersection inside extent.xy (infinite plane if extent is empty).
  - Analytic AA grid (a major line every 5 cells), subtle checker, belt animation via offset_x, fog relative to cam.distance.
  - base.A is ground opacity and line.A is line opacity.
  - tex is mapped u↔X over [lo.x,hi.x] and v↔Y, and does not move with the belt.
  - Depth-tested; writes depth where coverage > 0.5.
- draw_box_wire(fb, cam, box, color, width=1): 12 AA lines.
- draw_arrow(fb, cam, from, to, color, width=2.5): AA shaft plus a shaded cone head (radius ≥ ~2.4·width px). Does not overwrite last_mesh_stats().
- screen_space_edges(fb, vp, strength): cheap SSAO with 12 neighbours. Only neighbours 0.4–30% nearer occlude; no halos on the background. Flat oblique surfaces darken slightly.

Extensions (backward compatible):
- void fxaa(Framebuffer&, Rect): FXAA-lite, ~0.85 ms at 1500×1150.
- struct RasterStats + RasterStats last_mesh_stats(): timings and counters of the last draw_mesh.
- MeshStyle::debug_overdraw: every covered fragment adds +1 to the colour, with no depth test or shading.
- Contract changes made in this review: raster.hpp comments only (translucent order, the no-normal back-face rule, the viewport-edge rewrite note). No signature changes.
- New test file: tests/test_raster_equiv.cpp. It must be built with -DRZ_EXPERIMENTS on the raster*.cpp sources.

Tests are in /home/lucas/cfd/tests/test_raster.cpp and /home/lucas/cfd/tests/test_raster_equiv.cpp. The doc is /home/lucas/cfd/docs/opt/raster.md.

### API (implementador)
Namespace cfd::render. All functions are internally parallel (thread pool). They clip to cam.vp ∩ framebuffer and never write outside it. They share persistent scratch buffers, so call them from ONE thread at a time (the render thread). Degenerate input (NaN/inf, 1e30 coordinates, out-of-range indices, absurd widths/sizes, invalid polyline ranges) is discarded or clamped: no crash, no out-of-viewport writes (verified under ASan/UBSan).

- `draw_mesh(fb, cam, mesh, light, style)`
  - 28.4 fixed point, top-left rule (watertight), real near-plane clipping in view space (plus a ±8192 px guard band), perspective-correct interpolation.
  - Per-pixel Blinn-Phong: hemispheric ambient, camera fill light (only where the key light doesn't reach), specular, rim.
  - `two_sided=true` shades back faces with flipped normals; `false` culls them and is ~20% faster on closed meshes.
  - `alpha<1` blends without writing depth, in submission order (front faces first, then back faces), not depth-sorted.
  - Missing `mesh.nrm` → normal faces the camera. Missing `mesh.color` or `use_vertex_color=false` → `base_color`. Vertex-color alpha is ignored.
  - Requires `fb.depth`.
- `draw_textured_quad(fb, cam, corners[4], tex, tw, th, alpha, depth_test, depth_write)`
  - Corners in order (0,0) (1,0) (1,1) (0,1); texel i centre at u=(i+½)/tw, row 0 = v0, clamp-to-edge bilinear. Double-sided.
  - `depth_write` writes depth where final alpha ≥ 0.5, so transparent texels never occlude.
- `draw_lines(pts pairs, colors, per_vertex, width, depth_test)` / `draw_polylines(pts, colors per vertex, starts, counts, width, depth_test)`
  - Anti-aliased by coverage, round caps; the colour's A channel is opacity.
  - width<1 → 1 px with attenuated alpha; width clamped to 256. Missing colours repeat the last one (or white).
  - No depth write; depth test uses a bias of 0.2% + 0.05.
- `draw_points(pts, colors, size, additive)`
  - Soft (1−r²/R²)² splats. `additive` → saturating add (vpaddusb); otherwise alpha blend.
  - Depth-tested with bias, no depth write. Per-point colour if colors.size() ≥ pts.size(), else colors[0] (or white). size clamped to [1, 256] (below 1 → attenuated).
- `draw_ground(fb, cam, z0, extent, spacing, offset_x, base, line, tex, tw, th)`
  - Per-pixel ray/plane intersection inside extent.xy (infinite plane if extent is empty).
  - Analytic anti-aliased edges and grid lines (width from screen-space derivatives), subtle checker, major line every 5 cells, belt animation via offset_x, fog relative to cam.distance.
  - base.A = ground opacity, line.A = line opacity. tex is mapped with u↔X over [lo.x,hi.x] and v↔Y (matches flowvis GroundFootprint) and does not move with the belt.
  - Depth-tested; writes depth where coverage > 0.5.
- `draw_box_wire` → 12 anti-aliased lines.
- `draw_arrow` → anti-aliased shaft plus a shaded 3D cone head (radius ≥ ~2.4·width px). It does not overwrite last_mesh_stats().
- `screen_space_edges(fb, vp, strength)` → cheap SSAO with 12 neighbours. Only neighbours 0.4–30% closer occlude; no halo on background.

Extensions (backward-compatible):
- `void fxaa(Framebuffer&, Rect)` — FXAA-lite.
- `struct RasterStats` + `RasterStats last_mesh_stats()` — timings and counters of the last draw_mesh.
- `MeshStyle::debug_overdraw` (appended field) — each covered fragment adds 1 to the colour, with no depth test or shading; used for watertightness and overdraw maps.

Implementation notes (internal, namespace rz — unity-build safe):
- Tiles are 64×64. Counting-sort binning with no atomics gives deterministic order, with separate front/back lists.
- LPT tile scheduling. Hi-Z for back faces.
- Deferred shading with a per-thread 25 KB FP16 G-buffer.
- Optional instrumentation macros RZ_STATS and RZ_EXPERIMENTS; both are zero-cost when undefined.

### Cambios de contrato
- src/render/raster.hpp, compatible extensions only:
  - MeshStyle::debug_overdraw appended as the last field (default false).
  - New struct RasterStats and the function RasterStats last_mesh_stats().
  - New function void fxaa(Framebuffer&, Rect).
  - The existing function comments now spell out exact behaviour: threading rule, conventions, alpha/depth handling, clamping. No existing signature or default changed.
- New private header src/render/raster_internal.hpp, included only by my raster*.cpp files. Everything in it lives in cfd::render::rz; verified to compile as a single-TU unity build.
- No changes to core, geom, sdf or any other module's headers, and none are needed.
- Documentation location: I wrote docs/opt/raster.md as the task instructs. docs/ARCHITECTURE.md instead says each module adds a section to docs/OPTIMIZACIONES.md, so the lead may want to link or merge it there.
- I did not touch src/render/colormap.hpp, even though ARCHITECTURE.md lists it under render core.

### Problemas conocidos
- Timing noise: the machine carried heavy unrelated load throughout (load average 4–25). Absolute medians vary up to 2× between runs, and up to 10× when saturated. For that reason the test judges perf targets on the minimum of 31 runs, and when load average > ncpu/2 a missed target prints AVISO instead of FAIL. A perf regression on an overloaded machine can therefore go unflagged.
- Translucent meshes (alpha < 1) are not depth-sorted: order is submission order with front faces before back faces.
- No MSAA: mesh silhouettes are aliased unless fxaa() is applied (~0.85 ms at 1500×1150).
- Raster functions share persistent scratch buffers, so they must not be called concurrently from different threads (documented in raster.hpp).
- Framebuffer size is limited to 8128 px per side (7-bit tile coordinates in the packed rect). The SIMD binning path requires fewer than 2^28 vertices; larger meshes fall back to the scalar path automatically.
- draw_mesh ignores vertex-colour alpha (use style.alpha) and requires fb.depth.
- Deferred shading stores the interpolated base colour as 8 bits per channel before lighting (tiny quantization, not visible in the tests).
- Hi-Z is applied only to the back-face list. Front faces hidden behind other front faces still pay setup and the depth test.
- Polylines use round caps per segment, so at joints of semi-transparent or anti-aliased lines the fringe pixels are blended twice (slightly thicker fringe at joints; not visible at 1.5 px in the PNGs).
- FXAA copies the whole viewport colour into scratch each call. A halo-row scheme could cut ~0.3 ms, but it is not implemented.
- RZ_STATS and RZ_EXPERIMENTS instrumentation hooks remain in the code. They are compile-time zero-cost when undefined.
- As an integration check I ran the flowvis module's test once. It regenerated its own PNGs in build/flowvis/; no source files were touched.

### Riesgos restantes (revisor)
- Unity build of the whole project still fails in OTHER modules, not raster: src/models/objects.cpp:14 redefines cfd::models::detail::{anon}::deg, and src/render/flowvis_lines.cpp:66 redefines cfd::flowvis::{anon}::Norm. The lead should route these to the models and flowvis owners before `make unity`.
- Every 3D primitive re-writes, with the same value, the pixels (colour and depth) of the 8-px aligned blocks that straddle the viewport edge. The default 3D viewport width of 1500 is not a multiple of 8, so up to 7 px of the UI panel are read and rewritten. This is safe with the documented single render thread, but UI drawing must not run concurrently with 3D rasterisation. Now documented in raster.hpp.
- The SSAO darkens flat oblique surfaces by 1–10% (worst near the horizon). This is a quality limitation of the cheap depth-difference kernel.
- Polyline joints double-blend AA fringes, and the whole stroke when the colour is translucent (lines behind translucent slices). Pre-existing, documented.
- Translucent meshes are not depth-sorted within each face list. Back faces are now blended before front faces, which is correct for convex closed meshes, but a non-convex car still shows submission-order artefacts.
- The scalar vertex path (used only for the n%8 tail vertices and the A/B switch) is not bit-exact with the AVX2 path. Each vertex is transformed exactly once, so this cannot create seams.
- Performance targets are judged on the min of 31 runs and downgraded to AVISO when load > ncpu/2, so a real regression on a loaded machine can go unflagged. I could only confirm the absolute numbers in one quiet window.
- Reconstructed original sources for the A/B test remain under build/raster-review/orig/ (only under build/, not picked up by the Makefile's src/ wildcards).

## flowvis

### API (revisor)
namespace cfd::flowvis, header src/render/flowvis.hpp (documented in Spanish, with a per-frame usage block at the top). Implementation in flowvis.cpp, flowvis_lines.cpp, flowvis_volume.cpp, and flowvis_draw.cpp (the only file that calls raster.hpp).

Colour scales
- ColorScale {Colormap map; float lo, hi; bool diverging}: with diverging, 0 sits at the centre of the map.
- map_color(scale, v) and map_color8(scale, f8) give bit-identical results; NaN maps to index 0.

Quantities
- enum class Quantity {Speed, Ux, Uz, Cp, Cp0, Vorticity, QCriterion}.
- quantity_info(q) returns {name, short_name, description, default ColorScale}; also default_scale(q).
- Normalisation: velocities / U_inf; Cp = 2(rho-1)/(3U^2); Cp0 = Cp + rho|u|^2/U^2; vorticity |w|dx/U; Q as Q dx^2/U^2.
- cell_quantity(f, q, x, y, z): NaN in solid cells.
- sample_quantity(f, q, p): fluid-weighted trilinear.
- probe(f, p) returns Probe.

FlowSampler (FP16 packed copy of the field)
- update(FieldView): call once per solver macro update.
- sample(__m128), velocity(Vec3), outside(), solid(), prefetch(). The const sampling functions are thread-safe.
- Clamp now guarantees x0 <= n-2 for any n.

Seeding rakes
- Rake {origin, du, dv, nu, nv}, with Rake::line and Rake::grid.
- Helpers: rake_upstream, rake_floor, rake_vertical.

Streamlines
- params: max_steps, step/min/max, max_turn_deg, min_speed, both_directions, color_by, scale, width.
- set_rakes / set_seeds.
- compute(FlowSampler) is the fast path; compute_reference(FieldView) is FP32.
- draw(fb, cam, depth_test, behind).
- points(), colors(), starts(), counts() feed render::draw_polylines directly.

Particles (smoke)
- params: capacity, rate, max_age, jitter, time_scale, max_cells_per_substep, max_substeps, color_by, scale, intensity (clamped to [0,1]), point_size, additive.
- set_emitters, reset, step(sampler, lattice_steps), draw(fb, cam, behind), points(), colors().
- stats() conservation invariant: emitted == alive + died_* + overwritten.

Slices
- enum class Axis {X, Y, Z}.
- SliceParams: axis, pos (NaN maps to 0), quantity, scale, auto_range (1-99% of finite values), opacity, solid_color, fade_below, lic, lic_length, lic_contrast.
- SliceView methods: set_quantity, update(f), draw(fb, cam), texture/tex_w/tex_h, values/val_w/val_h (NaN = solid), corners(), effective_scale().
- More SliceView methods: value_min/max (finite values only), plane_pos, NEW axis(), value_at, value_at_world, pick, translucent_plane.
- NEW semantics: every param except opacity takes effect at update(); corners, pick, value_at* and translucent_plane describe the last computed slice.
- Plane axes: X -> (u=y, v=z), Y -> (x, z), Z -> (x, y).
- pick_plane(cam, sx, sy, axis, pos, hit).

Translucent planes
- TranslucentPlane {axis, pos, opacity, lo, hi} and plane_transmission(cam, p, planes).

Ground footprint
- GroundParams {quantity, scale, auto_range, z}.
- GroundFootprint: update(f), texture/tex_w/tex_h, extent() (z = 0.5 wall plane), draw(fb, cam, spacing, offset_x, base, line) via render::draw_ground.

Wake
- wake_survey(f, x) returns WakeStats {loss_area, crossflow_area, min_cp0, centroid, cells}.
- wake_slice_params(x).

Mesh colouring
- SurfaceMode {Cp, Speed, Component, Solid}; SurfaceParams {mode, scale, offset, solid_color, group_colors (256 entries)}.
- color_mesh(Mesh&, f, p): parallel over vertices.

Vortex volume
- VolumeParams: field, style Surface/Cloud, downsample, full, threshold, density, surface_alpha, color_by, color_scale, step, half_res, shading, hide_near_wall, light_dir.
- VortexVolume: update(f); render(fb, cam, behind) composites inside cam.vp ∩ framebuffer, reads fb.depth, and must be called LAST; stats().
- NEW documented split: update-time params are field, downsample, full, color_by, color_scale, hide_near_wall. Render-time params (no recompute) are style, threshold, density, surface_alpha, step, half_res, shading, light_dir.

Name helpers
- volume_field_name, volume_color_name, volume_style_name, surface_mode_name, axis_name.

detail namespace (internal)
- quantity_row.
- robust_range: finite values only; plo <= 0 and phi >= 1 returns exact min/max without a histogram.
- NEW mask_near_wall.

Recommended frame order
- mesh -> footprint.draw -> slice.draw -> lines.draw(behind) -> smoke.draw(behind) -> vortices.render(behind) -> UI.

Threading and memory
- update/compute/step/render are internally parallel via cfd::pool() and must be called from the main thread.
- No allocations in hot loops.
- Requires flags != nullptr and n >= 2 per axis.

### API (implementador)
namespace cfd::flowvis (header src/render/flowvis.hpp, fully documented in Spanish with a per-frame usage block at the top).
- ColorScale {Colormap map; float lo, hi; bool diverging} (diverging => 0 maps to colormap centre even for asymmetric ranges, e.g. Cp [-2.5,1]); map_color(scale, v) scalar and map_color8(scale, f8) AVX2 gather (bit-identical results).
- enum class Quantity {Speed, Ux, Uz, Cp, Cp0, Vorticity, QCriterion}; quantity_info(q) -> {name (Spanish), short_name, description, default ColorScale}; default_scale(q). Normalisation: velocities/U_inf, Cp=2(rho-1)/(3U^2), Cp0 = Cp + rho|u|^2/U^2 (1 = no loss), |w|*dx/U, Q*dx^2/U^2 (per-cell units).
- cell_quantity(f,q,x,y,z) (NaN in solids), sample_quantity(f,q,p) (fluid-weighted trilinear), Probe probe(f,p) (all quantities at a point).
- FlowSampler: update(FieldView) once per solver macro update (~0.8 ms at 3.1 M cells, parallel); packs (ux,uy,uz,rho-1) as 4xFP16/cell; sample(__m128 p)->__m128 (thread-safe const), velocity(Vec3), outside(), solid(), prefetch(). Used by Streamlines/Particles.
- Rake {origin,du,dv,nu,nv}; Rake::line/grid; helpers rake_upstream(objAabbCells, gap, nu, nv), rake_floor(obj, gap, z, n), rake_vertical(obj, gap, y, n).
- Streamlines: params (max_steps, step/min/max_step, max_turn_deg, min_speed, both_directions, color_by Speed/Ux/Uz/Cp/Cp0, scale, width); set_rakes/set_seeds; compute(const FlowSampler&) [fast path], compute_reference(FieldView) [FP32 validation]; draw(fb, cam, depth_test, span<TranslucentPlane> behind); points()/colors()/starts()/counts() spans ready for render::draw_polylines.
- Particles: params (capacity, rate per lattice step (0=auto), max_age (0=auto), jitter, time_scale, max_cells_per_substep, max_substeps, color_by, scale, intensity (alpha channel), point_size, additive); set_emitters(rakes); reset(); step(const FlowSampler&, float lattice_steps_advanced); draw(fb, cam, behind); points()/colors() compacted; stats() {emitted, died_outside, died_solid, died_age, overwritten, alive} with conservation invariant.
- enum class Axis {X,Y,Z}; SliceParams {axis, pos (cells, interpolated between layers), quantity, scale, auto_range (robust 1-99% percentiles), opacity, solid_color (dark grey), fade_below, lic (0 off, 2..4 subtexels/cell), lic_length, lic_contrast}; SliceView: set_quantity(q) (also resets scale), update(FieldView), draw(fb,cam) (depth write only if opaque), texture()/tex_w()/tex_h() (texture, may be LIC-upsampled), values()/val_w()/val_h() (1 value per cell, NaN=solid), corners(Vec3[4]) (quad over [-0.5,n-0.5], texel centres on cells, matches draw_textured_quad convention), effective_scale() (after auto range, for legends), value_min/max, value_at(u,v), value_at_world(p), pick(cam, sx, sy, hit, value) (mouse probe), translucent_plane(). Plane axes: X->(u=y,v=z), Y->(u=x,v=z), Z->(u=x,v=y).
- TranslucentPlane {axis,pos,opacity,lo,hi}; plane_transmission(cam,p,planes). Pass slice.translucent_plane() to lines/smoke/volume draws so things behind a translucent slice are attenuated.
- GroundFootprint: params {quantity (Cp), scale, auto_range, z (<0 auto = first layer above ground)}; update(f); texture()/tex_w()/tex_h()/extent() (u<->X, v<->Y, z=0.5 wall plane) and draw(fb, cam, spacing, offset_x, base, line) -> render::draw_ground with texture.
- WakeStats wake_survey(f, x_cells) {loss_area = integral(1-Cp0)dA, crossflow_area = integral((v^2+w^2)/U^2)dA (cells^2; *dx^2 for m^2), min_cp0, centroid, cells}; SliceParams wake_slice_params(x) preset (X axis, Cp0, Turbo [0,1]).
- SurfaceMode {Cp, Speed, Component, Solid}; SurfaceParams {mode, scale, offset (cells along normal), solid_color, group_colors span(256)}; color_mesh(Mesh&, FieldView, SurfaceParams) fills mesh.color in parallel.
- VortexVolume: params {field Q/Vorticity, style Surface (lit iso-surfaces)/Cloud, downsample 1|2, full (quantisation saturation; needs update), threshold (render-time, no update needed), density, surface_alpha, color_by Streamwise (ux/U, color_scale)/Magnitude, step, half_res, shading, hide_near_wall, light_dir}; update(FieldView) when the field changes (~2-3 ms); render(fb, cam, behind) LAST (reads fb.depth, composites into cam.vp); stats(). Name helpers: volume_field_name/volume_color_name/volume_style_name, surface_mode_name, axis_name.
Recommended frame order: mesh/ground(footprint) -> slice.draw -> lines.draw(behind) -> smoke.draw(behind) -> vortices.render(behind) -> UI.
Threading: all update/compute/step/render functions are internally parallel via cfd::pool() and must be called from the main thread (not re-entrant on the same object); const sampling functions are thread-safe. No allocations in hot loops (buffers grow only when sizes change). Requires FieldView.flags != nullptr and n >= 2 per axis. Relies on the solver writing wall velocity (0 / belt / wheel) and rho=1 into solid cells (true for current lbm/solver.cpp).

### Cambios de contrato
No shared/core headers were modified. New contract header src/render/flowvis.hpp (owned by this module) defines the whole API. Draw calls live in the new src/render/flowvis_draw.cpp so computations link without the rasterizer (the Makefile wildcard src/render/*.cpp picks it up). Suggested (not made) change to lbm/field.hpp: document that solid cells carry rho=1 and the wall velocity (0 / belt / wheel) after each macro step — lbm/solver.cpp does this today and flowvis gradients and sampler rely on it.

### Problemas conocidos
- Timings are noisy: the machine had constant unrelated load (1-min load 6-23). Reported numbers are medians from the cleanest runs; under heavy contention parallel timings go bimodal / 3-5x slower (e.g. streamlines 21.7 ms at load 21). Perf targets are enforced in the test only when load < 6 (or FLOWVIS_STRICT_PERF=1); otherwise they print AVISO.
- FP16 packed field has <= 2^-11 relative velocity error (U=0.08 stored as 0.0800171); particle/streamline positions inherit that bias (tests use relative tolerance 4e-4).
- flowvis relies on the solver writing wall velocity and rho=1 into solid cells (true for current lbm/solver.cpp). If another FieldView producer leaves garbage in solid cells, near-wall gradients/Cp sampling would be wrong (the volume hides near-wall cells anyway).
- hide_near_wall (default on) removes Q/|w| in fluid cells adjacent to solids to kill voxel-staircase noise; genuine vortices within 1 cell of a surface are hidden too.
- Translucent-plane attenuation (volume, lines, smoke) is an approximation, exact only when the slice colour equals the scene colour behind it; lines/points use per-vertex alpha so a segment crossing the plane gets interpolated attenuation.
- Half-resolution volume: silhouettes of vortex surfaces can be 1-2 px jagged; depth-aware compositing uses the nearest low-res first-hit depth.
- Vorticity and Q are normalised per cell (|w|dx/U, Q dx^2/U^2): default ranges depend on grid resolution; auto-range or rescaling needed when the grid changes.
- LIC is the most expensive slice mode (5-8 ms at 512x384 subtexels): recompute only when the field/slice changes or use lic=2.
- Particle RK2 substeps are capped at max_substeps (4): with very many lattice steps per frame each substep exceeds max_cells_per_substep, so accuracy near bodies drops (use time_scale < 1).
- No 8-wide AVX2 particle advection path: implemented only as a microbenchmark and measured slower than the packed FP16 per-particle path, so not integrated.
- GroundFootprint does not expose LIC options.
- Composed-image visual checks used the render-core module's raster_*.cpp as it existed during this session (in progress, concurrent work).

### Riesgos restantes (revisor)
- Contract change needed in a core header I may not edit: lbm/field.hpp FieldView::sample/velocity clamp to n-1-1e-4, which rounds to n-1 for n-1 >= 2048, so they read one cell out of bounds. Streamlines::compute_reference and any other FieldView consumer inherit this. Suggested fix: clamp to std::bit_cast<float>(std::bit_cast<u32>(float(n-1)) - 1).
- VortexVolume dens_at/col_at and the LIC use a v-1-1e-3 clamp: safe up to about 16k voxels per axis, same pattern beyond that (not reachable on this machine).
- Timings are noisy: background load was 4-21 during the review, including another engineer's test_lbm at 1800% CPU. Parallel timings are bimodal under contention.
- Design limitations reported by the implementer and confirmed: translucent-plane attenuation is approximate; half-res volume silhouettes are 1-2 px jagged; hide_near_wall hides real vortices within 1 cell of a wall; FP16 velocity bias up to 2^-11; particle substeps capped at 4 (particles can tunnel through thin solids with many lattice steps per frame).
- Visual note: with the default footprint scale (CoolWarm [-2.5, 1]), moderate ground-effect suction (Cp about -1) looks pale (ground_effect.png). The app may want auto_range or a tighter range for the footprint.
- Streamlines::draw and Particles::draw are const but write mutable scratch buffers, so they must not be called concurrently on the same object (documented as main-thread only). Not every user parameter is sanitised against NaN (e.g. lic_length).
- Not reviewed in integration: there is no app code yet. The composed-image tests used the render-core module's raster_*.cpp as it existed during this session.

## ui

### API (revisor)
PLATFORM (src/platform/platform.hpp, cfd::platform)
- Create a window with create_x11_window() (returns nullptr without DISPLAY) or create_headless_window(). Open it with w->open(title, fb_w, fb_h, detect_pixel_scale()); scale is 2 when the screen is at least 2560 px wide, and CFD_SCALE=1..4 overrides it.
- Every frame: in.begin_frame(); w->poll(in); if (in.resized) fb.resize(in.fb_w, in.fb_h); ... w->present(fb).
- Input is in framebuffer pixels. Mouse deltas keep the sub-pixel remainder.
- key_pressed includes auto-repeat. Key codes: printable ASCII in lowercase, Latin-1 160..255, then KeyEscape.. and the F-keys.
- text is Latin-1 and includes the tool's own dead-key composition. double_click and wheel are filled in; quit is set on WM_DELETE_WINDOW.
- Extensions: PresentStats present_stats(), set_fullscreen(bool), fullscreen(), and upscale_argb(src, sw, sh, sstride, dst, dw, dh, dstride, scale, parallel=true, nontemporal=true).
- Environment variables: CFD_NO_SHM=1 forces XPutImage; CFD_NO_NT=1 disables non-temporal stores. Link with -lX11 -lXext.

DRAW2D (src/render/draw2d.hpp, cfd::render)
- Painter p(fb) draws immediately, with push_clip/pop_clip/set_clip (32-deep stack, always inside the framebuffer).
- Primitives:
  - rects: fill_rect (opaque AVX2 fast path; translucent AVX2 blend, bit-exact with the scalar blend_px, destination alpha always 0xFF), rect, hline/vline;
  - rounded: fill_round_rect(r, radius, c, corner mask), round_rect (never paints outside r), shadow;
  - gradients: gradient_v (Bayer-dithered, phase-correct), gradient_h;
  - anti-aliased shapes: line, polyline, fill_circle, circle, fill_triangle, fill_area;
  - images: blit, blit_scaled;
  - text: text, text_glyphs, text_shadow, text_aligned.
- Text takes UTF-8 input with Font::Small (8×16) or Font::Large (16×32), Latin-1 plus 31 extra glyphs. Invalid sequences show '?'; orphan continuation bytes produce no glyph, so utf8_count == decoded glyphs == text_width/font_w for any input.
- Helpers: utf8_to_glyphs, utf8_count, utf8_prefix_bytes, font_w/h/center/baseline, lerp_color, with_alpha, mul_alpha, blend_px, alpha256. Colours are 0xAARRGGBB.
- DrawList has the same recording API with zero allocations in steady state. dl.render(fb) or DrawList::render(fb, lists, n, bands) rasterises in parallel row bands, bit-identical to execute(Painter&) and to issuing the same calls on a Painter directly. Images are not copied and must stay alive until render. Main thread only.

UI (src/ui/ui.hpp, cfd::ui::Context)
- Per frame: ui.begin_frame(input, fb.w, fb.h, now_sec()); if (ui.begin_panel("id", rect)) { ...widgets...; ui.end_panel(); } ui.end_frame(). Then, if (!ui.wants_mouse()), handle the camera; if (!ui.wants_keyboard()), handle app shortcuts. Draw the 3D view, and call ui.render(fb) last.
- wants_keyboard() is true while editing a value, while the Tab focus ring is visible, or in the frame where the UI consumed Esc. A click in the viewport releases keyboard focus.
- Widgets:
  - text: title, header (collapsible, salted id), label, text, text_colored, text_wrapped, value, value_colored, metric;
  - buttons and toggles: button(ButtonKind), toggle_button, toggle, checkbox, radio, segmented;
  - sliders: slider_float and slider_int. Shift gives fine control on both. Double-click or ctrl+click types a value, and a comma decimal is accepted.
  - combo, with the popup drawn on top; an outside click closes it and is consumed, and Esc closes it;
  - progress, tooltip (applies to the last item after 0.5 s);
  - plot_lines(PlotSeries[]): up to 8 ring-buffer series; the offset is normalized, and null or empty series are ignored;
  - bar_chart(Bar[]), colorbar, toast, set_toast_area.
- Layout: row(n), spacing, separator, indent/unindent, next_rect(h) plus draw() for custom drawing.
- IDs are FNV-1a of the label plus push_id/pop_id, with the '##' and '###' conventions. set_scale(2.0f) switches to the 16×32 font and doubled metrics.
- ui/demo.hpp provides demo_panel/demo_viewport as a reference panel.
- Files changed by the reviewer:
  - /home/lucas/cfd/src/render/draw2d.cpp
  - /home/lucas/cfd/src/ui/ui.cpp
  - /home/lucas/cfd/src/ui/ui.hpp
  - /home/lucas/cfd/tests/test_ui.cpp
  - /home/lucas/cfd/docs/opt/ui.md (new review section, corrected file:line references)

### API (implementador)
PLATFORM (platform/platform.hpp): auto w = create_x11_window() (nullptr without DISPLAY) or create_headless_window(); w->open(title, fb_w, fb_h, detect_pixel_scale()); every frame: input.begin_frame(); w->poll(input); if (input.resized) fb.resize(input.fb_w, input.fb_h); ... w->present(fb). Mouse coords/deltas are in framebuffer pixels (deltas keep the sub-pixel remainder). key_pressed includes auto-repeat (XKB detectable autorepeat). input.text is Latin-1 and composes Spanish dead keys itself. double_click is set by the platform. quit is set on WM_DELETE_WINDOW. Extensions: w->present_stats() (upscale/put/wait/total ms, shm, window size), w->set_fullscreen(bool), and upscale_argb(src,sw,sh,sstride,dst,dw,dh,dstride,scale,parallel,nontemporal). Env vars: CFD_SCALE=n, CFD_NO_SHM=1, CFD_NO_NT=1.

DRAW2D (render/draw2d.hpp, namespace cfd::render): Painter p(fb) draws immediately, with push_clip/pop_clip/set_clip. Primitives: fill_rect, rect, hline/vline, fill_round_rect(r, radius, c, corners mask), round_rect, shadow, gradient_v (Bayer-dithered)/gradient_h, AA line, polyline (max-coverage joins), fill_circle, circle, fill_triangle, fill_area (under an x-monotonic curve), blit/blit_scaled (with alpha), text/text_shadow/text_aligned with UTF-8 input and Font::Small 8x16 / Font::Large 16x32. Helpers: text_width, utf8_count, font_center/baseline, blend_px, lerp_color, with_alpha, mul_alpha. Colours are 0xAARRGGBB; alpha 255 takes the opaque fast path. DrawList has the same recording API and costs 0 allocations after warm-up. dl.render(fb) or DrawList::render(fb, lists, n) rasterises in parallel row bands and is bit-exact with serial execute(Painter&). Images passed to blit are NOT copied: they must stay alive until render. All calls are main-thread only. render uses the pool internally.

UI (ui/ui.hpp, cfd::ui::Context): ui.begin_frame(input, fb.w, fb.h, now_sec()); if (ui.begin_panel("id", rect)) { ... ui.end_panel(); } ui.end_frame(); then if (!ui.wants_mouse()) handle the camera; draw the 3D view; ui.render(fb) last. Layers: panel, then popups, then tooltips/toasts. Widgets: title, header (collapsible), label, text, text_colored, text_wrapped, value, value_colored, metric (big number card), button(ButtonKind), toggle_button, toggle (switch), checkbox, radio, segmented, slider_float/slider_int (shift = fine; double-click or ctrl+click to type a value; comma decimal accepted; bipolar fill when lo<0<hi), combo (popup on top, closes on outside click and consumes it), progress, tooltip (applies to the last item after 0.5 s), plot_lines(PlotSeries ring buffers: auto-scale, 1-2-5 grid, legend with current value, hover cursor, min/max decimation), bar_chart(Bar[] signed horizontal bars), colorbar(Colormap, lo, hi, unit), toast(color, fmt), set_toast_area(viewport). Layout: row(n), spacing, separator, indent, next_rect(h) + draw() for custom drawing. IDs are FNV-1a of the label + push_id/pop_id, with the '##' and '###' conventions. Section headers use salted ids. Without NDEBUG, duplicate ids in one frame print a stderr warning. wants_keyboard() is true while typing a value or while Tab focus is visible. set_scale(2.0f) doubles the metrics and uses the 16x32 font (for a native 4K framebuffer). ui/demo.hpp: demo_panel/demo_viewport is a reference panel with every widget, for the app developer.

### Cambios de contrato
1) platform.hpp (my contract), backward-compatible additions:
- struct PresentStats.
- Virtual methods with default bodies: Window::present_stats(), set_fullscreen(bool), fullscreen().
- Free function upscale_argb(...).
- Documented env var CFD_SCALE for detect_pixel_scale.
- Added `#include "../core/mathx.hpp"` before framebuffer.hpp.
- The Key enum semantics are extended so that Latin-1 letters 160-255 (e.g. 'ñ' = 241) map to their code. Those values are below KeyEscape and did not previously occur.

2) New module contracts: render/draw2d.hpp (Painter, DrawList, Font, text helpers), ui/ui.hpp (Context, Style, PlotSeries, Bar, ButtonKind) and ui/demo.hpp (reference panel).

3) Needed in core/shared headers, which I did NOT edit; worked around locally:
(a) render/framebuffer.hpp uses saturate() and Vec3 without including ../core/mathx.hpp, so any TU that includes it first fails to compile. It should add `#include "../core/mathx.hpp"`.
(b) core/threadpool.hpp: ThreadPool::parallel_for does not compile with an lvalue callable (F deduces to T& and `F* fn` becomes a pointer to a reference). It should use `std::remove_reference_t<F>* fn` (same issue in run_slots when forwarding). I pass temporary lambdas instead.
(c) Optional: docs/ARCHITECTURE.md asks each module to add a section to docs/OPTIMIZACIONES.md. I only wrote docs/opt/ui.md as instructed; the lead may merge it.

### Problemas conocidos
- There is no general text-entry widget. Typed input exists only for slider values (numeric characters, comma or dot). No clipboard support. Only one popup (combo) can be open at a time.
- Dead-key composition (´ ¨ ~ ` ^) uses an internal table instead of XIM, so other compose sequences or input methods are not supported. AltGr characters work through XLookupString.
- wants_mouse() is true whenever a combo popup is open, even with the pointer over the viewport. This is by design, because the next click closes the popup and is consumed.
- In the 20-thread microbenchmark, non-temporal stores in the upscaler were inconclusive (x0.67-1.40 vs normal stores). They are kept because they win slightly end-to-end in X11 and clearly win single-threaded (x1.6-2.1). CFD_NO_NT=1 turns them off.
- XWayland needs ~9-10 ms to consume each 37 MB SHM image. This caps presentation at roughly 100-200 FPS. It does not matter at the app's ~30 FPS target, but it shows up as ShmCompletion wait in ui_demo.
- The AVX2 opaque fill only ties GCC's autovectorized std::fill_n; the gain is only against non-vectorized scalar code.
- Inside real frame loops the UI costs 0.08-0.47 ms, not the 0.07-0.1 ms of the tight test loop. Pool workers fall asleep in the futex after 0.5 ms idle and each wake adds latency. That is a threadpool/app scheduling matter.
- The test's 1.5 ms timing check could in theory fail under extreme machine load (measured margin ~15x).
- perf was unavailable (no permissions), so profiling was done with per-primitive stopwatches in tests/test_ui_bench.cpp.
- Title faux-bold (the title drawn twice at a 1 px offset) is only used at scale >= 1.5, where the title and body fonts coincide. There is no larger bitmap font than 16x32.

### Riesgos restantes (revisor)
- Core/shared headers still need the fixes the implementer reported, which I did not make: render/framebuffer.hpp should include ../core/mathx.hpp (platform.hpp works around this), and core/threadpool.hpp parallel_for/run_slots break with lvalue callables (use std::remove_reference_t<F>*).
- Behaviour changes that integrators must know about: wants_keyboard() is now also true in the frame where the UI consumed Esc, so the app must read it after end_frame, as documented. A click in the viewport now hides the Tab focus ring. utf8_to_glyphs silently drops orphan continuation bytes instead of showing '?'. plot_lines draws at most 8 series.
- X11 input handling (dead keys, auto-repeat, double click, LeaveNotify, FocusOut) is not covered by automated tests; only the 60-frame smoke test and code reading. The dead-key compose table drops the accent when the next key does not compose (known_issue).
- wait_image gives up after 100 ms and writes into a buffer that may still be busy. This can tear under a stalled compositor but prevents a hang. It is a documented trade-off.
- Timings are noisy (load average 7–22 during the review). The 1.5 ms perf check has about 15× margin, but scale-2 figures ranged from 1.3 to 6.8 ms depending on load.
- No keyboard navigation inside an open combo popup, no general text entry, and only one popup at a time (implementer's known_issues remain valid).
