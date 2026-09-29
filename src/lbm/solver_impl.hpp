// ============================================================================
//  lbm/solver_impl.hpp — estado interno del solver (Solver::Impl). Privado del módulo lbm: lo comparten
//  lbm/solver.cpp (kernel, reconstrucción, contorno, API) y lbm/refine.cpp (refinamiento local por bloques).
//  Con refinamiento cada NIVEL (rejilla uniforme anidada) es otra instancia de Impl: la red base es la raíz y
//  las rejillas finas cuelgan de ella (Impl::kids / Impl::par). Ver docs/FISICA.md §1.6.
// ============================================================================
#pragma once

#include "kernel.hpp"
#include "../core/mem.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace cfd::lbm {

using namespace detail;

// ---- Refinamiento local: datos de la interfaz de una rejilla fina con su padre (lbm/refine.cpp) ----------------
// Estado compacto de una celda para interpolar / restringir (kMS floats): ρ, u, X (6: tensor simétrico). X guarda el
// no equilibrio como Π^neq/(ρ·τ) (desviadora, τ efectiva de cortante con Smagorinsky) y traza/(ρ·τ_b), en unidades
// de tiempo de la rejilla FINA de la interfaz (∝ tensor de deformación · dt: X_fina = X_gruesa / 2).
inline constexpr int kMS = 10;
// Celda fantasma de la rejilla fina: plantilla trilineal sobre el padre. idx[0..nm) → celdas activas del padre (lista
// mcells, interpoladas en el tiempo entre M_T y M_T+1); idx[nm..nm+nr) → celdas cubiertas del padre (lista rcells: la
// media actual de sus 8 celdas finas). Pesos ya normalizados (sin las esquinas sólidas).
struct IGhost {
    u32 n = 0;
    u32 dmask = 0;          // direcciones k cuyo post-colisión LEE alguien (n + c_k es una celda fluida): sólo esas se escriben
    u8 nm = 0, nr = 0, layer = 0, pad = 0;
    u32 idx[8] = {};
    float w[8] = {};
};
// Celda del padre en la capa cubierta junto a la interfaz ("esclava"): su post-colisión se escribe en cada paso del
// padre a partir de la media de sus 8 hijas (c0 = hija de menor índice; las demás c0 + {0,1} + {0,nx} + {0,nx·ny}).
struct IRcell {
    u32 np = 0, c0 = 0;
    u32 dmask = 0;          // direcciones k con np + c_k fluida en el padre (las únicas que se leen)
    u8 nfl = 0, layer = 0, pad0 = 0, pad1 = 0;   // nfl: hijas fluidas (0 → sin datos: equilibrio en reposo)
};

// ================================================================================================
struct Solver::Impl {
    Config cfg;
    Tuning tun;
    int nx = 0, ny = 0, nz = 0, nbx = 0;
    i64 N = 0, nxny = 0, S = 0, P = 0;       // celdas, zancada por dirección, relleno
    i64 off[L::Q] = {};
    Buffer<u8> ddf;                          // 19·S elementos de float o u16
    Buffer<u8> flags, flags_old, sid, user_sid, cls;
    Buffer<float> rho, ux, uy, uz;
    Buffer<float> tau0, tau0sq, omc0;
    Buffer<u32> rows_all, rows_active;
    i64 n_rows_active = 0;
    std::vector<WNode> wnodes;               // nodos junto a paredes (rebote, fuerzas, modelo de pared)
    WallSdf sdf;                             // distancia a la superficie real (rebote interpolado)
    Buffer<FAcc> facc;                       // kMaxForceChunks × 256 acumuladores
    Buffer<u8> ftouch;                       // kMaxForceChunks × 256: id tocado en el trozo
    WallMotion user_motion[256];
    bool user_moving[256] = {};
    Motion motion_now[256];
    Vec3 moment_ref{0, 0, 0};
    ForceSample fs, fs_mean;
    double acc_f[256][3] = {}, acc_m[256][3] = {};
    u64 t = 0;
    float u_from = 0, u_to = 0;
    u64 ramp_t0 = 0;
    double mlups = 0, t_kernel = 0, t_force = 0;
    std::atomic<u32> bad{0};
    // (fase 3) Backend externo: sincronización previa a leer/modificar el estado y campos macro sustitutos.
    u64 revision = 1;
    ExternalSync ext = nullptr;
    void* ext_ctx = nullptr;
    const float* ov[4] = {nullptr, nullptr, nullptr, nullptr};
    void sync_ext() const { if (ext) ext(ext_ctx); }

    usize esize() const { return cfg.precision == Precision::FP32 ? 4 : 2; }
    void* dir_base(int k) { return ddf.data() + static_cast<usize>((static_cast<i64>(k) * S + P)) * esize(); }
    const void* dir_base(int k) const { return ddf.data() + static_cast<usize>((static_cast<i64>(k) * S + P)) * esize(); }

    float u_at(u64 step) const {
        if (cfg.ramp_steps <= 0 || step >= ramp_t0 + static_cast<u64>(cfg.ramp_steps)) return u_to;
        const float s = smoothstep(0.0f, static_cast<float>(cfg.ramp_steps), static_cast<float>(step - ramp_t0));
        return u_from + (u_to - u_from) * s;
    }

    // Punteros de carga/escritura por paridad.
    void step_ptrs(u64 step, const void* ld[L::Q], void* st[L::Q]) const {
        const int p = static_cast<int>(step & 1);
        const usize es = esize();
        auto base = [&](int k, i64 o) -> const u8* {
            return ddf.data() + static_cast<usize>(static_cast<i64>(k) * S + P + o) * es;
        };
        ld[0] = base(0, 0);
        for (int i = 1; i < L::Q; i += 2) {
            ld[i] = p ? base(i, 0) : base(i + 1, 0);
            ld[i + 1] = p ? base(i + 1, off[i]) : base(i, off[i]);
        }
        st[0] = const_cast<void*>(ld[0]);
        for (int i = 1; i < L::Q; i += 2) {
            st[i] = const_cast<void*>(ld[i + 1]);
            st[i + 1] = const_cast<void*>(ld[i]);
        }
    }

    void build_tau() {
        ++revision;
        const float nu = std::max(cfg.nu, 1e-7f);
        const int xs = static_cast<int>(static_cast<float>(nx) * (1.0f - std::clamp(cfg.sponge_frac, 0.0f, 0.9f)));
        const float nu_max = std::max(nu, kSpongeNu);
        for (int x = 0; x < nx; ++x) {
            float v = nu;
            if (cfg.sponge_frac > 0.0f && x > xs && nx - 1 > xs) {
                const float s = smoothstep(static_cast<float>(xs), static_cast<float>(nx - 1), static_cast<float>(x));
                v = nu + (nu_max - nu) * s * s;
            }
            tau0[x] = 3.0f * v + 0.5f;
            tau0sq[x] = tau0[x] * tau0[x];
            omc0[x] = 1.0f - 2.0f / (tau0[x] + tau0[x]);   // misma expresión que con K=0 en one_minus_omega
        }
    }

    void update_motion_table() { fill_motion(t, motion_now); }
    void fill_motion(u64 step, Motion* dst) const {
        // Las velocidades de pared se dan a la u∞ objetivo: durante la rampa se escalan con u(t)/u∞.
        // Rampa hacia 0 (parar el túnel): se escalan respecto a la velocidad de partida → paran también.
        const float uc = u_at(step);
        const float r = u_to != 0.0f ? uc / u_to : (u_from != 0.0f ? uc / u_from : 1.0f);
        const Vec3 vg = cfg.ground == GroundMode::Moving ? Vec3(uc, 0, 0) : Vec3(0, 0, 0);
        for (int id = 0; id < 255; ++id) {
            const WallMotion& m = user_motion[id];
            dst[id].v = m.v * r;
            dst[id].omega = m.omega * r;
            dst[id].center = m.center;
            dst[id].contact_z = m.contact_z;
            dst[id].vc = vg;
            dst[id].conserve = m.impermeable;
        }
        dst[255] = Motion{};
        dst[255].v = vg;
    }

    bool id_moving(int id) const {
        if (id == k_ground_id) return cfg.ground == GroundMode::Moving;
        return user_moving[id];
    }

    void reset_fill();
    void rebuild(bool transitions);
    void run_kernel(bool macro);
    void compute_wall_geometry();
    // Rebote explícito (+ fuerzas) tras el paso `tt` (el que dejó en memoria su post-colisión). write_only:
    // sólo escribe las poblaciones entrantes (arranque / tras cambiar la geometría), sin fuerzas.
    void boundary_pass(bool accumulate, u64 tt, bool write_only = false);
    // Reserva y reconstruye una rejilla con la configuración c (init de la red base y de cada rejilla fina).
    void setup(const Config& c);
    double mass() const;                          // masa de la rejilla (sus unidades): ver Solver::total_mass

    // ---- Refinamiento local (lbm/refine.cpp). Cada rejilla es un Impl; la raíz es la red base. ----------------
    Impl* par = nullptr;                          // rejilla padre (nullptr = raíz)
    std::vector<Impl*> kids;                      // hijas directas (no propietario)
    std::vector<std::unique_ptr<Impl>> owned;     // (raíz) rejillas finas g = 1.. (padres antes que hijas)
    std::vector<LevelBox> cover;                  // cajas de las hijas, en celdas de ESTA rejilla
    int gid = 0, depth = 0;
    LevelBox box{};                               // (hija) celdas cubiertas del padre
    bool nested = false, gattached = false;       // hija / apoyada en el suelo
    Vec3 org{0, 0, 0};                            // centro de la celda (0,0,0) en celdas de la red base
    float scale = 1.0f;                           // dx / dx de la red base
    Buffer<u8> lay;                               // (refinada) capa de 2.º orden por celda (también fantasmas/esclavas)
    Buffer<u8> vflags, vsid;                      // (refinada) flags/ids de visualización
    std::vector<IGhost> ghosts;                   // (hija) capa fantasma
    std::vector<IRcell> rcells;                   // (hija) celdas esclavas del padre
    std::vector<u32> mcells;                      // (hija) celdas activas del padre en las plantillas
    // Grupos de 8 consecutivos en memoria (caras y/z: filas en x) para las rutas AVX2 de fantasmas y esclavas; el resto
    // (caras x, esquinas, sólidos, celdas junto a paredes móviles) va por la ruta escalar.
    std::vector<u32> gvec, gsca, rvec, rsca;
    Buffer<float> Mprev, Mnext, Rst;              // estados kMS: padre en T y T+1 (unidades finas), medias actuales
    Buffer<u32> tapidx;                           // (hija) bloque → ranura en tap (bloques con hijas de esclavas)
    Buffer<float> tap;                            // (hija) salida del kernel en esos bloques (kTapFloats por bloque)
    bool iface_dirty = true;                      // (raíz) hay que reconstruir las interfaces
    std::vector<u32> fown;                        // (con hijas) por nodo de pared: enlaces cuya fuerza cuenta aquí
    double t_iface = 0;                           // tiempo de las pasadas de interfaz de esta rejilla (último step())
    double t_iface_total = 0;                     // (raíz) suma de todas las rejillas
    int nsub = 1;                                 // pasos de esta rejilla por paso de la red base (2^depth)

    bool refined() const { return nested || !cover.empty(); }
    bool is_covered(int x, int y, int z) const;   // dentro de alguna caja hija
    Impl& grid(int g) { return g == 0 ? *this : *owned[static_cast<usize>(g - 1)]; }
    const Impl& grid(int g) const { return g == 0 ? *this : *owned[static_cast<usize>(g - 1)]; }
    int n_grids() const { return 1 + static_cast<int>(owned.size()); }
    void refine_plan(Config& c);                  // (raíz) normaliza las cajas y fija las listas `cover`
    void refine_create();                         // (raíz) crea las rejillas finas
    void iface_build();                           // (hija) plantillas de fantasmas, esclavas y celdas activas del padre
    void iface_build_all();                       // (raíz) todas las interfaces + flags de visualización + fown
    void iface_R();                               // (hija) Rst ← medias de las hijas de cada esclava (pre-colisión actual)
    void iface_M(Buffer<float>& dst, u64 pstep); // (hija) estado pre-colisión del padre en su paso pstep (mcells)
    void iface_restrict();                        // (hija) post-colisión de las esclavas en el paso actual del padre
    void iface_ghost(float a, bool macro);        // (hija) post-colisión de los fantasmas en el paso actual
    void advance(bool macro, float a);            // (raíz) un paso de la red base y, recursivamente, de las finas
    void step_kernel(bool macro);                 // fase A: kernel (con taps) y, en una hija, sus medias Rst
    void step_rest(bool macro, float a);          // fase B: esclavas, fantasmas, contorno y subpasos de las hijas
    void fill_vis();                              // (con hijas) celdas cubiertas del campo de visualización
    void build_fown();                            // (con hijas) propiedad de las fuerzas por enlace
    WallMotion to_grid(const WallMotion& m) const;   // movimiento de pared de la red base → esta rejilla
};

} // namespace cfd::lbm
