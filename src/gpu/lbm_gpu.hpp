// ============================================================================
//  gpu/lbm_gpu.hpp — backend iGPU del solver LBM (Vulkan de cómputo propio).
//
//  Diseño: el lbm::Solver de la CPU sigue siendo el DUEÑO del estado (config,
//  geometría, flags, nodos de pared con su q de Bouzidi, rampa, tiempo, fuerzas
//  publicadas). LbmGpu se engancha a él (Solver::set_external) y avanza los
//  pasos con su propia copia de las poblaciones en memoria de la iGPU (UMA:
//  misma LPDDR5x, sin copias PCIe). Cuando la CPU va a leer o cambiar el estado
//  (set_geometry, reset_flow, set_viscosity...), el gancho termina el lote en
//  vuelo y devuelve las poblaciones a la CPU; el siguiente lote detecta el cambio
//  de Solver::revision() y vuelve a subir todo.
//
//  Solapamiento CPU/GPU: submit(n) encola n pasos y vuelve enseguida; wait()
//  espera el semáforo de línea temporal y publica fuerzas y campos. Los campos
//  macro (ρ, u) van en DOBLE búfer en memoria HOST_CACHED: la CPU dibuja el lote
//  anterior mientras la GPU escribe el siguiente (Solver::field() apunta a él).
//
//  Alcance (paridad con la CPU): FP32 y FP16S, BGK y Regularizada + Smagorinsky,
//  esponja, entrada/campo lejano/salida, rebote implícito, cinta móvil, ruedas
//  (Ladd implícito o Bouzidi interpolado), ley de pared LogLaw/Slip y fuerzas
//  manométricas + galileanas por id. Todo lo que ofrece lbm::Config.
// ============================================================================
#pragma once

#include "../lbm/solver.hpp"
#include "vk.hpp"

#include <string>

namespace cfd::gpu {

struct LbmGpuTuning {
    u32 sg = 16;          // subgrupo del kernel de celdas (medido: 16 > 32 > 8 en Xe-LPG, ver docs/opt/gpu.md)
    u32 wg = 256;         // grupo de trabajo del kernel de celdas
    u32 wg_nodes = 64;    // grupo de trabajo del kernel de nodos de pared
    bool pair = true;        // 2 celdas por hilo con palabras f16×2 (ver docs/opt/gpu.md); false = 1 celda por hilo
    bool reuse_cmd = true;   // búferes de comandos grabados una vez por (n, paridad) y reenviados
    bool profile = false;    // marcas de tiempo por kernel (añade serialización mínima)
    vk::Mem ddf_mem = vk::Mem::Cached;   // tipo de memoria de las poblaciones
};

struct LbmGpuStats {
    int steps = 0;               // pasos del último lote publicado
    double gpu_ms = 0;           // tiempo de GPU del lote (marcas de tiempo)
    double wall_ms = 0;          // submit → publicación (reloj de pared)
    double mlups = 0;            // N·pasos / tiempo de GPU
    double gbs = 0;              // tráfico teórico de poblaciones (2·19·esize por celda y paso) / tiempo de GPU
    double record_us = 0;        // grabación del búfer de comandos (0 si se reutilizó)
    double publish_ms = 0;       // lectura de fuerzas + invalidación de campos en la CPU
    double upload_ms = 0, download_ms = 0;   // última subida / bajada completa del estado
    // Perfil por kernel (con LbmGpuTuning::profile): medias por paso en ms
    double k_step_ms = 0, k_boundary_ms = 0, k_reduce_ms = 0;
    usize nodes = 0, records = 0, chunks = 0, multi_id_nodes = 0, overflow_nodes = 0;
    double pure_groups = 0;      // fracción de grupos de trabajo "puros" (flags deducidos de las coordenadas)
};

class LbmGpu {
public:
    LbmGpu();
    ~LbmGpu();
    LbmGpu(const LbmGpu&) = delete;
    LbmGpu& operator=(const LbmGpu&) = delete;

    bool init(std::string* err = nullptr, bool verbose = false);   // dispositivo Vulkan
    bool ok() const;
    const vk::DeviceInfo& device() const;

    void set_tuning(const LbmGpuTuning& t);   // aplicar antes de attach (regenera pipelines si cambia)
    const LbmGpuTuning& tuning() const;

    // Engancha el solver (sube estado, genera y compila los shaders para su dominio y física).
    bool attach(lbm::Solver& s, std::string* err = nullptr);
    // Termina el trabajo en vuelo, devuelve poblaciones y campos a la CPU y desengancha.
    void detach();
    bool attached() const;

    // Encola n pasos (el último escribe ρ,u). Si hay un lote en vuelo lo publica antes. false si falla.
    bool submit(int n);
    // Espera el lote en vuelo y lo publica en el solver (fuerzas, t, divergencia, campos). Pasos publicados.
    int wait();
    // Solapamiento sin huecos: espera el lote en vuelo, encola n pasos más ENSEGUIDA y sólo entonces publica el
    // terminado (fuerzas en regiones alternas, campos en mitades alternas). Pasos publicados (−1 si falló el envío).
    int cycle(int n);
    bool busy() const;
    bool step(int n) { if (!submit(n)) return false; wait(); return true; }

    const LbmGpuStats& stats() const;
    const std::string& error() const;

    struct Impl;   // (público sólo para el gancho de sincronización del solver)
private:
    Impl* impl_;
};

// ¿Hay una iGPU/GPU Vulkan 1.3 utilizable? (prueba cacheada; `why` = motivo si no)
bool gpu_available(std::string* why = nullptr);

} // namespace cfd::gpu
