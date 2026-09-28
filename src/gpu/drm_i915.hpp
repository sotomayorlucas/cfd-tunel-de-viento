// ============================================================================
//  gpu/drm_i915.hpp — acceso "bare metal" a la iGPU Intel por ioctl DRM crudos
//  (driver i915 del kernel), SIN libdrm ni ninguna otra biblioteca.
//
//  Ruta EXPERIMENTAL de diagnóstico (ver docs/DRM.md). No es la ruta de cómputo del
//  simulador (esa es Vulkan compute, src/gpu/vk.*): aquí sólo se habla con el
//  command streamer (CS) de la GPU mediante buffers de lotes (batch buffers) escritos
//  a mano con comandos MI_* / XY_FAST_COPY_BLT documentados en los PRM públicos de
//  Intel, para medir latencia de envío y ancho de banda de memoria de la iGPU.
//
//  Estructuras: se usan las cabeceras UAPI del kernel instaladas en
//  /usr/include/drm (i915_drm.h, drm.h — son UAPI, no libdrm). Se comprueban los
//  tamaños con static_assert en drm_i915.cpp para detectar un ABI inesperado.
//
//  Todo falla con mensaje claro (Device::err) si un ioctl no está permitido; nada
//  lanza excepciones. No es ruta caliente: se permiten std::vector/std::string.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cfd::gpu::drm {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

// ---- Motores (clases de i915_drm.h: I915_ENGINE_CLASS_*) --------------------------
enum class EngineClass : u16 { Render = 0, Copy = 1, Video = 2, VideoEnhance = 3, Compute = 4 };
const char* engine_class_name(u16 cls);   // "rcs", "bcs", "vcs", "vecs", "ccs"

// Base MMIO de cada motor en Gen12+ (Xe-LP/Xe-LPG): el registro RING_TIMESTAMP está en
// base + 0x358. Sólo se conocen las instancias 0 de rcs/bcs/ccs/vcs/vecs.
// Devuelve 0 si no se conoce.
u32 engine_mmio_base(u16 cls, u16 instance);
inline constexpr u32 kRingTimestamp = 0x358;

struct EngineInfo { u16 cls = 0, instance = 0; u16 logical = 0; u64 caps = 0; };

struct MemRegion { u16 cls = 0, instance = 0; u64 probed = 0, unallocated = 0; };

struct Topology {
    bool ok = false;
    int max_slices = 0, max_subslices = 0, max_eus_per_subslice = 0;
    int slices = 0, subslices = 0, eus = 0;   // presentes (bits a 1 en las máscaras)
};

// ---- Dispositivo -------------------------------------------------------------------
struct Device {
    int fd = -1;
    std::string err;          // último error legible (vacío si todo fue bien)
    std::string drv_name, drv_date, drv_desc;
    int ver_major = 0, ver_minor = 0, ver_patch = 0;

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { close(); }

    // Abre el nodo (p. ej. "/dev/dri/renderD128") y hace DRM_IOCTL_VERSION.
    // Comprueba que el driver sea "i915".
    bool open(const char* path);
    void close();

    // ioctl con reintento en EINTR/EAGAIN. Devuelve 0 ó -errno.
    int ioctl(unsigned long req, void* arg) const;

    // I915_GETPARAM. Devuelve 0 ó -errno; en éxito escribe *out.
    int getparam(int param, int* out) const;

    // DRM_IOCTL_I915_QUERY de un único item (dos pasadas: tamaño y datos).
    // Devuelve 0 ó -errno (también si el kernel devuelve length < 0 en el item).
    int query(u64 query_id, u32 flags, std::vector<u8>& out) const;

    int query_engines(std::vector<EngineInfo>& out) const;
    int query_memory_regions(std::vector<MemRegion>& out) const;
    int query_topology(Topology& out) const;
    // Blob HWCONFIG del GuC: pares (clave, valores[]).
    int query_hwconfig(std::vector<std::pair<u32, std::vector<u32>>>& out) const;
    int query_guc_version(u32& branch, u32& major, u32& minor, u32& patch) const;

    // I915_REG_READ (lista blanca del kernel: en la práctica sólo RING_TIMESTAMP de rcs).
    int reg_read(u64 offset, u64* out) const;

    // VM (espacio de direcciones PPGTT) y contexto con mapa de motores + VM propia.
    int vm_create(u32* vm_id) const;
    void vm_destroy(u32 vm_id) const;
    // engines: lista (clase, instancia) → el índice en esta lista es el selector de
    // motor de execbuf. vm_id = 0 → VM por defecto del contexto.
    int context_create(const std::vector<EngineInfo>& engines, u32 vm_id, u32* ctx_id) const;
    void context_destroy(u32 ctx_id) const;
    int context_getparam(u32 ctx_id, u64 param, u64* value) const;
};

// ---- Objeto GEM (buffer) ------------------------------------------------------------
enum class MapMode : u32 { None = 0xFFFFFFFFu, WC = 1, WB = 2, UC = 3 };  // I915_MMAP_OFFSET_*

struct Bo {
    const Device* dev = nullptr;
    u32 handle = 0;
    u64 size = 0;
    u64 gpu_va = 0;           // dirección fija (softpin) en la VM del contexto
    void* map = nullptr;
    MapMode mode = MapMode::None;

    Bo() = default;
    Bo(const Bo&) = delete;
    Bo& operator=(const Bo&) = delete;
    Bo(Bo&& o) noexcept { *this = static_cast<Bo&&>(o); }
    Bo& operator=(Bo&& o) noexcept;
    ~Bo() { destroy(); }

    // I915_GEM_CREATE_EXT (memoria de sistema). pat_index < 0 → PAT por defecto del
    // kernel; ≥ 0 → extensión I915_GEM_CREATE_EXT_SET_PAT (sólo MTL+).
    int create(const Device& d, u64 bytes, int pat_index = -1);
    // I915_GEM_MMAP_OFFSET + mmap(MAP_SHARED) del nodo DRM.
    int mmap(MapMode m);
    void destroy();

    u32* u32p() const { return static_cast<u32*>(map); }
};

// ---- Envío -------------------------------------------------------------------------
struct ExecObj { const Bo* bo; bool write; };

// DRM_IOCTL_I915_GEM_EXECBUFFER2 con softpin (EXEC_OBJECT_PINNED, sin relocalizaciones:
// en Xe-HPG/Xe-LPG el kernel ya no admite relocs). El batch va en el último objeto.
// engine_idx = índice en el mapa de motores del contexto. batch_offset = inicio del lote
// dentro de su BO (múltiplo de 8): permite tener muchos lotes distintos en un solo BO.
int execbuf(const Device& d, u32 ctx_id, u32 engine_idx, const ExecObj* objs, u32 n_objs,
            u32 batch_len_bytes, u32 batch_offset = 0);

// I915_GEM_WAIT: espera a que el objeto quede ocioso. 0 = listo, -ETIME = timeout.
int gem_wait(const Device& d, u32 handle, i64 timeout_ns);

// ---- Constructor de batch (comandos MI_* / BLT documentados, PRM Gen12) --------------
// Todas las direcciones son virtuales de la PPGTT (48 bits, softpin).
struct Batch {
    u32* p = nullptr;   // mapeo CPU del BO del batch
    u32 n = 0;          // dwords escritos
    u32 cap = 0;        // capacidad en dwords

    void dw(u32 v) { if (n < cap) p[n] = v; ++n; }
    bool overflow() const { return n > cap; }

    void mi_noop() { dw(0); }
    // MI_STORE_DATA_IMM (0x20): escribe un dword inmediato en addr.
    void mi_store_dword(u64 addr, u32 v);
    // MI_STORE_REGISTER_MEM (0x24): copia un registro MMIO de 32 bits a memoria.
    void mi_store_reg(u32 reg, u64 addr);
    // MI_COPY_MEM_MEM (0x2E): copia un dword memoria→memoria desde el CS.
    void mi_copy_mem_mem(u64 dst, u64 src);
    // MI_FLUSH_DW (0x26): barrera/flush del motor de copia (sin post-sync).
    void mi_flush_dw();
    // XY_FAST_COPY_BLT (cliente 2D, 0x42), superficies lineales de 32 bpp.
    // pitch en bytes (< 65536), w/h en píxeles (< 65536).
    void xy_fast_copy(u64 dst, u64 src, u32 pitch, u32 w, u32 h);
    // MI_BATCH_BUFFER_END (0x0A) + relleno a múltiplo de 8 bytes.
    void end();
    u32 bytes() const { return n * 4u; }
};

// Conversión de ticks de RING_TIMESTAMP a ns (freq de I915_PARAM_CS_TIMESTAMP_FREQUENCY).
inline double ticks_to_ns(i64 ticks, int freq_hz) { return double(ticks) * 1e9 / double(freq_hz); }

// Texto del errno (strerror) para mensajes.
std::string errstr(int neg_errno);

}  // namespace cfd::gpu::drm
