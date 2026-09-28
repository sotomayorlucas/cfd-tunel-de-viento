// ============================================================================
//  gpu/drm_i915.cpp — ioctl DRM/i915 crudos (ver drm_i915.hpp y docs/DRM.md).
// ============================================================================
#include "gpu/drm_i915.hpp"

#include <drm/drm.h>
#include <drm/i915_drm.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace cfd::gpu::drm {

// ---- Comprobación del ABI (x86-64) ---------------------------------------------------
// Si la cabecera UAPI instalada no coincide con estos tamaños (los del kernel 6.x/7.x),
// preferimos no compilar a pasar estructuras corruptas al kernel.
static_assert(sizeof(drm_version) == 64);
static_assert(sizeof(drm_gem_close) == 8);
static_assert(sizeof(drm_i915_getparam_t) == 16);
static_assert(sizeof(i915_user_extension) == 32);
static_assert(sizeof(drm_i915_gem_create_ext) == 24);
static_assert(sizeof(drm_i915_gem_create_ext_set_pat) == 40);
static_assert(sizeof(drm_i915_gem_mmap_offset) == 32);
static_assert(sizeof(drm_i915_gem_exec_object2) == 56);
static_assert(sizeof(drm_i915_gem_execbuffer2) == 64);
static_assert(sizeof(drm_i915_gem_wait) == 16);
static_assert(sizeof(drm_i915_query) == 16);
static_assert(sizeof(drm_i915_query_item) == 24);
static_assert(sizeof(drm_i915_query_topology_info) == 16);
static_assert(sizeof(drm_i915_engine_info) == 56);
static_assert(sizeof(drm_i915_query_engine_info) == 16);
static_assert(sizeof(drm_i915_memory_region_info) == 88);
static_assert(sizeof(drm_i915_query_memory_regions) == 16);
static_assert(sizeof(drm_i915_reg_read) == 16);
static_assert(sizeof(drm_i915_gem_vm_control) == 16);
static_assert(sizeof(drm_i915_gem_context_param) == 24);
static_assert(sizeof(drm_i915_gem_context_create_ext) == 16);
static_assert(sizeof(drm_i915_gem_context_create_ext_setparam) == 56);
static_assert(sizeof(i915_engine_class_instance) == 4);
static_assert(offsetof(drm_i915_gem_execbuffer2, flags) == 40);
static_assert(offsetof(drm_i915_gem_exec_object2, offset) == 24);

std::string errstr(int neg_errno) {
    const int e = neg_errno < 0 ? -neg_errno : neg_errno;
    std::string s = std::strerror(e);
    s += " (errno ";
    s += std::to_string(e);
    s += ")";
    return s;
}

const char* engine_class_name(u16 cls) {
    switch (cls) {
        case I915_ENGINE_CLASS_RENDER: return "rcs";
        case I915_ENGINE_CLASS_COPY: return "bcs";
        case I915_ENGINE_CLASS_VIDEO: return "vcs";
        case I915_ENGINE_CLASS_VIDEO_ENHANCE: return "vecs";
        case I915_ENGINE_CLASS_COMPUTE: return "ccs";
        default: return "?";
    }
}

// Bases MMIO Gen12+ (i915: RENDER_RING_BASE, BLT_RING_BASE, GEN12_COMPUTE0_RING_BASE,
// GEN11_BSD_RING_BASE, GEN11_VEBOX_RING_BASE). En MTL vcs/vecs viven en la GT de
// medios (gt1) con un desplazamiento adicional 0x380000.
u32 engine_mmio_base(u16 cls, u16 instance) {
    if (instance != 0) return 0;
    switch (cls) {
        case I915_ENGINE_CLASS_RENDER: return 0x02000;
        case I915_ENGINE_CLASS_COPY: return 0x22000;
        case I915_ENGINE_CLASS_COMPUTE: return 0x1a000;
        default: return 0;
    }
}

// ---- Device -----------------------------------------------------------------------
int Device::ioctl(unsigned long req, void* arg) const {
    for (;;) {
        const int r = ::ioctl(fd, req, arg);
        if (r != -1) return 0;
        if (errno == EINTR || errno == EAGAIN) continue;
        return -errno;
    }
}

bool Device::open(const char* path) {
    close();
    fd = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        const int e = errno;
        err = std::string("no se puede abrir ") + path + ": " + errstr(-e);
        if (e == EACCES || e == EPERM)
            err += " — hace falta pertenecer al grupo 'render' (o una ACL en el nodo)";
        return false;
    }
    char name[64] = {}, date[64] = {}, desc[256] = {};
    drm_version v{};
    v.name = name; v.name_len = sizeof name - 1;
    v.date = date; v.date_len = sizeof date - 1;
    v.desc = desc; v.desc_len = sizeof desc - 1;
    if (int r = ioctl(DRM_IOCTL_VERSION, &v); r) {
        err = "DRM_IOCTL_VERSION falló: " + errstr(r);
        close();
        return false;
    }
    drv_name = name; drv_date = date; drv_desc = desc;
    ver_major = v.version_major; ver_minor = v.version_minor; ver_patch = v.version_patchlevel;
    if (drv_name != "i915") {
        err = "el driver del nodo es '" + drv_name + "', no 'i915' (¿xe?): esta ruta sólo habla i915";
        close();
        return false;
    }
    err.clear();
    return true;
}

void Device::close() {
    if (fd >= 0) ::close(fd);
    fd = -1;
}

int Device::getparam(int param, int* out) const {
    int v = 0;
    drm_i915_getparam_t gp{};
    gp.param = param;
    gp.value = &v;
    const int r = ioctl(DRM_IOCTL_I915_GETPARAM, &gp);
    if (r == 0) *out = v;
    return r;
}

int Device::query(u64 query_id, u32 flags, std::vector<u8>& out) const {
    drm_i915_query_item item{};
    item.query_id = query_id;
    item.flags = flags;
    drm_i915_query q{};
    q.num_items = 1;
    q.items_ptr = reinterpret_cast<u64>(&item);
    if (int r = ioctl(DRM_IOCTL_I915_QUERY, &q); r) return r;
    if (item.length < 0) return item.length;   // -EINVAL/-ENODEV por item
    if (item.length == 0) return -ENODATA;
    out.assign(static_cast<size_t>(item.length), 0);
    item.data_ptr = reinterpret_cast<u64>(out.data());
    if (int r = ioctl(DRM_IOCTL_I915_QUERY, &q); r) return r;
    if (item.length < 0) return item.length;
    out.resize(static_cast<size_t>(item.length));
    return 0;
}

int Device::query_engines(std::vector<EngineInfo>& out) const {
    std::vector<u8> buf;
    if (int r = query(DRM_I915_QUERY_ENGINE_INFO, 0, buf); r) return r;
    drm_i915_query_engine_info hdr;
    std::memcpy(&hdr, buf.data(), sizeof hdr);
    out.clear();
    for (u32 i = 0; i < hdr.num_engines; ++i) {
        const size_t off = sizeof hdr + i * sizeof(drm_i915_engine_info);
        if (off + sizeof(drm_i915_engine_info) > buf.size()) break;
        drm_i915_engine_info e;
        std::memcpy(&e, buf.data() + off, sizeof e);
        EngineInfo ei;
        ei.cls = e.engine.engine_class;
        ei.instance = e.engine.engine_instance;
        ei.logical = (e.flags & I915_ENGINE_INFO_HAS_LOGICAL_INSTANCE) ? e.logical_instance : ei.instance;
        ei.caps = e.capabilities;
        out.push_back(ei);
    }
    return 0;
}

int Device::query_memory_regions(std::vector<MemRegion>& out) const {
    std::vector<u8> buf;
    if (int r = query(DRM_I915_QUERY_MEMORY_REGIONS, 0, buf); r) return r;
    drm_i915_query_memory_regions hdr;
    std::memcpy(&hdr, buf.data(), sizeof hdr);
    out.clear();
    for (u32 i = 0; i < hdr.num_regions; ++i) {
        const size_t off = sizeof hdr + i * sizeof(drm_i915_memory_region_info);
        if (off + sizeof(drm_i915_memory_region_info) > buf.size()) break;
        drm_i915_memory_region_info m;
        std::memcpy(&m, buf.data() + off, sizeof m);
        out.push_back({m.region.memory_class, m.region.memory_instance, m.probed_size,
                       m.unallocated_size});
    }
    return 0;
}

int Device::query_topology(Topology& t) const {
    std::vector<u8> buf;
    t = Topology{};
    if (int r = query(DRM_I915_QUERY_TOPOLOGY_INFO, 0, buf); r) return r;
    drm_i915_query_topology_info h;
    std::memcpy(&h, buf.data(), sizeof h);
    const u8* data = buf.data() + sizeof h;
    const size_t dn = buf.size() - sizeof h;
    auto bit = [&](size_t byte, int b) { return byte < dn && ((data[byte] >> b) & 1); };
    t.max_slices = h.max_slices;
    t.max_subslices = h.max_subslices;
    t.max_eus_per_subslice = h.max_eus_per_subslice;
    for (int s = 0; s < h.max_slices; ++s) {
        if (!bit(size_t(s / 8), s % 8)) continue;
        ++t.slices;
        for (int ss = 0; ss < h.max_subslices; ++ss) {
            if (!bit(h.subslice_offset + size_t(s) * h.subslice_stride + size_t(ss / 8), ss % 8)) continue;
            ++t.subslices;
            for (int eu = 0; eu < h.max_eus_per_subslice; ++eu) {
                const size_t base = h.eu_offset + (size_t(s) * h.max_subslices + size_t(ss)) * h.eu_stride;
                if (bit(base + size_t(eu / 8), eu % 8)) ++t.eus;
            }
        }
    }
    t.ok = true;
    return 0;
}

int Device::query_hwconfig(std::vector<std::pair<u32, std::vector<u32>>>& out) const {
    std::vector<u8> buf;
    if (int r = query(DRM_I915_QUERY_HWCONFIG_BLOB, 0, buf); r) return r;
    out.clear();
    const size_t nd = buf.size() / 4;
    std::vector<u32> d(nd);
    std::memcpy(d.data(), buf.data(), nd * 4);
    // Formato KLV: clave, longitud (en dwords), valores[longitud].
    for (size_t i = 0; i + 2 <= nd;) {
        const u32 key = d[i], len = d[i + 1];
        if (i + 2 + len > nd) break;
        out.emplace_back(key, std::vector<u32>(d.begin() + long(i + 2), d.begin() + long(i + 2 + len)));
        i += 2 + len;
    }
    return 0;
}

int Device::query_guc_version(u32& branch, u32& major, u32& minor, u32& patch) const {
    std::vector<u8> buf;
    if (int r = query(DRM_I915_QUERY_GUC_SUBMISSION_VERSION, 0, buf); r) return r;
    if (buf.size() < sizeof(drm_i915_query_guc_submission_version)) return -ENODATA;
    drm_i915_query_guc_submission_version v;
    std::memcpy(&v, buf.data(), sizeof v);
    branch = v.branch; major = v.major; minor = v.minor; patch = v.patch;
    return 0;
}

int Device::reg_read(u64 offset, u64* out) const {
    drm_i915_reg_read rr{};
    rr.offset = offset;
    const int r = ioctl(DRM_IOCTL_I915_REG_READ, &rr);
    if (r == 0) *out = rr.val;
    return r;
}

int Device::vm_create(u32* vm_id) const {
    drm_i915_gem_vm_control c{};
    const int r = ioctl(DRM_IOCTL_I915_GEM_VM_CREATE, &c);
    if (r == 0) *vm_id = c.vm_id;
    return r;
}

void Device::vm_destroy(u32 vm_id) const {
    drm_i915_gem_vm_control c{};
    c.vm_id = vm_id;
    (void)ioctl(DRM_IOCTL_I915_GEM_VM_DESTROY, &c);
}

int Device::context_create(const std::vector<EngineInfo>& engines, u32 vm_id, u32* ctx_id) const {
    // i915_context_param_engines = { u64 extensions; i915_engine_class_instance[N] }.
    // Se construye a mano (la macro I915_DEFINE_CONTEXT_PARAM_ENGINES es C).
    std::vector<u8> eng(8 + 4 * engines.size(), 0);
    for (size_t i = 0; i < engines.size(); ++i) {
        const i915_engine_class_instance ci{engines[i].cls, engines[i].instance};
        std::memcpy(eng.data() + 8 + 4 * i, &ci, 4);
    }
    drm_i915_gem_context_create_ext_setparam p_eng{};
    p_eng.base.name = I915_CONTEXT_CREATE_EXT_SETPARAM;
    p_eng.param.param = I915_CONTEXT_PARAM_ENGINES;
    p_eng.param.size = u32(eng.size());
    p_eng.param.value = reinterpret_cast<u64>(eng.data());

    drm_i915_gem_context_create_ext_setparam p_vm{};
    p_vm.base.name = I915_CONTEXT_CREATE_EXT_SETPARAM;
    p_vm.param.param = I915_CONTEXT_PARAM_VM;
    p_vm.param.value = vm_id;

    if (vm_id) p_eng.base.next_extension = reinterpret_cast<u64>(&p_vm);

    drm_i915_gem_context_create_ext c{};
    c.flags = I915_CONTEXT_CREATE_FLAGS_USE_EXTENSIONS;
    c.extensions = engines.empty() ? (vm_id ? reinterpret_cast<u64>(&p_vm) : 0)
                                   : reinterpret_cast<u64>(&p_eng);
    const int r = ioctl(DRM_IOCTL_I915_GEM_CONTEXT_CREATE_EXT, &c);
    if (r == 0) *ctx_id = c.ctx_id;
    return r;
}

void Device::context_destroy(u32 ctx_id) const {
    drm_i915_gem_context_destroy d{};
    d.ctx_id = ctx_id;
    (void)ioctl(DRM_IOCTL_I915_GEM_CONTEXT_DESTROY, &d);
}

int Device::context_getparam(u32 ctx_id, u64 param, u64* value) const {
    drm_i915_gem_context_param p{};
    p.ctx_id = ctx_id;
    p.param = param;
    const int r = ioctl(DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM, &p);
    if (r == 0) *value = p.value;
    return r;
}

// ---- Bo ---------------------------------------------------------------------------
Bo& Bo::operator=(Bo&& o) noexcept {
    if (this != &o) {
        destroy();
        dev = o.dev; handle = o.handle; size = o.size; gpu_va = o.gpu_va; map = o.map; mode = o.mode;
        o.dev = nullptr; o.handle = 0; o.map = nullptr; o.size = 0; o.mode = MapMode::None;
    }
    return *this;
}

int Bo::create(const Device& d, u64 bytes, int pat_index) {
    destroy();
    drm_i915_gem_create_ext_set_pat pat{};
    pat.base.name = I915_GEM_CREATE_EXT_SET_PAT;
    pat.pat_index = pat_index < 0 ? 0u : u32(pat_index);
    drm_i915_gem_create_ext c{};
    c.size = (bytes + 4095) & ~u64(4095);
    c.extensions = pat_index >= 0 ? reinterpret_cast<u64>(&pat) : 0;
    const int r = d.ioctl(DRM_IOCTL_I915_GEM_CREATE_EXT, &c);
    if (r) return r;
    dev = &d;
    handle = c.handle;
    size = c.size;
    return 0;
}

int Bo::mmap(MapMode m) {
    if (!dev || !handle) return -EINVAL;
    if (map) { ::munmap(map, size); map = nullptr; }
    drm_i915_gem_mmap_offset mo{};
    mo.handle = handle;
    mo.flags = u64(m);
    if (int r = dev->ioctl(DRM_IOCTL_I915_GEM_MMAP_OFFSET, &mo); r) return r;
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, off_t(mo.offset));
    if (p == MAP_FAILED) return -errno;
    map = p;
    mode = m;
    return 0;
}

void Bo::destroy() {
    if (map) ::munmap(map, size);
    map = nullptr;
    if (dev && handle) {
        drm_gem_close c{};
        c.handle = handle;
        (void)dev->ioctl(DRM_IOCTL_GEM_CLOSE, &c);
    }
    handle = 0;
    dev = nullptr;
}

// ---- Envío ------------------------------------------------------------------------
int execbuf(const Device& d, u32 ctx_id, u32 engine_idx, const ExecObj* objs, u32 n_objs,
            u32 batch_len_bytes, u32 batch_offset) {
    constexpr u32 kMax = 16;
    if (n_objs == 0 || n_objs > kMax) return -EINVAL;
    drm_i915_gem_exec_object2 eo[kMax] = {};
    for (u32 i = 0; i < n_objs; ++i) {
        eo[i].handle = objs[i].bo->handle;
        eo[i].offset = objs[i].bo->gpu_va;
        eo[i].flags = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS |
                      (objs[i].write ? EXEC_OBJECT_WRITE : 0);
    }
    drm_i915_gem_execbuffer2 eb{};
    eb.buffers_ptr = reinterpret_cast<u64>(eo);
    eb.buffer_count = n_objs;
    eb.batch_start_offset = batch_offset;
    eb.batch_len = batch_len_bytes;
    eb.flags = u64(engine_idx & I915_EXEC_RING_MASK) | I915_EXEC_NO_RELOC;
    i915_execbuffer2_set_context_id(eb, ctx_id);
    return d.ioctl(DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb);
}

int gem_wait(const Device& d, u32 handle, i64 timeout_ns) {
    drm_i915_gem_wait w{};
    w.bo_handle = handle;
    w.timeout_ns = timeout_ns;
    return d.ioctl(DRM_IOCTL_I915_GEM_WAIT, &w);
}

// ---- Batch ------------------------------------------------------------------------
// MI_INSTR(op, len) = op << 23 | len, con len = dwords totales − 2.
static constexpr u32 mi(u32 op, u32 len) { return (op << 23) | len; }

void Batch::mi_store_dword(u64 addr, u32 v) {
    dw(mi(0x20, 2));
    dw(u32(addr)); dw(u32(addr >> 32));
    dw(v);
}

void Batch::mi_store_reg(u32 reg, u64 addr) {
    dw(mi(0x24, 2));
    dw(reg);
    dw(u32(addr)); dw(u32(addr >> 32));
}

void Batch::mi_copy_mem_mem(u64 dst, u64 src) {
    dw(mi(0x2E, 3));
    dw(u32(dst)); dw(u32(dst >> 32));
    dw(u32(src)); dw(u32(src >> 32));
}

void Batch::mi_flush_dw() {
    dw(mi(0x26, 2));
    dw(0); dw(0); dw(0);
}

void Batch::xy_fast_copy(u64 dst, u64 src, u32 pitch, u32 w, u32 h) {
    // DW0: cliente 2 (2D) | opcode 0x42 | longitud 10−2; tiling lineal en src y dst.
    dw((2u << 29) | (0x42u << 22) | 8u);
    dw((3u << 24) | (pitch & 0xFFFFu));      // 32 bpp | pitch destino (bytes)
    dw(0);                                   // y1<<16 | x1 destino
    dw((h << 16) | (w & 0xFFFFu));           // y2<<16 | x2 destino (exclusivo)
    dw(u32(dst)); dw(u32(dst >> 32));
    dw(0);                                   // y1<<16 | x1 origen
    dw(pitch & 0xFFFFu);                     // pitch origen
    dw(u32(src)); dw(u32(src >> 32));
}

void Batch::end() {
    dw(mi(0x0A, 0));                         // MI_BATCH_BUFFER_END
    if (n & 1) mi_noop();                    // longitud múltiplo de 8 bytes
}

}  // namespace cfd::gpu::drm
