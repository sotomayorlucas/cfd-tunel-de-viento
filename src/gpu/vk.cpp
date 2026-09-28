// ============================================================================
//  gpu/vk.cpp — carga dinámica de Vulkan y contexto de cómputo (ver vk.hpp).
// ============================================================================
#include "vk.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace cfd::gpu::vk {

namespace {

template <class F>
bool load_inst(const Fns& f, VkInstance inst, const char* name, F& out) {
    out = reinterpret_cast<F>(f.GetInstanceProcAddr(inst, name));
    return out != nullptr;
}
template <class F>
bool load_dev(const Fns& f, VkDevice dev, const char* name, F& out) {
    out = reinterpret_cast<F>(f.GetDeviceProcAddr(dev, name));
    return out != nullptr;
}

void set_err(std::string* err, const char* msg) { if (err) *err = msg; }

} // namespace

Context::~Context() { destroy(); }

void Context::destroy() {
    if (dev_) {
        f_.DeviceWaitIdle(dev_);
        if (qpool_) f_.DestroyQueryPool(dev_, qpool_, nullptr);
        if (tl_) f_.DestroySemaphore(dev_, tl_, nullptr);
        if (dpool_) f_.DestroyDescriptorPool(dev_, dpool_, nullptr);
        if (pool_) f_.DestroyCommandPool(dev_, pool_, nullptr);
        f_.DestroyDevice(dev_, nullptr);
    }
    if (inst_ && f_.DestroyInstance) f_.DestroyInstance(inst_, nullptr);
    if (lib_) dlclose(lib_);
    lib_ = nullptr; inst_ = nullptr; pd_ = nullptr; dev_ = nullptr; queue_ = nullptr;
    pool_ = dpool_ = tl_ = qpool_ = 0;
    qcount_ = 0; tl_value_ = 0;
    f_ = Fns{};
}

bool Context::init(std::string* err, bool verbose) {
    destroy();
    // ---- Cargador ----
    lib_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib_) { set_err(err, "no se encontró libvulkan.so.1"); return false; }
    f_.GetInstanceProcAddr = reinterpret_cast<decltype(f_.GetInstanceProcAddr)>(dlsym(lib_, "vkGetInstanceProcAddr"));
    if (!f_.GetInstanceProcAddr) { set_err(err, "libvulkan sin vkGetInstanceProcAddr"); destroy(); return false; }
    load_inst(f_, nullptr, "vkEnumerateInstanceVersion", f_.EnumerateInstanceVersion);
    if (!load_inst(f_, nullptr, "vkCreateInstance", f_.CreateInstance)) { set_err(err, "sin vkCreateInstance"); destroy(); return false; }
    u32 iver = vk_api(1, 0);
    if (f_.EnumerateInstanceVersion) f_.EnumerateInstanceVersion(&iver);
    if (iver < vk_api(1, 3)) { set_err(err, "el cargador de Vulkan no llega a 1.3"); destroy(); return false; }

    // ---- Instancia (sin capas ni extensiones) ----
    VkApplicationInfo app{ST_APPLICATION_INFO, nullptr, "cfd-tunel", 1, "cfd", 1, vk_api(1, 3)};
    VkInstanceCreateInfo ici{ST_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    if (f_.CreateInstance(&ici, nullptr, &inst_) != VK_SUCCESS) { set_err(err, "vkCreateInstance falló"); inst_ = nullptr; destroy(); return false; }
    bool okf = true;
    okf &= load_inst(f_, inst_, "vkDestroyInstance", f_.DestroyInstance);
    okf &= load_inst(f_, inst_, "vkEnumeratePhysicalDevices", f_.EnumeratePhysicalDevices);
    okf &= load_inst(f_, inst_, "vkGetPhysicalDeviceProperties2", f_.GetPhysicalDeviceProperties2);
    okf &= load_inst(f_, inst_, "vkGetPhysicalDeviceFeatures2", f_.GetPhysicalDeviceFeatures2);
    okf &= load_inst(f_, inst_, "vkGetPhysicalDeviceQueueFamilyProperties", f_.GetPhysicalDeviceQueueFamilyProperties);
    okf &= load_inst(f_, inst_, "vkGetPhysicalDeviceMemoryProperties", f_.GetPhysicalDeviceMemoryProperties);
    okf &= load_inst(f_, inst_, "vkEnumerateDeviceExtensionProperties", f_.EnumerateDeviceExtensionProperties);
    okf &= load_inst(f_, inst_, "vkCreateDevice", f_.CreateDevice);
    okf &= load_inst(f_, inst_, "vkGetDeviceProcAddr", f_.GetDeviceProcAddr);
    if (!okf) { set_err(err, "faltan funciones de instancia"); destroy(); return false; }

    // ---- Dispositivo físico: iGPU Intel (se descarta llvmpipe y cualquier CPU) ----
    u32 npd = 0;
    f_.EnumeratePhysicalDevices(inst_, &npd, nullptr);
    std::vector<VkPhysicalDevice> pds(npd);
    if (npd) f_.EnumeratePhysicalDevices(inst_, &npd, pds.data());
    int best = -1, best_score = -1;
    for (u32 i = 0; i < npd; ++i) {
        VkPhysicalDeviceProperties2 p{};
        p.sType = ST_PHYSICAL_DEVICE_PROPERTIES_2;
        f_.GetPhysicalDeviceProperties2(pds[i], &p);
        const auto& P = p.properties;
        if (verbose) std::fprintf(stderr, "[gpu] dispositivo %u: %s (tipo %u, vendor 0x%04x, API %u.%u)\n", i, P.deviceName, P.deviceType,
                                  P.vendorID, P.apiVersion >> 22, (P.apiVersion >> 12) & 1023);
        if (P.deviceType == PHYSICAL_DEVICE_TYPE_CPU || P.apiVersion < vk_api(1, 3)) continue;
        int score = 1;
        if (P.deviceType == PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 4;
        if (P.vendorID == 0x8086) score += 2;
        if (score > best_score) { best_score = score; best = static_cast<int>(i); }
    }
    if (best < 0) { set_err(err, "no hay GPU Vulkan 1.3 (sólo CPU/llvmpipe)"); destroy(); return false; }
    pd_ = pds[static_cast<usize>(best)];

    // ---- Propiedades ----
    VkPhysicalDeviceFloatControlsProperties fc{};
    fc.sType = ST_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlProperties scp{};
    scp.sType = ST_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    scp.pNext = &fc;
    VkPhysicalDeviceSubgroupProperties sgp{};
    sgp.sType = ST_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    sgp.pNext = &scp;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = ST_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sgp;
    f_.GetPhysicalDeviceProperties2(pd_, &p2);
    const auto& P = p2.properties;
    std::memcpy(info_.name, P.deviceName, sizeof info_.name);
    info_.vendor = P.vendorID; info_.device = P.deviceID; info_.api = P.apiVersion; info_.driver = P.driverVersion; info_.type = P.deviceType;
    info_.timestamp_period_ns = P.limits.timestampPeriod;
    info_.max_push = P.limits.maxPushConstantsSize;
    info_.max_shared = P.limits.maxComputeSharedMemorySize;
    info_.max_wg_invocations = P.limits.maxComputeWorkGroupInvocations;
    for (int a = 0; a < 3; ++a) info_.max_wg_count[a] = P.limits.maxComputeWorkGroupCount[a];
    info_.max_storage_range = P.limits.maxStorageBufferRange;
    info_.storage_align = P.limits.minStorageBufferOffsetAlignment;
    info_.max_storage_buffers = P.limits.maxPerStageDescriptorStorageBuffers;
    info_.subgroup_size = sgp.subgroupSize;
    info_.subgroup_ops = sgp.supportedOperations;
    info_.sg_min = scp.minSubgroupSize;
    info_.sg_max = scp.maxSubgroupSize;
    info_.max_wg_subgroups = scp.maxComputeWorkgroupSubgroups;
    info_.sg_control = (scp.requiredSubgroupSizeStages & SHADER_STAGE_COMPUTE_BIT) != 0;
    info_.rte16 = fc.shaderRoundingModeRTEFloat16 != 0;
    info_.denorm_preserve16 = fc.shaderDenormPreserveFloat16 != 0;
    // Comprobación de cordura del ABI escrito a mano: valores imposibles → estructura mal declarada.
    if (!(info_.timestamp_period_ns > 0.0f && info_.timestamp_period_ns < 1e6f) || info_.max_wg_invocations < 128 ||
        info_.max_wg_invocations > 65536 || info_.subgroup_size == 0 || info_.subgroup_size > 128) {
        set_err(err, "propiedades del dispositivo incoherentes (¿ABI?)");
        destroy();
        return false;
    }

    // ---- Características: consulta y activación (sólo lo que se usa) ----
    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = ST_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = ST_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &f13;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = ST_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = ST_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    f_.GetPhysicalDeviceFeatures2(pd_, &f2);
    info_.f16 = f12.v[F12_shaderFloat16];
    info_.i8 = f12.v[F12_shaderInt8];
    info_.i16 = f2.features.v[FEAT_shaderInt16];
    info_.s16 = f11.storageBuffer16BitAccess;
    info_.s8 = f12.v[F12_storageBuffer8BitAccess];
    info_.timeline = f12.v[F12_timelineSemaphore];
    info_.host_query_reset = f12.v[F12_hostQueryReset];
    info_.sync2 = f13.synchronization2;
    info_.full_subgroups = f13.computeFullSubgroups;
    info_.sg_control = info_.sg_control && f13.subgroupSizeControl;
    if (!info_.timeline) { set_err(err, "el dispositivo no tiene semáforos de línea temporal"); destroy(); return false; }
    if (!info_.s16 || !info_.s8) { set_err(err, "el dispositivo no admite búferes de 8/16 bits"); destroy(); return false; }

    VkPhysicalDeviceVulkan13Features e13{};
    e13.sType = ST_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    e13.subgroupSizeControl = info_.sg_control;
    e13.computeFullSubgroups = info_.full_subgroups;
    e13.synchronization2 = info_.sync2;
    VkPhysicalDeviceVulkan12Features e12{};
    e12.sType = ST_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    e12.pNext = &e13;
    e12.v[F12_shaderFloat16] = info_.f16;
    e12.v[F12_shaderInt8] = info_.i8;
    e12.v[F12_storageBuffer8BitAccess] = 1;
    e12.v[F12_timelineSemaphore] = 1;
    e12.v[F12_hostQueryReset] = info_.host_query_reset;
    VkPhysicalDeviceVulkan11Features e11{};
    e11.sType = ST_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    e11.pNext = &e12;
    e11.storageBuffer16BitAccess = 1;
    VkPhysicalDeviceFeatures2 e2{};
    e2.sType = ST_PHYSICAL_DEVICE_FEATURES_2;
    e2.pNext = &e11;
    e2.features.v[FEAT_shaderInt16] = info_.i16;

    // ---- Cola: la primera familia con cómputo ----
    u32 nqf = 0;
    f_.GetPhysicalDeviceQueueFamilyProperties(pd_, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    f_.GetPhysicalDeviceQueueFamilyProperties(pd_, &nqf, qfs.data());
    int qf = -1;
    for (u32 i = 0; i < nqf; ++i)
        if (qfs[i].queueFlags & QUEUE_COMPUTE_BIT) { qf = static_cast<int>(i); break; }
    if (qf < 0) { set_err(err, "sin cola de cómputo"); destroy(); return false; }
    qf_ = static_cast<u32>(qf);
    info_.timestamp_bits = qfs[qf_].timestampValidBits;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ST_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qf_, 1, &prio};
    VkDeviceCreateInfo dci{ST_DEVICE_CREATE_INFO, &e2, 0, 1, &qci, 0, nullptr, 0, nullptr, nullptr};
    if (f_.CreateDevice(pd_, &dci, nullptr, &dev_) != VK_SUCCESS) { set_err(err, "vkCreateDevice falló"); dev_ = nullptr; destroy(); return false; }

    // ---- Funciones de dispositivo (directas al controlador: sin el trampolín del cargador) ----
    okf = true;
#define CFD_VKD(name) okf &= load_dev(f_, dev_, "vk" #name, f_.name)
    CFD_VKD(DestroyDevice); CFD_VKD(GetDeviceQueue); CFD_VKD(CreateBuffer); CFD_VKD(DestroyBuffer); CFD_VKD(GetBufferMemoryRequirements);
    CFD_VKD(AllocateMemory); CFD_VKD(FreeMemory); CFD_VKD(BindBufferMemory); CFD_VKD(MapMemory); CFD_VKD(UnmapMemory);
    CFD_VKD(FlushMappedMemoryRanges); CFD_VKD(InvalidateMappedMemoryRanges); CFD_VKD(CreateShaderModule); CFD_VKD(DestroyShaderModule);
    CFD_VKD(CreateDescriptorSetLayout); CFD_VKD(DestroyDescriptorSetLayout); CFD_VKD(CreatePipelineLayout); CFD_VKD(DestroyPipelineLayout);
    CFD_VKD(CreateComputePipelines); CFD_VKD(DestroyPipeline); CFD_VKD(CreateDescriptorPool); CFD_VKD(DestroyDescriptorPool);
    CFD_VKD(AllocateDescriptorSets); CFD_VKD(ResetDescriptorPool); CFD_VKD(UpdateDescriptorSets); CFD_VKD(CreateCommandPool); CFD_VKD(DestroyCommandPool);
    CFD_VKD(AllocateCommandBuffers); CFD_VKD(FreeCommandBuffers); CFD_VKD(BeginCommandBuffer); CFD_VKD(EndCommandBuffer);
    CFD_VKD(ResetCommandBuffer); CFD_VKD(CmdBindPipeline); CFD_VKD(CmdBindDescriptorSets); CFD_VKD(CmdPushConstants); CFD_VKD(CmdDispatch);
    CFD_VKD(CmdPipelineBarrier); CFD_VKD(CmdWriteTimestamp); CFD_VKD(CmdResetQueryPool); CFD_VKD(CmdFillBuffer); CFD_VKD(CmdCopyBuffer);
    CFD_VKD(CreateQueryPool); CFD_VKD(DestroyQueryPool); CFD_VKD(GetQueryPoolResults); CFD_VKD(QueueSubmit); CFD_VKD(QueueWaitIdle);
    CFD_VKD(DeviceWaitIdle); CFD_VKD(CreateSemaphore); CFD_VKD(DestroySemaphore); CFD_VKD(WaitSemaphores); CFD_VKD(GetSemaphoreCounterValue);
#undef CFD_VKD
    if (info_.sync2) load_dev(f_, dev_, "vkCmdPipelineBarrier2", f_.CmdPipelineBarrier2);
    if (!f_.CmdPipelineBarrier2) info_.sync2 = false;
    if (info_.host_query_reset) load_dev(f_, dev_, "vkResetQueryPool", f_.ResetQueryPool);
    if (!okf) { set_err(err, "faltan funciones de dispositivo"); destroy(); return false; }
    f_.GetDeviceQueue(dev_, qf_, 0, &queue_);

    // ---- Memoria: tipos DEVICE_LOCAL|HOST_VISIBLE (UMA) ----
    VkPhysicalDeviceMemoryProperties mp{};
    f_.GetPhysicalDeviceMemoryProperties(pd_, &mp);
    info_.mem_type_count = mp.memoryTypeCount;
    for (u32 i = 0; i < mp.memoryTypeCount && i < 32; ++i) {
        const u32 fl = mp.memoryTypes[i].propertyFlags;
        info_.mem_flags[i] = fl;
        const bool hv = fl & MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if (!hv) continue;
        if ((fl & MEMORY_PROPERTY_HOST_COHERENT_BIT) && type_coherent_ < 0) type_coherent_ = static_cast<int>(i);
        if ((fl & MEMORY_PROPERTY_HOST_CACHED_BIT) && type_cached_ < 0) type_cached_ = static_cast<int>(i);
    }
    for (u32 h = 0; h < mp.memoryHeapCount && h < 16; ++h) info_.heap_size = info_.heap_size > mp.memoryHeaps[h].size ? info_.heap_size : mp.memoryHeaps[h].size;
    if (type_coherent_ < 0 && type_cached_ < 0) { set_err(err, "sin memoria visible desde la CPU"); destroy(); return false; }

    // ---- Pools, semáforo de línea temporal ----
    VkCommandPoolCreateInfo cpi{ST_COMMAND_POOL_CREATE_INFO, nullptr, COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf_};
    if (f_.CreateCommandPool(dev_, &cpi, nullptr, &pool_) != VK_SUCCESS) { set_err(err, "vkCreateCommandPool falló"); destroy(); return false; }
    VkDescriptorPoolSize ps{DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
    VkDescriptorPoolCreateInfo dpi{ST_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 1 /*FREE_DESCRIPTOR_SET*/, 256, 1, &ps};
    if (f_.CreateDescriptorPool(dev_, &dpi, nullptr, &dpool_) != VK_SUCCESS) { set_err(err, "vkCreateDescriptorPool falló"); destroy(); return false; }
    VkSemaphoreTypeCreateInfo sti{ST_SEMAPHORE_TYPE_CREATE_INFO, nullptr, SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo sci{ST_SEMAPHORE_CREATE_INFO, &sti, 0};
    if (f_.CreateSemaphore(dev_, &sci, nullptr, &tl_) != VK_SUCCESS) { set_err(err, "vkCreateSemaphore falló"); destroy(); return false; }
    tl_value_ = 0;
    if (verbose) std::fprintf(stderr, "%s", describe(info_).c_str());
    return true;
}

int Context::mem_type(Mem pref) const {
    if (pref == Mem::Cached) return type_cached_ >= 0 ? type_cached_ : type_coherent_;
    return type_coherent_ >= 0 ? type_coherent_ : type_cached_;
}

bool Context::create_buffer(Buffer& b, u64 size, Mem pref, u32 extra_usage) {
    destroy_buffer(b);
    if (size == 0) size = 16;
    size = (size + 63) & ~u64(63);
    VkBufferCreateInfo bi{ST_BUFFER_CREATE_INFO, nullptr, 0, size,
                          BUFFER_USAGE_STORAGE_BUFFER_BIT | BUFFER_USAGE_TRANSFER_SRC_BIT | BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage,
                          SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (f_.CreateBuffer(dev_, &bi, nullptr, &b.buf) != VK_SUCCESS) { b.buf = 0; return false; }
    VkMemoryRequirements mr{};
    f_.GetBufferMemoryRequirements(dev_, b.buf, &mr);
    int t = mem_type(pref);
    if (t < 0 || !(mr.memoryTypeBits & (1u << t))) {
        t = -1;
        for (u32 i = 0; i < info_.mem_type_count; ++i)
            if ((mr.memoryTypeBits & (1u << i)) && (info_.mem_flags[i] & MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { t = static_cast<int>(i); break; }
    }
    if (t < 0) { destroy_buffer(b); return false; }
    VkMemoryAllocateInfo ai{ST_MEMORY_ALLOCATE_INFO, nullptr, mr.size, static_cast<u32>(t)};
    if (f_.AllocateMemory(dev_, &ai, nullptr, &b.mem) != VK_SUCCESS) { b.mem = 0; destroy_buffer(b); return false; }
    if (f_.BindBufferMemory(dev_, b.buf, b.mem, 0) != VK_SUCCESS) { destroy_buffer(b); return false; }
    if (f_.MapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map) != VK_SUCCESS) { b.map = nullptr; destroy_buffer(b); return false; }
    b.size = size;
    b.type = static_cast<u32>(t);
    b.coherent = info_.mem_flags[t] & MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return true;
}

void Context::destroy_buffer(Buffer& b) {
    if (!dev_) { b = Buffer{}; return; }
    if (b.map) f_.UnmapMemory(dev_, b.mem);
    if (b.buf) f_.DestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) f_.FreeMemory(dev_, b.mem, nullptr);
    b = Buffer{};
}

void Context::flush(const Buffer& b, u64 off, u64 size) const {
    if (b.coherent || !b.mem) return;
    VkMappedMemoryRange r{ST_MAPPED_MEMORY_RANGE, nullptr, b.mem, off & ~u64(63), size};
    if (size != VK_WHOLE_SIZE) r.size = ((off + size + 63) & ~u64(63)) - r.offset;
    f_.FlushMappedMemoryRanges(dev_, 1, &r);
}

void Context::invalidate(const Buffer& b, u64 off, u64 size) const {
    if (b.coherent || !b.mem) return;
    VkMappedMemoryRange r{ST_MAPPED_MEMORY_RANGE, nullptr, b.mem, off & ~u64(63), size};
    if (size != VK_WHOLE_SIZE) r.size = ((off + size + 63) & ~u64(63)) - r.offset;
    f_.InvalidateMappedMemoryRanges(dev_, 1, &r);
}

bool Context::create_pipeline(Pipeline& p, const std::vector<u32>& spirv, u32 nbind, u32 push_bytes, u32 required_subgroup, std::string* err) {
    destroy_pipeline(p);
    std::vector<VkDescriptorSetLayoutBinding> b(nbind);
    for (u32 i = 0; i < nbind; ++i) b[i] = {i, DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, nbind, b.data()};
    if (f_.CreateDescriptorSetLayout(dev_, &dl, nullptr, &p.dsl) != VK_SUCCESS) { set_err(err, "vkCreateDescriptorSetLayout"); return false; }
    VkPushConstantRange pr{SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pl{ST_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &p.dsl, push_bytes ? 1u : 0u, &pr};
    if (f_.CreatePipelineLayout(dev_, &pl, nullptr, &p.layout) != VK_SUCCESS) { set_err(err, "vkCreatePipelineLayout"); destroy_pipeline(p); return false; }
    VkShaderModule mod = 0;
    VkShaderModuleCreateInfo smi{ST_SHADER_MODULE_CREATE_INFO, nullptr, 0, spirv.size() * 4, spirv.data()};
    if (f_.CreateShaderModule(dev_, &smi, nullptr, &mod) != VK_SUCCESS) { set_err(err, "vkCreateShaderModule (SPIR-V rechazado)"); destroy_pipeline(p); return false; }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs{ST_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr, required_subgroup};
    const bool req = required_subgroup != 0 && info_.sg_control;
    VkComputePipelineCreateInfo ci{};
    ci.sType = ST_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = {ST_PIPELINE_SHADER_STAGE_CREATE_INFO, req ? &rs : nullptr,
                (req && info_.full_subgroups) ? u32(PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0u,
                SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    ci.layout = p.layout;
    ci.basePipelineIndex = -1;
    const VkResult r = f_.CreateComputePipelines(dev_, 0, 1, &ci, nullptr, &p.pipe);
    f_.DestroyShaderModule(dev_, mod, nullptr);
    if (r != VK_SUCCESS) {
        if (err) { char b2[96]; std::snprintf(b2, sizeof b2, "vkCreateComputePipelines falló (%d)", r); *err = b2; }
        p.pipe = 0;
        destroy_pipeline(p);
        return false;
    }
    p.nbind = nbind;
    p.push = push_bytes;
    return true;
}

void Context::destroy_pipeline(Pipeline& p) {
    if (dev_) {
        if (p.pipe) f_.DestroyPipeline(dev_, p.pipe, nullptr);
        if (p.layout) f_.DestroyPipelineLayout(dev_, p.layout, nullptr);
        if (p.dsl) f_.DestroyDescriptorSetLayout(dev_, p.dsl, nullptr);
    }
    p = Pipeline{};
}

VkDescriptorSet Context::make_set(const Pipeline& p, const Bind* binds, u32 n) {
    VkDescriptorSet s = 0;
    VkDescriptorSetAllocateInfo ai{ST_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dpool_, 1, &p.dsl};
    if (f_.AllocateDescriptorSets(dev_, &ai, &s) != VK_SUCCESS) return 0;
    std::vector<VkDescriptorBufferInfo> bi(n);
    std::vector<VkWriteDescriptorSet> w(n);
    for (u32 i = 0; i < n; ++i) {
        bi[i] = {binds[i].b->buf, binds[i].off, binds[i].range ? binds[i].range : VK_WHOLE_SIZE};
        w[i] = {ST_WRITE_DESCRIPTOR_SET, nullptr, s, i, 0, 1, DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
    }
    f_.UpdateDescriptorSets(dev_, n, w.data(), 0, nullptr);
    return s;
}

void Context::reset_sets() { if (dev_ && dpool_) f_.ResetDescriptorPool(dev_, dpool_, 0); }

VkCommandBuffer Context::alloc_cmd() {
    VkCommandBuffer c = nullptr;
    VkCommandBufferAllocateInfo ai{ST_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool_, COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (f_.AllocateCommandBuffers(dev_, &ai, &c) != VK_SUCCESS) return nullptr;
    return c;
}
void Context::free_cmd(VkCommandBuffer c) { if (c && dev_) f_.FreeCommandBuffers(dev_, pool_, 1, &c); }

void Context::begin(VkCommandBuffer c, bool one_time) const {
    f_.ResetCommandBuffer(c, 0);
    VkCommandBufferBeginInfo bi{ST_COMMAND_BUFFER_BEGIN_INFO, nullptr, one_time ? u32(COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) : 0u, nullptr};
    f_.BeginCommandBuffer(c, &bi);
}
void Context::end(VkCommandBuffer c) const { f_.EndCommandBuffer(c); }

void Context::barrier_compute(VkCommandBuffer c) const {
    if (info_.sync2) {
        VkMemoryBarrier2 mb{ST_MEMORY_BARRIER_2, nullptr, PIPELINE_STAGE_COMPUTE_SHADER_BIT, ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            PIPELINE_STAGE_COMPUTE_SHADER_BIT, ACCESS_2_SHADER_STORAGE_READ_BIT | ACCESS_2_SHADER_STORAGE_WRITE_BIT};
        VkDependencyInfo di{ST_DEPENDENCY_INFO, nullptr, 0, 1, &mb, 0, nullptr, 0, nullptr};
        f_.CmdPipelineBarrier2(c, &di);
    } else {
        VkMemoryBarrier mb{ST_MEMORY_BARRIER, nullptr, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT | ACCESS_SHADER_WRITE_BIT};
        f_.CmdPipelineBarrier(c, PIPELINE_STAGE_COMPUTE_SHADER_BIT, PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
}

void Context::barrier_host(VkCommandBuffer c) const {
    VkMemoryBarrier mb{ST_MEMORY_BARRIER, nullptr, ACCESS_SHADER_WRITE_BIT | ACCESS_TRANSFER_WRITE_BIT, ACCESS_HOST_READ_BIT};
    f_.CmdPipelineBarrier(c, PIPELINE_STAGE_COMPUTE_SHADER_BIT | PIPELINE_STAGE_TRANSFER_BIT, PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

u64 Context::submit(VkCommandBuffer c) { return submit(&c, 1); }

u64 Context::submit(const VkCommandBuffer* cs, u32 n) {
    const u64 v = tl_value_ + 1;
    VkTimelineSemaphoreSubmitInfo ts{ST_TIMELINE_SEMAPHORE_SUBMIT_INFO, nullptr, 0, nullptr, 1, &v};
    VkSubmitInfo si{ST_SUBMIT_INFO, &ts, 0, nullptr, nullptr, n, cs, 1, &tl_};
    if (f_.QueueSubmit(queue_, 1, &si, 0) != VK_SUCCESS) return 0;
    tl_value_ = v;
    return v;
}

bool Context::wait(u64 value, u64 timeout_ns) const {
    if (value == 0) return true;
    VkSemaphoreWaitInfo wi{ST_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &tl_, &value};
    return f_.WaitSemaphores(dev_, &wi, timeout_ns) == VK_SUCCESS;
}

u64 Context::completed() const {
    u64 v = 0;
    f_.GetSemaphoreCounterValue(dev_, tl_, &v);
    return v;
}

void Context::wait_idle() const { if (dev_) f_.DeviceWaitIdle(dev_); }

VkQueryPool Context::timestamps(u32 n) {
    if (qpool_ && qcount_ >= n) return qpool_;
    if (qpool_) { f_.DestroyQueryPool(dev_, qpool_, nullptr); qpool_ = 0; }
    VkQueryPoolCreateInfo qi{ST_QUERY_POOL_CREATE_INFO, nullptr, 0, QUERY_TYPE_TIMESTAMP, n, 0};
    if (f_.CreateQueryPool(dev_, &qi, nullptr, &qpool_) != VK_SUCCESS) { qpool_ = 0; return 0; }
    qcount_ = n;
    if (f_.ResetQueryPool) f_.ResetQueryPool(dev_, qpool_, 0, n);
    return qpool_;
}

bool Context::read_timestamps(u32 first, u32 n, double* ns) const {
    if (!qpool_ || first + n > qcount_) return false;
    std::vector<u64> t(n);
    if (f_.GetQueryPoolResults(dev_, qpool_, first, n, n * 8, t.data(), 8, QUERY_RESULT_64_BIT) != VK_SUCCESS) return false;
    const u64 mask = info_.timestamp_bits >= 64 ? ~0ull : ((1ull << info_.timestamp_bits) - 1);
    for (u32 i = 0; i < n; ++i) ns[i] = static_cast<double>(t[i] & mask) * info_.timestamp_period_ns;
    return true;
}

std::string describe(const DeviceInfo& d) {
    char b[2048];
    int k = std::snprintf(b, sizeof b,
                          "[gpu] %s (vendor 0x%04x, device 0x%04x, Vulkan %u.%u.%u, tipo %u)\n"
                          "[gpu]   subgrupo %u (control %u..%u, obligatorio %s, completos %s), ops 0x%x, WG máx %u, compartida %u KB, push %u B\n"
                          "[gpu]   marcas de tiempo %.3f ns/tic (%u bits), rango de búfer %.0f MB (alineación %llu B), búferes/etapa %u, montón %.2f GB\n"
                          "[gpu]   f16 %d · i8 %d · i16 %d · búfer16 %d · búfer8 %d · timeline %d · sync2 %d · RTE16 %d · denorm16 %d\n",
                          d.name, d.vendor, d.device, d.api >> 22, (d.api >> 12) & 1023, d.api & 4095, d.type, d.subgroup_size, d.sg_min,
                          d.sg_max, d.sg_control ? "sí" : "no", d.full_subgroups ? "sí" : "no", d.subgroup_ops, d.max_wg_invocations,
                          d.max_shared / 1024, d.max_push, static_cast<double>(d.timestamp_period_ns), d.timestamp_bits,
                          static_cast<double>(d.max_storage_range) / 1048576.0, static_cast<unsigned long long>(d.storage_align), d.max_storage_buffers, static_cast<double>(d.heap_size) / 1e9,
                          d.f16, d.i8, d.i16, d.s16, d.s8, d.timeline, d.sync2, d.rte16, d.denorm_preserve16);
    for (u32 i = 0; i < d.mem_type_count && k > 0 && k < static_cast<int>(sizeof b) - 80; ++i)
        k += std::snprintf(b + k, sizeof b - static_cast<usize>(k), "[gpu]   memoria tipo %u: flags 0x%x%s%s%s%s\n", i, d.mem_flags[i],
                           (d.mem_flags[i] & 1) ? " DEVICE_LOCAL" : "", (d.mem_flags[i] & 2) ? " HOST_VISIBLE" : "",
                           (d.mem_flags[i] & 4) ? " COHERENT" : "", (d.mem_flags[i] & 8) ? " CACHED" : "");
    return b;
}

} // namespace cfd::gpu::vk
