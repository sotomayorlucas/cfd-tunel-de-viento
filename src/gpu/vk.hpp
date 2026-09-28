// ============================================================================
//  gpu/vk.hpp — binding MÍNIMO de Vulkan escrito a mano (sin cabeceras del SDK).
//
//  Sólo lo que usa el backend de cómputo: tipos, enumeraciones, estructuras con
//  su ABI EXACTA de x86-64 (static_assert de tamaños y desplazamientos) y
//  punteros a función cargados con dlopen("libvulkan.so.1") +
//  vkGetInstanceProcAddr / vkGetDeviceProcAddr. Sin capas de validación.
//
//  Encima, `Context`: instancia + dispositivo (iGPU Intel, se salta llvmpipe),
//  cola única de cómputo, memoria (UMA: DEVICE_LOCAL|HOST_VISIBLE), búferes
//  mapeados persistentemente, pipelines de cómputo a partir de SPIR-V propio
//  (gpu/spirv.hpp), semáforo de línea temporal y consultas de marcas de tiempo.
// ============================================================================
#pragma once

#include "../core/config.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cfd::gpu::vk {

// ---- Tipos básicos ---------------------------------------------------------------------------
using VkBool32 = u32;
using VkFlags = u32;
using VkFlags64 = u64;
using VkDeviceSize = u64;
using VkResult = i32;
using VkStructureType = u32;

// Manejadores despachables (punteros) y no despachables (64 bits).
struct VkInstance_T; struct VkPhysicalDevice_T; struct VkDevice_T; struct VkQueue_T; struct VkCommandBuffer_T;
using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkCommandBuffer = VkCommandBuffer_T*;
using VkBuffer = u64;
using VkDeviceMemory = u64;
using VkShaderModule = u64;
using VkPipeline = u64;
using VkPipelineLayout = u64;
using VkPipelineCache = u64;
using VkDescriptorSetLayout = u64;
using VkDescriptorPool = u64;
using VkDescriptorSet = u64;
using VkCommandPool = u64;
using VkSemaphore = u64;
using VkFence = u64;
using VkQueryPool = u64;

inline constexpr VkResult VK_SUCCESS = 0;
inline constexpr VkResult VK_TIMEOUT = 2;
inline constexpr VkResult VK_INCOMPLETE = 5;
inline constexpr u64 VK_WHOLE_SIZE = ~0ull;
inline constexpr u32 vk_api(u32 major, u32 minor) { return (major << 22) | (minor << 12); }

// ---- VkStructureType -------------------------------------------------------------------------
enum : VkStructureType {
    ST_APPLICATION_INFO = 0,
    ST_INSTANCE_CREATE_INFO = 1,
    ST_DEVICE_QUEUE_CREATE_INFO = 2,
    ST_DEVICE_CREATE_INFO = 3,
    ST_SUBMIT_INFO = 4,
    ST_MEMORY_ALLOCATE_INFO = 5,
    ST_MAPPED_MEMORY_RANGE = 6,
    ST_SEMAPHORE_CREATE_INFO = 9,
    ST_QUERY_POOL_CREATE_INFO = 11,
    ST_BUFFER_CREATE_INFO = 12,
    ST_SHADER_MODULE_CREATE_INFO = 16,
    ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    ST_COMPUTE_PIPELINE_CREATE_INFO = 29,
    ST_PIPELINE_LAYOUT_CREATE_INFO = 30,
    ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    ST_DESCRIPTOR_POOL_CREATE_INFO = 33,
    ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    ST_WRITE_DESCRIPTOR_SET = 35,
    ST_COMMAND_POOL_CREATE_INFO = 39,
    ST_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    ST_COMMAND_BUFFER_BEGIN_INFO = 42,
    ST_MEMORY_BARRIER = 46,
    ST_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES = 49,
    ST_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51,
    ST_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53,
    ST_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
    ST_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
    ST_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000,
    ST_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES = 1000197000,
    ST_SEMAPHORE_TYPE_CREATE_INFO = 1000207002,
    ST_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003,
    ST_SEMAPHORE_WAIT_INFO = 1000207004,
    ST_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES = 1000225000,
    ST_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO = 1000225001,
    ST_MEMORY_BARRIER_2 = 1000314000,
    ST_DEPENDENCY_INFO = 1000314003,
};

// ---- Enumeraciones y bits ----------------------------------------------------------------------
enum : u32 {
    PHYSICAL_DEVICE_TYPE_OTHER = 0, PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1, PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3, PHYSICAL_DEVICE_TYPE_CPU = 4,
};
enum : u32 { QUEUE_GRAPHICS_BIT = 1, QUEUE_COMPUTE_BIT = 2, QUEUE_TRANSFER_BIT = 4 };
enum : u32 { MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1, MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2, MEMORY_PROPERTY_HOST_COHERENT_BIT = 4,
             MEMORY_PROPERTY_HOST_CACHED_BIT = 8 };
enum : u32 { BUFFER_USAGE_TRANSFER_SRC_BIT = 1, BUFFER_USAGE_TRANSFER_DST_BIT = 2, BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x10,
             BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20, BUFFER_USAGE_INDIRECT_BUFFER_BIT = 0x100 };
enum : u32 { SHARING_MODE_EXCLUSIVE = 0 };
enum : u32 { DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6, DESCRIPTOR_TYPE_STORAGE_BUFFER = 7 };
enum : u32 { SHADER_STAGE_COMPUTE_BIT = 0x20 };
enum : u32 { PIPELINE_BIND_POINT_COMPUTE = 1 };
enum : u32 { COMMAND_BUFFER_LEVEL_PRIMARY = 0 };
enum : u32 { COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 2 };
enum : u32 { COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 1, COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT = 4 };
enum : u32 { QUERY_TYPE_TIMESTAMP = 2 };
enum : u32 { QUERY_RESULT_64_BIT = 1, QUERY_RESULT_WAIT_BIT = 2 };
enum : u32 { SEMAPHORE_TYPE_TIMELINE = 1 };
enum : u32 { PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT = 1, PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 2 };
// Etapas y accesos (sincronización 1 y 2 comparten los valores de los bits bajos).
enum : u32 {
    PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x1, PIPELINE_STAGE_DRAW_INDIRECT_BIT = 0x2, PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800,
    PIPELINE_STAGE_TRANSFER_BIT = 0x1000, PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x2000, PIPELINE_STAGE_HOST_BIT = 0x4000,
    PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x10000,
};
enum : u32 {
    ACCESS_INDIRECT_COMMAND_READ_BIT = 0x1, ACCESS_UNIFORM_READ_BIT = 0x8, ACCESS_SHADER_READ_BIT = 0x20, ACCESS_SHADER_WRITE_BIT = 0x40,
    ACCESS_TRANSFER_READ_BIT = 0x800, ACCESS_TRANSFER_WRITE_BIT = 0x1000, ACCESS_HOST_READ_BIT = 0x2000,
    ACCESS_HOST_WRITE_BIT = 0x4000, ACCESS_MEMORY_READ_BIT = 0x8000, ACCESS_MEMORY_WRITE_BIT = 0x10000,
};
inline constexpr u64 ACCESS_2_SHADER_STORAGE_READ_BIT = 0x200000000ull;
inline constexpr u64 ACCESS_2_SHADER_STORAGE_WRITE_BIT = 0x400000000ull;
enum : u32 { SUBGROUP_FEATURE_BASIC_BIT = 1, SUBGROUP_FEATURE_VOTE_BIT = 2, SUBGROUP_FEATURE_ARITHMETIC_BIT = 4,
             SUBGROUP_FEATURE_BALLOT_BIT = 8, SUBGROUP_FEATURE_SHUFFLE_BIT = 0x10, SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x20 };

// ---- Estructuras (ABI x86-64 LP64) -------------------------------------------------------------
struct VkApplicationInfo {
    VkStructureType sType; const void* pNext; const char* pApplicationName; u32 applicationVersion;
    const char* pEngineName; u32 engineVersion; u32 apiVersion;
};
static_assert(sizeof(VkApplicationInfo) == 48);

struct VkInstanceCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; const VkApplicationInfo* pApplicationInfo;
    u32 enabledLayerCount; const char* const* ppEnabledLayerNames; u32 enabledExtensionCount; const char* const* ppEnabledExtensionNames;
};
static_assert(sizeof(VkInstanceCreateInfo) == 64);

struct VkPhysicalDeviceLimits {
    u32 maxImageDimension1D, maxImageDimension2D, maxImageDimension3D, maxImageDimensionCube, maxImageArrayLayers;
    u32 maxTexelBufferElements, maxUniformBufferRange, maxStorageBufferRange, maxPushConstantsSize, maxMemoryAllocationCount;
    u32 maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity, sparseAddressSpaceSize;
    u32 maxBoundDescriptorSets, maxPerStageDescriptorSamplers, maxPerStageDescriptorUniformBuffers, maxPerStageDescriptorStorageBuffers;
    u32 maxPerStageDescriptorSampledImages, maxPerStageDescriptorStorageImages, maxPerStageDescriptorInputAttachments, maxPerStageResources;
    u32 maxDescriptorSetSamplers, maxDescriptorSetUniformBuffers, maxDescriptorSetUniformBuffersDynamic, maxDescriptorSetStorageBuffers;
    u32 maxDescriptorSetStorageBuffersDynamic, maxDescriptorSetSampledImages, maxDescriptorSetStorageImages, maxDescriptorSetInputAttachments;
    u32 maxVertexInputAttributes, maxVertexInputBindings, maxVertexInputAttributeOffset, maxVertexInputBindingStride, maxVertexOutputComponents;
    u32 maxTessellationGenerationLevel, maxTessellationPatchSize, maxTessellationControlPerVertexInputComponents;
    u32 maxTessellationControlPerVertexOutputComponents, maxTessellationControlPerPatchOutputComponents;
    u32 maxTessellationControlTotalOutputComponents, maxTessellationEvaluationInputComponents, maxTessellationEvaluationOutputComponents;
    u32 maxGeometryShaderInvocations, maxGeometryInputComponents, maxGeometryOutputComponents, maxGeometryOutputVertices;
    u32 maxGeometryTotalOutputComponents, maxFragmentInputComponents, maxFragmentOutputAttachments, maxFragmentDualSrcAttachments;
    u32 maxFragmentCombinedOutputResources, maxComputeSharedMemorySize, maxComputeWorkGroupCount[3], maxComputeWorkGroupInvocations;
    u32 maxComputeWorkGroupSize[3], subPixelPrecisionBits, subTexelPrecisionBits, mipmapPrecisionBits, maxDrawIndexedIndexValue;
    u32 maxDrawIndirectCount;
    float maxSamplerLodBias, maxSamplerAnisotropy;
    u32 maxViewports, maxViewportDimensions[2];
    float viewportBoundsRange[2];
    u32 viewportSubPixelBits;
    std::size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment, minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment;
    i32 minTexelOffset; u32 maxTexelOffset; i32 minTexelGatherOffset; u32 maxTexelGatherOffset;
    float minInterpolationOffset, maxInterpolationOffset;
    u32 subPixelInterpolationOffsetBits, maxFramebufferWidth, maxFramebufferHeight, maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts, framebufferDepthSampleCounts, framebufferStencilSampleCounts, framebufferNoAttachmentsSampleCounts;
    u32 maxColorAttachments;
    VkFlags sampledImageColorSampleCounts, sampledImageIntegerSampleCounts, sampledImageDepthSampleCounts, sampledImageStencilSampleCounts;
    VkFlags storageImageSampleCounts;
    u32 maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    u32 maxClipDistances, maxCullDistances, maxCombinedClipAndCullDistances, discreteQueuePriorities;
    float pointSizeRange[2], lineWidthRange[2], pointSizeGranularity, lineWidthGranularity;
    VkBool32 strictLines, standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment, optimalBufferCopyRowPitchAlignment, nonCoherentAtomSize;
};
static_assert(sizeof(VkPhysicalDeviceLimits) == 504);
static_assert(offsetof(VkPhysicalDeviceLimits, bufferImageGranularity) == 48);
static_assert(offsetof(VkPhysicalDeviceLimits, maxComputeSharedMemorySize) == 216);
static_assert(offsetof(VkPhysicalDeviceLimits, minMemoryMapAlignment) == 304);
static_assert(offsetof(VkPhysicalDeviceLimits, timestampPeriod) == 424);

struct VkPhysicalDeviceSparseProperties { VkBool32 v[5]; };
struct VkPhysicalDeviceProperties {
    u32 apiVersion, driverVersion, vendorID, deviceID, deviceType;
    char deviceName[256];
    u8 pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};
static_assert(offsetof(VkPhysicalDeviceProperties, limits) == 296);
static_assert(sizeof(VkPhysicalDeviceProperties) == 824);

struct VkPhysicalDeviceProperties2 { VkStructureType sType; void* pNext; VkPhysicalDeviceProperties properties; };
static_assert(sizeof(VkPhysicalDeviceProperties2) == 840);

struct VkPhysicalDeviceSubgroupProperties {
    VkStructureType sType; void* pNext; u32 subgroupSize; VkFlags supportedStages; VkFlags supportedOperations; VkBool32 quadOperationsInAllStages;
};
static_assert(sizeof(VkPhysicalDeviceSubgroupProperties) == 32);

struct VkPhysicalDeviceSubgroupSizeControlProperties {
    VkStructureType sType; void* pNext; u32 minSubgroupSize, maxSubgroupSize, maxComputeWorkgroupSubgroups; VkFlags requiredSubgroupSizeStages;
};
static_assert(sizeof(VkPhysicalDeviceSubgroupSizeControlProperties) == 32);

struct VkPhysicalDeviceFloatControlsProperties {
    VkStructureType sType; void* pNext; u32 denormBehaviorIndependence, roundingModeIndependence;
    VkBool32 shaderSignedZeroInfNanPreserveFloat16, shaderSignedZeroInfNanPreserveFloat32, shaderSignedZeroInfNanPreserveFloat64;
    VkBool32 shaderDenormPreserveFloat16, shaderDenormPreserveFloat32, shaderDenormPreserveFloat64;
    VkBool32 shaderDenormFlushToZeroFloat16, shaderDenormFlushToZeroFloat32, shaderDenormFlushToZeroFloat64;
    VkBool32 shaderRoundingModeRTEFloat16, shaderRoundingModeRTEFloat32, shaderRoundingModeRTEFloat64;
    VkBool32 shaderRoundingModeRTZFloat16, shaderRoundingModeRTZFloat32, shaderRoundingModeRTZFloat64;
};
static_assert(sizeof(VkPhysicalDeviceFloatControlsProperties) == 88);

// VkPhysicalDeviceFeatures: 55 VkBool32 (índices de los que se usan).
struct VkPhysicalDeviceFeatures { VkBool32 v[55]; };
inline constexpr int FEAT_shaderFloat64 = 39, FEAT_shaderInt64 = 40, FEAT_shaderInt16 = 41;
struct VkPhysicalDeviceFeatures2 { VkStructureType sType; void* pNext; VkPhysicalDeviceFeatures features; };
static_assert(sizeof(VkPhysicalDeviceFeatures2) == 240);

struct VkPhysicalDeviceVulkan11Features {
    VkStructureType sType; void* pNext;
    VkBool32 storageBuffer16BitAccess, uniformAndStorageBuffer16BitAccess, storagePushConstant16, storageInputOutput16, multiview;
    VkBool32 multiviewGeometryShader, multiviewTessellationShader, variablePointersStorageBuffer, variablePointers, protectedMemory;
    VkBool32 samplerYcbcrConversion, shaderDrawParameters;
};
static_assert(sizeof(VkPhysicalDeviceVulkan11Features) == 64);

// VkPhysicalDeviceVulkan12Features: 47 VkBool32 (índices de los que se usan).
struct VkPhysicalDeviceVulkan12Features { VkStructureType sType; void* pNext; VkBool32 v[47]; };
static_assert(sizeof(VkPhysicalDeviceVulkan12Features) == 208);
inline constexpr int F12_storageBuffer8BitAccess = 2, F12_uniformAndStorageBuffer8BitAccess = 3, F12_shaderFloat16 = 7, F12_shaderInt8 = 8,
                     F12_scalarBlockLayout = 31, F12_shaderSubgroupExtendedTypes = 34, F12_hostQueryReset = 36, F12_timelineSemaphore = 37,
                     F12_bufferDeviceAddress = 38, F12_vulkanMemoryModel = 41, F12_subgroupBroadcastDynamicId = 46;

struct VkPhysicalDeviceVulkan13Features {
    VkStructureType sType; void* pNext;
    VkBool32 robustImageAccess, inlineUniformBlock, descriptorBindingInlineUniformBlockUpdateAfterBind, pipelineCreationCacheControl;
    VkBool32 privateData, shaderDemoteToHelperInvocation, shaderTerminateInvocation, subgroupSizeControl, computeFullSubgroups;
    VkBool32 synchronization2, textureCompressionASTC_HDR, shaderZeroInitializeWorkgroupMemory, dynamicRendering;
    VkBool32 shaderIntegerDotProduct, maintenance4;
};
static_assert(sizeof(VkPhysicalDeviceVulkan13Features) == 80);

struct VkExtent3D { u32 width, height, depth; };
struct VkQueueFamilyProperties { VkFlags queueFlags; u32 queueCount; u32 timestampValidBits; VkExtent3D minImageTransferGranularity; };
static_assert(sizeof(VkQueueFamilyProperties) == 24);

struct VkMemoryType { VkFlags propertyFlags; u32 heapIndex; };
struct VkMemoryHeap { VkDeviceSize size; VkFlags flags; };
struct VkPhysicalDeviceMemoryProperties { u32 memoryTypeCount; VkMemoryType memoryTypes[32]; u32 memoryHeapCount; VkMemoryHeap memoryHeaps[16]; };
static_assert(sizeof(VkPhysicalDeviceMemoryProperties) == 520);

struct VkExtensionProperties { char extensionName[256]; u32 specVersion; };
static_assert(sizeof(VkExtensionProperties) == 260);

struct VkDeviceQueueCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 queueFamilyIndex; u32 queueCount; const float* pQueuePriorities;
};
static_assert(sizeof(VkDeviceQueueCreateInfo) == 40);

struct VkDeviceCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 queueCreateInfoCount; const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    u32 enabledLayerCount; const char* const* ppEnabledLayerNames; u32 enabledExtensionCount; const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
};
static_assert(sizeof(VkDeviceCreateInfo) == 72);

struct VkBufferCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; VkDeviceSize size; VkFlags usage; u32 sharingMode;
    u32 queueFamilyIndexCount; const u32* pQueueFamilyIndices;
};
static_assert(sizeof(VkBufferCreateInfo) == 56);

struct VkMemoryRequirements { VkDeviceSize size, alignment; u32 memoryTypeBits; };
static_assert(sizeof(VkMemoryRequirements) == 24);

struct VkMemoryAllocateInfo { VkStructureType sType; const void* pNext; VkDeviceSize allocationSize; u32 memoryTypeIndex; };
static_assert(sizeof(VkMemoryAllocateInfo) == 32);

struct VkMappedMemoryRange { VkStructureType sType; const void* pNext; VkDeviceMemory memory; VkDeviceSize offset, size; };
static_assert(sizeof(VkMappedMemoryRange) == 40);

struct VkShaderModuleCreateInfo { VkStructureType sType; const void* pNext; VkFlags flags; std::size_t codeSize; const u32* pCode; };
static_assert(sizeof(VkShaderModuleCreateInfo) == 40);

struct VkSpecializationInfo;
struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 stage; VkShaderModule module; const char* pName;
    const VkSpecializationInfo* pSpecializationInfo;
};
static_assert(sizeof(VkPipelineShaderStageCreateInfo) == 48);

struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { VkStructureType sType; void* pNext; u32 requiredSubgroupSize; };
static_assert(sizeof(VkPipelineShaderStageRequiredSubgroupSizeCreateInfo) == 24);

struct VkComputePipelineCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; VkPipelineShaderStageCreateInfo stage; VkPipelineLayout layout;
    VkPipeline basePipelineHandle; i32 basePipelineIndex;
};
static_assert(offsetof(VkComputePipelineCreateInfo, stage) == 24);
static_assert(sizeof(VkComputePipelineCreateInfo) == 96);

struct VkDescriptorSetLayoutBinding { u32 binding; u32 descriptorType; u32 descriptorCount; VkFlags stageFlags; const void* pImmutableSamplers; };
static_assert(sizeof(VkDescriptorSetLayoutBinding) == 24);

struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 bindingCount; const VkDescriptorSetLayoutBinding* pBindings;
};
static_assert(sizeof(VkDescriptorSetLayoutCreateInfo) == 32);

struct VkPushConstantRange { VkFlags stageFlags; u32 offset; u32 size; };
static_assert(sizeof(VkPushConstantRange) == 12);

struct VkPipelineLayoutCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 setLayoutCount; const VkDescriptorSetLayout* pSetLayouts;
    u32 pushConstantRangeCount; const VkPushConstantRange* pPushConstantRanges;
};
static_assert(sizeof(VkPipelineLayoutCreateInfo) == 48);

struct VkDescriptorPoolSize { u32 type; u32 descriptorCount; };
struct VkDescriptorPoolCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 maxSets; u32 poolSizeCount; const VkDescriptorPoolSize* pPoolSizes;
};
static_assert(sizeof(VkDescriptorPoolCreateInfo) == 40);

struct VkDescriptorSetAllocateInfo {
    VkStructureType sType; const void* pNext; VkDescriptorPool descriptorPool; u32 descriptorSetCount; const VkDescriptorSetLayout* pSetLayouts;
};
static_assert(sizeof(VkDescriptorSetAllocateInfo) == 40);

struct VkDescriptorBufferInfo { VkBuffer buffer; VkDeviceSize offset, range; };
static_assert(sizeof(VkDescriptorBufferInfo) == 24);

struct VkWriteDescriptorSet {
    VkStructureType sType; const void* pNext; VkDescriptorSet dstSet; u32 dstBinding; u32 dstArrayElement; u32 descriptorCount;
    u32 descriptorType; const void* pImageInfo; const VkDescriptorBufferInfo* pBufferInfo; const void* pTexelBufferView;
};
static_assert(sizeof(VkWriteDescriptorSet) == 64);

struct VkCommandPoolCreateInfo { VkStructureType sType; const void* pNext; VkFlags flags; u32 queueFamilyIndex; };
static_assert(sizeof(VkCommandPoolCreateInfo) == 24);

struct VkCommandBufferAllocateInfo { VkStructureType sType; const void* pNext; VkCommandPool commandPool; u32 level; u32 commandBufferCount; };
static_assert(sizeof(VkCommandBufferAllocateInfo) == 32);

struct VkCommandBufferBeginInfo { VkStructureType sType; const void* pNext; VkFlags flags; const void* pInheritanceInfo; };
static_assert(sizeof(VkCommandBufferBeginInfo) == 32);

struct VkMemoryBarrier { VkStructureType sType; const void* pNext; VkFlags srcAccessMask, dstAccessMask; };
static_assert(sizeof(VkMemoryBarrier) == 24);

struct VkMemoryBarrier2 {
    VkStructureType sType; const void* pNext; VkFlags64 srcStageMask, srcAccessMask, dstStageMask, dstAccessMask;
};
static_assert(sizeof(VkMemoryBarrier2) == 48);

struct VkDependencyInfo {
    VkStructureType sType; const void* pNext; VkFlags dependencyFlags; u32 memoryBarrierCount; const VkMemoryBarrier2* pMemoryBarriers;
    u32 bufferMemoryBarrierCount; const void* pBufferMemoryBarriers; u32 imageMemoryBarrierCount; const void* pImageMemoryBarriers;
};
static_assert(sizeof(VkDependencyInfo) == 64);

struct VkBufferCopy { VkDeviceSize srcOffset, dstOffset, size; };

struct VkSemaphoreTypeCreateInfo { VkStructureType sType; const void* pNext; u32 semaphoreType; u64 initialValue; };
static_assert(sizeof(VkSemaphoreTypeCreateInfo) == 32);
struct VkSemaphoreCreateInfo { VkStructureType sType; const void* pNext; VkFlags flags; };
static_assert(sizeof(VkSemaphoreCreateInfo) == 24);
struct VkTimelineSemaphoreSubmitInfo {
    VkStructureType sType; const void* pNext; u32 waitSemaphoreValueCount; const u64* pWaitSemaphoreValues;
    u32 signalSemaphoreValueCount; const u64* pSignalSemaphoreValues;
};
static_assert(sizeof(VkTimelineSemaphoreSubmitInfo) == 48);
struct VkSemaphoreWaitInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 semaphoreCount; const VkSemaphore* pSemaphores; const u64* pValues;
};
static_assert(sizeof(VkSemaphoreWaitInfo) == 40);

struct VkSubmitInfo {
    VkStructureType sType; const void* pNext; u32 waitSemaphoreCount; const VkSemaphore* pWaitSemaphores; const VkFlags* pWaitDstStageMask;
    u32 commandBufferCount; const VkCommandBuffer* pCommandBuffers; u32 signalSemaphoreCount; const VkSemaphore* pSignalSemaphores;
};
static_assert(sizeof(VkSubmitInfo) == 72);

struct VkQueryPoolCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; u32 queryType; u32 queryCount; VkFlags pipelineStatistics;
};
static_assert(sizeof(VkQueryPoolCreateInfo) == 32);

// ---- Punteros a función ---------------------------------------------------------------------------
using PFN_vkVoidFunction = void (*)();
struct Fns {
    // Globales / instancia
    PFN_vkVoidFunction (*GetInstanceProcAddr)(VkInstance, const char*) = nullptr;
    VkResult (*EnumerateInstanceVersion)(u32*) = nullptr;
    VkResult (*CreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = nullptr;
    void (*DestroyInstance)(VkInstance, const void*) = nullptr;
    VkResult (*EnumeratePhysicalDevices)(VkInstance, u32*, VkPhysicalDevice*) = nullptr;
    void (*GetPhysicalDeviceProperties2)(VkPhysicalDevice, VkPhysicalDeviceProperties2*) = nullptr;
    void (*GetPhysicalDeviceFeatures2)(VkPhysicalDevice, VkPhysicalDeviceFeatures2*) = nullptr;
    void (*GetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, u32*, VkQueueFamilyProperties*) = nullptr;
    void (*GetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*) = nullptr;
    VkResult (*EnumerateDeviceExtensionProperties)(VkPhysicalDevice, const char*, u32*, VkExtensionProperties*) = nullptr;
    VkResult (*CreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*) = nullptr;
    PFN_vkVoidFunction (*GetDeviceProcAddr)(VkDevice, const char*) = nullptr;
    // Dispositivo
    void (*DestroyDevice)(VkDevice, const void*) = nullptr;
    void (*GetDeviceQueue)(VkDevice, u32, u32, VkQueue*) = nullptr;
    VkResult (*CreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*) = nullptr;
    void (*DestroyBuffer)(VkDevice, VkBuffer, const void*) = nullptr;
    void (*GetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*) = nullptr;
    VkResult (*AllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*) = nullptr;
    void (*FreeMemory)(VkDevice, VkDeviceMemory, const void*) = nullptr;
    VkResult (*BindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) = nullptr;
    VkResult (*MapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**) = nullptr;
    void (*UnmapMemory)(VkDevice, VkDeviceMemory) = nullptr;
    VkResult (*FlushMappedMemoryRanges)(VkDevice, u32, const VkMappedMemoryRange*) = nullptr;
    VkResult (*InvalidateMappedMemoryRanges)(VkDevice, u32, const VkMappedMemoryRange*) = nullptr;
    VkResult (*CreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*) = nullptr;
    void (*DestroyShaderModule)(VkDevice, VkShaderModule, const void*) = nullptr;
    VkResult (*CreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*) = nullptr;
    void (*DestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*) = nullptr;
    VkResult (*CreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*) = nullptr;
    void (*DestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*) = nullptr;
    VkResult (*CreateComputePipelines)(VkDevice, VkPipelineCache, u32, const VkComputePipelineCreateInfo*, const void*, VkPipeline*) = nullptr;
    void (*DestroyPipeline)(VkDevice, VkPipeline, const void*) = nullptr;
    VkResult (*CreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*) = nullptr;
    void (*DestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*) = nullptr;
    VkResult (*AllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*) = nullptr;
    VkResult (*ResetDescriptorPool)(VkDevice, VkDescriptorPool, VkFlags) = nullptr;
    void (*UpdateDescriptorSets)(VkDevice, u32, const VkWriteDescriptorSet*, u32, const void*) = nullptr;
    VkResult (*CreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*) = nullptr;
    void (*DestroyCommandPool)(VkDevice, VkCommandPool, const void*) = nullptr;
    VkResult (*AllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*) = nullptr;
    void (*FreeCommandBuffers)(VkDevice, VkCommandPool, u32, const VkCommandBuffer*) = nullptr;
    VkResult (*BeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*) = nullptr;
    VkResult (*EndCommandBuffer)(VkCommandBuffer) = nullptr;
    VkResult (*ResetCommandBuffer)(VkCommandBuffer, VkFlags) = nullptr;
    void (*CmdBindPipeline)(VkCommandBuffer, u32, VkPipeline) = nullptr;
    void (*CmdBindDescriptorSets)(VkCommandBuffer, u32, VkPipelineLayout, u32, u32, const VkDescriptorSet*, u32, const u32*) = nullptr;
    void (*CmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, u32, u32, const void*) = nullptr;
    void (*CmdDispatch)(VkCommandBuffer, u32, u32, u32) = nullptr;
    void (*CmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, u32, const VkMemoryBarrier*, u32, const void*, u32, const void*) = nullptr;
    void (*CmdPipelineBarrier2)(VkCommandBuffer, const VkDependencyInfo*) = nullptr;
    void (*CmdWriteTimestamp)(VkCommandBuffer, VkFlags, VkQueryPool, u32) = nullptr;
    void (*CmdResetQueryPool)(VkCommandBuffer, VkQueryPool, u32, u32) = nullptr;
    void (*CmdFillBuffer)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, u32) = nullptr;
    void (*CmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, u32, const VkBufferCopy*) = nullptr;
    VkResult (*CreateQueryPool)(VkDevice, const VkQueryPoolCreateInfo*, const void*, VkQueryPool*) = nullptr;
    void (*DestroyQueryPool)(VkDevice, VkQueryPool, const void*) = nullptr;
    VkResult (*GetQueryPoolResults)(VkDevice, VkQueryPool, u32, u32, std::size_t, void*, VkDeviceSize, VkFlags) = nullptr;
    void (*ResetQueryPool)(VkDevice, VkQueryPool, u32, u32) = nullptr;
    VkResult (*QueueSubmit)(VkQueue, u32, const VkSubmitInfo*, VkFence) = nullptr;
    VkResult (*QueueWaitIdle)(VkQueue) = nullptr;
    VkResult (*DeviceWaitIdle)(VkDevice) = nullptr;
    VkResult (*CreateSemaphore)(VkDevice, const VkSemaphoreCreateInfo*, const void*, VkSemaphore*) = nullptr;
    void (*DestroySemaphore)(VkDevice, VkSemaphore, const void*) = nullptr;
    VkResult (*WaitSemaphores)(VkDevice, const VkSemaphoreWaitInfo*, u64) = nullptr;
    VkResult (*GetSemaphoreCounterValue)(VkDevice, VkSemaphore, u64*) = nullptr;
};

// ================================================================================================
//  Contexto de cómputo
// ================================================================================================
struct DeviceInfo {
    char name[256] = {};
    u32 vendor = 0, device = 0, api = 0, driver = 0, type = 0;
    float timestamp_period_ns = 0.0f;       // ns por tic del reloj de marcas de tiempo
    u32 timestamp_bits = 0;
    u32 subgroup_size = 0, sg_min = 0, sg_max = 0, max_wg_subgroups = 0;
    u32 subgroup_ops = 0;                   // VkSubgroupFeatureFlags
    bool sg_control = false, full_subgroups = false;
    u32 max_push = 0, max_shared = 0, max_wg_invocations = 0, max_wg_count[3] = {};
    u64 max_storage_range = 0;
    u64 storage_align = 0;                  // minStorageBufferOffsetAlignment
    u32 max_storage_buffers = 0;
    bool f16 = false, i8 = false, i16 = false, s16 = false, s8 = false, timeline = false, sync2 = false, host_query_reset = false;
    bool rte16 = false, denorm_preserve16 = false;
    u32 mem_type_count = 0;
    u32 mem_flags[32] = {};
    u64 heap_size = 0;
};

// Búfer con su memoria propia (una asignación por búfer: pocas y grandes) mapeado persistentemente.
struct Buffer {
    VkBuffer buf = 0;
    VkDeviceMemory mem = 0;
    void* map = nullptr;
    u64 size = 0;
    u32 type = 0;          // índice de tipo de memoria
    bool coherent = true;  // HOST_COHERENT (si no: flush/invalidate explícitos)
    template <class T> T* as() const { return static_cast<T*>(map); }
};

// Preferencias de tipo de memoria (UMA): COHERENT = escritura combinada (WC) en la CPU, sin flush;
// CACHED = cacheada en la CPU, flush tras escribir / invalidate antes de leer lo que escribió la GPU.
enum class Mem : u8 { Coherent, Cached };

class Context {
public:
    Context() = default;
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Crea instancia y dispositivo. false (con motivo en err) si no hay Vulkan o no hay iGPU apta.
    bool init(std::string* err = nullptr, bool verbose = false);
    bool ok() const { return dev_ != nullptr; }
    void destroy();

    const Fns& fn() const { return f_; }
    const DeviceInfo& info() const { return info_; }
    VkDevice device() const { return dev_; }
    VkQueue queue() const { return queue_; }
    u32 queue_family() const { return qf_; }

    // Búferes (tamaño en bytes). Se mapean siempre (UMA).
    bool create_buffer(Buffer& b, u64 size, Mem pref, u32 extra_usage = 0);
    void destroy_buffer(Buffer& b);
    void flush(const Buffer& b, u64 off = 0, u64 size = VK_WHOLE_SIZE) const;        // CPU → GPU (no coherente)
    void invalidate(const Buffer& b, u64 off = 0, u64 size = VK_WHOLE_SIZE) const;   // GPU → CPU (no coherente)
    int mem_type(Mem pref) const;   // índice elegido (−1 si no hay)

    // Pipeline de cómputo: SPIR-V propio, un único conjunto de descriptores de `nbind` búferes de
    // almacenamiento (bindings 0..nbind-1) y `push_bytes` de constantes de empuje.
    struct Pipeline {
        VkPipeline pipe = 0;
        VkPipelineLayout layout = 0;
        VkDescriptorSetLayout dsl = 0;
        u32 nbind = 0, push = 0;
    };
    bool create_pipeline(Pipeline& p, const std::vector<u32>& spirv, u32 nbind, u32 push_bytes, u32 required_subgroup = 0,
                         std::string* err = nullptr);
    void destroy_pipeline(Pipeline& p);
    // Conjunto de descriptores para `p` con los búferes dados (rangos completos o [off, off+range)).
    struct Bind { const Buffer* b; u64 off; u64 range; };
    VkDescriptorSet make_set(const Pipeline& p, const Bind* binds, u32 n);
    void reset_sets();   // libera TODOS los conjuntos de descriptores (vkResetDescriptorPool)

    // Comandos
    VkCommandBuffer alloc_cmd();
    void free_cmd(VkCommandBuffer c);
    void begin(VkCommandBuffer c, bool one_time) const;
    void end(VkCommandBuffer c) const;
    // Barrera global cómputo → cómputo (sync2 si está; si no, la clásica).
    void barrier_compute(VkCommandBuffer c) const;
    // Barrera cómputo → host (antes de que la CPU lea resultados).
    void barrier_host(VkCommandBuffer c) const;
    // Envía y señaliza el semáforo de línea temporal con un nuevo valor (que devuelve).
    u64 submit(VkCommandBuffer c);
    u64 submit(const VkCommandBuffer* cs, u32 n);
    bool wait(u64 value, u64 timeout_ns = ~0ull) const;     // espera a que el semáforo llegue a value
    u64 completed() const;                                   // valor actual del semáforo
    u64 last_submitted() const { return tl_value_; }
    void wait_idle() const;

    // Marcas de tiempo (pool único de `n` consultas).
    VkQueryPool timestamps(u32 n);
    // Lee `n` marcas desde `first` en ns (false si no están disponibles).
    bool read_timestamps(u32 first, u32 n, double* ns) const;

private:
    void* lib_ = nullptr;
    Fns f_;
    VkInstance inst_ = nullptr;
    VkPhysicalDevice pd_ = nullptr;
    VkDevice dev_ = nullptr;
    VkQueue queue_ = nullptr;
    u32 qf_ = 0;
    DeviceInfo info_;
    VkCommandPool pool_ = 0;
    VkDescriptorPool dpool_ = 0;
    VkSemaphore tl_ = 0;
    u64 tl_value_ = 0;
    VkQueryPool qpool_ = 0;
    u32 qcount_ = 0;
    int type_coherent_ = -1, type_cached_ = -1;
};

// Informe legible del dispositivo (para --gpu-info / tests).
std::string describe(const DeviceInfo& d);

} // namespace cfd::gpu::vk
