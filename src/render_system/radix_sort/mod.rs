// #include <vk_radix_sort.h>

// #include <utility>

// #include "generated/upsweep_comp.h"
// #include "generated/spine_comp.h"
// #include "generated/downsweep_comp.h"
// #include "generated/downsweep_key_value_comp.h"

// namespace {

// constexpr uint32_t RADIX = 256;
// constexpr int WORKGROUP_SIZE = 512;
// constexpr int PARTITION_DIVISION = 8;
// constexpr int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

// uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// VkDeviceSize HistogramSize(uint32_t elementCount) {
//   return (1 + 4 * RADIX + RoundUp(elementCount, PARTITION_SIZE) * RADIX) *
//          sizeof(uint32_t);
// }

// VkDeviceSize InoutSize(uint32_t elementCount) {
//   return elementCount * sizeof(uint32_t);
// }

// void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
//              uint32_t elementCount, VkBuffer indirectBuffer,
//              VkDeviceSize indirectOffset, VkBuffer buffer, VkDeviceSize offset,
//              VkBuffer valueBuffer, VkDeviceSize valueOffset,
//              VkBuffer storageBuffer, VkDeviceSize storageOffset,
//              VkQueryPool queryPool, uint32_t query);

// }  // namespace

// struct VrdxSorter_T {
//   VkDevice device = VK_NULL_HANDLE;

//   VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

//   VkPipeline upsweepPipeline = VK_NULL_HANDLE;
//   VkPipeline spinePipeline = VK_NULL_HANDLE;
//   VkPipeline downsweepPipeline = VK_NULL_HANDLE;
//   VkPipeline downsweepKeyValuePipeline = VK_NULL_HANDLE;

//   uint32_t maxWorkgroupSize = 0;
// };

// struct PushConstants {
//   uint32_t pass;
//   VkDeviceAddress elementCountReference;
//   VkDeviceAddress globalHistogramReference;
//   VkDeviceAddress partitionHistogramReference;
//   VkDeviceAddress keysInReference;
//   VkDeviceAddress keysOutReference;
//   VkDeviceAddress valuesInReference;
//   VkDeviceAddress valuesOutReference;
// };

// void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo,
//                       VrdxSorter* pSorter) {
//   VkDevice device = pCreateInfo->device;
//   VkPipelineCache pipelineCache = pCreateInfo->pipelineCache;

//   // shader specialization constants and defaults
//   VkPhysicalDeviceVulkan11Properties physicalDeviceVulkan11Properties = {
//       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
//   VkPhysicalDeviceProperties2 physicalDeviceProperties = {
//       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
//   physicalDeviceProperties.pNext = &physicalDeviceVulkan11Properties;

//   vkGetPhysicalDeviceProperties2(pCreateInfo->physicalDevice,
//                                  &physicalDeviceProperties);

//   // TODO: max workgroup size
//   uint32_t maxWorkgroupSize =
//       physicalDeviceProperties.properties.limits.maxComputeWorkGroupSize[0];
//   uint32_t subgroupSize = physicalDeviceVulkan11Properties.subgroupSize;

//   // pipeline layout
//   VkPushConstantRange pushConstants = {};
//   pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
//   pushConstants.offset = 0;
//   pushConstants.size = sizeof(PushConstants);

//   VkPipelineLayout pipelineLayout;
//   VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
//       VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
//   pipelineLayoutInfo.pushConstantRangeCount = 1;
//   pipelineLayoutInfo.pPushConstantRanges = &pushConstants;
//   vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

//   // pipelines
//   VkPipeline upsweepPipeline;
//   {
//     VkShaderModule shaderModule;
//     VkShaderModuleCreateInfo shaderModuleInfo = {
//         VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
//     shaderModuleInfo.codeSize = sizeof(upsweep_comp);
//     shaderModuleInfo.pCode = upsweep_comp;
//     vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

//     VkComputePipelineCreateInfo pipelineInfo = {
//         VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
//     pipelineInfo.stage.sType =
//         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//     pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//     pipelineInfo.stage.module = shaderModule;
//     pipelineInfo.stage.pName = "main";
//     pipelineInfo.layout = pipelineLayout;

//     vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
//                              &upsweepPipeline);

//     vkDestroyShaderModule(device, shaderModule, NULL);
//   }

//   VkPipeline spinePipeline;
//   {
//     VkShaderModule shaderModule;
//     VkShaderModuleCreateInfo shaderModuleInfo = {
//         VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
//     shaderModuleInfo.codeSize = sizeof(spine_comp);
//     shaderModuleInfo.pCode = spine_comp;
//     vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

//     VkComputePipelineCreateInfo pipelineInfo = {
//         VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
//     pipelineInfo.stage.sType =
//         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//     pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//     pipelineInfo.stage.module = shaderModule;
//     pipelineInfo.stage.pName = "main";
//     pipelineInfo.layout = pipelineLayout;

//     vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
//                              &spinePipeline);

//     vkDestroyShaderModule(device, shaderModule, NULL);
//   }

//   VkPipeline downsweepPipeline;
//   VkPipeline downsweepKeyValuePipeline;
//   {
//     VkShaderModule shaderModules[2];
//     VkShaderModuleCreateInfo shaderModuleInfo = {
//         VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
//     shaderModuleInfo.codeSize = sizeof(downsweep_comp);
//     shaderModuleInfo.pCode = downsweep_comp;
//     vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[0]);

//     shaderModuleInfo.codeSize = sizeof(downsweep_key_value_comp);
//     shaderModuleInfo.pCode = downsweep_key_value_comp;
//     vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[1]);

//     VkComputePipelineCreateInfo pipelineInfos[2];
//     pipelineInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
//     pipelineInfos[0].stage.sType =
//         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//     pipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//     pipelineInfos[0].stage.module = shaderModules[0];
//     pipelineInfos[0].stage.pName = "main";
//     pipelineInfos[0].layout = pipelineLayout;

//     pipelineInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
//     pipelineInfos[1].stage.sType =
//         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//     pipelineInfos[1].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//     pipelineInfos[1].stage.module = shaderModules[1];
//     pipelineInfos[1].stage.pName = "main";
//     pipelineInfos[1].layout = pipelineLayout;

//     VkPipeline pipelines[2];
//     vkCreateComputePipelines(device, pipelineCache, 2, pipelineInfos, NULL,
//                              pipelines);
//     downsweepPipeline = pipelines[0];
//     downsweepKeyValuePipeline = pipelines[1];

//     for (auto shaderModule : shaderModules)
//       vkDestroyShaderModule(device, shaderModule, NULL);
//   }

//   *pSorter = new VrdxSorter_T();
//   (*pSorter)->device = device;
//   (*pSorter)->pipelineLayout = pipelineLayout;

//   (*pSorter)->upsweepPipeline = upsweepPipeline;
//   (*pSorter)->spinePipeline = spinePipeline;
//   (*pSorter)->downsweepPipeline = downsweepPipeline;
//   (*pSorter)->downsweepKeyValuePipeline = downsweepKeyValuePipeline;

//   (*pSorter)->maxWorkgroupSize = maxWorkgroupSize;
// }

// void vrdxDestroySorter(VrdxSorter sorter) {
//   vkDestroyPipeline(sorter->device, sorter->upsweepPipeline, NULL);
//   vkDestroyPipeline(sorter->device, sorter->spinePipeline, NULL);
//   vkDestroyPipeline(sorter->device, sorter->downsweepPipeline, NULL);
//   vkDestroyPipeline(sorter->device, sorter->downsweepKeyValuePipeline, NULL);

//   vkDestroyPipelineLayout(sorter->device, sorter->pipelineLayout, NULL);
//   delete sorter;
// }

// void vrdxGetSorterStorageRequirements(
//     VrdxSorter sorter, uint32_t maxElementCount,
//     VrdxSorterStorageRequirements* requirements) {
//   VkDevice device = sorter->device;

//   VkDeviceSize elementCountSize = sizeof(uint32_t);
//   VkDeviceSize histogramSize = HistogramSize(maxElementCount);
//   VkDeviceSize inoutSize = InoutSize(maxElementCount);

//   VkDeviceSize histogramOffset = elementCountSize;
//   VkDeviceSize inoutOffset = histogramOffset + histogramSize;
//   VkDeviceSize storageSize = inoutOffset + inoutSize;

//   requirements->size = storageSize;
//   requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
//                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
// }

// void vrdxGetSorterKeyValueStorageRequirements(
//     VrdxSorter sorter, uint32_t maxElementCount,
//     VrdxSorterStorageRequirements* requirements) {
//   VkDevice device = sorter->device;

//   VkDeviceSize elementCountSize = sizeof(uint32_t);
//   VkDeviceSize histogramSize = HistogramSize(maxElementCount);
//   VkDeviceSize inoutSize = InoutSize(maxElementCount);

//   VkDeviceSize histogramOffset = elementCountSize;
//   VkDeviceSize inoutOffset = histogramOffset + histogramSize;
//   // 2x for key value
//   VkDeviceSize storageSize = inoutOffset + 2 * inoutSize;

//   requirements->size = storageSize;
//   requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
//                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
// }

// void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
//                  uint32_t elementCount, VkBuffer keysBuffer,
//                  VkDeviceSize keysOffset, VkBuffer storageBuffer,
//                  VkDeviceSize storageOffset, VkQueryPool queryPool,
//                  uint32_t query) {
//   gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
//           NULL, 0, storageBuffer, storageOffset, queryPool, query);
// }

// void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
//                          uint32_t maxElementCount, VkBuffer indirectBuffer,
//                          VkDeviceSize indirectOffset, VkBuffer keysBuffer,
//                          VkDeviceSize keysOffset, VkBuffer storageBuffer,
//                          VkDeviceSize storageOffset, VkQueryPool queryPool,
//                          uint32_t query) {
//   gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer,
//           indirectOffset, keysBuffer, keysOffset, NULL, 0, storageBuffer,
//           storageOffset, queryPool, query);
// }

// void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
//                          uint32_t elementCount, VkBuffer keysBuffer,
//                          VkDeviceSize keysOffset, VkBuffer valuesBuffer,
//                          VkDeviceSize valuesOffset, VkBuffer storageBuffer,
//                          VkDeviceSize storageOffset, VkQueryPool queryPool,
//                          uint32_t query) {
//   gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
//           valuesBuffer, valuesOffset, storageBuffer, storageOffset, queryPool,
//           query);
// }

// void vrdxCmdSortKeyValueIndirect(
//     VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t maxElementCount,
//     VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer keysBuffer,
//     VkDeviceSize keysOffset, VkBuffer valuesBuffer, VkDeviceSize valuesOffset,
//     VkBuffer storageBuffer, VkDeviceSize storageOffset, VkQueryPool queryPool,
//     uint32_t query) {
//   gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer,
//           indirectOffset, keysBuffer, keysOffset, valuesBuffer, valuesOffset,
//           storageBuffer, storageOffset, queryPool, query);
// }

// namespace {

// void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
//              uint32_t elementCount, VkBuffer indirectBuffer,
//              VkDeviceSize indirectOffset, VkBuffer keysBuffer,
//              VkDeviceSize keysOffset, VkBuffer valuesBuffer,
//              VkDeviceSize valuesOffset, VkBuffer storageBuffer,
//              VkDeviceSize storageOffset, VkQueryPool queryPool,
//              uint32_t query) {
//   VkDevice device = sorter->device;
//   uint32_t partitionCount =
//       RoundUp(indirectBuffer ? elementCount : elementCount, PARTITION_SIZE);

//   VkDeviceSize elementCountSize = sizeof(uint32_t);
//   VkDeviceSize histogramSize = HistogramSize(elementCount);
//   VkDeviceSize inoutSize = InoutSize(elementCount);

//   VkDeviceSize elementCountOffset = storageOffset;
//   VkDeviceSize histogramOffset = elementCountOffset + elementCountSize;
//   VkDeviceSize inoutOffset = histogramOffset + histogramSize;

//   if (queryPool) {
//     vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
//                         queryPool, query + 0);
//   }

//   if (indirectBuffer) {
//     // copy elementCount
//     VkBufferCopy region;
//     region.srcOffset = indirectOffset;
//     region.dstOffset = elementCountOffset;
//     region.size = sizeof(uint32_t);
//     vkCmdCopyBuffer(commandBuffer, indirectBuffer, storageBuffer, 1, &region);
//   } else {
//     // set element count
//     vkCmdUpdateBuffer(commandBuffer, storageBuffer, elementCountOffset,
//                       sizeof(elementCount), &elementCount);
//   }

//   // reset global histogram. partition histogram is set by shader.
//   vkCmdFillBuffer(commandBuffer, storageBuffer, histogramOffset,
//                   4 * RADIX * sizeof(uint32_t), 0);

//   VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
//   memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
//   memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//   vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
//                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
//                        &memoryBarrier, 0, NULL, 0, NULL);

//   if (queryPool) {
//     vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
//                         queryPool, query + 1);
//   }

//   VkBufferDeviceAddressInfo deviceAddressInfo = {
//       VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
//   deviceAddressInfo.buffer = storageBuffer;
//   VkDeviceAddress storageAddress =
//       vkGetBufferDeviceAddress(device, &deviceAddressInfo);

//   deviceAddressInfo.buffer = keysBuffer;
//   VkDeviceAddress keysAddress =
//       vkGetBufferDeviceAddress(device, &deviceAddressInfo);

//   VkDeviceAddress valuesAddress = 0;
//   if (valuesBuffer) {
//     deviceAddressInfo.buffer = valuesBuffer;
//     valuesAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);
//   }

//   PushConstants pushConstants;
//   pushConstants.elementCountReference = storageAddress + elementCountOffset;
//   pushConstants.globalHistogramReference = storageAddress + histogramOffset;
//   pushConstants.partitionHistogramReference =
//       storageAddress + histogramOffset + sizeof(uint32_t) * 4 * RADIX;

//   for (int i = 0; i < 4; ++i) {
//     pushConstants.pass = i;
//     pushConstants.keysInReference = keysAddress + keysOffset;
//     pushConstants.keysOutReference = storageAddress + inoutOffset;
//     pushConstants.valuesInReference = valuesAddress + valuesOffset;
//     pushConstants.valuesOutReference = storageAddress + inoutOffset + inoutSize;

//     if (i % 2 == 1) {
//       std::swap(pushConstants.keysInReference, pushConstants.keysOutReference);
//       std::swap(pushConstants.valuesInReference,
//                 pushConstants.valuesOutReference);
//     }

//     vkCmdPushConstants(commandBuffer, sorter->pipelineLayout,
//                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
//                        &pushConstants);

//     // upsweep
//     vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
//                       sorter->upsweepPipeline);

//     vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

//     if (queryPool) {
//       vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                           queryPool, query + 2 + 3 * i + 0);
//     }

//     // spine
//     memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
//     memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
//     memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//     vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
//                          &memoryBarrier, 0, NULL, 0, NULL);

//     vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
//                       sorter->spinePipeline);

//     vkCmdDispatch(commandBuffer, RADIX, 1, 1);

//     if (queryPool) {
//       vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                           queryPool, query + 2 + 3 * i + 1);
//     }

//     // downsweep
//     memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
//     memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
//     memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//     vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
//                          &memoryBarrier, 0, NULL, 0, NULL);

//     if (valuesBuffer) {
//       vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
//                         sorter->downsweepKeyValuePipeline);
//     } else {
//       vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
//                         sorter->downsweepPipeline);
//     }

//     vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

//     if (queryPool) {
//       vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                           queryPool, query + 2 + 3 * i + 2);
//     }

//     if (i < 3) {
//       memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
//       memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
//       memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
//       vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
//                            &memoryBarrier, 0, NULL, 0, NULL);
//     }
//   }

//   if (queryPool) {
//     vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
//                         queryPool, query + 14);
//   }
// }

// }  // namespace

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, FillBufferInfo,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
    },
    shader::{ShaderModule, ShaderStages},
    sync::{GpuFuture, self},
};

// Constants
const RADIX: u32 = 256;
const WORKGROUP_SIZE: u32 = 512;
const PARTITION_DIVISION: u32 = 8;
const PARTITION_SIZE: u32 = PARTITION_DIVISION * WORKGROUP_SIZE;

// Utility functions
fn round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

fn histogram_size(element_count: u32) -> u64 {
    (1 + 4 * RADIX + round_up(element_count, PARTITION_SIZE) * RADIX) as u64 * std::mem::size_of::<u32>() as u64
}

fn inout_size(element_count: u32) -> u64 {
    element_count as u64 * std::mem::size_of::<u32>() as u64
}

// Sorter struct to hold all necessary objects
pub struct Sorter {
    device: Arc<Device>,
    upsweep_pipeline: Arc<ComputePipeline>,
    spine_pipeline: Arc<ComputePipeline>,
    downsweep_pipeline: Arc<ComputePipeline>,
    downsweep_key_value_pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    max_workgroup_size: u32,
}

// Storage requirements struct
pub struct SorterStorageRequirements {
    pub size: u64,
    pub usage: BufferUsage,
}

impl Sorter {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        // Need shader modules as input since we don't have the shader source in this example
        upsweep_shader: Arc<ShaderModule>,
        spine_shader: Arc<ShaderModule>,
        downsweep_shader: Arc<ShaderModule>,
        downsweep_key_value_shader: Arc<ShaderModule>,
    ) -> Self {
        // Get device properties (limit of workgroup size)
        let properties = device.physical_device().properties();
        let max_workgroup_size = properties.max_compute_work_group_size[0];

        // Create pipeline layout
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                push_constant_ranges: vec![push_constant_range],
                ..Default::default()
            },
        )
        .unwrap();

        // Create upsweep pipeline
        let upsweep_pipeline = ComputePipeline::new(
            device.clone(),
            upsweep_shader.entry_point("main").unwrap(),
            &(),
            Some(pipeline_layout.clone()),
            None,
        )
        .unwrap();

        // Create spine pipeline
        let spine_pipeline = ComputePipeline::new(
            device.clone(),
            spine_shader.entry_point("main").unwrap(),
            &(),
            Some(pipeline_layout.clone()),
            None,
        )
        .unwrap();

        // Create downsweep pipeline
        let downsweep_pipeline = ComputePipeline::new(
            device.clone(),
            downsweep_shader.entry_point("main").unwrap(),
            &(),
            Some(pipeline_layout.clone()),
            None,
        )
        .unwrap();

        // Create downsweep key-value pipeline
        let downsweep_key_value_pipeline = ComputePipeline::new(
            device.clone(),
            downsweep_key_value_shader.entry_point("main").unwrap(),
            &(),
            Some(pipeline_layout.clone()),
            None,
        )
        .unwrap();

        Self {
            device,
            pipeline_layout,
            upsweep_pipeline,
            spine_pipeline,
            downsweep_pipeline,
            downsweep_key_value_pipeline,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            max_workgroup_size,
        }
    }

    pub fn get_storage_requirements(&self, max_element_count: u32) -> SorterStorageRequirements {
        let element_count_size = std::mem::size_of::<u32>() as u64;
        let histogram_size = histogram_size(max_element_count);
        let inout_size = inout_size(max_element_count);

        let histogram_offset = element_count_size;
        let inout_offset = histogram_offset + histogram_size;
        let storage_size = inout_offset + inout_size;

        SorterStorageRequirements {
            size: storage_size,
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::DEVICE_ADDRESS,
        }
    }

    pub fn get_key_value_storage_requirements(&self, max_element_count: u32) -> SorterStorageRequirements {
        let element_count_size = std::mem::size_of::<u32>() as u64;
        let histogram_size = histogram_size(max_element_count);
        let inout_size = inout_size(max_element_count);

        let histogram_offset = element_count_size;
        let inout_offset = histogram_offset + histogram_size;
        // 2x for key value
        let storage_size = inout_offset + 2 * inout_size;

        SorterStorageRequirements {
            size: storage_size,
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::DEVICE_ADDRESS,
        }
    }

    pub fn sort<T: GpuFuture>(
        &self,
        previous_future: T,
        queue: Arc<Queue>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        keys_offset: u64,
        storage_buffer: Subbuffer<[u8]>,
        storage_offset: u64,
    ) -> Box<dyn GpuFuture> {
        self.gpu_sort(
            previous_future,
            queue,
            element_count,
            None,
            0,
            keys_buffer,
            keys_offset,
            None,
            0,
            storage_buffer,
            storage_offset,
        )
    }

    pub fn sort_key_value<T: GpuFuture>(
        &self,
        previous_future: T,
        queue: Arc<Queue>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        keys_offset: u64,
        values_buffer: Subbuffer<[u32]>,
        values_offset: u64,
        storage_buffer: Subbuffer<[u8]>,
        storage_offset: u64,
    ) -> Box<dyn GpuFuture> {
        self.gpu_sort(
            previous_future,
            queue,
            element_count,
            None,
            0,
            keys_buffer,
            keys_offset,
            Some(values_buffer),
            values_offset,
            storage_buffer,
            storage_offset,
        )
    }
    
    // Implementation of sort algorithm
    fn gpu_sort<T: GpuFuture>(
        &self,
        previous_future: T,
        queue: Arc<Queue>,
        element_count: u32,
        indirect_buffer: Option<Subbuffer<[u32]>>,
        indirect_offset: u64,
        keys_buffer: Subbuffer<[u32]>,
        keys_offset: u64,
        values_buffer: Option<Subbuffer<[u32]>>,
        values_offset: u64,
        storage_buffer: Subbuffer<[u8]>,
        storage_offset: u64,
    ) -> Box<dyn GpuFuture> {
        let partition_count = round_up(
            if indirect_buffer.is_some() { element_count } else { element_count },
            PARTITION_SIZE
        );

        let element_count_size = std::mem::size_of::<u32>() as u64;
        let histogram_size = histogram_size(element_count);
        let inout_size = inout_size(element_count);

        let element_count_offset = storage_offset;
        let histogram_offset = element_count_offset + element_count_size;
        let inout_offset = histogram_offset + histogram_size;

        // Begin building command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Set element count
        if let Some(indirect_buf) = indirect_buffer {
            // Copy from indirect buffer
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    indirect_buf,
                    storage_buffer.clone(),
                ))
                .unwrap();
        } else {
            // Use provided element count
            let element_count_bytes = element_count.to_ne_bytes();
            builder
                .update_buffer(storage_buffer.clone(), element_count_offset, &element_count_bytes)
                .unwrap();
        }

        // Reset global histogram
        builder
            .fill_buffer(FillBufferInfo::buffer(
                storage_buffer.clone(),
                histogram_offset..(histogram_offset + 4 * RADIX as u64 * std::mem::size_of::<u32>() as u64),
            ))
            .unwrap();

        // Get buffer device addresses
        let storage_address = storage_buffer.device_address().unwrap().get();
        let keys_address = keys_buffer.device_address().unwrap().get();
        let values_address = values_buffer.map(|buf| buf.device_address().unwrap().get()).unwrap_or(0);

        // Sort in 4 passes (for 32 bit keys)
        for i in 0..4 {
            let pass = i;
            
            // Set up push constants
            let mut keys_in_reference = keys_address + keys_offset;
            let mut keys_out_reference = storage_address + inout_offset;
            let mut values_in_reference = values_address + values_offset;
            let mut values_out_reference = storage_address + inout_offset + inout_size;

            // Swap references for odd passes
            if i % 2 == 1 {
                std::mem::swap(&mut keys_in_reference, &mut keys_out_reference);
                std::mem::swap(&mut values_in_reference, &mut values_out_reference);
            }

            let push_constants = PushConstants {
                pass,
                element_count_reference: storage_address + element_count_offset,
                global_histogram_reference: storage_address + histogram_offset,
                partition_histogram_reference: storage_address + histogram_offset + std::mem::size_of::<u32>() as u64 * 4 * RADIX as u64,
                keys_in_reference,
                keys_out_reference,
                values_in_reference,
                values_out_reference,
            };

            // Upsweep pass
            builder
                .bind_pipeline_compute(self.upsweep_pipeline.clone())
                .unwrap()
                .push_constants(self.pipeline_layout.clone(), 0, push_constants)
                .unwrap()
                .dispatch([partition_count, 1, 1])
                .unwrap();

            // Spine pass
            builder
                .bind_pipeline_compute(self.spine_pipeline.clone())
                .unwrap()
                .push_constants(self.pipeline_layout.clone(), 0, push_constants)
                .unwrap()
                .dispatch([RADIX, 1, 1])
                .unwrap();

            // Downsweep pass
            if values_buffer.is_some() {
                builder
                    .bind_pipeline_compute(self.downsweep_key_value_pipeline.clone())
                    .unwrap();
            } else {
                builder
                    .bind_pipeline_compute(self.downsweep_pipeline.clone())
                    .unwrap();
            }
            
            builder
                .push_constants(self.pipeline_layout.clone(), 0, push_constants)
                .unwrap()
                .dispatch([partition_count, 1, 1])
                .unwrap();
        }

        // Execute command buffer
        let command_buffer = builder.build().unwrap();

        previous_future
            .then_execute(queue, command_buffer)
            .unwrap()
            .boxed()
    }
}