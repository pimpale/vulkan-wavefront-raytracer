// Source:
// https://github.com/jaesung-cs/vulkan_radix_sort

use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::GpuFuture,
};

mod shader;
use shader::downsweep;
use shader::downsweep_key_value;
use shader::spine;
use shader::upsweep;

// Constants
const RADIX: u32 = 256;
const WORKGROUP_SIZE: u32 = 512;
const PARTITION_DIVISION: u32 = 8;
const PARTITION_SIZE: u32 = PARTITION_DIVISION * WORKGROUP_SIZE;

// No need for custom round_up function anymore as we'll use Rust's div_ceil

fn histogram_size(element_count: u32) -> u64 {
    (1 + 4 * RADIX + element_count.div_ceil(PARTITION_SIZE) * RADIX) as u64
        * std::mem::size_of::<u32>() as u64
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
    ) -> Self {
        // Get device properties (limit of workgroup size)
        let properties = device.physical_device().properties();
        let max_workgroup_size = properties.max_compute_work_group_size[0];

        // Create upsweep pipeline
        let upsweep_pipeline = {
            let cs = upsweep::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        // Create spine pipeline
        let spine_pipeline = {
            let cs = spine::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        // Create downsweep pipeline
        let downsweep_pipeline = {
            let cs = downsweep::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        // Create downsweep key value pipeline
        let downsweep_key_value_pipeline = {
            let cs = downsweep_key_value::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        Self {
            device,
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
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        }
    }

    pub fn get_key_value_storage_requirements(
        &self,
        max_element_count: u32,
    ) -> SorterStorageRequirements {
        let element_count_size = std::mem::size_of::<u32>() as u64;
        let histogram_size = histogram_size(max_element_count);
        let inout_size = inout_size(max_element_count);

        let histogram_offset = element_count_size;
        let inout_offset = histogram_offset + histogram_size;
        // 2x for key value
        let storage_size = inout_offset + 2 * inout_size;

        SorterStorageRequirements {
            size: storage_size,
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        }
    }

    pub fn sort<T: GpuFuture>(
        &self,
        previous_future: T,
        queue: Arc<Queue>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        storage_buffer: Subbuffer<[u32]>,
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
        values_buffer: Subbuffer<[u32]>,
        storage_buffer: Subbuffer<[u32]>,
    ) -> Box<dyn GpuFuture> {
        self.gpu_sort(
            previous_future,
            queue,
            element_count,
            None,
            keys_buffer,
            Some(values_buffer),
            storage_buffer,
        )
    }

    // Implementation of sort algorithm
    fn gpu_sort<T: GpuFuture>(
        &self,
        previous_future: T,
        queue: Arc<Queue>,
        element_count: u32,
        indirect_buffer: Option<Subbuffer<[u32]>>,
        keys_buffer: Subbuffer<[u32]>,
        values_buffer: Option<Subbuffer<[u32]>>,
        storage_buffer: Subbuffer<[u32]>,
    ) -> Box<dyn GpuFuture> {
        let partition_count = if indirect_buffer.is_some() {
            element_count.div_ceil(PARTITION_SIZE)
        } else {
            element_count.div_ceil(PARTITION_SIZE)
        };

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
                .update_buffer(
                    storage_buffer.slice(element_count_offset..(element_count_offset + std::mem::size_of::<u32>() as u64)),
                    element_count_offset,
                    &element_count_bytes,
                )
                .unwrap();
        }

        // Reset global histogram
        builder
            .fill_buffer(
                storage_buffer.slice(
                    histogram_offset
                        ..(histogram_offset + 4 * RADIX as u64 * std::mem::size_of::<u32>() as u64),
                ),
                0,
            )
            .unwrap();

        // Get buffer device addresses
        let storage_address = storage_buffer.device_address().unwrap().get();
        let keys_address = keys_buffer.device_address().unwrap().get();
        let values_address = values_buffer
            .map(|buf| buf.device_address().unwrap().get())
            .unwrap_or(0);

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

            let element_count_reference = storage_address + element_count_offset;
            let global_histogram_reference = storage_address + histogram_offset;
            let partition_histogram_reference = storage_address
                + histogram_offset
                + std::mem::size_of::<u32>() as u64 * 4 * RADIX as u64;

            // Upsweep pass
            unsafe {
                builder
                    .bind_pipeline_compute(self.upsweep_pipeline.clone())
                    .unwrap()
                    .push_constants(
                        self.upsweep_pipeline.layout().clone(),
                        0,
                        upsweep::PushConstant {
                            pass,
                            elementCountReference: element_count_reference,
                            globalHistogramReference: global_histogram_reference,
                            partitionHistogramReference: partition_histogram_reference,
                            keysInReference: keys_in_reference,
                        },
                    )
                    .unwrap()
                    .dispatch([partition_count, 1, 1])
                    .unwrap();
            }

            // Spine pass
            unsafe {
            builder
                .bind_pipeline_compute(self.spine_pipeline.clone())
                .unwrap()
                .push_constants(
                    self.spine_pipeline.layout().clone(),
                    0,
                    spine::PushConstant {
                        pass,
                        elementCountReference: element_count_reference,
                        globalHistogramReference: global_histogram_reference,
                        partitionHistogramReference: partition_histogram_reference,
                    },
                )
                .unwrap()
                .dispatch([RADIX, 1, 1])
                .unwrap();
            }

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

            unsafe {
            builder
                .dispatch([partition_count, 1, 1])
                .unwrap();
            }
        }

        // Execute command buffer
        let command_buffer = builder.build().unwrap();

        previous_future
            .then_execute(queue, command_buffer)
            .unwrap()
            .boxed()
    }
}
