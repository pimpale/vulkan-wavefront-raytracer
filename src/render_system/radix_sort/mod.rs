// Source:
// https://github.com/jaesung-cs/vulkan_radix_sort

use std::mem::size_of;
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer, RecordingCommandBuffer
    },
    descriptor_set::{allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayoutCreateFlags, WriteDescriptorSet},
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
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
}

// Sorter struct to hold all necessary objects
pub struct Sorter {
    device: Arc<Device>,
    upsweep_pipeline: Arc<ComputePipeline>,
    spine_pipeline: Arc<ComputePipeline>,
    downsweep_pipeline: Arc<ComputePipeline>,
    downsweep_key_value_pipeline: Arc<ComputePipeline>,
    max_workgroup_size: u32,
}

// Storage requirements struct
pub struct SorterStorageRequirements {
    pub size: u64,
    pub usage: BufferUsage,
}

impl Sorter {
    pub fn new(device: Arc<Device>) -> Self {
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

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // enable push descriptor for set 0
                layout_create_info.set_layouts[0].flags |=
                    DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;

                PipelineLayout::new(
                    device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            };

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

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // enable push descriptor for set 0
                layout_create_info.set_layouts[0].flags |=
                    DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;

                PipelineLayout::new(
                    device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            };

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

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // enable push descriptor for set 0
                layout_create_info.set_layouts[0].flags |=
                    DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;

                PipelineLayout::new(
                    device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            };

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

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // enable push descriptor for set 0
                layout_create_info.set_layouts[0].flags |=
                    DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;

                PipelineLayout::new(
                    device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            };

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
            max_workgroup_size,
        }
    }

    pub fn get_storage_requirements(&self, max_element_count: u32) -> SorterStorageRequirements {
        let element_count_size = 1;
        let histogram_size = histogram_size(max_element_count);

        SorterStorageRequirements {
            size: element_count_size + histogram_size,
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        }
    }

    pub fn sort(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        storage_buffer: Subbuffer<[u32]>,
        keys_out_buffer: Subbuffer<[u32]>,
    ) {
        self.gpu_sort(
            builder,
            element_count,
            None,
            keys_buffer,
            None,
            storage_buffer,
            keys_out_buffer,
            None,
        )
    }

    pub fn sort_key_value(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        values_buffer: Subbuffer<[u32]>,
        storage_buffer: Subbuffer<[u32]>,
        keys_out_buffer: Subbuffer<[u32]>,
        values_out_buffer: Subbuffer<[u32]>,
    ) {
        self.gpu_sort(
            builder,
            element_count,
            None,
            keys_buffer,
            Some(values_buffer),
            storage_buffer,
            keys_out_buffer,
            Some(values_out_buffer),
        )
    }

    // Implementation of sort algorithm
    fn gpu_sort(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        element_count: u32,
        indirect_buffer: Option<Subbuffer<[u32]>>,
        keys_buffer: Subbuffer<[u32]>,
        values_buffer: Option<Subbuffer<[u32]>>,
        storage_buffer: Subbuffer<[u32]>,
        keys_out_buffer: Subbuffer<[u32]>,
        values_out_buffer: Option<Subbuffer<[u32]>>,
    ) {
        let partition_count = element_count.div_ceil(PARTITION_SIZE);

        let element_count_size = 1;

        // Define offsets
        let element_count_offset = 0;
        let histogram_offset = element_count_offset + element_count_size;
        let partition_histogram_offset = histogram_offset + 4 * RADIX as u64;

        // Set element count
        if let Some(indirect_buf) = indirect_buffer {
            // Copy from indirect buffer
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    indirect_buf,
                    storage_buffer
                        .clone()
                        .slice(element_count_offset..(element_count_offset + 1)),
                ))
                .unwrap();
        } else {
            // Use provided element count
            builder
                .fill_buffer(
                    storage_buffer
                        .clone()
                        .slice(element_count_offset..(element_count_offset + 1)),
                    element_count,
                )
                .unwrap();
        }

        // Reset global histogram
        builder
            .fill_buffer(
                storage_buffer
                    .clone()
                    .slice(histogram_offset..(histogram_offset + 4 * RADIX as u64)),
                0,
            )
            .unwrap();

        // Sort in 4 passes (for 32 bit keys)
        for i in 0..4 {
            let pass = i;

            // Set up push constants using buffer slices
            let mut keys_in_reference = keys_buffer.device_address().unwrap().get();
            let mut keys_out_reference = keys_out_buffer.device_address().unwrap().get();
            let mut values_in_reference = values_buffer
                .as_ref()
                .map(|buf| buf.device_address().unwrap().get())
                .unwrap_or(0);
            let mut values_out_reference = values_out_buffer
                .as_ref()
                .map(|buf| buf.device_address().unwrap().get())
                .unwrap_or(0);

            // Swap references for odd passes
            if i % 2 == 1 {
                std::mem::swap(&mut keys_in_reference, &mut keys_out_reference);
                std::mem::swap(&mut values_in_reference, &mut values_out_reference);
            }

            let element_count_reference = storage_buffer
                .clone()
                .slice(element_count_offset..)
                .device_address()
                .unwrap()
                .get();
            let global_histogram_reference = storage_buffer
                .clone()
                .slice(histogram_offset..)
                .device_address()
                .unwrap()
                .get();
            let partition_histogram_reference = storage_buffer
                .clone()
                .slice(partition_histogram_offset..)
                .device_address()
                .unwrap()
                .get();

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
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.upsweep_pipeline.layout().clone(),
                        0,
                        vec![
                            WriteDescriptorSet::buffer(0, storage_buffer.clone()),
                            WriteDescriptorSet::buffer(1, keys_buffer.clone()),
                            WriteDescriptorSet::buffer(2, keys_out_buffer.clone()),
                        ]
                        .into(),
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
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.spine_pipeline.layout().clone(),
                        0,
                        vec![
                            WriteDescriptorSet::buffer(0, storage_buffer.clone()),
                            WriteDescriptorSet::buffer(1, keys_buffer.clone()),
                            WriteDescriptorSet::buffer(2, keys_out_buffer.clone()),
                        ]
                        .into(),
                    )
                    .unwrap()
                    .dispatch([RADIX, 1, 1])
                    .unwrap();
            }

            // Downsweep pass
            if values_buffer.is_some() {
                builder
                    .bind_pipeline_compute(self.downsweep_key_value_pipeline.clone())
                    .unwrap()
                    .push_constants(
                        self.downsweep_key_value_pipeline.layout().clone(),
                        0,
                        downsweep_key_value::PushConstant {
                            pass,
                            elementCountReference: element_count_reference,
                            globalHistogramReference: global_histogram_reference,
                            partitionHistogramReference: partition_histogram_reference,
                            keysInReference: keys_in_reference,
                            keysOutReference: keys_out_reference,
                            valuesInReference: values_in_reference,
                            valuesOutReference: values_out_reference,
                        },
                    )
                    .unwrap()
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.downsweep_key_value_pipeline.layout().clone(),
                        0,
                        vec![
                            WriteDescriptorSet::buffer(0, storage_buffer.clone()),
                            WriteDescriptorSet::buffer(1, keys_buffer.clone()),
                            WriteDescriptorSet::buffer(2, keys_out_buffer.clone()),
                            WriteDescriptorSet::buffer(3, values_buffer.as_ref().unwrap().clone()),
                            WriteDescriptorSet::buffer(
                                4,
                                values_out_buffer.as_ref().unwrap().clone(),
                            ),
                        ]
                        .into(),
                    )
                    .unwrap();
            } else {
                builder
                    .bind_pipeline_compute(self.downsweep_pipeline.clone())
                    .unwrap()
                    .push_constants(
                        self.downsweep_pipeline.layout().clone(),
                        0,
                        downsweep::PushConstant {
                            pass,
                            elementCountReference: element_count_reference,
                            globalHistogramReference: global_histogram_reference,
                            partitionHistogramReference: partition_histogram_reference,
                            keysInReference: keys_in_reference,
                            keysOutReference: keys_out_reference,
                        },
                    )
                    .unwrap()
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.downsweep_pipeline.layout().clone(),
                        0,
                        vec![
                            WriteDescriptorSet::buffer(0, storage_buffer.clone()),
                            WriteDescriptorSet::buffer(1, keys_buffer.clone()),
                            WriteDescriptorSet::buffer(2, keys_out_buffer.clone()),
                        ]
                        .into(),
                    )
                    .unwrap();
            }

            unsafe {
                builder.dispatch([partition_count, 1, 1]).unwrap();
            }
        }
    }

    // Implementation of sort algorithm
    pub fn gpu_sort_dbg(
        &self,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        element_count: u32,
        keys_buffer: Subbuffer<[u32]>,
        values_buffer: Option<Subbuffer<[u32]>>,
        storage_buffer: Subbuffer<[u32]>,
        keys_out_buffer: Subbuffer<[u32]>,
        values_out_buffer: Option<Subbuffer<[u32]>>,
    ) {
        let partition_count = element_count.div_ceil(PARTITION_SIZE);

        let element_count_size = 1;

        // Define offsets
        let element_count_offset = 0;
        let histogram_offset = element_count_offset + element_count_size;
        let partition_histogram_offset = histogram_offset + 4 * RADIX as u64;

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Use provided element count
        builder
            .fill_buffer(
                storage_buffer
                    .clone()
                    .slice(element_count_offset..(element_count_offset + 1)),
                element_count,
            )
            .unwrap();

        // Reset global histogram
        builder
            .fill_buffer(
                storage_buffer
                    .clone()
                    .slice(histogram_offset..(histogram_offset + 4 * RADIX as u64)),
                0,
            )
            .unwrap();

        // immediately submit and wait for completion on builder
        let cmd_buf = builder.build().unwrap();
        let future = vulkano::sync::now(self.device.clone())
            .then_execute(queue.clone(), cmd_buf)
            .unwrap();
        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // print element count
        println!("Element count: {}", element_count);

        // Sort in 4 passes (for 32 bit keys)
        for i in 0..4 {
            let pass = i;

            // Set up push constants using buffer slices
            let mut keys_in_reference = keys_buffer.device_address().unwrap().get();
            let mut keys_out_reference = keys_out_buffer.device_address().unwrap().get();
            let mut values_in_reference = values_buffer
                .as_ref()
                .map(|buf| buf.device_address().unwrap().get())
                .unwrap_or(0);
            let mut values_out_reference = values_out_buffer
                .as_ref()
                .map(|buf| buf.device_address().unwrap().get())
                .unwrap_or(0);

            // Swap references for odd passes
            if i % 2 == 1 {
                std::mem::swap(&mut keys_in_reference, &mut keys_out_reference);
                std::mem::swap(&mut values_in_reference, &mut values_out_reference);
            }

            let element_count_reference = storage_buffer
                .clone()
                .slice(element_count_offset..)
                .device_address()
                .unwrap()
                .get();
            let global_histogram_reference = storage_buffer
                .clone()
                .slice(histogram_offset..)
                .device_address()
                .unwrap()
                .get();
            let partition_histogram_reference = storage_buffer
                .clone()
                .slice(partition_histogram_offset..)
                .device_address()
                .unwrap()
                .get();

            // Upsweep pass
            unsafe {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

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

                let cmd_buf = builder.build().unwrap();
                let future = vulkano::sync::now(self.device.clone())
                    .then_execute(queue.clone(), cmd_buf)
                    .unwrap();
                future
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
            }

            // Spine pass
            unsafe {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

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

                let cmd_buf = builder.build().unwrap();
                let future = vulkano::sync::now(self.device.clone())
                    .then_execute(queue.clone(), cmd_buf)
                    .unwrap();
                future
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // Downsweep pass
            if values_buffer.is_some() {
                builder
                    .bind_pipeline_compute(self.downsweep_key_value_pipeline.clone())
                    .unwrap()
                    .push_constants(
                        self.downsweep_key_value_pipeline.layout().clone(),
                        0,
                        downsweep_key_value::PushConstant {
                            pass,
                            elementCountReference: element_count_reference,
                            globalHistogramReference: global_histogram_reference,
                            partitionHistogramReference: partition_histogram_reference,
                            keysInReference: keys_in_reference,
                            keysOutReference: keys_out_reference,
                            valuesInReference: values_in_reference,
                            valuesOutReference: values_out_reference,
                        },
                    )
                    .unwrap();
            } else {
                builder
                    .bind_pipeline_compute(self.downsweep_pipeline.clone())
                    .unwrap()
                    .push_constants(
                        self.downsweep_pipeline.layout().clone(),
                        0,
                        downsweep::PushConstant {
                            pass,
                            elementCountReference: element_count_reference,
                            globalHistogramReference: global_histogram_reference,
                            partitionHistogramReference: partition_histogram_reference,
                            keysInReference: keys_in_reference,
                            keysOutReference: keys_out_reference,
                        },
                    )
                    .unwrap();
            }

            unsafe {
                builder.dispatch([partition_count, 1, 1]).unwrap();
            }

            let cmd_buf = builder.build().unwrap();
            let future = vulkano::sync::now(self.device.clone())
                .then_execute(queue.clone(), cmd_buf)
                .unwrap();
            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }
    }
}
