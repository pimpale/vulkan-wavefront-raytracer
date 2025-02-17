use std::sync::Arc;

use ash::vk::Fence;
use image::RgbaImage;
use nalgebra::{Point3, Vector3};
use rand::RngCore;
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceOwned, Queue,
        QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, sampler::Sampler, view::ImageView},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{GpuFuture, future::FenceSignalFuture},
};
use winit::window::Window;

use crate::camera::RenderingPreferences;

use super::{
    bvh::BvhNode,
    scene::Scene,
    shader::{nee_pdf, outgoing_radiance, postprocess, raygen, raytrace},
    vertex::InstanceData,
};

const MIN_IMAGE_COUNT: usize = 3;

pub fn get_device_for_rendering_on(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
    let device_extensions = DeviceExtensions {
        khr_acceleration_structure: true,
        khr_ray_query: true,
        khr_swapchain: true,
        khr_push_descriptor: true,
        ..DeviceExtensions::empty()
    };
    let features = DeviceFeatures {
        acceleration_structure: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        ray_query: true,
        shader_int8: true,
        shader_int64: true,
        shader_float64: true,
        storage_buffer8_bit_access: true,
        uniform_and_storage_buffer8_bit_access: true,
        runtime_descriptor_array: true,
        descriptor_binding_variable_descriptor_count: true,
        ..DeviceFeatures::empty()
    };
    let (physical_device, general_queue_family_index, transfer_queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // find a general purpose queue
            let general_queue_family_index = p
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags
                        .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                });

            // find a transfer-only queue (this will be fast for transfers)
            let transfer_queue_family_index = p
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // has transfer and sparse binding only
                    q.queue_flags == QueueFlags::TRANSFER | QueueFlags::SPARSE_BINDING
                });

            match (general_queue_family_index, transfer_queue_family_index) {
                (Some(q), Some(t)) => Some((p, q as u32, t as u32)),
                _ => None,
            }
        })
        .min_by_key(|(p, _, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: features,
            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index: general_queue_family_index,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_queue_family_index,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    )
    .unwrap();

    let general_queue = queues.next().unwrap();
    let transfer_queue = queues.next().unwrap();

    (device, general_queue, transfer_queue)
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    // Querying the capabilities of the surface. When we create the swapchain we can only
    // pass values that are allowed by the capabilities.
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: MIN_IMAGE_COUNT as u32,
            image_format: Format::B8G8R8A8_UNORM,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::STORAGE,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),

            ..Default::default()
        },
    )
    .unwrap()
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup<T: BufferContents>(
    memory_allocator: Arc<StandardMemoryAllocator>,
    images: &[Arc<Image>],
    transfer_src: bool,
    scale: u32,
    channels: u32,
) -> Vec<Subbuffer<[T]>> {
    let render_dests = images
        .iter()
        .map(|image| {
            let extent = image.extent();
            let xsize = extent[0] * scale;
            let ysize = extent[1] * scale;

            Buffer::new_slice::<T>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: if transfer_src {
                        BufferUsage::STORAGE_BUFFER
                            | BufferUsage::TRANSFER_SRC
                            | BufferUsage::TRANSFER_DST
                    } else {
                        BufferUsage::STORAGE_BUFFER
                    },
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (xsize * ysize * channels) as u64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    render_dests
}

pub fn get_surface_extent(surface: &Surface) -> [u32; 2] {
    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
    window.inner_size().into()
}

pub struct Renderer {
    scale: u32,
    num_bounces: u32,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    swapchain: Arc<Swapchain>,
    material_descriptor_set: Arc<DescriptorSet>,
    ray_origins: Vec<Subbuffer<[f32]>>,
    ray_directions: Vec<Subbuffer<[f32]>>,
    bounce_normals: Vec<Subbuffer<[f32]>>,
    bounce_emissivity: Vec<Subbuffer<[f32]>>,
    bounce_reflectivity: Vec<Subbuffer<[f32]>>,
    // balance heuristic weight to give to nee
    bounce_nee_mis_weight: Vec<Subbuffer<[f32]>>,
    // the pdf of the selected ray direction only considering the bsdf
    bounce_bsdf_pdf: Vec<Subbuffer<[f32]>>,
    // the pdf of the selected ray direction only considering light sources
    bounce_nee_pdf: Vec<Subbuffer<[f32]>>,
    // the outgoing radiance at each bounce point
    bounce_outgoing_radiance: Vec<Subbuffer<[f32]>>,
    // the sampling pdf of the next direction
    bounce_omega_sampling_pdf: Vec<Subbuffer<[f32]>>,
    debug_info: Vec<Subbuffer<[f32]>>,
    frame_finished_rendering: Vec<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    raygen_pipeline: Arc<ComputePipeline>,
    raytrace_pipeline: Arc<ComputePipeline>,
    nee_pdf_pipeline: Arc<ComputePipeline>,
    outgoing_radiance_pipeline: Arc<ComputePipeline>,
    postprocess_pipeline: Arc<ComputePipeline>,
    wdd_needs_rebuild: bool,
    frame_count: usize,
    rng: rand::prelude::ThreadRng,
}

fn load_textures(
    textures: Vec<RgbaImage>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView>> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut image_views = vec![];

    for texture in textures {
        let extent = [texture.width(), texture.height(), 1];

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            texture.into_raw(),
        )
        .unwrap();

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
            ))
            .unwrap();

        image_views.push(ImageView::new_default(image).unwrap());
    }

    let future = builder.build().unwrap().execute(queue.clone()).unwrap();

    future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    image_views
}

impl Renderer {
    pub fn new(
        surface: Arc<Surface>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        texture_atlas: Vec<(RgbaImage, RgbaImage, RgbaImage)>,
    ) -> Renderer {
        let texture_atlas = texture_atlas
            .into_iter()
            .flat_map(|(reflectivity, emissivity, metallicity)| {
                [reflectivity, emissivity, metallicity]
            })
            .collect::<Vec<_>>();

        let device = memory_allocator.device().clone();

        let (swapchain, swapchain_images) = create_swapchain(device.clone(), surface.clone());
        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>();

        let raygen_pipeline = {
            let cs = raygen::load(device.clone())
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

        let raytrace_pipeline = {
            let cs = raytrace::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // Adjust the info for set 0, binding 1 to make it variable with texture_atlas.len() descriptors.
                let binding = layout_create_info.set_layouts[0]
                    .bindings
                    .get_mut(&1)
                    .unwrap();
                binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                binding.descriptor_count = texture_atlas.len() as u32;

                // enable push descriptor for set 1
                layout_create_info.set_layouts[1].flags |=
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

        let nee_pdf_pipeline = {
            let cs = nee_pdf::load(device.clone())
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

        let outgoing_radiance_pipeline = {
            let cs = outgoing_radiance::load(device.clone())
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

        let postprocess_pipeline = {
            let cs = postprocess::load(device.clone())
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

        let texture_atlas = load_textures(
            texture_atlas,
            queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
        );

        let sampler = Sampler::new(device.clone(), Default::default()).unwrap();

        let material_descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
            raytrace_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            texture_atlas.len() as u32,
            [
                WriteDescriptorSet::sampler(0, sampler),
                WriteDescriptorSet::image_view_array(1, 0, texture_atlas),
            ],
            [],
        )
        .unwrap();

        let frame_futures = (0..swapchain_images.len()).map(|_| None).collect();

        let mut renderer = Renderer {
            scale: 1,
            num_bounces: 6,
            surface,
            command_buffer_allocator,
            device,
            queue,
            swapchain,
            raygen_pipeline,
            raytrace_pipeline,
            nee_pdf_pipeline,
            outgoing_radiance_pipeline,
            postprocess_pipeline,
            descriptor_set_allocator,
            swapchain_images,
            swapchain_image_views,
            frame_finished_rendering: frame_futures,
            memory_allocator,
            wdd_needs_rebuild: false,
            material_descriptor_set,
            frame_count: 0,
            // buffers (to be created)
            ray_origins: vec![],
            ray_directions: vec![],
            bounce_normals: vec![],
            bounce_emissivity: vec![],
            bounce_reflectivity: vec![],
            bounce_nee_mis_weight: vec![],
            bounce_bsdf_pdf: vec![],
            bounce_nee_pdf: vec![],
            bounce_outgoing_radiance: vec![],
            bounce_omega_sampling_pdf: vec![],
            debug_info: vec![],
            rng: rand::rng(),
        };

        // create buffers
        renderer.create_buffers();

        renderer
    }

    pub fn n_swapchain_images(&self) -> usize {
        self.swapchain_images.len()
    }

    pub fn rebuild(&mut self, extent: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>();
        self.create_buffers();
    }

    pub fn create_buffers(&mut self) {
        self.ray_origins = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * (self.num_bounces + 1),
        );
        self.ray_directions = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * (self.num_bounces + 1),
        );
        self.bounce_normals = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * self.num_bounces,
        );
        self.bounce_emissivity = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * self.num_bounces,
        );
        self.bounce_reflectivity = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * self.num_bounces,
        );
        self.bounce_nee_mis_weight = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * self.num_bounces,
        );
        self.bounce_bsdf_pdf = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * self.num_bounces,
        );
        self.bounce_nee_pdf = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * self.num_bounces,
        );
        self.bounce_outgoing_radiance = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3 * self.num_bounces,
        );
        self.bounce_omega_sampling_pdf = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * self.num_bounces,
        );
        // debug info (single image)
        self.debug_info = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3,
        );
    }

    fn group_count(&self, extent: &[u32; 2]) -> [u32; 3] {
        [
            extent[0].div_ceil(32) * self.scale,
            extent[1].div_ceil(32) * self.scale,
            1,
        ]
    }

    pub fn render(
        &mut self,
        scene: &mut Scene<u32>,
        eye: Point3<f32>,
        front: Vector3<f32>,
        right: Vector3<f32>,
        up: Vector3<f32>,
        rendering_preferences: RenderingPreferences,
    ) {
        let (
            top_level_acceleration_structure,
            light_top_level_acceleration_structure,
            instance_data,
            luminance_bvh,
        ) = scene.get_tlas();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if self.wdd_needs_rebuild {
            self.rebuild(get_surface_extent(&self.surface));
            self.wdd_needs_rebuild = false;
            println!("rebuilt swapchain");
        }

        // Do not draw frame when screen dimensions are zero.
        // On Windows, this can occur from minimizing the application.
        let win_extent = get_surface_extent(&self.surface);
        if win_extent[0] == 0 || win_extent[1] == 0 {
            return;
        }

        // This operation returns the index of the image that we are allowed to draw upon.
        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    println!("swapchain out of date (at acquire)");
                    self.wdd_needs_rebuild = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            self.wdd_needs_rebuild = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let extent_3d = self.swapchain_images[image_index as usize].extent();
        let extent = [extent_3d[0], extent_3d[1]];
        let rt_extent = [extent[0] * self.scale, extent[1] * self.scale];

        unsafe {
            builder
                .bind_pipeline_compute(self.raygen_pipeline.clone())
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Compute,
                    self.raygen_pipeline.layout().clone(),
                    0,
                    vec![
                        WriteDescriptorSet::buffer(
                            0,
                            self.ray_origins[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            1,
                            self.ray_directions[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                    ]
                    .into(),
                )
                .unwrap()
                .push_constants(
                    self.raygen_pipeline.layout().clone(),
                    0,
                    raygen::PushConstants {
                        camera: raygen::Camera {
                            eye: eye.coords,
                            front,
                            right,
                            up,
                            screen_size: rt_extent.into(),
                        },
                        invocation_seed: self.rng.next_u32(),
                    },
                )
                .unwrap()
                .dispatch(self.group_count(&rt_extent))
                .unwrap();
        }

        let sect_sz = (size_of::<f32>() as u32 * rt_extent[0] * rt_extent[1]) as u64;

        // bind raytrace pipeline
        builder
            .bind_pipeline_compute(self.raytrace_pipeline.clone())
            .unwrap()
            // bind material descriptor set
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.raytrace_pipeline.layout().clone(),
                0,
                self.material_descriptor_set.clone(),
            )
            .unwrap();

        // dispatch raytrace pipeline
        for bounce in 0..self.num_bounces {
            // for bounce in 0..0 {
            let b = bounce as u64;

            unsafe {
                builder
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.raytrace_pipeline.layout().clone(),
                        1,
                        vec![
                            WriteDescriptorSet::acceleration_structure(
                                0,
                                top_level_acceleration_structure.clone(),
                            ),
                            WriteDescriptorSet::buffer(1, instance_data.clone()),
                            // input ray origin
                            WriteDescriptorSet::buffer_with_range(
                                2,
                                DescriptorBufferInfo {
                                    buffer: self.ray_origins[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            // input ray direction
                            WriteDescriptorSet::buffer_with_range(
                                3,
                                DescriptorBufferInfo {
                                    buffer: self.ray_directions[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            // output ray origin
                            WriteDescriptorSet::buffer_with_range(
                                4,
                                DescriptorBufferInfo {
                                    buffer: self.ray_origins[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            // output ray direction
                            WriteDescriptorSet::buffer_with_range(
                                5,
                                DescriptorBufferInfo {
                                    buffer: self.ray_directions[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                6,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_normals[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                7,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_emissivity
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                8,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_reflectivity
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                9,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_nee_mis_weight
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                10,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_bsdf_pdf
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer(
                                11,
                                self.debug_info[self.frame_count % MIN_IMAGE_COUNT].clone(),
                            ),
                        ]
                        .into(),
                    )
                    .unwrap()
                    .push_constants(
                        self.raytrace_pipeline.layout().clone(),
                        0,
                        raytrace::PushConstants {
                            nee_type: rendering_preferences.nee_type,
                            bounce: bounce,
                            xsize: rt_extent[0],
                            ysize: rt_extent[1],
                            invocation_seed: (self.frame_count as u32) * self.num_bounces + bounce,
                            tl_bvh_addr: luminance_bvh.device_address().unwrap().get(),
                        },
                    )
                    .unwrap()
                    .dispatch(self.group_count(&rt_extent))
                    .unwrap();
            }
        }

        // bind nee pdf pipeline
        // this is done in a separate pass for better memory access patterns
        builder
            .bind_pipeline_compute(self.nee_pdf_pipeline.clone())
            .unwrap();

        // dispatch nee pdf pipeline
        for bounce in 0..(self.num_bounces - 1) {
            // for bounce in 0..0 {
            let b = bounce as u64;
            unsafe {
                // compute nee pdf
                builder
                    .push_descriptor_set(
                        PipelineBindPoint::Compute,
                        self.nee_pdf_pipeline.layout().clone(),
                        0,
                        vec![
                            WriteDescriptorSet::acceleration_structure(
                                0,
                                light_top_level_acceleration_structure.clone(),
                            ),
                            WriteDescriptorSet::buffer(1, instance_data.clone()),
                            // input intersection normal
                            WriteDescriptorSet::buffer_with_range(
                                2,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_normals[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b) * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            // input intersection location
                            WriteDescriptorSet::buffer_with_range(
                                3,
                                DescriptorBufferInfo {
                                    buffer: self.ray_origins[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            // input intersection outgoing direction
                            WriteDescriptorSet::buffer_with_range(
                                4,
                                DescriptorBufferInfo {
                                    buffer: self.ray_directions[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            // input nee mis weight
                            WriteDescriptorSet::buffer_with_range(
                                5,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_nee_mis_weight
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            // output nee pdf
                            WriteDescriptorSet::buffer_with_range(
                                6,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_nee_pdf[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                        ]
                        .into(),
                    )
                    .unwrap()
                    .push_constants(
                        self.nee_pdf_pipeline.layout().clone(),
                        0,
                        nee_pdf::PushConstants {
                            nee_type: rendering_preferences.nee_type,
                            xsize: rt_extent[0],
                            ysize: rt_extent[1],
                            tl_bvh_addr: luminance_bvh.device_address().unwrap().get(),
                        },
                    )
                    .unwrap()
                    .dispatch(self.group_count(&rt_extent))
                    .unwrap();
            }
        }

        // compute the outgoing radiance at all bounces
        unsafe {
            builder
                .bind_pipeline_compute(self.outgoing_radiance_pipeline.clone())
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Compute,
                    self.outgoing_radiance_pipeline.layout().clone(),
                    0,
                    vec![
                        // WriteDescriptorSet::buffer(
                        //     0,
                        //     self.bounce_origins[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        // ),
                        WriteDescriptorSet::buffer(
                            1,
                            self.ray_directions[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            2,
                            self.bounce_emissivity[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            3,
                            self.bounce_reflectivity[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            4,
                            self.bounce_nee_mis_weight[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            5,
                            self.bounce_bsdf_pdf[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            6,
                            self.bounce_nee_pdf[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            7,
                            self.bounce_outgoing_radiance[self.frame_count % MIN_IMAGE_COUNT]
                                .clone(),
                        ),
                        WriteDescriptorSet::buffer(
                            8,
                            self.bounce_omega_sampling_pdf[self.frame_count % MIN_IMAGE_COUNT]
                                .clone(),
                        ),
                    ]
                    .into(),
                )
                .unwrap()
                .push_constants(
                    self.outgoing_radiance_pipeline.layout().clone(),
                    0,
                    outgoing_radiance::PushConstants {
                        num_bounces: self.num_bounces,
                        xsize: rt_extent[0],
                        ysize: rt_extent[1],
                    },
                )
                .unwrap()
                .dispatch(self.group_count(&rt_extent))
                .unwrap();
        }

        // aggregate the samples and write to swapchain image
        unsafe {
            builder
                .bind_pipeline_compute(self.postprocess_pipeline.clone())
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Compute,
                    self.postprocess_pipeline.layout().clone(),
                    0,
                    vec![
                        WriteDescriptorSet::buffer_with_range(
                            0,
                            DescriptorBufferInfo {
                                buffer: self.bounce_outgoing_radiance
                                    [self.frame_count % MIN_IMAGE_COUNT]
                                    .as_bytes()
                                    .clone(),
                                range: 0 * sect_sz * 3..1 * sect_sz * 3,
                            },
                        ),
                        WriteDescriptorSet::buffer(
                            1,
                            self.debug_info[self.frame_count % MIN_IMAGE_COUNT].clone(),
                        ),
                        WriteDescriptorSet::image_view(
                            2,
                            self.swapchain_image_views[image_index as usize].clone(),
                        ),
                    ]
                    .into(),
                )
                .unwrap()
                .push_constants(
                    self.postprocess_pipeline.layout().clone(),
                    0,
                    postprocess::PushConstants {
                        debug_view: rendering_preferences.debug_view,
                        srcscale: 4*self.scale,
                        dstscale: 4,
                        xsize: extent[0]/4,
                        ysize: extent[1]/4,
                    },
                )
                .unwrap()
                .dispatch(self.group_count(&[extent[0]/4, &extent[1]/4]))
                .unwrap();
        }

        let command_buffer = builder.build().unwrap();

        let last_cycle_future = std::mem::replace(
            &mut self.frame_finished_rendering[self.frame_count % MIN_IMAGE_COUNT],
            None,
        );

        match last_cycle_future {
            Some(f) => f.wait(None).unwrap(),
            None => {}
        }

        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .boxed()
            .then_signal_fence_and_flush();

        self.frame_finished_rendering[self.frame_count % MIN_IMAGE_COUNT] =
            match future.map_err(Validated::unwrap) {
                Ok(future) => Some(future),
                Err(VulkanError::OutOfDate) => {
                    self.wdd_needs_rebuild = true;
                    println!("swapchain out of date (at flush)");
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    None
                }
            };

        self.frame_count = self.frame_count.wrapping_add(1);
    }
}
