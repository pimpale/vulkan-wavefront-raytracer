use std::sync::Arc;

use ash::vk::{Fence, PresentInfoKHR, SubmitInfo};
use image::RgbaImage;
use nalgebra::{Point3, Vector3};
use rand::RngCore;
use vulkano::{
    Validated, VulkanError, VulkanObject,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
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
    swapchain::{
        self, AcquireNextImageInfo, AcquiredImage, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{
        self, GpuFuture,
        fence::{FenceCreateFlags, FenceCreateInfo},
        future::FenceSignalFuture,
    },
};
use winit::window::Window;

use crate::camera::RenderingPreferences;

use super::{
    bvh::BvhNode,
    radix_sort::{Sorter, SorterStorageRequirements},
    scene::Scene,
    shader::{compact, nee_pdf, outgoing_radiance, postprocess, raygen, raytrace},
    vertex::InstanceData,
};

const MIN_IMAGE_COUNT: usize = 2;

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
                            | BufferUsage::SHADER_DEVICE_ADDRESS
                    } else {
                        BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS
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
    // sorter (used to sort the bounces)
    sorter: Sorter,
    sorter_storage: Vec<Subbuffer<[u32]>>,
    ray_origins: Vec<Subbuffer<[f32]>>,
    ray_directions: Vec<Subbuffer<[f32]>>,
    // each thread looks up the memory location of the bounce in the this array
    bounce_indices: Vec<Subbuffer<[u32]>>,
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
    // the sort keys for each bounce
    sort_keys: Vec<Subbuffer<[u32]>>,
    debug_info: Vec<Subbuffer<[f32]>>,
    frame_finished_rendering2: Vec<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>,
    frame_swapchain_image_acquired_semaphore: Vec<Arc<sync::semaphore::Semaphore>>,
    frame_finished_rendering_semaphore: Vec<Arc<sync::semaphore::Semaphore>>,
    frame_finished_rendering_fence: Vec<Arc<sync::fence::Fence>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    raygen_pipeline: Arc<ComputePipeline>,
    compact_pipeline: Arc<ComputePipeline>,
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

        let compact_pipeline = {
            let cs = compact::load(device.clone())
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

        let frame_swapchain_image_acquired_semaphore = (0..swapchain_images.len())
            .map(|_| {
                Arc::new(
                    sync::semaphore::Semaphore::new(device.clone(), Default::default()).unwrap(),
                )
            })
            .collect();

        let frame_finished_rendering_semaphore = (0..swapchain_images.len())
            .map(|_| {
                Arc::new(
                    sync::semaphore::Semaphore::new(device.clone(), Default::default()).unwrap(),
                )
            })
            .collect();

        // note that all fences start signaled. This is because we want to wait for the fence to be signaled before we can present the image.
        let frame_finished_rendering_fence = (0..swapchain_images.len())
            .map(|_| {
                Arc::new(
                    sync::fence::Fence::new(
                        device.clone(),
                        FenceCreateInfo {
                            flags: FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            })
            .collect();

        let sorter = Sorter::new(device.clone());

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
            compact_pipeline,
            nee_pdf_pipeline,
            outgoing_radiance_pipeline,
            postprocess_pipeline,
            descriptor_set_allocator,
            swapchain_images,
            swapchain_image_views,
            frame_swapchain_image_acquired_semaphore,
            frame_finished_rendering_semaphore,
            frame_finished_rendering_fence,
            frame_finished_rendering2: frame_futures,
            memory_allocator,
            wdd_needs_rebuild: false,
            material_descriptor_set,
            frame_count: 0,
            sorter,
            sorter_storage: vec![],
            // buffers (to be created)
            ray_origins: vec![],
            ray_directions: vec![],
            bounce_indices: vec![],
            bounce_normals: vec![],
            bounce_emissivity: vec![],
            bounce_reflectivity: vec![],
            bounce_nee_mis_weight: vec![],
            bounce_bsdf_pdf: vec![],
            bounce_nee_pdf: vec![],
            bounce_outgoing_radiance: vec![],
            bounce_omega_sampling_pdf: vec![],
            sort_keys: vec![],
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
        self.bounce_indices = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * (self.num_bounces + 1),
        );
        self.bounce_indices = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1 * self.num_bounces,
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
        self.sort_keys = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            1,
        );
        // debug info (single image)
        self.debug_info = window_size_dependent_setup(
            self.memory_allocator.clone(),
            &self.swapchain_images,
            true,
            self.scale,
            3,
        );

        self.sorter_storage = self
            .swapchain_images
            .iter()
            .map(|image| image.extent())
            .map(|extent| {
                let SorterStorageRequirements { size, usage } =
                    self.sorter.get_storage_requirements(
                        extent[0] * extent[1] * self.scale * self.scale as u32,
                    );

                Buffer::new_slice::<u32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    size,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
    }

    fn group_count_1d(&self, extent: &[u32; 2]) -> [u32; 3] {
        [
            (extent[0] * extent[1] * self.scale * self.scale).div_ceil(1024),
            1,
            1,
        ]
    }

    fn group_count_2d(&self, extent: &[u32; 2]) -> [u32; 3] {
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
        unsafe {
            // wait for the last fence to be signaled (signaled = not in flight)
            self.frame_finished_rendering_fence[(self.frame_count) % MIN_IMAGE_COUNT]
                .wait(None)
                .unwrap();
            self.frame_finished_rendering_fence[(self.frame_count) % MIN_IMAGE_COUNT]
                .reset()
                .unwrap();


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
            let AcquiredImage {
                image_index,
                is_suboptimal,
            } = {
                match self
                    .swapchain
                    .acquire_next_image(&AcquireNextImageInfo {
                        semaphore: Some(
                            self.frame_swapchain_image_acquired_semaphore
                                [self.frame_count % MIN_IMAGE_COUNT]
                                .clone(),
                        ),
                        ..Default::default()
                    })
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        println!("swapchain out of date (at acquire)");
                        self.wdd_needs_rebuild = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                }
            };

            if is_suboptimal {
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

            // blank the debug info buffer
            builder
                .fill_buffer(
                    self.debug_info[self.frame_count % MIN_IMAGE_COUNT]
                        .clone()
                        .reinterpret::<[u32]>(),
                    0,
                )
                .unwrap();

            // dispatch raygen pipeline
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
                        WriteDescriptorSet::buffer(
                            2,
                            self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT].clone(),
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
                .dispatch(self.group_count_2d(&rt_extent))
                .unwrap();

            let ray_count = (rt_extent[0] * rt_extent[1]) as u64;
            let sect_sz = size_of::<f32>() as u64 * ray_count;

            // dispatch raytrace pipeline
            for bounce in 0..self.num_bounces {
                // for bounce in 0..0 {
                let b = bounce as u64;

                // sort the rays (if we are not the first bounce)
                if bounce > 0 {
                    // NOTE: for now, we just copy
                    builder
                        .copy_buffer(CopyBufferInfo::buffers(
                            self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT]
                                .clone()
                                .slice(0..ray_count),
                            self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT]
                                .clone()
                                .slice(b * ray_count..(b + 1) * ray_count),
                        ))
                        .unwrap();

                    // self.sorter.sort_key_value(
                    //     &mut builder,
                    //     ray_count as u32,
                    //     // keys in (morton codes)
                    //     self.sort_keys[self.frame_count % MIN_IMAGE_COUNT].clone(),
                    //     // values in (index of the ray in memory (which is the same as the bounce index at the first bounce)
                    //     self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT]
                    //         .clone()
                    //         .slice(0..ray_count),
                    //     self.sorter_storage[self.frame_count % MIN_IMAGE_COUNT].clone(),
                    //     // keys out (we don't care about the sorted keys)
                    //     self.debug_info[self.frame_count % MIN_IMAGE_COUNT]
                    //         .clone()
                    //         .reinterpret(),
                    //     // values out (needs to be written to the bounce indices buffer that will be used for the next bounce)
                    //     self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT]
                    //         .clone()
                    //         .slice(b * ray_count..(b + 1) * ray_count),
                    // );
                }

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
                    .unwrap()
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
                            // input bounce index
                            WriteDescriptorSet::buffer_with_range(
                                4,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_indices[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            // output ray origin
                            WriteDescriptorSet::buffer_with_range(
                                5,
                                DescriptorBufferInfo {
                                    buffer: self.ray_origins[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            // output ray direction
                            WriteDescriptorSet::buffer_with_range(
                                6,
                                DescriptorBufferInfo {
                                    buffer: self.ray_directions[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: (b + 1) * 3 * sect_sz..(b + 2) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                7,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_normals[self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                8,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_emissivity
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                9,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_reflectivity
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * 3 * sect_sz..(b + 1) * 3 * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                10,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_nee_mis_weight
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer_with_range(
                                11,
                                DescriptorBufferInfo {
                                    buffer: self.bounce_bsdf_pdf
                                        [self.frame_count % MIN_IMAGE_COUNT]
                                        .as_bytes()
                                        .clone(),
                                    range: b * sect_sz..(b + 1) * sect_sz,
                                },
                            ),
                            WriteDescriptorSet::buffer(
                                12,
                                self.sort_keys[self.frame_count % MIN_IMAGE_COUNT].clone(),
                            ),
                            WriteDescriptorSet::buffer(
                                13,
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
                    .dispatch(self.group_count_1d(&rt_extent))
                    .unwrap();
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
                    .dispatch(self.group_count_2d(&rt_extent))
                    .unwrap();
            }

            // compute the outgoing radiance at all bounces
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
                .dispatch(self.group_count_2d(&rt_extent))
                .unwrap();

            // aggregate the samples and write to swapchain image
            let poolsize = 1;
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
                        srcscale: poolsize * self.scale,
                        dstscale: poolsize,
                        xsize: extent[0] / poolsize,
                        ysize: extent[1] / poolsize,
                    },
                )
                .unwrap()
                .dispatch(self.group_count_2d(&[extent[0] / poolsize, &extent[1] / poolsize]))
                .unwrap();

            unsafe {
                let submit_fn = self.queue.device().fns().v1_0.queue_submit;
                let present_fn = self.queue.device().fns().khr_swapchain.queue_present_khr;

                let command_buffer_handle = builder.build().unwrap().handle();

                submit_fn(
                    self.queue.handle(),
                    1,
                    &SubmitInfo::default()
                        // we wait for the swapchain image to be acquired before submitting the command buffer
                        // since the command buffer will write to the swapchain image
                        .wait_semaphores(&[self.frame_swapchain_image_acquired_semaphore
                            [self.frame_count % MIN_IMAGE_COUNT]
                            .handle()])
                        .command_buffers(&[command_buffer_handle])
                        // we wait for
                        .signal_semaphores(&[self.frame_finished_rendering_semaphore
                            [self.frame_count % MIN_IMAGE_COUNT]
                            .handle()]) as *const _,
                    // submitting causes the fence to go to an unsignaled state
                    // once it is finished processing, the fence will be signaled again
                    // the reason we need both a fence and the semaphore is because the swapchain present function only accepts a semaphore
                    // we need to be able to check the state of the fence on the next frame though, so we need to use both
                    self.frame_finished_rendering_fence[self.frame_count % MIN_IMAGE_COUNT]
                        .handle(),
                )
                .result()
                .unwrap();

                // now we can present the image
                present_fn(
                    self.queue.handle(),
                    &PresentInfoKHR::default()
                        .wait_semaphores(&[self.frame_finished_rendering_semaphore
                            [self.frame_count % MIN_IMAGE_COUNT]
                            .handle()])
                        .swapchains(&[self.swapchain.handle()])
                        .image_indices(&[image_index]) as *const _,
                )
                .result()
                .unwrap();
            }

            // let command_buffer_handle = builder.build().unwrap().handle();

            // // try executing command buffer on queue directly, yielding a fence
            // let fence = unsafe {
            //     let submit_fn = self.queue.device().fns().v1_0.queue_submit;

            //     let fence = sync::fence::Fence::new(
            //         self.queue.device().clone(),
            //         FenceCreateInfo {
            //             ..Default::default()
            //         },
            //     )
            //     .unwrap();

            //     submit_fn(
            //         self.queue.handle(),
            //         1,
            //         &SubmitInfo::default().command_buffers(&[command_buffer_handle]) as *const _,
            //         fence.handle(),
            //     )
            //     .result()
            //     .unwrap();

            //     fence
            // };

            // let command_buffer = builder.build().unwrap();

            // let last_cycle_future = std::mem::replace(
            //     &mut self.frame_finished_rendering2[self.frame_count % MIN_IMAGE_COUNT],
            //     None,
            // );

            // match last_cycle_future {
            //     Some(f) => f.wait(None).unwrap(),
            //     None => {}
            // }

            // let future = acquire_future
            //     .then_execute(self.queue.clone(), command_buffer)
            //     .unwrap()
            //     .then_swapchain_present(
            //         self.queue.clone(),
            //         SwapchainPresentInfo::swapchain_image_index(
            //             self.swapchain.clone(),
            //             image_index,
            //         ),
            //     )
            //     .boxed()
            //     .then_signal_fence_and_flush();

            // self.frame_finished_rendering2[self.frame_count % MIN_IMAGE_COUNT] =
            //     match future.map_err(Validated::unwrap) {
            //         Ok(future) => Some(future),
            //         Err(VulkanError::OutOfDate) => {
            //             self.wdd_needs_rebuild = true;
            //             println!("swapchain out of date (at flush)");
            //             None
            //         }
            //         Err(e) => {
            //             println!("failed to flush future: {e}");
            //             None
            //         }
            //     };

            self.frame_count = self.frame_count.wrapping_add(1);
        }
    }
}
