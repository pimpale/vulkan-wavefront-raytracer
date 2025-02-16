#![feature(array_chunks)]
use std::{error::Error, sync::Arc};

use game_system::game_world::{EntityCreationData, EntityPhysicsData, GameWorld};
use nalgebra::{Isometry3, Point3, Vector3};
use rapier3d::{dynamics::RigidBodyType, geometry::ColliderBuilder};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo
    }, descriptor_set::allocator::StandardDescriptorSetAllocator, device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags
    }, image::{view::ImageView, Image, ImageUsage}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, subpass::PipelineRenderingCreateInfo, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo
        }, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo
    }, render_pass::{AttachmentLoadOp, AttachmentStoreOp}, swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo
    }, sync::{self, GpuFuture}, Validated, Version, VulkanError, VulkanLibrary
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod camera;
mod game_system;
mod handle_user_input;
mod render_system;
mod utils;

fn build_scene(
    general_queue: Arc<vulkano::device::Queue>,
    transfer_queue: Arc<vulkano::device::Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    surface: Arc<Surface>,
) -> GameWorld {
    let rd: Vec<Point3<f32>> = vec![
        [0.0, 0.0, 0.0].into(),
        [1.0, 0.0, 0.0].into(),
        [2.0, 0.0, 0.0].into(),
        [3.0, 0.0, 0.0].into(),
        [4.0, 0.0, 0.0].into(),
        [5.0, 0.0, 0.0].into(),
        [6.0, 0.0, 0.0].into(),
        [7.0, 0.0, 0.0].into(),
        [8.0, 0.0, 0.0].into(),
        [9.0, 0.0, 0.0].into(),
        [10.0, 0.0, 0.0].into(),
        [11.0, 0.0, 0.0].into(),
        [12.0, 0.0, 0.0].into(),
        [13.0, 0.0, 0.0].into(),
        [14.0, 0.0, 0.0].into(),
        [15.0, 0.0, 0.0].into(),
        [15.0, 0.0, 1.0].into(),
        [15.0, 0.0, 2.0].into(),
        [15.0, 0.0, 3.0].into(),
        [15.0, 0.0, 4.0].into(),
        [15.0, 0.0, 5.0].into(),
        [15.0, 0.0, 6.0].into(),
        [15.0, 0.0, 7.0].into(),
        [15.0, 0.0, 8.0].into(),
        [15.0, 0.0, 9.0].into(),
        [15.0, 0.0, 10.0].into(),
        [15.0, 0.0, 11.0].into(),
        [15.0, 0.0, 12.0].into(),
        [15.0, 0.0, 13.0].into(),
        [15.0, 0.0, 14.0].into(),
        [15.0, 0.0, 15.0].into(),
    ];

    let g: Vec<Point3<f32>> = vec![[0.0, -0.1, -50.0].into(), [0.0, -0.1, 50.0].into()];

    let mut world = GameWorld::new(
        general_queue,
        transfer_queue,
        command_buffer_allocator,
        memory_allocator.clone(),
        descriptor_set_allocator,
        0,
        surface,
        Box::new(camera::SphericalCamera::new()),
    );

    let uploader = world.scene_uploader().clone();

    // add ego agent
    let ego_mesh = utils::unitcube();
    world.add_entity(
        0,
        EntityCreationData {
            physics: Some(EntityPhysicsData {
                rigid_body_type: RigidBodyType::KinematicVelocityBased,
                hitbox: utils::get_aabb_hitbox(&ego_mesh),
                //                hitbox: ColliderBuilder::capsule_y(0.5, 0.5).build(),
                linvel: Vector3::zeros(),
                angvel: Vector3::zeros(),
                controlled: true,
                grounded: false,
            }),
            mesh: uploader.upload_object(utils::unitcube()),
            isometry: Isometry3::translation(0.0, 5.0, 0.0),
        },
    );

    // // add road
    // world.add_entity(
    //     1,
    //     EntityCreationData {
    //         physics: None,
    //         mesh: utils::flat_polyline(rd.clone(), 1.0, [0.5, 0.5, 0.5]),
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // // add road yellow line
    // world.add_entity(
    //     2,
    //     EntityCreationData {
    //         physics: None,
    //         mesh: utils::flat_polyline(
    //             rd.iter().map(|v| v + Vector3::new(0.0, 0.1, 0.0)).collect(),
    //             0.1,
    //             [1.0, 1.0, 0.0],
    //         ),
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // // add ground
    // let ground_mesh = utils::flat_polyline(g.clone(), 50.0, [0.5, 1.0, 0.5]);
    // world.add_entity(
    //     3,
    //     EntityCreationData {
    //         physics: Some(EntityPhysicsData {
    //             rigid_body_type: rapier3d::dynamics::RigidBodyType::Fixed,
    //             hitbox: utils::get_aabb_hitbox(&ground_mesh),
    //             linvel: Vector3::zeros(),
    //             angvel: Vector3::zeros(),
    //             controlled: false,
    //             grounded: false,
    //         }),
    //         mesh: ground_mesh,
    //         isometry: Isometry3::identity(),
    //     },
    // );

    // add blas test
    // let blas_test_mesh = render_system::bvh::build::test_blas();
    // world.add_entity(
    //     4,
    //     EntityCreationData {
    //         physics: None,
    //         mesh: uploader.upload_object(blas_test_mesh),
    //         isometry: Isometry3::identity(),
    //     },
    // );

    world
}

// fn main() {
//     let library = VulkanLibrary::new().unwrap();
//     let event_loop = EventLoop::new().unwrap();
//     let required_extensions = Surface::required_extensions(&event_loop).unwrap();

//     let instance = Instance::new(
//         library,
//         InstanceCreateInfo {
//             flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
//             enabled_extensions: required_extensions,
//             ..Default::default()
//         },
//     )
//     .unwrap();

//     let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

//     let surface = Surface::from_window(instance.clone(), window).unwrap();

//     let (device, general_queue, transfer_queue) =
//         render_system::interactive_rendering::get_device_for_rendering_on(
//             instance.clone(),
//             surface.clone(),
//         );

//     //Print some info about the device currently being used
//     println!(
//         "Using device: {} (type: {:?})",
//         device.physical_device().properties().device_name,
//         device.physical_device().properties().device_type
//     );

//     let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
//     let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
//         device.clone(),
//         Default::default(),
//     ));
//     let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
//         device.clone(),
//         Default::default(),
//     ));

//     let mut start_time = std::time::Instant::now();
//     let mut frame_count = 0;

//     let mut world = build_scene(
//         general_queue.clone(),
//         transfer_queue.clone(),
//         command_buffer_allocator.clone(),
//         memory_allocator.clone(),
//         descriptor_set_allocator.clone(),
//         surface.clone(),
//     );

//     event_loop.set_control_flow(ControlFlow::Poll);

//     event_loop
//         .run(move |event, active_event_loop| match event {
//             Event::WindowEvent {
//                 event: WindowEvent::CloseRequested,
//                 ..
//             } => {
//                 active_event_loop.exit();
//             }
//             Event::WindowEvent { event, .. } => {
//                 world.handle_window_event(event);
//             }
//             Event::AboutToWait => {}
//             _ => (),
//         })
//         .unwrap();
// }

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    world: GameWorld,
    frame_count: u64,
    start_time: std::time::Instant,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();

        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw
        // to a window.
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        App {
            instance,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The objective of this example is to draw a triangle on a window. To do so, we first need
        // to create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
        //
        // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
        // object from it, which represents the drawable surface of a window. For that we must wrap
        // the `winit::window::Window` in an `Arc`.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let (device, general_queue, transfer_queue) =
            render_system::interactive_rendering::get_device_for_rendering_on(
                self.instance.clone(),
                surface.clone(),
            );

        //Print some info about the device currently being used
        println!(
            "Using device: {} (type: {:?})",
            device.physical_device().properties().device_name,
            device.physical_device().properties().device_type
        );

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let world = build_scene(
            general_queue.clone(),
            transfer_queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            surface.clone(),
        );

        self.rcx = Some(RenderContext {
            window,
            world,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // print fps
                rcx.frame_count += 1;
                let elapsed = rcx.start_time.elapsed();
                if elapsed.as_secs() >= 1 {
                    println!("fps: {}", rcx.frame_count);
                   rcx. frame_count = 0;
                    rcx.start_time = std::time::Instant::now();
                }

                // game step and render
                rcx.world.step();
            }
            _ => rcx.world.handle_window_event(event),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}
