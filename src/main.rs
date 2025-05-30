use std::{error::Error, sync::Arc};

use ash::vk::SubmitInfo;
use game_system::game_world::{EntityCreationData, EntityPhysicsData, GameWorld};
use nalgebra::{Isometry3, Point3, Vector3};
use rapier3d::dynamics::RigidBodyType;

use vulkano::VulkanObject;
use vulkano::command_buffer::{CommandBufferBeginInfo, CommandBufferLevel, RecordingCommandBuffer};
use vulkano::instance::debug::{
    DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
};
use vulkano::sync::fence::FenceCreateInfo;
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    VulkanLibrary,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::Surface,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod camera;
mod game_system;
mod handle_user_input;
mod render_system;
mod utils;

use rand::RngCore;

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

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    rcx: Option<RenderContext>,
    _callback: Option<DebugUtilsMessenger>,
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

        let _callback = DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                    dbg!("Debug callback: {:?}", callback_data.message);
                })
            }),
        )
        .ok();

        App {
            instance,
            _callback,
            rcx: None,
        }
    }
}

fn test_radix_sort(device: Arc<vulkano::device::Device>, queue: Arc<vulkano::device::Queue>) {
    use rand::RngCore;
    use render_system::radix_sort::Sorter;
    use vulkano::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        command_buffer::{
            AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
            allocator::StandardCommandBufferAllocator,
        },
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
        sync,
    };

    // Allocators
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // Instantiate the sorter once – it will be reused by the individual tests
    let sorter = Sorter::new(device.clone());

    // Convenience lambda to build & submit a command buffer and wait for completion
    let submit_and_wait = |builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
        let cmd_buf = builder.build().unwrap();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), cmd_buf)
            .unwrap();
        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    };

    // same but for raw command buffer
    let submit_and_wait_raw = |builder: RecordingCommandBuffer| unsafe {
        let command_buffer = builder.end().unwrap();

        let submit_fn = queue.device().fns().v1_0.queue_submit;

        // create fence
        let fence = sync::fence::Fence::new(
            device.clone(),
            FenceCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        submit_fn(
            queue.handle(),
            1,
            &SubmitInfo::default().command_buffers(&[command_buffer.handle()]) as *const _,
            fence.handle(),
        )
        .result()
        .unwrap();

        fence.wait(None).unwrap();
    };

    // ------------- Test 1: Keys-only sorting -------------
    {
        let keys: Vec<u32> = vec![10, 5, 3, 8, 2, 1, 7, 6, 4, 9];
        let mut expected = keys.clone();
        expected.sort();

        // Buffer usage flags common to all buffers we create
        let usage_common = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST
            | BufferUsage::SHADER_DEVICE_ADDRESS;
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        // Input / output buffers
        let keys_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            keys.clone(),
        )
        .unwrap();
        let keys_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            alloc_info.clone(),
            keys.len() as u64,
        )
        .unwrap();

        // Scratch buffer required by the sorter
        let storage_req = sorter.get_storage_requirements(keys.len() as u32);
        let storage_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: storage_req.usage,
                ..Default::default()
            },
            alloc_info.clone(),
            storage_req.size,
        )
        .unwrap();

        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe {
            sorter.sort(
                &mut builder,
                keys.len() as u32,
                keys_buffer.clone(),
                storage_buffer.clone(),
                keys_out_buffer.clone(),
            );
        }

        submit_and_wait_raw(builder);

        let gpu_sorted = keys_out_buffer.read().unwrap().to_vec();
        let alt_sorted = keys_buffer.read().unwrap().to_vec();

        if gpu_sorted == expected {
            println!("[RadixSort Test] Keys-only sort: PASS (output buffer)");
        } else if alt_sorted == expected {
            println!("[RadixSort Test] Keys-only sort: PASS (input buffer)");
        } else {
            println!(
                "[RadixSort Test] Keys-only sort: FAIL\n  Expected: {:?}\n  Got (out): {:?}\n  Got (in) : {:?}",
                expected, gpu_sorted, alt_sorted
            );
        }
    }

    // ------------- Test 2: Key-value sorting -------------
    {
        let keys: Vec<u32> = vec![3, 1, 4, 5, 9, 2, 6, 8, 7, 0];
        let values: Vec<u32> = (0..keys.len() as u32).collect();

        // Expected result computed on CPU
        let mut cpu_pairs: Vec<(u32, u32)> =
            keys.iter().copied().zip(values.iter().copied()).collect();
        cpu_pairs.sort_by_key(|&(k, _)| k);
        let expected_keys: Vec<u32> = cpu_pairs.iter().map(|&(k, _)| k).collect();
        let expected_vals: Vec<u32> = cpu_pairs.iter().map(|&(_, v)| v).collect();

        // Buffer setup
        let usage_common = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST
            | BufferUsage::SHADER_DEVICE_ADDRESS;
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let keys_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            keys.clone(),
        )
        .unwrap();
        let values_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            alloc_info.clone(),
            values.clone(),
        )
        .unwrap();
        let keys_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            alloc_info.clone(),
            keys.len() as u64,
        )
        .unwrap();
        let values_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            alloc_info.clone(),
            keys.len() as u64,
        )
        .unwrap();

        // Scratch buffer
        let storage_req = sorter.get_storage_requirements(keys.len() as u32);
        let storage_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: storage_req.usage,
                ..Default::default()
            },
            alloc_info.clone(),
            storage_req.size,
        )
        .unwrap();

        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe {
            sorter.sort_key_value(
                &mut builder,
                keys.len() as u32,
                keys_buffer.clone(),
                values_buffer.clone(),
                storage_buffer.clone(),
                keys_out_buffer.clone(),
                values_out_buffer.clone(),
            );
        }

        submit_and_wait_raw(builder);

        let gpu_keys_out = keys_out_buffer.read().unwrap().to_vec();
        let gpu_vals_out = values_out_buffer.read().unwrap().to_vec();
        let alt_keys = keys_buffer.read().unwrap().to_vec();
        let alt_vals = values_buffer.read().unwrap().to_vec();

        let pass_primary = gpu_keys_out == expected_keys && gpu_vals_out == expected_vals;
        let pass_alt = alt_keys == expected_keys && alt_vals == expected_vals;

        if pass_primary {
            println!("[RadixSort Test] Key-value sort: PASS (output buffers)");
        } else if pass_alt {
            println!("[RadixSort Test] Key-value sort: PASS (input buffers)");
        } else {
            println!(
                "[RadixSort Test] Key-value sort: FAIL\n  Expected keys : {:?}\n  Expected vals : {:?}\n  Got (out) keys: {:?}\n  Got (out) vals: {:?}",
                expected_keys, expected_vals, gpu_keys_out, gpu_vals_out
            );
        }
    }

    // ------------- Test 3: Large random keys-only sorting -------------
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        const N: usize = 3_145_729; // Non power-of-two (>3 million)
        println!(
            "[RadixSort Test] Generating {} random keys (keys-only test)...",
            N
        );

        // Deterministic RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let keys: Vec<u32> = (0..N).map(|_| rng.next_u32()).collect();

        let mut expected = keys.clone();
        expected.sort();

        let usage_common = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST
            | BufferUsage::SHADER_DEVICE_ADDRESS;

        // Input / output buffers
        let keys_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            keys.clone(),
        )
        .unwrap();

        let keys_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            N as u64,
        )
        .unwrap();

        // Scratch/storage buffer
        let storage_req = sorter.get_storage_requirements(N as u32);
        let storage_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: storage_req.usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            storage_req.size,
        )
        .unwrap();

        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe {
            sorter.sort(
                &mut builder,
                keys.len() as u32,
                keys_buffer.clone(),
                storage_buffer.clone(),
                keys_out_buffer.clone(),
            );
        }

        submit_and_wait_raw(builder);

        // Validate results (check whichever buffer holds the sorted data)
        let gpu_out = keys_out_buffer.read().unwrap().to_vec();
        if gpu_out.as_slice() == expected.as_slice() {
            println!("[RadixSort Test] Large keys-only sort: PASS (output buffer)");
        } else {
            let gpu_alt = keys_buffer.read().unwrap().to_vec();
            if gpu_alt.as_slice() == expected.as_slice() {
                println!("[RadixSort Test] Large keys-only sort: PASS (input buffer)");
            } else {
                println!("[RadixSort Test] Large keys-only sort: FAIL (mismatch detected)");
            }
        }
    }

    // ------------- Test 4: Large random key-value sorting -------------
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        const N: usize = 3_145_729; // Same non power-of-two size
        println!(
            "[RadixSort Test] Generating {} random key/value pairs (key-value test)...",
            N
        );

        let mut rng = StdRng::seed_from_u64(0xCAFEBABEu64);
        let keys: Vec<u32> = (0..N).map(|_| rng.next_u32()).collect();
        let values: Vec<u32> = (0..N as u32).collect();

        // Expected CPU sort
        let mut cpu_pairs: Vec<(u32, u32)> =
            keys.iter().copied().zip(values.iter().copied()).collect();
        cpu_pairs.sort_by_key(|&(k, _)| k);
        let expected_keys: Vec<u32> = cpu_pairs.iter().map(|&(k, _)| k).collect();
        let expected_vals: Vec<u32> = cpu_pairs.iter().map(|&(_, v)| v).collect();

        let usage_common = BufferUsage::STORAGE_BUFFER
            | BufferUsage::TRANSFER_SRC
            | BufferUsage::TRANSFER_DST
            | BufferUsage::SHADER_DEVICE_ADDRESS;

        // Input buffers
        let keys_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            keys.clone(),
        )
        .unwrap();

        let values_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            values.clone(),
        )
        .unwrap();

        // Output buffers
        let keys_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            N as u64,
        )
        .unwrap();

        let values_out_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage_common,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            N as u64,
        )
        .unwrap();

        // Scratch buffer
        let storage_req = sorter.get_storage_requirements(N as u32);
        let storage_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: storage_req.usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            storage_req.size,
        )
        .unwrap();

        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe {
            sorter.sort_key_value(
                &mut builder,
                keys.len() as u32,
                keys_buffer.clone(),
                values_buffer.clone(),
                storage_buffer.clone(),
                keys_out_buffer.clone(),
                values_out_buffer.clone(),
            );
        }

        submit_and_wait_raw(builder);

        // Validate results
        let gpu_keys_out = keys_out_buffer.read().unwrap().to_vec();
        let gpu_vals_out = values_out_buffer.read().unwrap().to_vec();

        let mut pass = gpu_keys_out.as_slice() == expected_keys.as_slice()
            && gpu_vals_out.as_slice() == expected_vals.as_slice();

        if !pass {
            let alt_keys = keys_buffer.read().unwrap().to_vec();
            let alt_vals = values_buffer.read().unwrap().to_vec();
            pass = alt_keys.as_slice() == expected_keys.as_slice()
                && alt_vals.as_slice() == expected_vals.as_slice();
        }

        if pass {
            println!("[RadixSort Test] Large key-value sort: PASS");
        } else {
            println!("[RadixSort Test] Large key-value sort: FAIL (mismatch detected)");
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
                .create_window(
                    Window::default_attributes()
                        .with_title("FLOATING")
                        .with_inner_size(PhysicalSize::new(1920, 1080)),
                )
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

        // // Run standalone radix-sort tests before we continue with the normal setup
        // println!("Running radix-sort tests...");
        // test_radix_sort(device.clone(), general_queue.clone());
        // // sleep for 1 seconds
        // println!("Sleeping for 1 seconds...");
        // std::thread::sleep(std::time::Duration::from_secs(1));
        // std::process::exit(0);

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
                    rcx.frame_count = 0;
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
