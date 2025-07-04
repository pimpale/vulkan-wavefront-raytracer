use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::fs;
use std::path::Path;

use nalgebra::Isometry3;
use nalgebra::Point3;
use nalgebra::Vector3;

use rapier3d::dynamics::RigidBodyType;
use rapier3d::geometry::Collider;
use threadpool::ThreadPool;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::device::DeviceOwned;
use vulkano::device::Queue;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;

use crate::camera::InteractiveCamera;
use crate::camera::RenderingPreferences;
use crate::game_system::block::BlockDefinitionTable;
use crate::game_system::block::BlockIdx;
use crate::game_system::chunk_manager::ChunkManager;
use crate::game_system::ego_controls_manager::EgoControlsManager;
use crate::game_system::manager::Manager;
use crate::game_system::manager::UpdateData;
use crate::game_system::physics_manager::PhysicsManager;
use crate::game_system::scene_manager::SceneManager;
use crate::render_system::interactive_rendering;
use crate::render_system::scene::Scene;
use crate::render_system::scene::SceneUploadedObjectHandle;
use crate::render_system::scene::SceneUploader;
use crate::utils;

#[derive(Clone)]
pub struct EntityPhysicsData {
    pub rigid_body_type: RigidBodyType,
    pub hitbox: Collider,
    pub linvel: Vector3<f32>,
    pub angvel: Vector3<f32>,
    pub controlled: bool,
    pub grounded: bool,
}

pub struct EntityCreationData {
    // if not specified then the object is visual only
    pub physics: Option<EntityPhysicsData>,
    // scene object handle
    pub mesh: SceneUploadedObjectHandle,
    // initial transformation
    // position and rotation in space
    pub isometry: Isometry3<f32>,
}

pub struct Entity {
    // mesh (untransformed)
    pub mesh: SceneUploadedObjectHandle,
    // transformation from origin
    pub isometry: Isometry3<f32>,
    // physics
    pub physics_data: Option<EntityPhysicsData>,
}

pub enum WorldChange {
    GlobalEntityAdd(u32, EntityCreationData),
    GlobalEntityRemove(u32),
    GlobalEntityUpdateIsometry(u32, Isometry3<f32>),
    GlobalEntityUpdateVelocity {
        id: u32,
        linvel: Vector3<f32>,
        angvel: Vector3<f32>,
    },
    GlobalEntityUpdateGroundedness(u32, bool),
    PhysicsSetVelocity {
        id: u32,
        linvel: Vector3<f32>,
        angvel: Vector3<f32>,
    },
    PhysicsApplyImpulse {
        id: u32,
        impulse: Vector3<f32>,
        torque_impulse: Vector3<f32>,
    },
    WorldSetBlock {
        global_coords: Point3<i32>,
        block_id: BlockIdx,
    },
}

pub struct GameWorld {
    general_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    entities: HashMap<u32, Entity>,
    ego_entity_id: u32,
    scene: Rc<RefCell<Scene<u32>>>,
    scene_uploader: SceneUploader,
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    surface: Arc<Surface>,
    renderer: interactive_rendering::Renderer,

    // manager data
    events_since_last_step: Vec<winit::event::WindowEvent>,
    changes_since_last_step: Vec<WorldChange>,
    managers: Vec<Box<dyn Manager>>,
}

impl GameWorld {
    pub fn wait_idle(&self) {
        dbg!("waiting for transfer queue to be idle");
        self.transfer_queue
            .with(|mut queue| queue.wait_idle().unwrap());
        dbg!("waiting for general queue to be idle");
        self.general_queue
            .with(|mut queue| queue.wait_idle().unwrap());
        dbg!("done waiting for queues to be idle");
    }

    pub fn new(
        general_queue: Arc<Queue>,
        transfer_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        ego_entity_id: u32,
        surface: Arc<Surface>,
        camera: Box<dyn InteractiveCamera>,
    ) -> GameWorld {
        let device = general_queue.device();

        assert!(device == memory_allocator.device());

        let mut texture_atlas = vec![];

        let block_table = Arc::new(BlockDefinitionTable::load_assets(
            "assets",
            &mut texture_atlas,
        ));

        let texture_luminances = utils::get_texture_luminances(&texture_atlas);

        let renderer = interactive_rendering::Renderer::new(
            surface.clone(),
            general_queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            texture_atlas,
        );

        let scene = Scene::new(
            general_queue.clone(),
            transfer_queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            renderer.n_swapchain_images(),
            texture_luminances,
        );

        let scene_uploader = scene.uploader();
        let scene = Rc::new(RefCell::new(scene));

        let threadpool = Arc::new(ThreadPool::new(15));

        let scene_manager = SceneManager::new(scene.clone());

        let camera = Rc::new(RefCell::new(camera));

        let (chunk_manager, chunk_querier) = ChunkManager::new(
            threadpool,
            scene_uploader.clone(),
            memory_allocator.clone(),
            0,
            block_table.clone(),
        );

        let physics_manager = PhysicsManager::new();

        let ego_movement_manager =
            EgoControlsManager::new(camera.clone(), chunk_querier, block_table.clone());

        GameWorld {
            general_queue,
            transfer_queue,
            entities: HashMap::new(),
            scene,
            scene_uploader,
            camera,
            ego_entity_id,
            renderer,
            surface,
            events_since_last_step: vec![],
            changes_since_last_step: vec![],
            managers: vec![
                Box::new(chunk_manager),
                Box::new(physics_manager),
                Box::new(ego_movement_manager),
                Box::new(scene_manager),
            ],
        }
    }

    fn get_reserve_closure<'a>(entities: &'a HashMap<u32, Entity>) -> impl FnMut() -> u32 + 'a {
        let reserved_ids = vec![];
        move || loop {
            let id = rand::random::<u32>();
            if !entities.contains_key(&id) && !reserved_ids.contains(&id) {
                return id;
            }
        }
    }

    pub fn update_entity_table(&mut self, changes: &Vec<WorldChange>) {
        for change in changes {
            match change {
                WorldChange::GlobalEntityAdd(entity_id, entity_creation_data) => {
                    self.entities.insert(
                        *entity_id,
                        Entity {
                            mesh: entity_creation_data.mesh.clone(),
                            isometry: entity_creation_data.isometry.clone(),
                            physics_data: entity_creation_data.physics.clone(),
                        },
                    );
                }
                WorldChange::GlobalEntityRemove(entity_id) => {
                    self.entities.remove(&entity_id);
                }
                WorldChange::GlobalEntityUpdateIsometry(entity_id, isometry) => {
                    if let Some(entity) = self.entities.get_mut(&entity_id) {
                        entity.isometry = isometry.clone();
                    }
                }
                WorldChange::GlobalEntityUpdateVelocity { id, linvel, angvel } => {
                    if let Some(entity) = self.entities.get_mut(id) {
                        if let Some(physics_data) = &mut entity.physics_data {
                            physics_data.linvel = linvel.clone();
                            physics_data.angvel = angvel.clone();
                        }
                    }
                }
                WorldChange::GlobalEntityUpdateGroundedness(id, grounded) => {
                    if let Some(entity) = self.entities.get_mut(id) {
                        if let Some(physics_data) = &mut entity.physics_data {
                            physics_data.grounded = *grounded;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn step(&mut self) {
        let new_changes = {
            let extent = interactive_rendering::get_surface_extent(&self.surface);
            let mut reserve_fn = Self::get_reserve_closure(&self.entities);

            let mut new_changes = vec![];
            for manager in self.managers.iter_mut() {
                let data = UpdateData {
                    entities: &self.entities,
                    window_events: &self.events_since_last_step,
                    world_changes: &self.changes_since_last_step,
                    ego_entity_id: self.ego_entity_id,
                    extent,
                    reserve_entity_id: &mut reserve_fn,
                };
                // run each manager, and store the changes required
                new_changes.extend(manager.update(data));
            }
            new_changes
        };

        // clear window events
        self.events_since_last_step.clear();

        // update entity table
        self.update_entity_table(&new_changes);
        self.changes_since_last_step = new_changes;

        // render to screen
        let ((eye, front, right, up), rendering_preferences) = {
            let camera = self.camera.borrow();
            (camera.eye_front_right_up(), camera.rendering_preferences())
        };
        {
            let mut mutscene = self.scene.borrow_mut();
            unsafe {
                self.renderer.render(
                    &mut mutscene,
                    eye,
                    front,
                    right,
                    up,
                    rendering_preferences.clone(),
                );
            }
        }
        if rendering_preferences.should_screenshot {
            let screenshot = self.renderer.screenshot();

            // Ensure the screenshots directory exists
            let screenshots_dir = Path::new("screenshots");
            fs::create_dir_all(&screenshots_dir).expect("Failed to create screenshots directory");

            // Determine the next file index
            let mut next_idx: u32 = 0;
            if let Ok(entries) = fs::read_dir(&screenshots_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("png")).unwrap_or(false) {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            // Accept either purely numeric names like "123" or prefixed names like "screenshot123"
                            let numeric_part = stem.trim_start_matches("screenshot");
                            if let Ok(n) = numeric_part.parse::<u32>() {
                                next_idx = next_idx.max(n + 1);
                            }
                        }
                    }
                }
            }

            let file_path = screenshots_dir.join(format!("{}.png", next_idx));
            screenshot.save(&file_path).expect("Failed to save screenshot");

            // reset the screenshot flag
            {
                self.camera
                    .borrow_mut()
                    .set_rendering_preferences(RenderingPreferences {
                        should_screenshot: false,
                        ..rendering_preferences
                    });
            }
        }

        // at this point we can now garbage collect removed entities from the last step (but not this step yet!)
        // this is because the the entities might potentially be in use until the next frame has started rendering
        // self.renderer.render will block until the current frame starts rendering
        unsafe {
            self.scene.borrow_mut().dispose_old_objects();
        }
    }

    // add a new entity to the world
    pub fn add_entity(&mut self, entity_id: u32, entity_creation_data: EntityCreationData) {
        self.entities.insert(
            entity_id,
            Entity {
                mesh: entity_creation_data.mesh.clone(),
                isometry: entity_creation_data.isometry.clone(),
                physics_data: entity_creation_data.physics.clone(),
            },
        );
        self.changes_since_last_step
            .push(WorldChange::GlobalEntityAdd(
                entity_id,
                entity_creation_data,
            ));
    }

    // remove an entity from the world
    pub fn remove_entity(&mut self, entity_id: u32) {
        self.entities.remove(&entity_id);
        self.changes_since_last_step
            .push(WorldChange::GlobalEntityRemove(entity_id));
    }

    pub fn handle_window_event(&mut self, input: winit::event::WindowEvent) {
        self.events_since_last_step.push(input);
    }

    pub fn scene_uploader(&self) -> &SceneUploader {
        &self.scene_uploader
    }
}
