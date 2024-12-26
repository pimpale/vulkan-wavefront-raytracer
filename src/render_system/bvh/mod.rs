use vulkano::buffer::BufferContents;

pub mod aabb;
pub mod build;

#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BvhNode {
    // if this is 0xFFFFFFFF, then this is a leaf node
    pub left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, right_node_idx_or_prim_idx is an `Index`
    // if this BVH represents a bottom level BVH, then `Index` is a GLSL PrimitiveIndex
    // if this BVH represents a top level BVH, then `Index` is a GLSL InstanceID
    // otherwise, it is the index of the right node
    // if this is 0xFFFFFFFF and left_node_idx is 0xFFFFFFFF, then this means that this is a dummy node
    pub right_node_idx_or_prim_idx: u32,
    // the min bound
    pub min: [f32; 3],
    // the power emitted by the entire object
    pub power: f32,
    // the max bound
    pub max: [f32; 3],
    // the parent index
    pub parent_idx: u32,
}

impl Default for BvhNode {
    fn default() -> BvhNode {
        BvhNode {
            left_node_idx: 0xFFFFFFFF,
            right_node_idx_or_prim_idx: 0xFFFFFFFF,
            min: [0.0; 3],
            max: [0.0; 3],
            power: 0.0,
            parent_idx: 0xFFFFFFFF,
        }
    }
}
