vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_ray_query: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputsBounceIndices {
    uint input_bounce_indices[];
};

layout(set = 0, binding = 1, scalar) writeonly restrict buffer OutputsBounceIndices {
    uint output_bounce_indices[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint xsize;
    uint ysize;
} push_constants;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if(index >= push_constants.xsize * push_constants.ysize) {
        return;
    }
    output_bounce_indices[index] = input_bounce_indices[index];
}

"
}