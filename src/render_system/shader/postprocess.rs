vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputOutgoingRadiance {
    vec3 input_outgoing_radiance[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputDebugInfo {
    vec3 input_debug_info[];
};

layout(set = 0, binding = 2) uniform writeonly image2D output_image;

layout(push_constant, scalar) uniform PushConstants {
    uint debug_view;
    // how much larger the source image is compared to the rendering resolution
    uint srcscale;
    // how much larger the output canvas is compared to the rendering resolution
    uint dstscale;
    uint xsize;
    uint ysize;
};

void main() {
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    const uint srcxsize = xsize * srcscale;
    const uint srcysize = ysize * srcscale;

    vec3 outgoing_radiance = vec3(0.0);
    vec3 debug_info = vec3(0.0);

    for (uint scaley = 0; scaley < srcscale; scaley++) {
        const uint srcy = gl_GlobalInvocationID.y * srcscale + scaley;
        for(uint scalex = 0; scalex < srcscale; scalex++) {
            const uint srcx = gl_GlobalInvocationID.x * srcscale + scalex;
            
            // compute id of the source pixel
            const uint id = srcy * srcxsize + srcx;

            // fetch the color for this sample
            outgoing_radiance += input_outgoing_radiance[id];
            // fetch the debug info for this sample
            debug_info += input_debug_info[id].xyz;
        }
    }

    vec3 pixel_color;
    if (debug_view == 0) {
        pixel_color = outgoing_radiance;
    } else {
        pixel_color = debug_info;
        // infinite loop to test crash layer
        while (true) {
            pixel_color += pixel_color;
        }
    }

    // average the samples
    pixel_color = pixel_color / float(srcscale*srcscale);

    // write to a patch of size dstscale*dstscale
    for (uint scaley = 0; scaley < dstscale; scaley++) {
        const uint dsty = gl_GlobalInvocationID.y * dstscale + scaley;
        for(uint scalex = 0; scalex < dstscale; scalex++) {
            const uint dstx = gl_GlobalInvocationID.x * dstscale + scalex;
            imageStore(output_image, ivec2(dstx, dsty), vec4(pixel_color, 1.0));
        }
    }
}
",
}
