vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) readonly buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 0, binding = 1, scalar) readonly buffer InputsDirection {
    vec3 input_direction[];
};

layout(set = 0, binding = 2, scalar) readonly buffer InputsEmissivity {
    vec3 input_emissivity[];
};

layout(set = 0, binding = 3, scalar) readonly buffer InputsReflectivity {
    vec3 input_reflectivity[];
};

layout(set = 0, binding = 4) readonly buffer InputsNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 5) readonly buffer InputsBsdfPdf {
    float input_bsdf_pdf[];
};

layout(set = 0, binding = 6) readonly buffer InputsNeePdf {
    float input_nee_pdf[];
};

layout(set = 0, binding = 7, scalar) readonly buffer InputsDebugInfo {
    vec4 input_debug_info[];
};

layout(set = 0, binding = 8) writeonly buffer OutputsImage {
    u8vec4 output_image[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint frame;
    uint num_bounces;
    uint scale;
    uint xsize;
    uint ysize;
};

void main() {
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    const uint srcxsize = xsize * scale;
    const uint srcysize = ysize * scale;

    vec3 color = vec3(0.0);
    for (uint scaley = 0; scaley < scale; scaley++) {
        const uint srcy = gl_GlobalInvocationID.y * scale + scaley;
        for(uint scalex = 0; scalex < scale; scalex++) {
            const uint srcx = gl_GlobalInvocationID.x * scale + scalex;
            // compute the color for this sample
            vec3 sample_color = vec3(0.0);
            for(int bounce = int(num_bounces)-1; bounce >= 0; bounce--) {            
                // tensor layout: [bounce, y, x, channel]
                const uint bid = bounce * srcysize * srcxsize 
                               + srcy   * srcxsize 
                               + srcx;
                // whether the ray is valid
                float ray_valid = input_direction[bid] == vec3(0.0) ? 0.0 : 1.0;

                // compute importance sampling data
                float bsdf_pdf = input_bsdf_pdf[bid];
                float nee_pdf = input_nee_pdf[bid];
                float nee_mis_weight = input_nee_mis_weight[bid];
                // this is our sampling distribution: 
                // mis_weight proportion of the time, we sample from the light source, and 1-mis_weight proportion of the time, we sample from the BSDF
                float qx = nee_pdf * nee_mis_weight + (1.0 - nee_mis_weight) * bsdf_pdf;
                // this is the distribution we are trying to compute the expectation over
                float px = bsdf_pdf;
                float reweighting_factor = px / qx;

                sample_color = input_emissivity[bid] + sample_color * input_reflectivity[bid] * reweighting_factor * ray_valid;
            }
            // // render debug info on even frames
            // if (frame % 100 < 50) {
            //     const uint bounce_to_render = 0;
            //     const uint bid = bounce_to_render * srcysize * srcxsize 
            //                    + srcy   * srcxsize 
            //                    + srcx;
            //     sample_color = vec3(input_nee_pdf[bid], input_reflectivity[bid].g*10, 0.0);
            // }

            color += sample_color;
        }
    }

    // average the samples
    vec3 pixel_color = color / float(scale*scale);
    output_image[gl_GlobalInvocationID.y*xsize + gl_GlobalInvocationID.x] = u8vec4(pixel_color.zyx*255, 255);
}
",
}
