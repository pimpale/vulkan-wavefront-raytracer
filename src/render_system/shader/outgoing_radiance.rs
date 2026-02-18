vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

#define M_PI 3.1415926535897932384626433832795

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputDirection {
    vec3 input_direction[];
};

layout(set = 0, binding = 2, scalar) readonly restrict buffer InputEmissivity {
    vec3 input_emissivity[];
};

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputAlbedo {
    vec3 input_albedo[];
};

layout(set = 0, binding = 4, scalar) readonly restrict buffer InputNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputBsdfPdf {
    float input_bsdf_pdf[];
};

layout(set = 0, binding = 6, scalar) readonly restrict buffer InputNeePdf {
    float input_nee_pdf[];
};

layout(set = 0, binding = 7, scalar) writeonly restrict buffer OutputOutgoingRadiance {
    vec3 output_outgoing_radiance[];
};

layout(set = 0, binding = 8, scalar) writeonly restrict buffer OutputOmegaSamplingPdf {
    float output_omega_sampling_pdf[];
};

layout(set = 0, binding = 9, scalar) restrict buffer AccumulatedOutgoingRadiance {
    vec3 accumulated_outgoing_radiance[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint always_zero;
    uint num_bounces;
    uint xsize;
    uint ysize;
    uint spp;
};

#define EPSILON 0.0001

void dummyUse() {
    if(always_zero != 0) {
        float d = input_origin[0].x
            + input_direction[0].x
            + input_emissivity[0].x
            + input_albedo[0].x
            + input_nee_mis_weight[0]
            + input_bsdf_pdf[0]
            + input_nee_pdf[0];
        output_outgoing_radiance[0] = vec3(d);
        output_omega_sampling_pdf[0] = d;
        accumulated_outgoing_radiance[0] = vec3(d);
    }
}

void main() {
    dummyUse();
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const float factor = 1.0 / float(spp);

    // compute the outgoing radiance (L_o)
    vec3 L_o = vec3(0.0);
    for(int bounce = int(num_bounces)-1; bounce >= 0; bounce--) {            
        // tensor layout: [bounce, y, x, channel]
        const uint bid = bounce * ysize * xsize 
                        + y   * xsize 
                        + x;

        // whether the ray is valid
        float ray_valid = float(input_direction[bid] != vec3(0.0));

        // incoming radiance (L_i) = last bounce's outgoing radiance
        // we zero it out if the ray is invalid
        const vec3 L_i = L_o * ray_valid;

        const float bsdf_pdf = input_bsdf_pdf[bid];
        const float nee_pdf = input_nee_pdf[bid];
        const float nee_mis_weight = input_nee_mis_weight[bid];
        const vec3 A = input_albedo[bid];
        const vec3 L_e = input_emissivity[bid];

        // compute the sampling pdf: this is how we are sampling rays
        // mis_weight proportion of the time, we sample from the light source, and 1-mis_weight proportion of the time, we sample from the bsdf pdf
        float sampling_pdf = nee_pdf * nee_mis_weight + (1.0 - nee_mis_weight) * bsdf_pdf;

        L_o = L_e + A * bsdf_pdf * L_i / sampling_pdf;
        
        // write raw per-sample value (for ReSTIR to read)
        output_outgoing_radiance[bid] = L_o;
        output_omega_sampling_pdf[bid] = sampling_pdf;

        // accumulate bounce 0 for vanilla PT display
        if (bounce == 0) {
            accumulated_outgoing_radiance[bid] += L_o * factor;
        }
    }
}
",
}
