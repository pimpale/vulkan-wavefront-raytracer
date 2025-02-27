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

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputReflectivity {
    vec3 input_reflectivity[];
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


layout(push_constant, scalar) uniform PushConstants {
    uint num_bounces;
    uint xsize;
    uint ysize;
};

void main() {
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    // compute the color for this sample
    vec3 outgoing_radiance = vec3(0.0);
    for(int bounce = int(num_bounces)-1; bounce >= 0; bounce--) {            
        // tensor layout: [bounce, y, x, channel]
        const uint bid = bounce * ysize * xsize 
                        + y   * xsize 
                        + x;

        // whether the ray is valid
        float ray_valid = float(input_direction[bid] != vec3(0.0));

        // compute importance sampling data
        float bsdf_pdf = input_bsdf_pdf[bid];
        float nee_pdf = input_nee_pdf[bid];
        float nee_mis_weight = input_nee_mis_weight[bid];
        // this is our sampling distribution: 
        // mis_weight proportion of the time, we sample from the light source, and 1-mis_weight proportion of the time, we sample from the bsdf pdf
        float q_omega = nee_pdf * nee_mis_weight + (1.0 - nee_mis_weight) * bsdf_pdf;
        // this is the distribution we are trying to compute the expectation over
        float p_omega = bsdf_pdf;
        float reweighting_factor = p_omega / q_omega;

        outgoing_radiance = input_emissivity[bid] + input_reflectivity[bid] * outgoing_radiance * reweighting_factor * ray_valid;
        
        // write to global memory
        output_outgoing_radiance[bid] = outgoing_radiance;
        output_omega_sampling_pdf[bid] = q_omega;
    }
}
",
}
