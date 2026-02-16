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


struct Sample {
    vec3 x_v;
    vec3 n_v;
    vec3 x_s;
    vec3 n_s;
    vec3 l_o_hat;
    float p_omega;
    uint seed;
};

struct Reservoir {
    Sample z;
    float ucw;
    uint m;
    float w_sum;
};

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputRayOrigins {
    vec3 ray_origins[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputBounceNormals {
    vec3 bounce_normals[];
};

layout(set = 0, binding = 2, scalar) readonly restrict buffer InputBounceOutgoingRadiance {
    vec3 bounce_outgoing_radiance[];
};

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputBounceOmegaSamplingPdf {
    float bounce_omega_sampling_pdf[];
};

layout(set = 0, binding = 4, scalar) restrict buffer TemporalReservoirBuffer {
    Reservoir temporal_reservoirs[];
};

layout(set = 0, binding = 5, scalar) restrict buffer OutputDebugInfo {
    vec3 debug_info[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint always_zero;
    uint invocation_seed;
    uint xsize;
    uint ysize;
};

// source: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu;
    const uint ieeeOne      = 0x3F800000u;

    m &= ieeeMantissa;
    m |= ieeeOne;

    float  f = uintBitsToFloat( m );
    return f - 1.0;
}

uint murmur3_combine(uint h, uint k) {
    k *= 0x1b873593;

    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    return h;
}

uint murmur3_finalize(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint murmur3_combinef(uint h, float k) {
    return murmur3_combine(h, floatBitsToUint(k));
}

float murmur3_finalizef(uint h) {
    return floatConstruct(murmur3_finalize(h));
}

float dummyUse() {
    if(always_zero == 0) {
        return 0;
    }
    return ray_origins[0].x
         + bounce_normals[0].x
         + bounce_outgoing_radiance[0].x
         + bounce_omega_sampling_pdf[0]
         + temporal_reservoirs[0].ucw
         + debug_info[0].x;
}

Sample loadInitialSample(uint id) {
    uint screen_size = xsize * ysize;
    return Sample(
        ray_origins[1 * screen_size + id],
        bounce_normals[0 * screen_size + id],
        ray_origins[2 * screen_size + id],
        bounce_normals[1 * screen_size + id],
        bounce_outgoing_radiance[1 * screen_size + id],
        bounce_omega_sampling_pdf[0 * screen_size + id],
        0u
    );
}

Reservoir loadTemporalReservoir(uint id) {
    return temporal_reservoirs[id];
}

void storeTemporalReservoir(uint id, Reservoir r) {
    temporal_reservoirs[id] = r;
}

void updateReservoir(
    uint seed,
    inout Reservoir r,
    Sample z,
    float w_new
) {
    r.w_sum += w_new;
    r.m += 1;
    if(floatConstruct(seed) < w_new / r.w_sum) {
        r.z = z;
    }
}

float luminance(vec3 v) {
    return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
}

float p_hat_q(Sample S) {
    return luminance(S.l_o_hat);
}

void main() {
    dummyUse();
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    uint id = gl_GlobalInvocationID.y * xsize + gl_GlobalInvocationID.x;
    
    uint pixel_seed = murmur3_combine(invocation_seed, id);

    Sample S = loadInitialSample(id);
    Reservoir R = loadTemporalReservoir(id);
    // re-initialize the reservoir with probability 0.01
    if(murmur3_finalizef(murmur3_combine(pixel_seed, 0)) < 0.01) {
        R.w_sum = 0.0;
        R.m = 0;
    }
    
    const float p_q = S.p_omega;
    const float w = p_hat_q(S) / p_q;

    updateReservoir(
        murmur3_finalize(murmur3_combine(pixel_seed, 1)),
        R,
        S,
        w
    );

    float p_hat_R_z = p_hat_q(R.z);
    if(p_hat_R_z > 0) {
       R.ucw = R.w_sum / (R.m * p_hat_R_z);
    } else {
        R.ucw = 0.0;
    }
    storeTemporalReservoir(id, R);
}
"
}
