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

layout(set = 0, binding = 0, scalar) readonly restrict buffer TemporalReservoirBuffer {
    Reservoir temporal_reservoirs[];
};

layout(set = 0, binding = 1, scalar) restrict buffer SpatialReservoirBuffer {
    Reservoir spatial_reservoirs[];
};

layout(set = 0, binding = 2, scalar) restrict buffer DebugInfo {
    vec3 debug_info[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint always_zero;
    uint num_iterations;
    uint invocation_seed;
    uint xsize;
    uint ysize;
};


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
    return temporal_reservoirs[0].ucw
         + spatial_reservoirs[0].ucw
         + debug_info[0].x;
}

Reservoir loadTemporalReservoir(uint id) {
    return temporal_reservoirs[id];
}

Reservoir loadSpatialReservoir(uint id) {
    return spatial_reservoirs[id];
}

void storeSpatialReservoir(uint id, Reservoir r) {
    spatial_reservoirs[id] = r;
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

void mergeReservoir(
    uint seed,
    inout Reservoir r,
    Reservoir r_new,
    float p_hat
) {
    uint m0 = r.m;
    updateReservoir(seed, r, r_new.z, p_hat*r_new.ucw*r_new.m);
    r.m = m0 + r_new.m;
}

bool geometricallySimilar(Sample a, Sample b) {
    if(a.n_v == vec3(0) || b.n_v == vec3(0)) {
        return false;
    }
    if(dot(a.n_v, b.n_v) < cos(radians(25.0))) {
        return false;
    }
    return true;
}

float luminance(vec3 v) {
    return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
}

float p_hat_q(Sample z) {
    return luminance(z.l_o_hat);
}

float computeJacobianQToR(Sample q, Sample r) {
    vec3 n = q.n_s;
    if(n == vec3(0)) {
        return 1;
    }
    vec3 q_v_to_s = q.x_v - q.x_s;
    vec3 r_v_to_s = r.x_v - q.x_s;
    float cos_phi_q_2 = abs(dot(normalize(q_v_to_s), n));
    float cos_phi_r_2 = abs(dot(normalize(r_v_to_s), n));
    return (cos_phi_r_2 / cos_phi_q_2) * (length(q_v_to_s) / length(r_v_to_s));
}

const uint maxIterations = 20;
const float spatialSearchRadius = 15;

void main() {
    dummyUse();

    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    uvec2 q = gl_GlobalInvocationID.xy;
    uint id = q.x + q.y*xsize;
    uint pixel_seed = murmur3_combine(invocation_seed, id);

    uint nQ = 0;
    uvec2 Q[maxIterations+1];
    Reservoir Q_reservoirs[maxIterations+1];
    
    Reservoir R_s = loadTemporalReservoir(id);
    
    // TESTING
    storeSpatialReservoir(id, R_s);
    return;


    Sample S = R_s.z;

    Q[0] = q;
    Q_reservoirs[0] = R_s;
    nQ++;

    float v_jacobian = 0;

    for(uint s = 0; s < num_iterations; s++) {
        uint iter_seed = murmur3_combine(pixel_seed, s);

        vec2 jitter = spatialSearchRadius*vec2(
            murmur3_finalizef(murmur3_combine(iter_seed, 0))-0.5,
            murmur3_finalizef(murmur3_combine(iter_seed, 1))-0.5
        );

        uvec2 q_n = uvec2(
            clamp(int(q.x + 0.5 + jitter.x), 0, int(xsize-1)),
            clamp(int(q.y + 0.5 + jitter.y), 0, int(ysize-1))
        );

        Reservoir R_n = loadTemporalReservoir(q_n.x + q_n.y*xsize);

        if(!geometricallySimilar(S, R_n.z)) {
            continue;
        }

        float jacobian = computeJacobianQToR(R_n.z, S);
        jacobian = clamp(jacobian, 0.1, 10);
        v_jacobian += jacobian;

        float p_hat_q_adj = p_hat_q(R_n.z) / jacobian;

        mergeReservoir(
            murmur3_finalize(murmur3_combine(iter_seed, 2)),
            R_s,
            R_n,
            p_hat_q_adj
        );

        Q[nQ] = q_n;
        Q_reservoirs[nQ] = R_n;
        nQ++;
    }

    uint Z = 0;
    for(uint i = 0; i < nQ; i++) {
        uvec2 q_n = Q[i];
        Reservoir R_n = Q_reservoirs[i];
        if(p_hat_q(R_n.z) > 0) {
            Z += R_n.m;
        }
    }

    if(p_hat_q(R_s.z) == 0) {
        R_s.ucw = 0;
    } else {
        R_s.ucw = R_s.w_sum / (Z * p_hat_q(R_s.z));
    }
    storeSpatialReservoir(id, R_s);
}
",
}
