vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_ray_query: require

#define EPSILON_BLOCK 0.001

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
    Sample Y;
    float W_Y; // the unbiased contribution weight of Y
    uint c; // confidence
    float w_sum;
};

layout(set = 0, binding = 0, scalar) readonly restrict buffer TemporalReservoirBuffer {
    Reservoir temporal_reservoirs[];
};

layout(set = 0, binding = 1, scalar) writeonly restrict buffer SpatialReservoirBuffer {
    Reservoir spatial_reservoirs[];
};

layout(set = 0, binding = 2, scalar) restrict buffer DebugInfo {
    vec3 debug_info[];
};

layout(set = 0, binding = 3) uniform accelerationStructureEXT top_level_acceleration_structure;

layout(push_constant, scalar) uniform PushConstants {
    uint always_zero;
    uint num_iterations;
    uint invocation_seed;
    uint xsize;
    uint ysize;
    vec3 cam_pos;
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
    spatial_reservoirs[0].W_Y = 0;

    rayQueryEXT dummy_rq;
    rayQueryInitializeEXT(dummy_rq, top_level_acceleration_structure, gl_RayFlagsNoneEXT, 0xFF, vec3(0), 0.0, vec3(1,0,0), 1.0);

    return temporal_reservoirs[0].W_Y
         + debug_info[0].x;
}

bool isVisible(vec3 origin, vec3 target) {
    vec3 dir = target - origin;
    float dist = length(dir);
    if(dist < EPSILON_BLOCK) {
        return true;
    }
    dir /= dist;

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query,
        top_level_acceleration_structure,
        gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF,
        origin,
        EPSILON_BLOCK,
        dir,
        dist - EPSILON_BLOCK
    );

    while(rayQueryProceedEXT(ray_query));

    return rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT;
}

void updateReservoir(
    float rand,
    inout Reservoir r,
    Sample x,
    float w,
    uint c
) {
    r.w_sum += w;
    r.c += c;
    if(rand < w / r.w_sum) {
        r.Y = x;
    }
}

void mergeReservoir(
    float rand,
    inout Reservoir r,
    Reservoir r_new,
    float p_hat
) {
    uint c0 = r.c;
    updateReservoir(rand, r, r_new.Y, p_hat*r_new.W_Y*r_new.c, r_new.c);
    r.c = c0 + r_new.c;
}

bool geometricallySimilar(vec3 cam_pos,Sample a, Sample b) {
    if(a.n_v == vec3(0) || b.n_v == vec3(0)) {
        return false;
    }
    // test that the two surface normals of the visible points are within 25 degrees of each other
    if(dot(a.n_v, b.n_v) < cos(radians(25.0))) {
        return false;
    }
    // test that the two normalized depths of the visible points are within 0.05 of each other
    float depth_a = length(a.x_v - cam_pos);
    float depth_b = length(b.x_v - cam_pos);
    if(abs(depth_a - depth_b) / max(depth_a, depth_b) > 0.05) {
        return false;
    }
    return true;
}

float luminance(vec3 v) {
    return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
}

float p_hat_q(Sample x) {
    return luminance(x.l_o_hat);
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
    float length_q_v_to_s = length(q_v_to_s);
    float length_r_v_to_s = length(r_v_to_s);
    return (cos_phi_r_2 / cos_phi_q_2) * ((length_q_v_to_s*length_q_v_to_s) / (length_r_v_to_s*length_r_v_to_s));
}

const uint maxIterations = 20;
const float spatialSearchRadius = 20;

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
    
    Reservoir R_s = temporal_reservoirs[id];

    Sample S = R_s.Y;

    Q[0] = q;
    Q_reservoirs[0] = R_s;
    nQ++;

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

        Reservoir R_n = temporal_reservoirs[q_n.x + q_n.y*xsize];

        if(!geometricallySimilar(cam_pos, S, R_n.Y)) {
            continue;
        }

        float jacobian = computeJacobianQToR(R_n.Y, S);

        float p_hat_q_adj = p_hat_q(R_n.Y) / jacobian;

        // if R_n's sample point is not visible from q's visible point, zero out p_hat
        if(!isVisible(S.x_v, R_n.Y.x_s)) {
            p_hat_q_adj = 0;
        }

        mergeReservoir(
            murmur3_finalizef(murmur3_combine(iter_seed, 2)),
            R_s,
            R_n,
            p_hat_q_adj
        );

        Q[nQ] = q_n;
        Q_reservoirs[nQ] = R_n;
        nQ++;
    }

    // Bias correction (Algorithm 4, lines 16-19)
    // Z counts the total confidence of neighbors that could have produced R_s.Y
    uint Z = 0;
    for(uint i = 0; i < nQ; i++) {
        Reservoir R_n = Q_reservoirs[i];
        if(p_hat_q(R_s.Y) > 0) {
            Z += R_n.c;
        }
    }

    if(p_hat_q(R_s.Y) == 0) {
        R_s.W_Y = 0;
    } else {
        R_s.W_Y = R_s.w_sum / (Z * p_hat_q(R_s.Y));
    }
    spatial_reservoirs[id] = R_s;
    debug_info[id] += vec3(0, float(R_s.c)/10.0, 0);

}
",
}
