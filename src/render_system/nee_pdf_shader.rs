vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_ray_query: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_nonuniform_qualifier: require

#define M_PI 3.1415926535897932384626433832795
#define EPSILON_BLOCK 0.001

layout(
    local_size_x_id = 1, 
    local_size_y_id = 2, 
    local_size_z_id = 3
) in;

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer Vertex {
    vec3 position;
    uint t;
    vec2 uv;
};

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer BvhNode {
    uint left_node_idx;
    uint right_node_idx_or_prim_idx;
    vec3 min_or_v0;
    vec3 max_or_v1;
    float left_luminance_or_v2_1;
    float right_luminance_or_v2_2;
    float down_luminance_or_v2_3;
    float up_luminance_or_prim_luminance;
    float back_luminance;
    float front_luminance;
};

struct InstanceData {
    // points to the device address of the vertex data for this instance
    uint64_t vertex_buffer_addr;
    // points to the device address of the light bvh data for this instance
    uint64_t bvh_node_buffer_addr;
    // the transform of this instance
    mat4x3 transform;
};

layout(set = 1, binding = 1, scalar) readonly buffer InstanceDataBuffer {
    InstanceData instance_data[];
};

// normal of the ray intersection location
layout(set = 1, binding = 2, scalar) readonly buffer InputsIntersectionNormal {
    vec3 intersection_normal[];
};

// the location of the intersection
layout(set = 1, binding = 3, scalar) readonly buffer InputsIntersectionPosition {
    vec3 input_intersection_position[];
};

// outgoing direction of the ray that just intersected
layout(set = 1, binding = 4, scalar) readonly buffer InputsIntersectionOutDirection {
    vec3 intersection_out_direction[];
};


layout(set = 1, binding = 5) writeonly buffer OutputsNeePdf {
    float output_bsdf_pdf[];
};

layout(set = 1, binding = 6) writeonly buffer OutputsDebugInfo {
    vec4 output_debug_info[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint bounce_seed;
    uint xsize;
    uint ysize;
    uint64_t tl_bvh_addr;
};

struct VisibleTriangles {
    uint num_visible;
    vec3[3] tri0;
    vec3[3] tri1;
};


// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
bool rayTriangleIntersect(
    vec3 orig, vec3 dir,
    vec3 v0, vec3 v1, vec3 v2,
    out float t
)
{
    const float EPS = 0.0000001;
    const float EPS2 = 0.0001;


    // Compute the plane's normal
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    // No need to normalize
    vec3 N = cross(v0v1, v0v2); // N
    float area2 = length(N);

    // Step 1: Finding P
    
    // Check if the ray and plane are parallel
    float NdotRayDirection = dot(N, dir);
    if (abs(NdotRayDirection) < EPS) // Almost 0
        return false; // They are parallel, so they don't intersect!

    // Compute d parameter using equation 2
    float d = -dot(N, v0);
    
    // Compute t (equation 3)
    t = -(dot(N, orig) + d) / NdotRayDirection;
    
    // Check if the triangle is behind the ray
    if (t < 0) return false; // The triangle is behind

    // Compute the intersection point using equation 1
    vec3 P = orig + t * dir;

    // Step 2: Inside-Outside Test
    vec3 C; // Vector perpendicular to triangle's plane

    // Edge 0
    vec3 edge0 = v1 - v0; 
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < -EPS2) return false; // P is on the right side

    // Edge 1
    vec3 edge1 = v2 - v1; 
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    if (dot(N, C) < -EPS2) return false; // P is on the right side

    // Edge 2
    vec3 edge2 = v0 - v2; 
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < -EPS2) return false; // P is on the right side

    return true; // This ray hits the triangle
}

bool rayVisibleTriangleIntersect(
    vec3 orig, vec3 dir,
    VisibleTriangles vt,
    out float t
) {
    if(vt.num_visible == 0) {
        return false;
    } else {
        bool success = rayTriangleIntersect(
            orig, dir,
            vt.tri0[0], vt.tri0[1], vt.tri0[2],
            t
        );
        if(!success && vt.num_visible == 2) {
            return rayTriangleIntersect(
                orig, dir,
                vt.tri1[0], vt.tri1[1], vt.tri1[2],
                t
            );
        } else {
            return success;
        }
    }
}

void main() {
    // return early if we are out of bounds
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    
    // tensor layout: [sample, y, x, channel]
    const uint bid =  
            + gl_GlobalInvocationID.z   * ysize * xsize 
            + gl_GlobalInvocationID.y   * xsize 
            + gl_GlobalInvocationID.x; 
            
    // const vec3 origin = input_origin[bid];
    // const vec3 direction = input_direction[bid];

    // // compute the ray pdf for the light
    // float ray_pdf_light = 0.0;
    // if(light_pdf_mis_weight > 0.0) {
    //     float t;
    //     if(rayVisibleTriangleIntersect(new_origin, new_direction, vt, t)) {
    //         vec3 sampled_light_point = new_origin + t*new_direction;
    //         float light_area = getVisibleTriangleArea(vt);
    //         float light_distance = length(sampled_light_point - new_origin);
    //         // what is the probability of picking this ray if we were picking a random point on the light?
    //         ray_pdf_light = light_distance*light_distance/(cos_theta*light_area);
    //     }
    // }
}
",
}
