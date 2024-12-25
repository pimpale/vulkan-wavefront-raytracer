vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460
#extension GL_EXT_ray_query: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_nonuniform_qualifier: require

#define M_PI 3.1415926535897932384626433832795
#define EPSILON_BLOCK 0.001

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform accelerationStructureEXT light_top_level_acceleration_structure;

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer Vertex {
    vec3 position;
    uint t;
    vec2 uv;
};

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer LightVertex {
    vec3 position;
    uint light_bvh_node_idx;
};

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer BvhNode {
    uint left_node_idx;
    uint right_node_idx_or_prim_idx;
    vec3 min;
    vec3 max;
    float luminance;
    uint parent_node_idx;
};

struct InstanceData {
    // points to the device address of the vertex data for this instance
    uint64_t vertex_buffer_addr;
    // points to the device address of the light vertex data for this instance
    uint64_t light_vertex_buffer_addr;
    // points to the device address of the light bvh data for this instance
    uint64_t bvh_node_buffer_addr;
    // the offset of the bottom level light bvh in the top level bvh
    uint light_bvh_tl_idx;
    // the transform of this instance
    mat4x3 transform;
};

layout(set = 0, binding = 1, scalar) readonly buffer InstanceDataBuffer {
    InstanceData instance_data[];
};

// normal of the ray intersection location
layout(set = 0, binding = 2, scalar) readonly buffer InputsIntersectionNormal {
    vec3 intersection_normal[];
};

// the location of the intersection
layout(set = 0, binding = 3, scalar) readonly buffer InputsIntersectionPosition {
    vec3 input_intersection_position[];
};

// outgoing direction of the ray that just intersected
layout(set = 0, binding = 4, scalar) readonly buffer InputsIntersectionOutDirection {
    vec3 intersection_out_direction[];
};

layout(set = 0, binding = 5) readonly buffer InputsNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 6) writeonly buffer OutputsNeePdf {
    float output_nee_pdf[];
};

layout(set = 0, binding = 7) writeonly buffer OutputsDebugInfo {
    vec4 output_debug_info[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint nee_type;
    uint bounce_seed;
    uint xsize;
    uint ysize;
    uint64_t tl_bvh_addr;
};

float lengthSquared(vec3 v) {
    return dot(v, v);
}

vec3[3] triangleTransform(mat4x3 transform, vec3[3] tri) {
    return vec3[3](
        transform * vec4(tri[0], 1.0),
        transform * vec4(tri[1], 1.0),
        transform * vec4(tri[2], 1.0)
    );
}

vec3 triangleCenter(vec3[3] tri) {
    return (tri[0] + tri[1] + tri[2]) / 3.0;
}

float triangleRadiusSquared(vec3 center, vec3[3] tri) {
    return max(
        max(
            lengthSquared(tri[0] - center),
            lengthSquared(tri[1] - center)
        ),
        lengthSquared(tri[2] - center)
    );
}

float nodeImportance(bool topLevel, vec3 point, vec3 normal, mat4x3 transform, BvhNode node) {
    // get corners
    vec3 v000 = transform * vec4(node.min, 1.0);
    vec3 v111 = transform * vec4(node.max, 1.0);

    float distance_sq = max(
        lengthSquared(v111 - v000),
        lengthSquared(0.5*(v000 + v111) - point)
    );

    return node.luminance / distance_sq;
}

float reverseTraverseBvh(
    // the point from which we're evaluating the importance
    vec3 shading_point,
    vec3 shading_normal,
    InstanceData id,
    // index of the bvh node in the instance
    uint bvh_node_idx
) {
    // root starts off as the bottom level bvh root,
    // once we reach the ascend through the entire instance, we will replace root with the top level bvh root
    BvhNode root = BvhNode(id.bvh_node_buffer_addr);

    // start off with 1 probability
    float probability = 1.0;
    mat4x3 transform = id.transform;
    bool topLevel = false;

    // loop works like this:
    // start in the bottom layer with a primitive node
    // ascend 1 level up
    // compute left and right importance
    // multiply probability
    // if we are at the top level, we are done

    BvhNode node = root[bvh_node_idx];
    uint node_idx = bvh_node_idx;

    while(true) {
        // ascend
        uint old_node_idx = node_idx;
        if(node.parent_node_idx == 0xFFFFFFFF) {
            if(topLevel) {
                // we are at the top level
                return probability;
            } else {
                // ascend to the top level
                topLevel = true;
                root = BvhNode(tl_bvh_addr);
                old_node_idx = id.light_bvh_tl_idx;
                
                // we are now at a leaf node
                // we need to ascend until we hit an internal node (one which has a choice)
                BvhNode tl_leaf_node = root[id.light_bvh_tl_idx];
                // note that the tl_leaf_node could be the top level root
                if(tl_leaf_node.parent_node_idx == 0xFFFFFFFF) {
                    // we are at the top level
                    return probability;
                }

                node_idx = tl_leaf_node.parent_node_idx;
                node = root[tl_leaf_node.parent_node_idx];
                transform = mat4x3(1.0);
            }
        } else {
            node_idx = node.parent_node_idx;
            node = root[node.parent_node_idx];
        }

        // look at left and right children
        BvhNode left = root[node.left_node_idx];
        BvhNode right = root[node.right_node_idx_or_prim_idx];

        float left_importance = nodeImportance(topLevel, shading_point, shading_normal, transform, left);
        float right_importance = nodeImportance(topLevel, shading_point, shading_normal, transform, right);
        float total_importance = left_importance + right_importance;
        float left_importance_normalized = left_importance / total_importance;
        float right_importance_normalized = right_importance / total_importance;

        if(old_node_idx == node.left_node_idx) {
            probability *= left_importance_normalized;
        } else {
            probability *= right_importance_normalized;
        }
    }
}

vec3 pointOnTriangle(vec3[3] tri, vec2 barycentric_coords) {
    return barycentric_coords.x*tri[0] + barycentric_coords.y*tri[1] + (1.0 - barycentric_coords.x - barycentric_coords.y)*tri[2];
}

float triangleArea(vec3[3] tri) {
    vec3 a = tri[0] - tri[1];
    vec3 b = tri[2] - tri[1];
    return 0.5*length(cross(a, b));
}

float computeNeePdf(
    vec3 shading_point,
    vec3 shading_normal,
    vec3 outgoing_direction,
    uint instance_id,
    uint light_primitive_id,
    vec2 barycentric_coords
) {

    // get light vertex data
    InstanceData id = instance_data[instance_id];
    vec3[3] tri_raw = vec3[3](
        LightVertex(id.light_vertex_buffer_addr)[3*light_primitive_id + 0].position,
        LightVertex(id.light_vertex_buffer_addr)[3*light_primitive_id + 1].position,
        LightVertex(id.light_vertex_buffer_addr)[3*light_primitive_id + 2].position
    );

    // transform triangle
    vec3[3] tri_light = triangleTransform(id.transform, tri_raw);

    // find point on triangle
    vec3 point_on_triangle = pointOnTriangle(tri_light, barycentric_coords);

    // probability of picking this ray if we were picking a random point on the light
    const float light_distance = length(point_on_triangle - shading_point);
    const float light_area = triangleArea(tri_light);
    const float cos_theta = dot(shading_normal, outgoing_direction);
    const float pointPickProbability = light_distance*light_distance/(cos_theta*light_area);

    // probability of picking the primitive
    uint light_bvh_node_idx = LightVertex(id.light_vertex_buffer_addr)[3*light_primitive_id].light_bvh_node_idx;
    const float primPickProbability = reverseTraverseBvh(
        shading_point,
        shading_normal,
        id,
        LightVertex(id.light_vertex_buffer_addr)[3*light_primitive_id].light_bvh_node_idx
    );

    return primPickProbability * pointPickProbability;
}

void main() {
    // return early if we are out of bounds
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    // tensor layout: [bounce, y, x, channel]
    const uint bid =  
            + gl_GlobalInvocationID.y   * xsize 
            + gl_GlobalInvocationID.x; 
            
    const vec3 direction = intersection_out_direction[bid];

    // return early if the ray is not valid (already terminated) or the mis weight is zero
    if(length(direction) == 0.0 || input_nee_mis_weight[bid] == 0.0) {
        output_nee_pdf[bid] = 0.0;
        return;
    }
    
    const vec3 origin = input_intersection_position[bid];
    const vec3 normal = intersection_normal[bid];

    // trace ray
    const float t_min = EPSILON_BLOCK;
    const float t_max = 1000.0;
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query,
        light_top_level_acceleration_structure,
        gl_RayFlagsCullBackFacingTrianglesEXT, // cull back faces (they have zero importance anyway)
        0xFF,
        origin,
        t_min,
        direction,
        t_max
    );

    float nee_pdf = 0.0;
    // trace ray
    while (rayQueryProceedEXT(ray_query)) {
        if (rayQueryGetIntersectionTypeEXT(ray_query, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint instance_id = rayQueryGetIntersectionInstanceIdEXT(ray_query, false);
            uint primitive_id = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);
            vec2 barycentric_coords = rayQueryGetIntersectionBarycentricsEXT(ray_query, false);
            nee_pdf += computeNeePdf(
                origin,
                normal,
                direction,
                instance_id,
                primitive_id,
                barycentric_coords
            );
        }
    }
    
    output_nee_pdf[bid] = nee_pdf;
}
",
}
