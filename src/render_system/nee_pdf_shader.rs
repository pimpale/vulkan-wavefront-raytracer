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
    vec3 min_or_v0;
    vec3 max_or_v1;
    float left_luminance_or_v2_1;
    float right_luminance_or_v2_2;
    float down_luminance_or_v2_3;
    float up_luminance_or_prim_luminance;
    float back_luminance;
    float front_luminance;
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

struct VisibleTriangles {
    uint num_visible;
    vec3[3] tri0;
    vec3[3] tri1;
};


vec3 line_plane_intersection(vec3 line_point, vec3 line_direction, vec3 plane_point, vec3 plane_normal) {
    float t = dot(plane_normal, line_point - plane_point) / dot(-line_direction, plane_normal);
    return line_point + t * line_direction;
}

// returns anywhere between 0 and 2 triangles, all of which are visible from the point in the direction of the normal
VisibleTriangles splitIntoVisibleTriangles(vec3 point, vec3 normal, vec3[3] tri) {
    vec3[3] tri_sorted = tri;

    // sort 3 vertices by cosine of angle between vertex and normal
    float cos0 = dot(tri_sorted[0]-point, normal);
    float cos1 = dot(tri_sorted[1]-point, normal);
    float cos2 = dot(tri_sorted[2]-point, normal);

    if(cos0 > cos2) {
        vec3 tmp = tri_sorted[0];
        tri_sorted[0] = tri_sorted[2];
        tri_sorted[2] = tmp;
        float tmp2 = cos0;
        cos0 = cos2;
        cos2 = tmp2;
    }

    if(cos0 > cos1) {
        vec3 tmp = tri_sorted[0];
        tri_sorted[0] = tri_sorted[1];
        tri_sorted[1] = tmp;
        float tmp2 = cos0;
        cos0 = cos1;
        cos1 = tmp2;
    }

    if(cos1 > cos2) {
        vec3 tmp = tri_sorted[1];
        tri_sorted[1] = tri_sorted[2];
        tri_sorted[2] = tmp;
        float tmp2 = cos1;
        cos1 = cos2;
        cos2 = tmp2;
    }

    vec3[3] null_tri = vec3[3](vec3(0.0), vec3(0.0), vec3(0.0));

    if(dot(tri_sorted[2]-point, normal) <= 0) {
        // none of the triangle's vertices are visible 
        return VisibleTriangles(0, null_tri, null_tri);
    } else if(dot(tri_sorted[1]-point, normal) <= 0) {
        // only one of the triangle's vertices (vertex 2) is visible
        // we can now construct a new triangle that is guaranteed to be visible by finding
        // the intersection of v2->v1 with the normal plane (new_v0)
        // and the intersection of v2->v0 with the normal plane (new_v1)

        // for the line plane intersection test, we would normally need to check if the determinant of the matrix
        // formed by the vectors is zero, but we know that the normal is not parallel to the plane, so we can skip that check

        vec3[3] tri0 = vec3[3](
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[1] - tri_sorted[2],
                point,
                normal
            ),
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            ),
            tri_sorted[2]
        );
        return VisibleTriangles(1, tri0, null_tri);
    } else if(dot(tri_sorted[0] - point, normal) <= 0) {
        // two of the triangle's vertices are visible
        
        // in this case we have two visible triangles:
        // the triangle formed by v2, v1, and the intersection of v2->v0 with the normal plane
        // and the triangle formed by v2, the intersection of v2->v0 with the normal plane, and the intersection of v1->v0 with the normal plane
        
        vec3[3] tri0 = vec3[3](
            tri_sorted[2],
            tri_sorted[1],
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            )
        );

        vec3[3] tri1 = vec3[3](
            tri_sorted[2],
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            ),
            line_plane_intersection(
                tri_sorted[1],
                tri_sorted[0] - tri_sorted[1],
                point,
                normal
            )
        );
        
        return VisibleTriangles(2, tri0, tri1);
    } else {
        // all of the triangle's vertices are visible
        // so return the original triangle
        return VisibleTriangles(1, tri, null_tri);
    }
}

// returns the area of the triangle that is visible from the point in the direction of the normal
float getVisibleTriangleArea(VisibleTriangles vt) {
    float area = 0.0;
    area += float(vt.num_visible >= 1)*0.5*length(cross(vt.tri0[1] - vt.tri0[0], vt.tri0[2] - vt.tri0[0]));
    area += float(vt.num_visible == 2)*0.5*length(cross(vt.tri1[1] - vt.tri1[0], vt.tri1[2] - vt.tri1[0]));
    return area;
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

// returns true if any part of the rect is visible from the point in the direction of the normal
bool rectIsVisible(vec3 point, vec3 normal, vec3[4] rect) {
    uint visible = 0;
    for(uint i = 0; i < 4; i++) {
        vec3 to_v = rect[i] - point;
        visible |= uint(dot(to_v, normal) >= EPSILON_BLOCK);
    }
    return bool(visible);
}

// gets the importance of a node relative to a point on a surface, specialized for leaf nodes
float nodeImportance(bool topLevel, vec3 point, vec3 normal, mat4x3 transform, BvhNode node) {
    // replace node with lower level node to get better bounds
    if(topLevel && node.left_node_idx == 0xFFFFFFFF) {
        InstanceData id = instance_data[node.right_node_idx_or_prim_idx];
        transform = id.transform;
        topLevel = false;
        node = BvhNode(id.bvh_node_buffer_addr);
    }
    
    if(topLevel || node.left_node_idx != 0xFFFFFFFF) {
        // get corners
        vec3 v000 = transform * vec4(node.min_or_v0, 1.0);
        vec3 v111 = transform * vec4(node.max_or_v1, 1.0);
        vec3 v001 = vec3(v000.x, v000.y, v111.z);
        vec3 v010 = vec3(v000.x, v111.y, v000.z);
        vec3 v011 = vec3(v000.x, v111.y, v111.z);
        vec3 v100 = vec3(v111.x, v000.y, v000.z);
        vec3 v101 = vec3(v111.x, v000.y, v111.z);
        vec3 v110 = vec3(v111.x, v111.y, v000.z);

        float distance_sq = max(
            lengthSquared(v111 - v000),
            lengthSquared(0.5*(v000 + v111) - point)
        );

        float luminance = 0.0;
        
        vec3 lv = v000 - v100;
        luminance +=
            // light contained in this part of the node
            node.left_luminance_or_v2_1
            // clamped scalar projection of point onto x axis (to reduce estimate if we are inside the node)
            * clamp(dot(point - v100, lv)/lengthSquared(lv), 0.0, 1.0)
            // very rough approximation of cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v100, v101, v111, v110)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v100), normalize(lv)));

        vec3 rv = v100 - v000;
        luminance +=
            node.right_luminance_or_v2_2 
            * clamp(dot(point - v000, rv)/lengthSquared(rv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v001, v011, v010)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(rv)));
        
        vec3 dv = v000 - v010;
        luminance +=
            node.down_luminance_or_v2_3 
            * clamp(dot(point - v010, dv)/lengthSquared(dv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v010, v011, v111, v110)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v010), normalize(dv)));

        vec3 uv = v010 - v000;
        luminance +=
            node.up_luminance_or_prim_luminance
            * clamp(dot(point - v000, uv)/lengthSquared(uv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v001, v101, v100)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(uv)));

        vec3 bv = v000 - v001;
        luminance +=
            node.back_luminance
            * clamp(dot(point - v001, bv)/lengthSquared(bv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v001, v011, v111, v101)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v001), normalize(bv)));


        vec3 fv = v001 - v000;
        luminance +=
            node.front_luminance
            * clamp(dot(point - v000, fv)/lengthSquared(fv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v010, v110, v100)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(fv)));


        return luminance / distance_sq;
    } else {
        // untransformed triangle
        vec3[3] tri_r = vec3[3](
            node.min_or_v0,
            node.max_or_v1,
            vec3(
                node.left_luminance_or_v2_1,
                node.right_luminance_or_v2_2,
                node.down_luminance_or_v2_3
            )
        );
        // transformed triangle visible to us
        vec3[3] tri = triangleTransform(transform, tri_r);
        VisibleTriangles vt = splitIntoVisibleTriangles(point, normal, tri);
        if(vt.num_visible == 0) {
            return 0.0;
        }

        vec3 tri_centroid = vt.num_visible == 1 
            ? triangleCenter(vt.tri0)
            : 0.5*(triangleCenter(vt.tri0) + triangleCenter(vt.tri1));

        vec3 tri_normal = normalize(cross(tri[1] - tri[0], tri[2] - tri[0]));

        // get total luminance of the triangle
        float emitted_light = getVisibleTriangleArea(vt)*node.up_luminance_or_prim_luminance;
        
        // https://en.wikipedia.org/wiki/View_factor
        float dist_to_tri = length(point - tri_centroid);
        float cos_theta_tri = dot(tri_normal, point-tri_centroid)/dist_to_tri;
        float cost_theta_surf = dot(normal, tri_centroid-point)/dist_to_tri;
        if (cost_theta_surf < 0.0 || cos_theta_tri < 0.0) {
            return 0.0;
        }

        float min_distance_sq = triangleRadiusSquared(tri_centroid, tri);
        float distance_sq = max(dist_to_tri*dist_to_tri, min_distance_sq);
        
        float visibility_coefficient = cos_theta_tri*cost_theta_surf;

        return emitted_light*visibility_coefficient / distance_sq;
    }
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

    // convert to visible triangle struct
    VisibleTriangles vt = splitIntoVisibleTriangles(shading_point, shading_normal, tri_light);

    // probability of picking this ray if we were picking a random point on the light
    const float light_distance = length(point_on_triangle - shading_point);
    const float light_area = getVisibleTriangleArea(vt);
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
