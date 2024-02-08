#pragma once
#include "src/kernels/tools/constants.cl"
#include "src/kernels/types/mat4.cl"
#include "src/kernels/objects/triangle.cl"
#include "src/kernels/objects/bvh.cl"

void intersect_scene(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    float3* ray_normal,
    float3* intersect_color,
    uint num_objects,
    uint* obj_mesh_ids,
    uint* obj_mat_ids,
    struct mat4* obj_transforms,
    struct mat4* obj_inv_transforms,
    uint* bvh_offsets,
    uint* mesh_offsets,
    float3* bvh_min_bounds,
    float3* bvh_max_bounds,
    uint* bvh_tri_counts,
    uint* bvh_left_firsts,
    uint* bvh_triangle_offsets,
    struct triangle* bvh_triangles,
    uint3* mesh_vertex_ids,
    float3* mesh_vertex_normals,
    uint* mat_offsets,
    uint* mat_colors,
    uchar* mat_reflectiveness,
    float* mat_refraction_index
)
{
    uint ray_tri_idx = MAX_UINT;
    uint ray_mesh_idx = MAX_UINT;
    uint ray_obj_idx = MAX_UINT;

    for (uint i = 0; i < num_objects; i++)
    {
        uint mesh_idx = obj_mesh_ids[i];
        struct mat4 obj_inv_transform = obj_inv_transforms[i];
        float3 new_origin = transform_position(ray_origin, &obj_inv_transform);
        float3 new_direction = transform_vector(ray_direction, &obj_inv_transform);

        uint bvh_offset = bvh_offsets[mesh_idx];
        uint triangle_offset = bvh_triangle_offsets[mesh_idx];

        if (!intersect_bvh(ray_t, &new_origin, &new_direction, &ray_tri_idx, bvh_min_bounds + bvh_offset, bvh_max_bounds + bvh_offset, bvh_tri_counts + bvh_offset, bvh_left_firsts + bvh_offset, bvh_triangles + triangle_offset))
        {
            continue;
        }

        ray_mesh_idx = ray_mesh_idx;
        ray_obj_idx = i;
    }

    if (ray_obj_idx == MAX_UINT)
    {
        return; // no intersection
    }

    *intersect_color = (float3)1;
}
