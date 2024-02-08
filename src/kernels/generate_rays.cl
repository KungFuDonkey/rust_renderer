#include "src/kernels/objects/scene.cl"

__kernel void generate_rays(
    uint glob_seed,
    uint screen_width,
    uint screen_height,
    uint num_bounces,
    float3 cam_position,
    float3 cam_top_left,
    float3 cam_bottom_left,
    float3 cam_top_right,
    __global uint* ray_write_back_ids,
    __global float* ray_ts,
    __global float3* ray_origins,
    __global float3* ray_directions,
    __global float3* ray_normals,
    __global float3* ray_intersection_colors,
    __global float3* ray_energies,
    __global float3* ray_write_back_lights,
    __global float3* albedo,
    __global float3* light,
    uint num_objects,
    __global uint* obj_mesh_ids,
    __global uint* obj_mat_ids,
    __global struct mat4* obj_transforms,
    __global struct mat4* obj_inv_transforms,
    __global uint* bvh_offsets,
    __global uint* mesh_offsets,
    __global float3* bvh_min_bounds,
    __global float3* bvh_max_bounds,
    __global uint* bvh_tri_counts,
    __global uint* bvh_left_firsts,
    __global uint* bvh_triangle_offsets,
    __global struct triangle* bvh_triangles,
    __global uint3* mesh_vertex_ids,
    __global float3* mesh_vertex_normals,
    __global uint* mat_offsets,
    __global uint* mat_colors,
    __global uchar* mat_reflectiveness,
    __global float* mat_refraction_index
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    // get uv coordinate
    float u = ((float)x) / (float)screen_width;
    float v = ((float)y) / (float)screen_height;

    // get direction
    float3 p = cam_top_left + (cam_top_right - cam_top_left) * u + (cam_bottom_left - cam_top_left) * v;
    float3 ray_direction = normalize(p - cam_position);
    uint idx = x + y * screen_width;
    uint max_idx = screen_width * screen_height;

    float ray_t = 1e30;
    float3 ray_origin = cam_position;
    float3 ray_normal = ray_direction;
    float3 ray_intersection_color = ray_direction;

    intersect_scene(
        &ray_t,
        &ray_origin,
        &ray_direction,
        &ray_normal,
        &ray_intersection_color,
        num_objects,
        obj_mesh_ids,
        obj_mat_ids,
        obj_transforms,
        obj_inv_transforms,
        bvh_offsets,
        mesh_offsets,
        bvh_min_bounds,
        bvh_max_bounds,
        bvh_tri_counts,
        bvh_left_firsts,
        bvh_triangle_offsets,
        bvh_triangles,
        mesh_vertex_ids,
        mesh_vertex_normals,
        mat_offsets,
        mat_colors,
        mat_reflectiveness,
        mat_refraction_index);

    albedo[idx] = ray_intersection_color;
    ray_write_back_ids[idx] = idx;
    ray_ts[idx] = ray_t;
    ray_origins[idx] = ray_origin;
    ray_directions[idx] = ray_direction;
    ray_normals[idx] = ray_normal;
    ray_energies[idx] = (float3)1;

    for (uint i = 0; i < num_bounces; i++)
    {
        light[max_idx * i + idx] = (float3)0;
    }

}