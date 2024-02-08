#include "src/kernels/objects/aabb.cl"
#include "src/kernels/objects/triangle.cl"
#include "src/kernels/types/mat4.cl"

// meshes.cl
// Contains all code related to rendering a mesh that has a BVH, similar to Jacco's code, so won't comment

bool intersect_bvh(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    uint* ray_tri_idx,
    float3* mesh_min_bounds,
    float3* mesh_max_bounds,
    uint* mesh_tri_counts,
    uint* mesh_left_firsts,
    struct triangle* mesh_triangles
)
{
    uint node = 0;

    {
        float3 min_bound = mesh_min_bounds[0];
        float3 max_bound = mesh_max_bounds[0];
        float t = intersect_aabb(ray_t, ray_origin, ray_direction, &min_bound, &max_bound);
        if (t == 1e30)
        {
            return false;
        }
    }

    uint stack[64];
    uint stack_ptr = 0;
    bool intersected = false;

    while (1)
    {
        uint tri_count = mesh_tri_counts[node];
        uint left_first = mesh_left_firsts[node];
        if (tri_count > 0)
        {
            for (int i = 0; i < tri_count; i++)
            {
                struct triangle tr = mesh_triangles[left_first + i];
                if (intersect_triangle(ray_t, ray_origin, ray_direction, &tr.vertex0, &tr.vertex1, &tr.vertex2))
                {
                    *ray_tri_idx = tr.idx;
                    intersected = true;
                }
            }

            if (stack_ptr == 0)
            {
                break;
            }
            node = stack[--stack_ptr];
            continue;
        }

        uint child1 = left_first;
        uint child2 = left_first + 1;

        float3 child1_min_bound = mesh_min_bounds[child1];
        float3 child2_min_bound = mesh_min_bounds[child2];
        float3 child1_max_bound = mesh_max_bounds[child1];
        float3 child2_max_bound = mesh_max_bounds[child2];

        float dist1 = intersect_aabb(ray_t, ray_origin, ray_direction, &child1_min_bound, &child1_max_bound);
        float dist2 = intersect_aabb(ray_t, ray_origin, ray_direction, &child2_min_bound, &child2_max_bound);

        if (dist1 > dist2)
        {
            float tmp = dist1;
            dist1 = dist2;
            dist2 = tmp;
            uint tmp2 = child1;
            child1 = child2;
            child2 = tmp2;
        }

        if (dist1 == 1e30)
        {
            if (stack_ptr == 0)
            {
                break;
            }
            node = stack[--stack_ptr];
            continue;
        }

        node = child1;
        if (dist2 != 1e30)
        {
            stack[stack_ptr++] = child2;
        }
    }

    return intersected;
}


bool occlude_bvh(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    float3* mesh_min_bounds,
    float3* mesh_max_bounds,
    uint* mesh_tri_counts,
    uint* mesh_left_firsts,
    struct triangle* mesh_triangles
)
{
    uint node = 0;

    {
        float3 min_bound = mesh_min_bounds[0];
        float3 max_bound = mesh_max_bounds[0];
        float t = intersect_aabb(ray_t, ray_origin, ray_direction, &min_bound, &max_bound);
        if (t == 1e30)
        {
            return false;
        }
    }

    uint stack[64];
    uint stack_ptr = 0;

    while (1)
    {
        uint tri_count = mesh_tri_counts[node];
        uint left_first = mesh_left_firsts[node];
        if (tri_count > 0)
        {
            for (int i = 0; i < tri_count; i++)
            {
                struct triangle tr = mesh_triangles[left_first + i];
                if (intersect_triangle(ray_t, ray_origin, ray_direction, &tr.vertex0, &tr.vertex1, &tr.vertex2))
                {
                    return true;
                }
            }

            if (stack_ptr == 0)
            {
                break;
            }
            node = stack[--stack_ptr];
            continue;
        }

        uint child1 = left_first;
        uint child2 = left_first + 1;

        float3 child1_min_bound = mesh_min_bounds[child1];
        float3 child2_min_bound = mesh_min_bounds[child2];
        float3 child1_max_bound = mesh_max_bounds[child1];
        float3 child2_max_bound = mesh_max_bounds[child2];

        float dist1 = intersect_aabb(ray_t, ray_origin, ray_direction, &child1_min_bound, &child1_max_bound);
        float dist2 = intersect_aabb(ray_t, ray_origin, ray_direction, &child2_min_bound, &child2_max_bound);

        if (dist1 > dist2)
        {
            float tmp = dist1;
            dist1 = dist2;
            dist2 = tmp;
            uint tmp2 = child1;
            child1 = child2;
            child2 = tmp2;
        }

        if (dist1 == 1e30)
        {
            if (stack_ptr == 0)
            {
                break;
            }
            node = stack[--stack_ptr];
            continue;
        }

        node = child1;
        if (dist2 != 1e30)
        {
            stack[stack_ptr++] = child2;
        }
    }

    return false;
}

bool occlude_meshes(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    uint num_meshes,
    uint* mesh_offsets,
    uint* mesh_triangle_offsets,
    struct mat4* mesh_inv_transforms,
    float3* mesh_min_bounds,
    float3* mesh_max_bounds,
    uint* mesh_tri_counts,
    uint* mesh_left_firsts,
    struct triangle* mesh_triangles
    )
{
    for (int i = 0; i < num_meshes; i++)
    {
        struct mat4 mesh_inv_transform = mesh_inv_transforms[i];
        float3 new_origin = transform_position(ray_origin, &mesh_inv_transform);
        float3 new_direction = transform_vector(ray_direction, &mesh_inv_transform);
        uint mesh_offset = mesh_offsets[i];
        uint triangle_offset = mesh_triangle_offsets[i];

        if (occlude_bvh(ray_t, &new_origin, &new_direction, mesh_min_bounds + mesh_offset, mesh_max_bounds + mesh_offset, mesh_tri_counts + mesh_offset, mesh_left_firsts + mesh_offset, mesh_triangles + triangle_offset))
        {
            return true;
        }
    }
    return false;
}
