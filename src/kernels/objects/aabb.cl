#pragma once

// aabb.cl
// Contains all code related to aabbs

// Compute intersection of AABB
float intersect_aabb(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    float3* min_bound,
    float3* max_bound)
{
    float3 r_direction = (float3)(1.0f / ray_direction->x, 1.0f / ray_direction->y, 1.0f / ray_direction->z);
    float tx1 = (min_bound->x - ray_origin->x) * r_direction.x;
    float tx2 = (max_bound->x - ray_origin->x) * r_direction.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (min_bound->y - ray_origin->y) * r_direction.y;
    float ty2 = (max_bound->y - ray_origin->y) * r_direction.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (min_bound->z - ray_origin->z) * r_direction.z;
    float tz2 = (max_bound->z - ray_origin->z) * r_direction.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
    if (tmax >= tmin && tmin < *ray_t && tmax > 0.0)
    {
        return tmin;
    }
    return 1e30;
}

// Check if point lies in AABB
bool point_in_aabb(
    float3* point,
    float3* min_bound,
    float3* max_bound
)
{
    return point->x >= min_bound->x && point->x <= max_bound->x &&
           point->y >= min_bound->y && point->y <= max_bound->y &&
           point->x >= min_bound->z && point->z <= max_bound->z;
}