#pragma once
// triangle.cl
// Contains all code related to rendering and intersecting a triangle

struct triangle
{
    float3 vertex0;
    float3 vertex1;
    float3 vertex2;
    uint idx;
};

bool intersect_triangle(
    float* ray_t,
    float3* ray_origin,
    float3* ray_direction,
    float3* vertex0,
    float3* vertex1,
    float3* vertex2)
{
    float3 edge1 = *vertex1 - *vertex0;
    float3 edge2 = *vertex2 - *vertex0;
    float3 h = cross(*ray_direction, edge2);
    float a = dot(edge1, h);
    if (a > -0.0001f && a < 0.0001)
    {
        return false;
    }

    float3 s = *ray_origin - *vertex0;
    float f = 1.0 / a;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    float3 q = cross(s, edge1);
    float v = f * dot(*ray_direction, q);
    if (v < 0.0 || u + v > 1.0)
    {
        return false;
    }

    float t = f * dot(edge2, q);
    if (t > 0.0001 && t < *ray_t)
    {
        *ray_t = t;
        return true;
    }
    return false;
}
