
__kernel void albedo(
    __global float3* albedo,
    __global uint* output_buffer
)
{
    uint idx = get_global_id(0);

    float3 rendered_color = albedo[idx];

    // clamp color under one and multiply to get byte values
    float3 one = (float3)1;
    float3 ranged_color = min(rendered_color, one) * 255.0f;

    uint r = (uint)ranged_color.x;
    uint g = (uint)ranged_color.y;
    uint b = (uint)ranged_color.z;
    output_buffer[idx] = (r << 16) + (g << 8) + b;
}

__kernel void finalize()
{

}