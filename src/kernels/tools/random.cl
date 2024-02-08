#pragma once
#include "src/kernels/tools/constants.cl"

// random.cl
// Contains all code related to rng

uint wang_hash(uint seed)
{
    uint v = (seed ^ 61) ^ (seed >> 16);
    v *= 9;
    v = v ^ (v >> 4);
    v *= 0x27d4eb2d;
    v = v ^ (v >> 15);
    return v;
}

uint init_seed(uint seed_base)
{
    return wang_hash((seed_base + 1) * 17);
}

uint random_uint(uint* seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

float random_float(uint* seed)
{
    return (float)(random_uint(seed)) * 2.3283064365387e-10;
}

float random_float_normal_distribution(uint* seed)
{
    float theta = 2.0f * PI * random_float(seed);
    float rho = sqrt(-2.0f * log(random_float(seed)));
    return rho * cos(theta);
}

float3 random_direction(uint* seed)
{
    float x = random_float_normal_distribution(seed);
    float y = random_float_normal_distribution(seed);
    float z = random_float_normal_distribution(seed);
    return normalize((float3)(x,y,z));
}

float3 random_hemisphere_direction(float3* normal, uint* seed)
{
    float3 dir = random_direction(seed);
    return dir * sign(dot(*normal, dir));
}

const float2 c_blue_noise_in_disk[64] = {
    (float2)(0.478712,0.875764),
    (float2)(-0.337956,-0.793959),
    (float2)(-0.955259,-0.028164),
    (float2)(0.864527,0.325689),
    (float2)(0.209342,-0.395657),
    (float2)(-0.106779,0.672585),
    (float2)(0.156213,0.235113),
    (float2)(-0.413644,-0.082856),
    (float2)(-0.415667,0.323909),
    (float2)(0.141896,-0.939980),
    (float2)(0.954932,-0.182516),
    (float2)(-0.766184,0.410799),
    (float2)(-0.434912,-0.458845),
    (float2)(0.415242,-0.078724),
    (float2)(0.728335,-0.491777),
    (float2)(-0.058086,-0.066401),
    (float2)(0.202990,0.686837),
    (float2)(-0.808362,-0.556402),
    (float2)(0.507386,-0.640839),
    (float2)(-0.723494,-0.229240),
    (float2)(0.489740,0.317826),
    (float2)(-0.622663,0.765301),
    (float2)(-0.010640,0.929347),
    (float2)(0.663146,0.647618),
    (float2)(-0.096674,-0.413835),
    (float2)(0.525945,-0.321063),
    (float2)(-0.122533,0.366019),
    (float2)(0.195235,-0.687983),
    (float2)(-0.563203,0.098748),
    (float2)(0.418563,0.561335),
    (float2)(-0.378595,0.800367),
    (float2)(0.826922,0.001024),
    (float2)(-0.085372,-0.766651),
    (float2)(-0.921920,0.183673),
    (float2)(-0.590008,-0.721799),
    (float2)(0.167751,-0.164393),
    (float2)(0.032961,-0.562530),
    (float2)(0.632900,-0.107059),
    (float2)(-0.464080,0.569669),
    (float2)(-0.173676,-0.958758),
    (float2)(-0.242648,-0.234303),
    (float2)(-0.275362,0.157163),
    (float2)(0.382295,-0.795131),
    (float2)(0.562955,0.115562),
    (float2)(0.190586,0.470121),
    (float2)(0.770764,-0.297576),
    (float2)(0.237281,0.931050),
    (float2)(-0.666642,-0.455871),
    (float2)(-0.905649,-0.298379),
    (float2)(0.339520,0.157829),
    (float2)(0.701438,-0.704100),
    (float2)(-0.062758,0.160346),
    (float2)(-0.220674,0.957141),
    (float2)(0.642692,0.432706),
    (float2)(-0.773390,-0.015272),
    (float2)(-0.671467,0.246880),
    (float2)(0.158051,0.062859),
    (float2)(0.806009,0.527232),
    (float2)(-0.057620,-0.247071),
    (float2)(0.333436,-0.516710),
    (float2)(-0.550658,-0.315773),
    (float2)(-0.652078,0.589846),
    (float2)(0.008818,0.530556),
    (float2)(-0.210004,0.519896)
};

#define BLUE_NOISE_TEXTURE_WIDTH_HEIGHT (64)
#define BLUE_NOISE_TEXTURE_MAX (BLUE_NOISE_TEXTURE_WIDTH_HEIGHT * BLUE_NOISE_TEXTURE_WIDTH_HEIGHT)
#define BLUE_NOISE_TEXTURE_SAMPLE_MASK (BLUE_NOISE_TEXTURE_MAX - 1)

// from: https://www.shadertoy.com/view/3sfBWs
// ONLY WORKS FOR ONE SHADOW RAY PER RAY
// MULTIPLE SHOULD DO THE THETA CALCULATIONS OUTSIDE
float2 random_blue_noise_point(__constant uchar* blue_noise_texture, uint frame_idx, uint pixel_x, uint pixel_y)
{
    uint sample_idx = (pixel_y & BLUE_NOISE_TEXTURE_SAMPLE_MASK) * BLUE_NOISE_TEXTURE_WIDTH_HEIGHT + (pixel_x & BLUE_NOISE_TEXTURE_SAMPLE_MASK);
    float blue_noise = ((float)blue_noise_texture[sample_idx] / 255.0f) + GOLDEN_RATIO * (float)frame_idx;
    blue_noise -= floor(blue_noise);
    float theta = blue_noise * 2.0f * PI;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float2 sample_position = c_blue_noise_in_disk[frame_idx & BLUE_NOISE_TEXTURE_SAMPLE_MASK];

    float2 disk_point = (float2)(
        sample_position.x * cos_theta - sample_position.y * sin_theta,
        sample_position.y * sin_theta + sample_position.y * cos_theta
    );

    return (disk_point + (float2)1) / 2;
}