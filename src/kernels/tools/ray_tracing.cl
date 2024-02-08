#pragma once

// ray_tracing.cl
// Contains code related to ray_tracing, such as reflect and refract

// reflect a ray
float3 reflect(float3 I, float3 N)
{
    return I - 2.0f * dot(I, N) * N;
}

// refract a ray
float3 refract(float3 I, float3 N, float eta)
{
    float cosI = -dot( N, I );
    float cosT2 = 1.0f - eta * eta * (1.0f - cosI * cosI);
    return (eta * I) + (eta * cosI - sqrt( cosT2 )) * N;
}

// compute fresnel
float fresnel_reflect_amount(float n1, float n2, float3 N, float3 I, float reflectivity)
{
    float r0 = (n1-n2) / (n1+n2);
    r0 *= r0;
    float cosX = -dot(N, I);
    if (n1 > n2)
    {
        float n = n1 / n2;
        float sinT2 = n * n * (1.0 - cosX * cosX);
        if (sinT2 > 1.0)
        {
            return 1.0;
        }
        cosX = sqrt(1.0-sinT2);
    }
    float x = 1.0 - cosX;
    float ret = r0 + (1.0 - r0) * x * x * x * x * x;

    // adjust reflect multiplier for object reflectivity
    ret = (reflectivity + (1.0 - reflectivity) * ret);
    return clamp(ret, 0.0, 1.0);
}