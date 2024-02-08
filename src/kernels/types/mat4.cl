#pragma once

// mat4.cl
// Contains all code related to matrices

struct mat4
{
    float cell[16];
};

#define IDENTITY_MATRIX {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}}

float4 multiply_matrix(float4* a, struct mat4* m)
{
    return (float4)(
        m->cell[0] * a->x + m->cell[1] * a->y + m->cell[2] * a->z + m->cell[3] * a->w,
        m->cell[4] * a->x + m->cell[5] * a->y + m->cell[6] * a->z + m->cell[7] * a->w,
        m->cell[8] * a->x + m->cell[9] * a->y + m->cell[10] * a->z + m->cell[11] * a->w,
        m->cell[12] * a->x + m->cell[13] * a->y + m->cell[14] * a->z + m->cell[15] * a->w
    );
}

float3 transform_position(float3* a, struct mat4* m)
{
    return (float3)(
        m->cell[0] * a->x + m->cell[1] * a->y + m->cell[2] * a->z + m->cell[3],
        m->cell[4] * a->x + m->cell[5] * a->y + m->cell[6] * a->z + m->cell[7],
        m->cell[8] * a->x + m->cell[9] * a->y + m->cell[10] * a->z + m->cell[11]
    );
}

float3 transform_vector(float3* a, struct mat4* m)
{
    return (float3)(
       m->cell[0] * a->x + m->cell[1] * a->y + m->cell[2] * a->z,
       m->cell[4] * a->x + m->cell[5] * a->y + m->cell[6] * a->z,
       m->cell[8] * a->x + m->cell[9] * a->y + m->cell[10] * a->z
   );
}

struct mat4 invert_mat4(struct mat4* other)
{
    float inv[16] = {
        other->cell[5] * other->cell[10] * other->cell[15] - other->cell[5] * other->cell[11] * other->cell[14] - other->cell[9] * other->cell[6] * other->cell[15] +
            other->cell[9] * other->cell[7] * other->cell[14] + other->cell[13] * other->cell[6] * other->cell[11] - other->cell[13] * other->cell[7] * other->cell[10],
        -other->cell[1] * other->cell[10] * other->cell[15] + other->cell[1] * other->cell[11] * other->cell[14] + other->cell[9] * other->cell[2] * other->cell[15] -
            other->cell[9] * other->cell[3] * other->cell[14] - other->cell[13] * other->cell[2] * other->cell[11] + other->cell[13] * other->cell[3] * other->cell[10],
        other->cell[1] * other->cell[6] * other->cell[15] - other->cell[1] * other->cell[7] * other->cell[14] - other->cell[5] * other->cell[2] * other->cell[15] +
            other->cell[5] * other->cell[3] * other->cell[14] + other->cell[13] * other->cell[2] * other->cell[7] - other->cell[13] * other->cell[3] * other->cell[6],
        -other->cell[1] * other->cell[6] * other->cell[11] + other->cell[1] * other->cell[7] * other->cell[10] + other->cell[5] * other->cell[2] * other->cell[11] -
            other->cell[5] * other->cell[3] * other->cell[10] - other->cell[9] * other->cell[2] * other->cell[7] + other->cell[9] * other->cell[3] * other->cell[6],
        -other->cell[4] * other->cell[10] * other->cell[15] + other->cell[4] * other->cell[11] * other->cell[14] + other->cell[8] * other->cell[6] * other->cell[15] -
            other->cell[8] * other->cell[7] * other->cell[14] - other->cell[12] * other->cell[6] * other->cell[11] + other->cell[12] * other->cell[7] * other->cell[10],
        other->cell[0] * other->cell[10] * other->cell[15] - other->cell[0] * other->cell[11] * other->cell[14] - other->cell[8] * other->cell[2] * other->cell[15] +
            other->cell[8] * other->cell[3] * other->cell[14] + other->cell[12] * other->cell[2] * other->cell[11] - other->cell[12] * other->cell[3] * other->cell[10],
        -other->cell[0] * other->cell[6] * other->cell[15] + other->cell[0] * other->cell[7] * other->cell[14] + other->cell[4] * other->cell[2] * other->cell[15] -
            other->cell[4] * other->cell[3] * other->cell[14] - other->cell[12] * other->cell[2] * other->cell[7] + other->cell[12] * other->cell[3] * other->cell[6],
        other->cell[0] * other->cell[6] * other->cell[11] - other->cell[0] * other->cell[7] * other->cell[10] - other->cell[4] * other->cell[2] * other->cell[11] +
            other->cell[4] * other->cell[3] * other->cell[10] + other->cell[8] * other->cell[2] * other->cell[7] - other->cell[8] * other->cell[3] * other->cell[6],
        other->cell[4] * other->cell[9] * other->cell[15] - other->cell[4] * other->cell[11] * other->cell[13] - other->cell[8] * other->cell[5] * other->cell[15] +
            other->cell[8] * other->cell[7] * other->cell[13] + other->cell[12] * other->cell[5] * other->cell[11] - other->cell[12] * other->cell[7] * other->cell[9],
        -other->cell[0] * other->cell[9] * other->cell[15] + other->cell[0] * other->cell[11] * other->cell[13] + other->cell[8] * other->cell[1] * other->cell[15] -
            other->cell[8] * other->cell[3] * other->cell[13] - other->cell[12] * other->cell[1] * other->cell[11] + other->cell[12] * other->cell[3] * other->cell[9],
        other->cell[0] * other->cell[5] * other->cell[15] - other->cell[0] * other->cell[7] * other->cell[13] - other->cell[4] * other->cell[1] * other->cell[15] +
            other->cell[4] * other->cell[3] * other->cell[13] + other->cell[12] * other->cell[1] * other->cell[7] - other->cell[12] * other->cell[3] * other->cell[5],
        -other->cell[0] * other->cell[5] * other->cell[11] + other->cell[0] * other->cell[7] * other->cell[9] + other->cell[4] * other->cell[1] * other->cell[11] -
            other->cell[4] * other->cell[3] * other->cell[9] - other->cell[8] * other->cell[1] * other->cell[7] + other->cell[8] * other->cell[3] * other->cell[5],
        -other->cell[4] * other->cell[9] * other->cell[14] + other->cell[4] * other->cell[10] * other->cell[13] + other->cell[8] * other->cell[5] * other->cell[14] -
            other->cell[8] * other->cell[6] * other->cell[13] - other->cell[12] * other->cell[5] * other->cell[10] + other->cell[12] * other->cell[6] * other->cell[9],
        other->cell[0] * other->cell[9] * other->cell[14] - other->cell[0] * other->cell[10] * other->cell[13] - other->cell[8] * other->cell[1] * other->cell[14] +
            other->cell[8] * other->cell[2] * other->cell[13] + other->cell[12] * other->cell[1] * other->cell[10] - other->cell[12] * other->cell[2] * other->cell[9],
        -other->cell[0] * other->cell[5] * other->cell[14] + other->cell[0] * other->cell[6] * other->cell[13] + other->cell[4] * other->cell[1] * other->cell[14] -
            other->cell[4] * other->cell[2] * other->cell[13] - other->cell[12] * other->cell[1] * other->cell[6] + other->cell[12] * other->cell[2] * other->cell[5],
        other->cell[0] * other->cell[5] * other->cell[10] - other->cell[0] * other->cell[6] * other->cell[9] - other->cell[4] * other->cell[1] * other->cell[10] +
            other->cell[4] * other->cell[2] * other->cell[9] + other->cell[8] * other->cell[1] * other->cell[6] - other->cell[8] * other->cell[2] * other->cell[5]
    };
    float det = other->cell[0] * inv[0] + other->cell[1] * inv[4] + other->cell[2] * inv[8] + other->cell[3] * inv[12];
    struct mat4 ret_val = IDENTITY_MATRIX;
    if (det != 0.0f)
    {
        float inv_det = 1.0 / det;
        for (int i = 0; i < 16; i++)
        {
            ret_val.cell[i] = inv[i] * inv_det;
        }
    }
    return ret_val;
}