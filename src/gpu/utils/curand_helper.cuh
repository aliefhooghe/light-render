#ifndef CURAND_HELPER_CUH
#define CURAND_HELPER_CUH

#include <curand_kernel.h>
#include <math_constants.h>
#include "gpu/model/float3_operators.cuh"

namespace Xrender {

    static __device__ __forceinline__ float rand_uniform(curandState *state, float min, float max)
    {
        return min + (max - min) * curand_uniform(state);
    }

    static __device__ float3 rand_unit_sphere_uniform(curandState *state)
    {
        float u1, u2, s;

        do
        {
            u1 = rand_uniform(state, -1.0f, 1.0f);
            u2 = rand_uniform(state, -1.0f, 1.0f);
        } while ((s = (u1 * u1 + u2 * u2)) >= 1.0f);

        float tmp = 2.0f * sqrtf(1.0f - s);

        return {u1 * tmp,
                u2 * tmp,
                1.0f - (2.0f * s)};
    }

    static __device__ float3 rand_unit_hemisphere_uniform(curandState *state, const float3& normal)
    {
        const float3 ret = rand_unit_sphere_uniform(state);
        return dot(ret, normal) >= 0.0f ? ret : -ret;
    }

    // density(omega) = cos(dot(normal,ret))/pi
    static __device__ float3 rand_unit_hemisphere_cos(curandState *state, const float3 &ab, const float3 &normal)
    {
        const auto basis_x = normalized(ab);
        const auto basis_y = cross(normal, basis_x);
        const float phi = 2.f * curand_uniform(state) * CUDART_PI_F;

        const float sin_theta = sqrtf(curand_uniform(state));
        const float cos_theta = sqrtf(1.f - sin_theta * sin_theta);

        const float cos_phi = cosf(phi);
        const float sin_phi = sinf(phi);

        return
            (cos_phi * sin_theta) * basis_x +
            (sin_phi * sin_theta) * basis_y +
            cos_theta * normal;
    }

    static __device__ float3 rand_unit_hemisphere_cos_pow(curandState *state, float exponent, const float3 &ab, const float3 &normal)
    {
        const auto basis_x = normalized(ab);
        const auto basis_y = cross(normal, basis_x);
        const float phi = 2.f * curand_uniform(state) * CUDART_PI_F;

        const float cos_theta = powf(curand_uniform(state), 1.f/(exponent + 1.f));
        const float sin_theta = sqrtf(1.f - cos_theta * cos_theta);

        const float cos_phi = cosf(phi);
        const float sin_phi = sinf(phi);

        return
            (cos_phi * sin_theta) * basis_x +
            (sin_phi * sin_theta) * basis_y +
            cos_theta * normal;
    }

    static __device__ float2 rand_unit_disc_uniform(curandState *state)
    {
        float2 ret;
        do {
            ret.x = rand_uniform(state, -1.f, 1.f);
            ret.y = rand_uniform(state, -1.f, 1.f);
        } while( ret.x*ret.x + ret.y*ret.y > 1.f);
        return ret;
    }

}

#endif