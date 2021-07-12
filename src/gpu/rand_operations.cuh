#ifndef RAND_OPERATIONS_CUH
#define RAND_OPERATIONS_CUH

#include <curand_kernel.h>

#include "vector_operations.cuh"

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
        return _dot(ret, normal) >= 0.0f ? ret : -ret;
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