#ifndef GPU_MTL_CUH
#define GPU_MTL_CUH

#include <cuda.h>
#include <curand_kernel.h>

#include "rand_operations.cuh"

namespace Xrender {

    struct source_mtl {
        float3 emission;
    };

    struct lambertian_mtl {
        float3 absorption;
    };

    struct glass_mtl {
        float reflexivity;
        float3 tf;
        float3 ks;
        float3 n;
    };

    struct gpu_mtl {
        enum {SOURCE, LAMBERTIAN, GLASS} type;
        union {
            source_mtl source;
            lambertian_mtl lambertian;
            glass_mtl glass;
        };
    };

    // Onmy lambertian / sources

    gpu_mtl gpu_make_source_material();
    gpu_mtl gpu_make_lambertian_materal(float3 absorption);
    gpu_mtl make_glass_material(float reflexivity, const float3& tf, const float3& ks, float a, float b);


    static __device__ bool gpu_mtl_is_source(const gpu_mtl& mtl)
    {
        return mtl.type == gpu_mtl::SOURCE;
    }

    static __device__ float3 gpu_preview_color(const gpu_mtl& mtl)
    {
        switch (mtl.type)
        {
        default:
        case gpu_mtl::SOURCE     : return {0.f, 1.f, 0.f};
        case gpu_mtl::LAMBERTIAN : return mtl.lambertian.absorption;
        case gpu_mtl::GLASS      : return mtl.glass.tf;
        }
    }

    static __device__ float3 gpu_source_brdf(const source_mtl& src)
    {
        return src.emission;
    }

    static __device__ float3 gpu_lambertian_brdf(curandState *state, const lambertian_mtl& lamb, const float3& normal, float3& edir)
    {
        edir = rand_unit_hemisphere_uniform(state, normal);
        return lamb.absorption;
    }

    static __device__ float3 gpu_glass_brdf(curandState *state, const glass_mtl& glass, const float3& normal, const float3& idir, float3& edir)
    {
        if (curand_uniform(state) >= glass.reflexivity) {
            // try transmition
            float n;
            float3 tf;

            switch (curand(state) % 3u)
            {
                case 0: n = glass.n.x; tf = {glass.tf.x, 0.f, 0.f}; break;
                case 1: n = glass.n.y; tf = {0.f, glass.tf.y, 0.f}; break;
                case 2: n = glass.n.z; tf = {0.f, 0.f, glass.tf.z}; break;
            }

            float ratio;
            float3 norm;

            const auto dot_dir_norm = _dot(idir, normal);

            if (dot_dir_norm < 0.f)
            {
                ratio = 1.f / n;
                norm = normal;
            }
            else
            {
                ratio = n;
                norm = -normal;
            }

            const auto c = fabs(dot_dir_norm);
            const auto tmp = 1.f - (ratio * ratio) * (1.f - c * c);

            if (tmp > 0.f)
            {
                edir = ratio * idir + (ratio * c - sqrtf(tmp)) * norm;
                return 3.f * tf;
            }
            // else total relfection
        }

        // total reflection
        edir = idir - 2.f * _dot(normal, idir) * normal;
        return glass.ks;
    }

    static __device__ float3 gpu_brdf(
        curandState *state,
        const gpu_mtl& mtl,
        const float3& normal,
        const float3& idir, float3& edir)
    {
        switch (mtl.type)
        {
        case gpu_mtl::SOURCE:       return gpu_source_brdf(mtl.source);
        case gpu_mtl::LAMBERTIAN :  return gpu_lambertian_brdf(state, mtl.lambertian, normal, edir);
        default: /* GLASS */        return gpu_glass_brdf(state, mtl.glass, normal, idir, edir); 
        }
    }

}

#endif