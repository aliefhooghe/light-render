#ifndef GPU_MTL_CUH
#define GPU_MTL_CUH

#include <cuda.h>
#include <curand_kernel.h>

namespace Xrender
{

    struct source_mtl
    {
        float3 emission;
    };

    struct lambertian_mtl
    {
        float3 absorption;
    };

    struct mirror_mtl
    {
        float3 reflection;
    };

    struct glass_mtl
    {
        float reflexivity;
        float3 tf;
        float3 ks;
        float3 n;
    };

    struct material
    {
        enum mtl_type
        {
            SOURCE,
            LAMBERTIAN,
            MIRROR,
            GLASS
        } type;

        union
        {
            source_mtl source;
            lambertian_mtl lambertian;
            mirror_mtl mirror;
            glass_mtl glass;
        };

        /** \todo + preview_color(), + brdf(), ...**/
    };

    // Onmy lambertian / sources

    material make_source_material();
    material make_lambertian_materal(float3 absorption);
    material make_mirror_material(float3 reflection);
    material make_glass_material(float reflexivity, const float3 &tf, const float3 &ks, float a, float b);

    static __device__ bool gpu_mtl_is_source(const material &mtl)
    {
        return mtl.type == material::SOURCE;
    }

    static __device__ float3 gpu_preview_color(const material &mtl)
    {
        switch (mtl.type)
        {
        default:
        case material::SOURCE:
            return {0.f, 1.f, 0.f};
        case material::LAMBERTIAN:
            return mtl.lambertian.absorption;
        case material::MIRROR:
            return mtl.mirror.reflection;
        case material::GLASS:
            return mtl.glass.tf;
        }
    }

}

#endif