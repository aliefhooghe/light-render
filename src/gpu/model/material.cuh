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

    struct phong_mtl
    {
        float3 specular;
        float ns;
    };

    struct mirror_mtl
    {
        float3 reflection;
    };

    struct dispersive_glass_mtl
    {
        float reflexivity;
        float3 tf;
        float3 ks;
        float3 n;
    };

    struct glass_mtl
    {
        float reflexivity;
        float3 tf;
        float3 ks;
        float n;
    };

    struct material
    {
        enum mtl_type
        {
            SOURCE,
            LAMBERTIAN,
            PHONG,
            MIRROR,
            GLASS,
            DISPERSIVE_GLASS
        } type;

        union
        {
            source_mtl source;
            lambertian_mtl lambertian;
            phong_mtl phong;
            mirror_mtl mirror;
            glass_mtl glass;
            dispersive_glass_mtl dispersive_glass;
        };

        /** \todo + preview_color(), + brdf(), ...**/
    };

    // Onmy lambertian / sources

    material make_source_material(float tkelvin);
    material make_phong_material(float3 specular, float n);
    material make_lambertian_materal(float3 absorption);
    material make_mirror_material(float3 reflection);
    material make_glass_material(float reflexivity, const float3 &tf, const float3 &ks, float n);
    material make_dispersive_glass_material(float reflexivity, const float3 &tf, const float3 &ks, float a, float b);

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
        case material::PHONG:
            return mtl.phong.specular;
        case material::MIRROR:
            return mtl.mirror.reflection;
        case material::GLASS:
            return mtl.glass.tf;
        }
    }

}

#endif