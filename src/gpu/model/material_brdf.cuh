#ifndef XRENDER_MTL_BRDF_CUH_
#define XRENDER_MTL_BRDF_CUH_

#include "gpu/utils/curand_helper.cuh"

#include "material.cuh"

namespace Xrender
{

    /**
     *  \brief Sample a material brdf :
     *      Choose edit with a given angular density and return an estimator:
     *  \return brdf(idir, edir) * cos(theta_e) / density(edir)
     */
    static __device__ float3 sample_brdf(
        curandState *state,
        const intersection &inter,
        const float3 &normal,
        const float3 &idir, float3 &edir);

    static __device__ float3 gpu_source_brdf(const source_mtl &src)
    {
        return src.emission;
    }

    static __device__ float3 gpu_lambertian_brdf(curandState *state, const intersection& inter, const float3 &normal, const float3 &idir, float3 &edir)
    {
        /**
         *  density = uniform = cos(theta_e) / pi
         *  brdf = lambertian = absorption/pi
         *
         *  => brdf(idir, edir) * cos(theta_e) / density(edir) = absorption
         */
        edir = rand_unit_hemisphere_cos(state, inter.ab, normal);
        return inter.mtl.lambertian.absorption;
    }

    static __device__ float3 gpu_mirror_bfdf(curandState *state, const mirror_mtl &mirror, const float3 &normal, const float3 &idir, float3 &edir)
    {
        edir = idir - 2.f * dot(normal, idir) * normal;
        return mirror.reflection;
    }

    static __device__ float3 gpu_glass_brdf(curandState *state, const glass_mtl &glass, const float3 &normal, const float3 &idir, float3 &edir)
    {
        const auto wavelentgth_idx = (curand(state) % 3u);
        const auto dot_dir_norm = dot(idir, normal);
        const auto cos_theta = fabs(dot_dir_norm);

        float n;
        float3 tf;
        float3 ks;

        // choose wavelength
        switch (wavelentgth_idx)
        {
        case 0:
            n = glass.n.x;
            tf = {glass.tf.x, 0.f, 0.f};
            ks = {glass.ks.x, 0.f, 0.f};
            break;
        case 1:
            n = glass.n.y;
            tf = {0.f, glass.tf.y, 0.f};
            ks = {0.f, glass.ks.y, 0.f};
            break;
        case 2:
            n = glass.n.z;
            tf = {0.f, 0.f, glass.tf.z};
            ks = {0.f, 0.f, glass.ks.z};
            break;
        }

        const auto tmp = (1.f - n) / (1.f + n);
        const auto r0 = tmp * tmp;
        const auto factor = 1.f - cos_theta;
        const auto factor2 = factor * factor;
        const float factor5 = factor2 * factor2 * factor;

        const float r = r0 + (1.f - r0) * factor5;

        if (curand_uniform(state) >= r)
        {
            float ratio;
            float3 norm;

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

            const auto tmp = 1.f - (ratio * ratio) * (1.f - cos_theta * cos_theta);

            if (tmp >= 0.f)
            {
                edir = ratio * idir + (ratio * cos_theta - sqrtf(tmp)) * norm;
                return 3.f * tf;
            }
            // else : total reflection
        }

        edir = idir - 2.f * dot(normal, idir) * normal;
        return 3.f * ks;
    }

    static __device__ float3 sample_brdf(
        curandState *state,
        const intersection &inter,
        const float3 &normal,
        const float3 &idir, float3 &edir)
    {
        switch (inter.mtl.type)
        {
        case material::SOURCE:
            return gpu_source_brdf(inter.mtl.source);
        case material::LAMBERTIAN:
            return gpu_lambertian_brdf(state, inter, normal, idir, edir);
        case material::MIRROR:
            return gpu_mirror_bfdf(state, inter.mtl.mirror, normal, idir, edir);
        default: /* GLASS */
            return gpu_glass_brdf(state, inter.mtl.glass, normal, idir, edir);
        }
    }

}

#endif