
#include "material.cuh"

namespace Xrender
{

    /**
     * Pixel component wavelength (in micro metter)
     */
    constexpr auto red_wl_micro = 0.662f;
    constexpr auto green_wl_micro = 0.532f;
    constexpr auto blue_wl_micro = 0.467f;

    static constexpr auto cauchy_law(float a, float b, float wavelength_micro)
    {
        return a + b / (wavelength_micro * wavelength_micro);
    }

    material make_source_material()
    {
        material mtl;
        mtl.type = material::SOURCE;
        mtl.source.emission = {1.f, 1.f, 1.f};
        return mtl;
    }

    material make_lambertian_materal(float3 absorption)
    {
        material mtl;
        mtl.type = material::LAMBERTIAN;
        mtl.lambertian.absorption = absorption;
        return mtl;
    }

    material make_glass_material(float reflexivity, const float3& tf, const float3& ks, float a, float b)
    {
        const float3 n = {
            cauchy_law(a, b, blue_wl_micro),
            cauchy_law(a, b, green_wl_micro),
            cauchy_law(a, b, red_wl_micro)};

        material mtl;
        mtl.type = material::GLASS;
        mtl.glass.reflexivity = reflexivity;
        mtl.glass.tf = tf;
        mtl.glass.ks = ks;
        mtl.glass.n = n;
        return mtl;
    }

}