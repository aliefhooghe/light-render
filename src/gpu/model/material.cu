
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

    static constexpr float wv_luminance_by_temperature(const float Tkelvin, const float wavelength)
    {
        const double H = 6.62607015E-34f;     //  planck
        const double C = 299792458.f;         //  light celerity
        const double K = 1.380649E-23f;       //  boltzmann
        return 1.f / (
            wavelength*wavelength*wavelength*wavelength*wavelength * (expf(H * C / (wavelength * K * Tkelvin)) - 1.f));
    }

    float3 luminance_by_temperature(const float tkelvin)
    {
        const float r = wv_luminance_by_temperature(tkelvin, red_wl_micro * 1E-6f);
        const float g = wv_luminance_by_temperature(tkelvin, green_wl_micro * 1E-6f);
        const float b = wv_luminance_by_temperature(tkelvin, blue_wl_micro * 1E-6f);
        const float scale = std::max(r, std::max(g, b));
        return {
            r / scale,
            g / scale,
            b / scale
        };
    }

    material make_source_material(float tkelvin)
    {
        material mtl;
        mtl.type = material::SOURCE;
        mtl.source.emission = luminance_by_temperature(tkelvin);
        return mtl;
    }

    material make_phong_material(float3 specular, float n)
    {
        material mtl;
        mtl.type = material::PHONG;
        mtl.phong.specular = specular;
        mtl.phong.ns = n;
        return mtl;
    }

    material make_lambertian_materal(float3 absorption)
    {
        material mtl;
        mtl.type = material::LAMBERTIAN;
        mtl.lambertian.absorption = absorption;
        return mtl;
    }

    material make_mirror_material(float3 reflection)
    {
        material mtl;
        mtl.type = material::MIRROR;
        mtl.mirror.reflection = reflection;
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