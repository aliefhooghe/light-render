
#include <iostream>

#include "random_generator.h"
#include "material.h"


namespace Xrender {

    constexpr auto red_wl_micro = 0.662f;
    constexpr auto green_wl_micro = 0.532f;
    constexpr auto blue_wl_micro = 0.467f;

    bool is_source(const material& mtl) noexcept
    {
        return std::holds_alternative<source_material>(mtl);
    }

    material make_source_material(float temperature)
    {
        return source_material{{1.f, 1.f, 1.f}};
    }

    material make_lambertian_material(const vecf& absorption)
    {
        return lambertian_material{absorption};
    }

    material make_mirror_material(const vecf& reflection)
    {
        return mirror_material{reflection};
    }

    static constexpr auto cauchy_law(float a, float b, float wavelength_micro)
    {
        return a + b / (wavelength_micro * wavelength_micro);
    }

    material make_glass_material(float reflexivity, const vecf& tf, const vecf& ks, float a, float b)
    {
        const float nr = cauchy_law(a, b, red_wl_micro);
        const float ng = cauchy_law(a, b, green_wl_micro);
        const float nb = cauchy_law(a, b, blue_wl_micro);
        return glass_material{reflexivity, tf, ks, nr, ng, nb};
    }

    struct preview_impl
    {
        vecf operator()(const source_material &) const noexcept { return {1.f, 1.f, 1.f}; };
        vecf operator()(const lambertian_material &lamb) const noexcept { return lamb.absorption; }
        vecf operator()(const mirror_material &mir) const noexcept { return mir.reflection; }
        vecf operator()(const glass_material &glass) const noexcept { return glass.tf; }
    };

    vecf material_preview_color(const material& mtl)
    {
        return std::visit(preview_impl{}, mtl);
    }

    struct brdf_impl
    {
        vecf brdf(const source_material& src, const vecf&, const vecf&, vecf&)
        {
            return src.emission;
        }

        vecf brdf(const lambertian_material& lamb, const vecf& normal, const vecf&, vecf& edir)
        {
            edir = rand::unit_hemisphere_uniform(normal);
            return lamb.absorption;// <why ? * 2.f * constant::pi_flt;
        }

        vecf brdf(const mirror_material& mir, const vecf& normal, const vecf& idir, vecf& edir)
        {
            edir = idir - 2.f * dot(normal, idir) * normal;
            return mir.reflection;
        }

        vecf brdf(const glass_material& glass, const vecf& normal, const vecf& idir, vecf& edir)
        {
            const auto seed = rand::uniform();

            // if choose tranmition
            if (seed >= glass.reflexivity)
            {
                float n;
                float ratio;
                vecf norm;
                vecf tf;

                // choose wavelength
                switch (rand::integer(0, 2))
                {
                    case 0: n = glass.nr; tf = {glass.tf.x, 0.f, 0.}; break;
                    case 1: n = glass.ng; tf = {0.f , glass.tf.y, 0.f}; break;
                    case 2: n = glass.nb; tf = {0.f, 0.f, glass.tf.z}; break;
                }

                //
                const auto dot_dir_norm = dot(idir, normal);

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

                // r = n/1 or 1/n

                const auto c = std::abs(dot_dir_norm);
                const auto tmp = 1.f - (ratio * ratio) * (1.f - c * c);

                if (tmp >= 0.f)
                {
                    edir = ratio * idir + (ratio * c - sqrtf(tmp)) * norm;
                    return 3.f * tf;
                }
                //else // total reflection
            }
            
            edir = idir - 2.f * dot(normal, idir) * normal;
            return glass.ks;
        }
    };

    vecf brdf(const material& mtl, const vecf& normal, const vecf& idir, vecf& edir)
    {
        return std::visit(
            [&normal, &idir, &edir](const auto& m) { return brdf_impl{}.brdf(m, normal, idir, edir); },
            mtl);
    }

} /* namespace Xrender */
