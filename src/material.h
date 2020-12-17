#ifndef XRENDER_MATERIAL_H_
#define XRENDER_MATERIAL_H_

#include <variant>
#include "vec.h"

namespace Xrender
{
    struct source_material {
        vecf emission;
    };

    struct lambertian_material {
        vecf absorption;
    };

    struct glass_material {
        float reflexivity;
        vecf tf;
        vecf ks;
        float nr;
        float ng;
        float nb;
    };

    struct mirror_material {
        vecf reflection;
    };

    using material =
        std::variant<
            lambertian_material,
            source_material,
            mirror_material,
            glass_material
        >;


    bool is_source(const material& mtl) noexcept;

    material make_source_material(float temperature);
    material make_lambertian_material(const vecf& absorption);
    material make_mirror_material(const vecf& reflection);
    material make_glass_material(float reflexivity, const vecf& tf, const vecf& ks, float a, float b);

    vecf material_preview_color(const material& mtl);


    /**
     * \brief Return an estimator of the brdf
     * \param[in] face material
     * \param[in] normal normal at intersection point
     * \param[in] idir incoming vector
     * \param[out] edir emitted vector according to some angular density
     * \return the estimator
     */
    vecf brdf(const material& mtl, const vecf& normal, const vecf& idir, vecf& edir);




} // namespace Xrender

#endif