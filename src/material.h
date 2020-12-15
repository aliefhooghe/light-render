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

    using material =
        std::variant<
            lambertian_material,
            source_material
        >;


    bool is_source(const material& mtl) noexcept;

    material make_source_material(float temperature = 4000.f);
    material make_lambertian_material(const vecf absorption = {0.8f, 0.8f, 0.8f});

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