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


    vecf brdf(const material& mtl, const vecf& normal, const vecf& idir, const vecf& edir);

} // namespace Xrender

#endif  