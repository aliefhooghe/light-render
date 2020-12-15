
#include "material.h"

namespace Xrender {

    bool is_source(const material& mtl) noexcept
    {
        return std::holds_alternative<source_material>(mtl);
    }

    material make_source_material(float temperature)
    {
        return source_material{{1.f, 1.f, 1.f}};
    }

    material make_lambertian_material(const vecf absorption)
    {
        return lambertian_material{absorption};
    }

    vecf material_preview_color(const material& mtl)
    {
        if (is_source(mtl))
            return {1.f, 0.f, 1.f};
        else
            return std::get<lambertian_material>(mtl).absorption;
    }

    vecf brdf(const material& mtl, const vecf& normal, const vecf& idir, const vecf& edir)
    {
        if (is_source(mtl))
            return std::get<source_material>(mtl).emission;
        else
            return std::get<lambertian_material>(mtl).absorption;
    }

} /* namespace Xrender */
