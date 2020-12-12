
#include "material.h"

namespace Xrender {

    bool is_source(const material& mtl) noexcept
    {
        return std::holds_alternative<source_material>(mtl);
    }

    material make_source_material(float temperature)
    {
        return lambertian_material{};
    }

    material make_lambertian_material(const vecf absorption)
    {
        return source_material{};
    }
} /* namespace Xrender */

