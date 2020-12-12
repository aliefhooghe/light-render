#ifndef XRENDER_MATERIAL_H_
#define XRENDER_MATERIAL_H_

#include <variant>
#include "vec.h"

namespace Xrender
{
    

    struct source_material {

    };

    struct lambertian_material {
        
    };

    using material = std::variant<
                        lambertian_material,
                        source_material
                    >;


    bool is_source(const material& mtl) noexcept;

    material make_source_material(float temperature = 4000.f);
    material make_lambertian_material(const vecf absorption = {0.8f, 0.8f, 0.8f});

} // namespace Xrender

#endif  