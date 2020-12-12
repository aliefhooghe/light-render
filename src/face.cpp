
#include <stdexcept>
#include "face.h"

namespace Xrender {

    face make_face(material mtl, vecf p1, vecf p2, vecf p3)
    {
        const vecf a = p2 - p1;
        const vecf b = p3 - p1;

        // sens direct
        const auto normal = cross(a, b);

        return {
                mtl,
                {p1, p2, p3},
                {normal, normal, normal},
                normal
        };
    }
    
    face make_face(
        material mtl,
        vecf p1, vecf p2, vecf p3,
        vecf n1, vecf n2, vecf n3)
    {
        const vecf a = p2 - p1;
        const vecf b = p3 - p1;

        // sens direct
        const auto normal = cross(a, b);

        return  {
                mtl,
                {p1, p2, p3},
                {n1, n2, n3},
                normal
            };
    }
    
    face make_face(
        material mtl,
        vecf p1, vecf p2, vecf p3,
        vecf n1, vecf n2, vecf n3,
        vecf normal)
    {
        return  {
                mtl,
                {p1, p2, p3},
                {n1, n2, n3},
                normal
            };
    }

} /* namespace Xrender */