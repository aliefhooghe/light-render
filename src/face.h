#ifndef XRENDER_FACE_H_
#define XRENDER_FACE_H_

#include <array>

#include "vec.h"
#include "material.h"

namespace Xrender {

    struct face
    {
        material mtl;
        std::array<vecf, 3> points;
        std::array<vecf, 3> normals;
        vecf normal;
    };

    face make_face(material mtl, vecf p1, vecf p2, vecf p3);
    
    face make_face(
        material mtl,
        vecf p1, vecf p2, vecf p3,
        vecf n1, vecf n2, vecf n3);
    
    face make_face(
        material mtl,
        vecf p1, vecf p2, vecf p3,
        vecf n1, vecf n2, vecf n3,
        vecf normal);

} /* namespace Xrender */


#endif /* MODEL_H_ */