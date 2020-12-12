#ifndef TRIANGLE_INTERSECTION_H_
#define TRIANGLE_INTERSECTION_H_

#include "face.h"

namespace Xrender {

    struct intersection
    {
        vecf pos;
        //vecf dir; // needed ?
        vecf normal;
        float distance;
        const face *triangle;
    };

    // dir is normalized !!
    bool intersect_ray_face(const face& f, const vecf& pos, const vecf& dir, intersection& inter);

} /* namespace Xrender */

#endif