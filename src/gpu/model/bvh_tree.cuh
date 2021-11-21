
#ifndef GPU_BVH_
#define GPU_BVH_

#include <math_constants.h>

namespace Xrender {

    struct aabb_box {
        float3 ext_min;
        float3 ext_max;

        __device__ bool intersect(const float3& pos, const float3& dir, float& distance) const
        {
            // rely on  IEEE 754 if divide by zero

            const float tx1 = (ext_min.x - pos.x) / dir.x;
            const float tx2 = (ext_max.x - pos.x) / dir.x;

            const float ty1 = (ext_min.y - pos.y) / dir.y;
            const float ty2 = (ext_max.y - pos.y) / dir.y;

            const float tz1 = (ext_min.z - pos.z) / dir.z;
            const float tz2 = (ext_max.z - pos.z) / dir.z;

            const float tmin = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
            const float tmax = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));

            distance = tmin;
            return (tmax >= 0 && tmin <= tmax);
        }

    };

    struct bvh_parent
    {
        aabb_box box;
        int skip_index;
    };

    struct bvh_node {
        enum {LEAF, BOX} type;
        union {
            bvh_parent node;
            int leaf;
        };
    };
}


#endif