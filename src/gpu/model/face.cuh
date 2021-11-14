#ifndef GPU_FACE_CUH
#define GPU_FACE_CUH

namespace Xrender {

    struct triangle {
        float3 points[3];
        float3 normals[3];
    };

    struct face {
        triangle geo;
        int mtl;
    };
}

#endif