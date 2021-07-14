#ifndef GPU_FACE_CUH
#define GPU_FACE_CUH

#include <math_constants.h>

#include "float3_operators.cuh"
#include "material.cuh"

namespace Xrender {

    struct face {
        material mtl;
        float3 points[3];
        float3 normals[3];
        float3 normal;
    };
}

#endif