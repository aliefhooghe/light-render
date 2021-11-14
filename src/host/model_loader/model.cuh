#ifndef XRENDER_MODEL_CUH
#define XRENDER_MODEL_CUH

#include <vector>

#include "gpu/model/face.cuh"
#include "gpu/model/material.cuh"

namespace Xrender
{
    struct model
    {
        std::vector<face> geometry;
        std::vector<material> mtl_bank;
    };
}

#endif