#ifndef FACE_BUILDER_CUH
#define FACE_BUILDER_CUH

#include "gpu/model/face.cuh"

namespace Xrender
{
    face make_face(material mtl, float3 p1, float3 p2, float3 p3);

    face make_face(
        material mtl,
        float3 p1, float3 p2, float3 p3,
        float3 n1, float3 n2, float3 n3);

}

#endif