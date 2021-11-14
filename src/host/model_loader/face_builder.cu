
#include "face_builder.cuh"
#include "gpu/model/float3_operators.cuh"

namespace Xrender
{
    face make_face(int mtl, float3 p1, float3 p2, float3 p3)
    {
        const auto a = p2 - p1;
        const auto b = p3 - p1;

        // sens direct
        const auto normal = normalized(cross(a, b));

        return {
            {
                {p1, p2, p3},
                {normal, normal, normal}
            },
            mtl
        };
    }

    face make_face(
        int mtl,
        float3 p1, float3 p2, float3 p3,
        float3 n1, float3 n2, float3 n3)
    {
        const auto a = p2 - p1;
        const auto b = p3 - p1;

        return {
            {
                {p1, p2, p3},
                {n1, n2, n3}
            },
            mtl
        };
    }
}
