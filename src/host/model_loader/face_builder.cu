
#include "face_builder.cuh"

namespace Xrender
{
    face make_face(material mtl, float3 p1, float3 p2, float3 p3)
    {
        const auto a = p2 - p1;
        const auto b = p3 - p1;

        // sens direct
        const auto normal = normalized(cross(a, b));

        return {
                mtl,
                {p1, p2, p3},
                {normal, normal, normal},
                normal
        };
    }

    face make_face(
        material mtl,
        float3 p1, float3 p2, float3 p3,
        float3 n1, float3 n2, float3 n3)
    {
        const auto a = p2 - p1;
        const auto b = p3 - p1;

        const auto normal = (n1 + n2 + n3) / 3.f;

        return  {
                mtl,
                {p1, p2, p3},
                {n1, n2, n3},
                normal
            };
    }

    face make_face(
        material mtl,
        float3 p1, float3 p2, float3 p3,
        float3 n1, float3 n2, float3 n3,
        float3 normal)
    {
        return  {
                mtl,
                {p1, p2, p3},
                {n1, n2, n3},
                normal
            };
    }
}