#ifndef XRENDER_FACE_INTERSECTION_CUH_
#define XRENDER_FACE_INTERSECTION_CUH_

#include "gpu/model/float3_operators.cuh"

namespace Xrender
{
    struct intersection
    {
        float2 uv;
        float distance;
    };

    static __device__ float3 interpolate_normal(
        const float3& incoming_dir,
        const float2& uv,
        const float3 normals[3])
    {
        auto w = 1.f - (uv.x + uv.y);
        const auto inter_normal = w * normals[0] + uv.x * normals[1] + uv.y * normals[2];
        return dot(inter_normal, incoming_dir) > 0.f ? -inter_normal : inter_normal;
    }

    static __device__ bool intersect_ray_face(const float3 points[3], const float3 &pos, const float3 &dir, intersection &inter)
    {
        const float EPSILON = 0.000001f;
        const auto point0 = points[0];
        const auto point1 = points[1];
        const auto point2 = points[2];
        float3 edge1, edge2, h, s, q;
        float a, f, u, v;
        edge1 = point1 - point0;
        edge2 = point2 - point0;
        h = cross(dir, edge2);
        a = dot(edge1, h);
        if (fabs(a) < EPSILON)
            return false; // This ray is parallel to this triangle.
        f = 1.0f / a;
        s = pos + EPSILON * dir - point0;
        u = f * dot(s, h);
        if (u < 0.f || u > 1.f)
            return false;
        q = cross(s, edge1);
        v = f * dot(dir, q);
        if (v < 0.f || u + v > 1.f)
            return false;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * dot(edge2, q);
        if (t > EPSILON) // ray intersection
        {
            inter.uv = make_float2(u, v);
            inter.distance = t;
            return true;
        }
        else {
            // This means that there is a line intersection but not a ray intersection.
            return false;
        }
    }
}

#endif