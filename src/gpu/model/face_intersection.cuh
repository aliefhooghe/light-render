#ifndef XRENDER_FACE_INTERSECTION_CUH_
#define XRENDER_FACE_INTERSECTION_CUH_

#include "face.cuh"

namespace Xrender
{
    struct intersection
    {
        float3 pos;
        float3 normal;
        float3 ab;
        float distance;
        material mtl;
    };

    static __device__ bool intersect_ray_face(const face &fa, const float3 &pos, const float3 &dir, intersection &inter)
    {
        const float EPSILON = 0.000001f;
        const float3 vertex0 = fa.points[0];
        const float3 vertex1 = fa.points[1];
        const float3 vertex2 = fa.points[2];
        float3 edge1, edge2, h, s, q;
        float a, f, u, v, w;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = cross(dir, edge2);
        a = dot(edge1, h);
        if (fabs(a) < EPSILON)
            return false; // This ray is parallel to this triangle.
        f = 1.0f / a;
        s = pos + EPSILON * dir - vertex0;
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
            w = 1.f - (u + v);
            inter.pos = pos + t * dir;
            inter.distance = t;
            inter.normal = w * fa.normals[0] + u * fa.normals[1] + v * fa.normals[2];
            inter.ab = fa.points[1] - fa.points[0];
            return true;
        }
        else {
            // This means that there is a line intersection but not a ray intersection.
            return false;
        }
    }

    static __device__ bool intersect_ray_model(
        const face *model, const int face_count, const float3& pos, const float3& dir, intersection& inter)
    {
        intersection tmp_inter;
        float nearest = CUDART_INF_F;
        bool hit = false;

        for (int i = 0u; i < face_count; ++i) {
            if (intersect_ray_face(model[i], pos, dir, tmp_inter)) {
                hit = true;
                if (nearest > tmp_inter.distance) {
                    nearest = tmp_inter.distance;
                    inter = tmp_inter;
                    inter.mtl = model[i].mtl;
                }
            }
        }

        return hit;
    }
}

#endif