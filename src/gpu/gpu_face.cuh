#ifndef GPU_FACE_CUH
#define GPU_FACE_CUH

#include <math_constants.h>

#include "vector_operations.cuh"
#include "gpu_mtl.cuh"

namespace Xrender {

    struct gpu_face {
        gpu_mtl mtl;
        float3 points[3];
        float3 normals[3];
        float3 normal;
    };

    struct gpu_intersection {
        float3 pos;
        float3  normal;
        float distance;
        const gpu_face *triangle;
    };

    static __device__ bool gpu_intersect_ray_face(const gpu_face& fa, const float3& pos, const float3& dir, gpu_intersection& inter)
    {
        const float EPSILON = 0.000001f;
        const float3 vertex0 = fa.points[0];
        const float3 vertex1 = fa.points[1];
        const float3 vertex2 = fa.points[2];
        float3 edge1, edge2, h, s, q;
        float a,f,u,v, w;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = _cross(dir, edge2);
        a = _dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return false;    // This ray is parallel to this triangle.
        f = 1.0f / a;
        s = pos - vertex0;
        u = f * _dot(s, h);
        if (u < 0.0 || u > 1.f)
            return false;
        q = _cross(s, edge1);
        v = f * _dot(dir, q);
        if (v < 0.f || u + v > 1.f)
            return false;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * _dot(edge2, q);
        if (t > EPSILON) // ray intersection
        {
            w = 1.f - (u + v);
            inter.pos = pos + t * dir;
            inter.triangle = &fa;
            inter.distance = t;
            inter.normal = w * fa.normals[0] + u * fa.normals[1] + v * fa.normals[2];
            return true;
        }
        else // This means that there is a line intersection but not a ray intersection.
            return false;
    }

    static __device__ bool gpu_intersect_ray_model(
        const gpu_face *model, const int face_count, const float3& pos, const float3& dir, gpu_intersection& inter)
    {
        gpu_intersection tmp_inter;
        float nearest = CUDART_INF_F;
        bool hit = false;

        for (int i = 0u; i < face_count; ++i) {
            if (gpu_intersect_ray_face(model[i], pos, dir, tmp_inter)) {
                hit = true;
                if (nearest > tmp_inter.distance) {
                    nearest = tmp_inter.distance;
                    inter = tmp_inter;
                }
            }
        }

        return hit;
    }

}

#endif