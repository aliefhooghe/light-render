
#include "ray_intersection.h"

namespace Xrender {

    bool intersect_ray_face(const face& fa, const vecf& pos, const vecf& dir, intersection& inter)
    {
        const float EPSILON = 0.000001f;
        vecf vertex0 = fa.points[0];
        vecf vertex1 = fa.points[1];
        vecf vertex2 = fa.points[2];
        vecf edge1, edge2, h, s, q;
        float a,f,u,v, w;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = cross(dir, edge2);
        a = dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return false;    // This ray is parallel to this triangle.
        f = 1.0f / a;
        s = pos - vertex0;
        u = f * dot(s, h);
        if (u < 0.0 || u > 1.f)
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
            inter.triangle = &fa;
            inter.distance = t;
            inter.normal = w * fa.normals[0] + u * fa.normals[1] + v * fa.normals[2];
            return true;
        }
        else // This means that there is a line intersection but not a ray intersection.
            return false;
    }

} /* Xrender */