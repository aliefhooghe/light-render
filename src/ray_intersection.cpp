
#include "ray_intersection.h"

namespace Xrender {

    bool intersect_ray_face(const face& f, const vecf& pos, const vecf& dir, intersection& inter)
    {
        const auto dot_dir_normal = dot(dir, f.normal);

        if (dot_dir_normal >= 0.0f)
            return false; // to do : mieux

        // triangle = abc

        //  Compute triangle plane intersection
        const auto pos_a = f.points[0] - pos;
        const float k =  // todo ne dépend pas du signe de dot_dir_normal
            dot(pos_a, f.normal) / dot_dir_normal;

        //  if ray does not intersect triangle plane
        if (k <= 0.f)
            return false;

        const auto intersect_point = pos + k * dir;

        const auto a_b = f.points[1] - f.points[0];
        const auto b_c = f.points[2] - f.points[1];
        const auto c_a = f.points[0] - f.points[2];

        const auto a_intersetc_point = intersect_point - f.points[0]; 
        const auto b_intersect_point = intersect_point - f.points[1];
        const auto c_intersect_point = intersect_point - f.points[2];

        float u, v, w; // barycentric coordinate into the triangle

        const auto test = cross(a_b, a_intersetc_point);
        if (dot(f.normal, test) < 0)
            return false;

        const auto test2 = cross(b_c, b_intersect_point);
        if ((u = dot(f.normal, test2)) < 0)
            return false;
    
        const auto test3 = cross(c_a, c_intersect_point);
        if ((v = dot(f.normal, test3)) < 0)
            return false;

        // compute the barycentric coordinate
        const auto double_area = cross(a_b, c_a).norm();
        u /= double_area;
        v /= double_area;
        w = 1.0f - (u + v);

        if (w < 0.f)
            return false;

        // fill intersection
        inter.pos = intersect_point;
        inter.distance = k;
        inter.triangle = &f;

        //  normal interpolation
        inter.normal = u * f.normals[0] + v * f.normals[1] + w * f.normals[2];

        return true;
    }

} /* Xrender */