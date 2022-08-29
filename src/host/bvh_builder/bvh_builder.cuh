
#include <vector>

#include "host_bvh_tree.cuh"

namespace Xrender
{
    __host__ float aabb_box_half_area(const aabb_box &box);
    __host__ std::unique_ptr<host_bvh_tree> build_bvh_tree(const std::vector<face>& geometry);
}
