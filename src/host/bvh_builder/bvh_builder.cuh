
#include <vector>

#include "host_bvh_tree.cuh"

namespace Xrender
{
    __host__ std::unique_ptr<host_bvh_tree> build_bvh_tree(const std::vector<face>& geometry);
}
