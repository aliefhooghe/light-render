
#include <vector>

#include "gpu/model/bvh_tree.cuh"

namespace Xrender
{
    __host__ std::vector<bvh_node> build_bvh_tree(const std::vector<face>& model);
    __host__ std::size_t bvh_tree_max_depth(const std::vector<bvh_node>& tree);
}