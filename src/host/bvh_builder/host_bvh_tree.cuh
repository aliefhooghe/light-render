#ifndef XRENDER_HOST8BVH_TREE_H_
#define XRENDER_HOST8BVH_TREE_H_

#include <variant>
#include <vector>
#include <memory>

#include "gpu/model/bvh_tree.cuh"

namespace Xrender
{

    struct host_bvh_tree
    {
        using leaf = face;
        using parent = std::unique_ptr<host_bvh_tree>;
        using node = std::variant<leaf, parent>;

        __host__ std::size_t max_depth() const noexcept;
        __host__ std::vector<bvh_node> to_gpu_bvh() const;

        aabb_box box;
        node left_child{};
        node right_child{};
    };



}

#endif /* XRENDER_HOST8BVH_TREE_H_ */