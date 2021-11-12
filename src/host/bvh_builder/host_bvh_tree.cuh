#ifndef XRENDER_HOST8BVH_TREE_H_
#define XRENDER_HOST8BVH_TREE_H_

#include <variant>
#include <vector>
#include <memory>

#include "gpu/model/bvh_tree.cuh"
#include "gpu/model/face.cuh"

namespace Xrender
{
    struct host_bvh_tree
    {
        struct gpu_compatible_bvh
        {
            gpu_compatible_bvh(std::vector<bvh_node>&& t, std::vector<face>&& m) noexcept
            :   tree{std::move(t)}, model{std::move(m)}
            {}

            gpu_compatible_bvh(const gpu_compatible_bvh&) = default;
            gpu_compatible_bvh(gpu_compatible_bvh&&) noexcept = default;

            std::vector<bvh_node> tree{};
            std::vector<face> model{};
        };

        using leaf = const face*;
        using parent = std::unique_ptr<host_bvh_tree>;
        using node = std::variant<leaf, parent>;

        __host__ std::size_t max_depth() const noexcept;
        __host__ gpu_compatible_bvh to_gpu_bvh() const;

        aabb_box box;
        node left_child{};
        node right_child{};
    };
}

#endif /* XRENDER_HOST8BVH_TREE_H_ */