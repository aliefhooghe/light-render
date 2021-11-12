
#include <stdexcept>
#include <iostream>
#include "host_bvh_tree.cuh"

namespace Xrender
{
    __host__ static void _push_node(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const host_bvh_tree::node& host_node);

    __host__ static void _push_branch(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const host_bvh_tree& host_branch)
    {
        // Push root
        bvh_node gpu_node;
        gpu_node.type = bvh_node::BOX;
        gpu_node.node.box = host_branch.box;

        const auto root_index = gpu_tree.size();
        gpu_tree.emplace_back(std::move(gpu_node));

        // Push left child
        _push_node(gpu_tree, gpu_model, host_branch.left_child);

        // Update root info
        gpu_tree[root_index].node.second_child_idx = gpu_tree.size();

        // Push right child
        _push_node(gpu_tree, gpu_model, host_branch.right_child);
    }

    __host__ static void _push_child(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const host_bvh_tree::parent& host_parent)
    {
        _push_branch(gpu_tree, gpu_model, *host_parent);
    }

    __host__ static void _push_child(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const host_bvh_tree::leaf& host_leaf)
    {
        // Push face on gpu model
        const int leaf_index = gpu_model.size();
        gpu_model.emplace_back(*host_leaf);

        // Push leaf on gpu tree
        bvh_node gpu_leaf;
        gpu_leaf.type = bvh_node::LEAF;
        gpu_leaf.leaf = leaf_index;
        gpu_tree.emplace_back(std::move(gpu_leaf));
    }

    __host__ static void _push_node(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const host_bvh_tree::node& host_node)
    {
        std::visit(
            [&gpu_tree, &gpu_model](auto& child) { _push_child(gpu_tree, gpu_model, child);},
            host_node);
    }

    __host__ host_bvh_tree::gpu_compatible_bvh host_bvh_tree::to_gpu_bvh() const
    {
        const auto depth = max_depth();
        if (depth > BVH_MAX_DEPTH)
        {
            std::cout << "Too much depth in tree for gpu: " << depth << std::endl;
            //throw std::invalid_argument("Bvh depth is too high for gpu");
        }

        std::vector<bvh_node> gpu_tree{};
        std::vector<face> gpu_model{};

        _push_branch(gpu_tree, gpu_model, *this);

        gpu_tree.shrink_to_fit();
        gpu_model.shrink_to_fit();

        return gpu_compatible_bvh
        {
            std::move(gpu_tree),
            std::move(gpu_model)
        };
    }

    __host__ static std::size_t _child_max_depth(const host_bvh_tree::parent& parent)
    {
        return parent->max_depth();
    }

    __host__ static std::size_t _child_max_depth(const host_bvh_tree::leaf& leaf)
    {
        return 1u;
    }

    __host__ static std::size_t _node_max_depth(const host_bvh_tree::node& host_node)
    {
        return std::visit(
            [](auto& child) { return _child_max_depth(child);},
            host_node);
    }

    __host__ std::size_t host_bvh_tree::max_depth() const noexcept
    {
        return 1 + std::max(
            _node_max_depth(left_child),
            _node_max_depth(right_child)
        );
    }

}