
#include <stdexcept>
#include <iostream>

#include "host_bvh_tree.cuh"
#include "bvh_builder.cuh"

namespace Xrender
{
    __host__ static void _push_node(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const aabb_box *parent_box,
        const host_bvh_tree::node& host_node);

    __host__ static void _push_branch(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const aabb_box *parent_box,
        const host_bvh_tree& host_branch)
    {
        const auto branch_root_index = gpu_tree.size();
        const bool skip_box =
            (parent_box != nullptr) &&
            ((aabb_box_half_area(host_branch.box) / aabb_box_half_area(*parent_box)) > 0.8f);

        // Push root
        if (!skip_box)
        {
            bvh_node gpu_node;
            gpu_node.type = bvh_node::BOX;
            gpu_node.node.box = host_branch.box;
            gpu_tree.emplace_back(std::move(gpu_node));
        }

        // Push left child
        _push_node(gpu_tree, gpu_model, &host_branch.box, host_branch.left_child);

        // Push right child
        _push_node(gpu_tree, gpu_model, &host_branch.box, host_branch.right_child);

        if (!skip_box)
        {
            // Update branch root skip index
            gpu_tree[branch_root_index].node.skip_index = gpu_tree.size();
        }
    }

    __host__ static void _push_child(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const aabb_box *parent_box,
        const host_bvh_tree::parent& host_parent)
    {
        _push_branch(gpu_tree, gpu_model, parent_box, *host_parent);
    }

    __host__ static void _push_child(
        std::vector<bvh_node>& gpu_tree,
        std::vector<face>& gpu_model,
        const aabb_box *parent_box,
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
        const aabb_box *parent_box,
        const host_bvh_tree::node& host_node)
    {
        std::visit(
            [&gpu_tree, &gpu_model, parent_box](auto& child)
            {
                _push_child(gpu_tree, gpu_model, parent_box, child);
            },
            host_node);
    }

    __host__ host_bvh_tree::gpu_compatible_bvh host_bvh_tree::to_gpu_bvh() const
    {
        std::vector<bvh_node> gpu_tree{};
        std::vector<face> gpu_model{};

        // Push the root on the tree
        _push_branch(gpu_tree, gpu_model, nullptr, *this);

        // Change index that skip to the end at -1
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