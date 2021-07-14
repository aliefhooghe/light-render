
#include <algorithm>
#include <iostream>

#include "random_generator.cuh"
#include "bvh_builder.cuh"

namespace Xrender
{
    /**
     * \brief return half the surface of a aabb box
     */
    static __host__ float aabb_box_half_area(const aabb_box &box)
    {
        const auto lengths = box.ext_max - box.ext_min;
        return (lengths.x * lengths.y +
                lengths.x * lengths.z +
                lengths.y * lengths.z);
    }

    /**
     * \brief build the smallest aabb box containing all given faces
     * \param begin iterator to a face pointer container
     */
    template <typename Titerator>
    static __host__ aabb_box make_aabb_box(Titerator begin, Titerator end)
    {
        aabb_box box = {
            {INFINITY, INFINITY, INFINITY},
            {-INFINITY, -INFINITY, -INFINITY}};

        for (auto it = begin; it != end; ++it)
        {
            box.ext_min = min(box.ext_min, min((*it)->points[2], min((*it)->points[1], (*it)->points[0])));
            box.ext_max = max(box.ext_max, max((*it)->points[2], max((*it)->points[1], (*it)->points[0])));
        }

        return box;
    }

    /**
     * \brief compute the coordinate of a face gravity center projected on a line
     * \param f face to be projected
     * \param dir the direction of the line
     */
    static __host__ float face_axis_value(const face *f, const float3 &dir)
    {
        return dot(dir, f->points[0] + f->points[1] + f->points[2]);
    }

    /**
     * \brief Compute the variance of the faces gravity centers projected on a line
     */
    template <typename Titerator>
    static __host__ float axis_variance(const float3 &dir, Titerator begin, Titerator end)
    {
        const float count = end - begin;
        float sum = 0.0f;
        float square_sum = 0.0f;

        for (auto it = begin; it != end; ++it)
        {
            const auto value = face_axis_value(*it, dir);
            sum += value;
            square_sum += (value * value);
        }

        return (square_sum - (sum * sum)) / (count * count);
    }



    /**
     * \brief Sample some random axis directions and return the one
     * with the greatest axis variance
     */
    template <typename Titerator>
    static __host__ float3 find_axis(Titerator begin, Titerator end, std::size_t sample_count)
    {
        float3 best_axis{};
        float best_variance = -INFINITY;

        for (auto i = 0u; i < sample_count; ++i)
        {
            const float3 axis = rand::unit_sphere_uniform();
            const float axis_var = axis_variance(axis, begin, end);

            if (axis_var > best_variance)
            {
                best_variance = axis_var;
                best_axis = axis;
            }
        }

        return best_axis;
    }

    /**
     * \brief compute the SAH heuristic for a given partition
     */
    template <typename Titerator>
    static __host__ float compute_partition_sah_heuristic(Titerator begin1, Titerator begin2, Titerator end)
    {
        const auto count1 = begin2 - begin1; // end
        const auto count2 = end - begin2;

        const auto box1 = make_aabb_box(begin1, begin2);
        const auto box2 = make_aabb_box(begin2, end);

        return (float)count1 * aabb_box_half_area(box1) + (float)count2 * aabb_box_half_area(box2);
    }

    /**
     * \brief Try to find the partition with the lower SAH heuristic
     */
    template <typename Titerator>
    static __host__ auto find_partition(Titerator begin, Titerator end, std::size_t axis_sample_count)
    {
        const auto count = end - begin;
        const auto pivot_test_count = std::min(count - 1l, 1000l);

        // Generate an axis along which face are spread
        const auto sort_axis = find_axis(begin, end, axis_sample_count);

        // Sort the faces according to the axis
        std::sort(
            begin, end,
            [&sort_axis](const face *f1, const face *f2) -> bool
            {
                return face_axis_value(f1, sort_axis) >
                       face_axis_value(f2, sort_axis);
            });

        // Find the best partition
        float best_sah = INFINITY;                    // less is better
        Titerator best_partition = begin + count / 2; // TODO

        for (auto i = 0; i < pivot_test_count; ++i)
        {
            const auto partition = begin + (i * (count - 1u)) / pivot_test_count;
            const float partition_sah = compute_partition_sah_heuristic(begin, partition, end);

            if (partition_sah < best_sah)
            {
                best_sah = partition_sah;
                best_partition = partition;
            }
        }

        return best_partition;
    }

    /**
     * \brief Build a branch with the given faces with branch's root at given position in tree
     */
    template <typename Titerator>
    static __host__ std::size_t build_branch(Titerator begin, Titerator end, std::vector<bvh_node> &tree, std::size_t root_index)
    {
        //  Reallocate nodes if needed
        if (tree.size() <= root_index)
            tree.resize(1u + 2u * tree.size());

        // Only one face
        if (begin + 1 == end)
        {
            tree[root_index].type = bvh_node::LEAF;
            tree[root_index].leaf = **begin;
            return root_index + 1u;
        }
        else
        {
            // find best partition
            const auto partition = find_partition(begin, end, 32);
            const auto node_box = make_aabb_box(begin, end);

            // build first child
            const auto first_child_index = root_index + 1u;
            const auto second_child_index = build_branch(begin, partition, tree, first_child_index);

            // fill node
            tree[root_index].type = bvh_node::BOX;
            tree[root_index].node = bvh_parent{
                node_box,
                static_cast<decltype(bvh_parent::second_child_idx)>(second_child_index)};

            // create second branch an return leading index
            return build_branch(partition, end, tree, second_child_index);
        }
    }

    __host__ std::vector<bvh_node> build_bvh_tree(const std::vector<face> &model)
    {
        const auto face_count = model.size();
        // Optmistic allaction (for a perfectly balanced tree)
        // build_branch will reallocate when needed
        std::vector<bvh_node> tree(2u * face_count);
        std::vector<const face *> model_faces{face_count};

        // Get faces ptr in a buffer (in order to sort them)
        std::transform(
            model.begin(), model.end(), model_faces.begin(),
            [](const auto& f) { return &f; });

        // Build tree : root branch
        const auto node_count = build_branch(model_faces.begin(), model_faces.end(), tree, 0u);
        tree.resize(node_count);

        std::cout << "Bvh tree was built : \n"
                  << "\tface_count : " << face_count << "\n"
                  << "\tnode_count : " << node_count << "\n"
                  << "\tmax depth  : " << bvh_tree_max_depth(tree) << std::endl;

        return tree;
    }

    static __host__ std::size_t branch_max_depth(const std::vector<bvh_node> &tree, std::size_t root_index)
    {
        if (tree[root_index].type == bvh_node::LEAF)
        {
            return 1u;
        }
        else
        {
            return 1u + std::max(
                branch_max_depth(tree, root_index + 1u),
                branch_max_depth(tree, tree[root_index].node.second_child_idx));
        }
    }

    __host__ std::size_t bvh_tree_max_depth(const std::vector<bvh_node> &tree)
    {
        return branch_max_depth(tree, 0u);
    }

}