
#include <algorithm>
#include <iostream>
#include <thread>
#include <iostream>

#include "gpu/model/float3_operators.cuh"
#include "random_generator.cuh"
#include "bvh_builder.cuh"

namespace Xrender
{
    struct bvh_builder_input
    {
        bvh_builder_input() noexcept
        : face{nullptr}
        {
        }

        bvh_builder_input(const face* f) noexcept
        : face{f}
        {}

        // Avoid copying boxes when sorting
        bvh_builder_input(const bvh_builder_input& other) noexcept
        : face{other.face}
        {
        }

        bvh_builder_input(bvh_builder_input&& other) noexcept
        : face{other.face}
        {
        }

        auto& operator= (const bvh_builder_input& other) noexcept
        {
            face = other.face;
            return *this;
        }

        auto& operator= (bvh_builder_input&& other) noexcept
        {
            face = other.face;
            return *this;
        }

        const face *face;
        aabb_box boxes[2]; // 0: left, 1: right
    };

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
            box.ext_min = min(box.ext_min, min(it->face->geo.points[2], min(it->face->geo.points[1], it->face->geo.points[0])));
            box.ext_max = max(box.ext_max, max(it->face->geo.points[2], max(it->face->geo.points[1], it->face->geo.points[0])));
        }

        return box;
    }

    template <typename Titerator>
    static __host__ void make_boxes(Titerator begin, Titerator end, unsigned int box_index)
    {
        aabb_box box = {
            {INFINITY, INFINITY, INFINITY},
            {-INFINITY, -INFINITY, -INFINITY}};

        for (auto it = begin; it != end; ++it)
        {
            box.ext_min = min(box.ext_min, min(it->face->geo.points[2], min(it->face->geo.points[1], it->face->geo.points[0])));
            box.ext_max = max(box.ext_max, max(it->face->geo.points[2], max(it->face->geo.points[1], it->face->geo.points[0])));
            it->boxes[box_index] = box;
        }
    }

    /**
     * \brief compute the coordinate of a face gravity center projected on a line
     * \param f face to be projected
     * \param dir the direction of the line
     */
    static __host__ float face_axis_value(const triangle& f, const float3 &dir)
    {
        return dot(dir, f.points[0] + f.points[1] + f.points[2]);
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
            const auto value = face_axis_value(it->face->geo, dir);
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
     * \brief Try to find the partition with the lower SAH heuristic
     */
    template <typename Titerator>
    static __host__ auto find_partition(Titerator begin, Titerator end, std::size_t axis_sample_count)
    {
        // Generate an axis along which face are spread
        const auto sort_axis = find_axis(begin, end, axis_sample_count);

        // Sort the faces according to the axis
        std::sort(
            begin, end,
            [&sort_axis](const bvh_builder_input& f1, const bvh_builder_input& f2) -> bool
            {
                return face_axis_value(f1.face->geo, sort_axis) >
                       face_axis_value(f2.face->geo, sort_axis);
            });

        // Compute all the boxes
        make_boxes(begin, end, 0); // left
        make_boxes(std::make_reverse_iterator(end), std::make_reverse_iterator(begin), 1); // right

        // Find the best partition
        const auto count = end - begin;
        auto best_pivot = begin + 1;
        auto best_sah = INFINITY;

        for (auto it = begin + 1; it != end - 1; ++it)
        {
            const float left_count = it - begin;
            const float right_count = end - it;
            const float sah =
                (float)left_count * aabb_box_half_area((it - 1)->boxes[0]) +
                (float)right_count * aabb_box_half_area(it->boxes[1]);

            if (sah < best_sah)
            {
                best_pivot = it;
                best_sah = sah;
            }
        }

        return best_pivot;
    }


    template <typename Titerator>
    static __host__ host_bvh_tree::node build_node(Titerator begin, Titerator end);

    /**
     * \brief Build a branch with the given faces (The must be more than one face)
     */
    template <typename Titerator>
    static __host__ std::unique_ptr<host_bvh_tree> build_branch(Titerator begin, Titerator end)
    {
        // find best partition and build childs
        const auto partition = find_partition(begin, end, 32);

        // Build node
        auto branch = std::make_unique<host_bvh_tree>();
        branch->box = make_aabb_box(begin, end);
        branch->left_child = build_node(begin, partition);
        branch->right_child = build_node(partition, end);

        return branch;
    }

    /**
     * \brief Build a node with the given faces
     */
    template <typename Titerator>
    static __host__ host_bvh_tree::node build_node(Titerator begin, Titerator end)
    {
        // Only one face
        if (begin + 1 == end)
        {
            // return face as leaf
            return begin->face;
        }
        else
        {
            return build_branch(begin, end);
        }
    }

    __host__ std::unique_ptr<host_bvh_tree> build_bvh_tree(const std::vector<face>& geometry)
    {
        const auto face_count = geometry.size();
        std::vector<bvh_builder_input> builder_input{face_count};

        // Get faces ptr in a buffer (in order to sort them)
        std::transform(
            geometry.begin(), geometry.end(), builder_input.begin(),
            [](const auto& f) { return bvh_builder_input{&f}; });

        return build_branch(builder_input.begin(), builder_input.end());
    }

}