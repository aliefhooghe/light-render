

#include <iostream>

#include "bvh_tree.h"
#include "random_generator.h"
#include "ray_intersection.h"

namespace Xrender
{

    /**
     * 
     *      AABB BOX 
     * 
     */

    /**
     * \brief ray/aabb box intersection
     */
    static bool intersect_ray_aabb_box(const aabb_box &box, const vecf &pos, const vecf &dir, float &distance)
    {
        // rely on  IEEE 754 if divide by zero

        const float tx1 = (box.ext_min.x - pos.x) / dir.x;
        const float tx2 = (box.ext_max.x - pos.x) / dir.x;

        const float ty1 = (box.ext_min.y - pos.y) / dir.y;
        const float ty2 = (box.ext_max.y - pos.y) / dir.y;

        const float tz1 = (box.ext_min.z - pos.z) / dir.z;
        const float tz2 = (box.ext_max.z - pos.z) / dir.z;

        const float tmin = std::max(std::max(std::min(tx1, tx2), std::min(ty1, ty2)), std::min(tz1, tz2));
        const float tmax = std::min(std::min(std::max(tx1, tx2), std::max(ty1, ty2)), std::max(tz1, tz2));

        distance = tmin;
        return (tmax >= 0 && tmin <= tmax);
    }

    /**
     * \brief return half the surface of a aabb box
     */
    static float aabb_box_half_area(const aabb_box &box)
    {
        const auto lengths = box.ext_max - box.ext_min;
        return (lengths.x * lengths.y +
                lengths.x * lengths.z +
                lengths.y * lengths.z);
    }

    /**
     * 
     *  Tree construction
     *
     */

    /**
     * \brief build the smallest aabb box containing all given faces 
     * \param begin iterator to a face pointer container
     */
    template <typename Titerator>
    aabb_box make_aabb_box(Titerator begin, Titerator end)
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
    static float face_axis_value(const face *f, const vecf &dir)
    {
        return dot(dir, f->points[0] + f->points[1] + f->points[2]);
    }

    /**
     * \brief Compute the variance of the faces gravity centers projected on a line
     */
    template <typename Titerator>
    float axis_variance(const vecf &dir, Titerator begin, Titerator end)
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
    vecf find_axis(Titerator begin, Titerator end, unsigned int sample_count)
    {
        vecf best_axis{};
        float best_variance = -INFINITY;

        for (auto i = 0u; i < sample_count; ++i)
        {
            const vecf axis = rand::unit_sphere_uniform();
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
    float compute_partition_sah_heuristic(Titerator begin1, Titerator begin2, Titerator end)
    {
        const auto count1 = begin2 - begin1; // end
        const auto count2 = end - begin2;

        const auto box1 = make_aabb_box(begin1, begin2);
        const auto box2 = make_aabb_box(begin2, end);

        return (float)count1 * aabb_box_half_area(box1) + (float)count2 * aabb_box_half_area(box2);
    }

    /**
     * \brief 
     */
    template <typename Titerator>
    auto find_partition(Titerator begin, Titerator end, unsigned int axis_sample_count)
    {
        const auto count = end - begin;
        const auto pivot_test_count = std::min(count - 1l, 1000l);

        // Generate an axis along which face are spread
        const auto sort_axis = find_axis(begin, end, axis_sample_count);

        // Sort the faces according to the axis
        std::sort(
            begin, end,
            [&sort_axis](const face *f1, const face *f2) -> bool {
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

    template <typename Titerator>
    unsigned int build_branch(Titerator begin, Titerator end, std::vector<bvh_tree::node> &tree, unsigned int root_index)
    {
        if (tree.size() <= root_index)
            tree.resize(1u + 2u * tree.size());

        // Only one face
        if (begin + 1 == end)
        {
            tree[root_index] = bvh_tree::leaf{*begin};
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
            tree[root_index] = bvh_tree::box{node_box, second_child_index};

            // create second branch an return leading index
            return build_branch(partition, end, tree, second_child_index);
        }
    }

    bvh_tree::bvh_tree(const std::vector<face> &model)
    {
        const auto face_count = model.size();
        std::vector<const face *> model_faces{face_count};

        for (auto i = 0u; i < face_count; ++i)
            model_faces[i] = &(model[i]);

        const auto node_count =
            build_branch(model_faces.begin(), model_faces.end(), _tree, 0u);

        std::printf("Bvh tree was built. Use %u/%lu allocated nodes\n", node_count, _tree.size());
        _tree.resize(node_count);
    }

    bool bvh_tree::_branch_intersect_ray(
        unsigned int root_index,
        const vecf &pos, const vecf &dir, float distance_max,
        intersection &inter) const noexcept
    {
        const auto &branch = _tree[root_index];

        if (std::holds_alternative<box>(branch))
        {
            const auto &parent = std::get<box>(branch);
            float box_distance;

            if (intersect_ray_aabb_box(parent.box, pos, dir, box_distance) &&
                //(box_distance < distance_max) &&
                _branch_intersect_ray(root_index + 1, pos, dir, distance_max, inter))
            {
                // something hit in left child. Search right child with updated max distance
                _branch_intersect_ray(parent.second_child_idx, pos, dir, inter.distance, inter);
                return true;
            }
            else
            {
                return _branch_intersect_ray(parent.second_child_idx, pos, dir, distance_max, inter);
            }
        }
        else
        {
            const face *f = std::get<leaf>(branch);
            intersection tmp;
            const bool hit = intersect_ray_face(*f, pos, dir, tmp);

            if (hit && tmp.distance < distance_max)
            {
                inter = tmp;
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    bool bvh_tree::intersect_ray(const vecf &pos, const vecf &dir, intersection &inter) const noexcept
    {
        return _branch_intersect_ray(0u, pos, dir, INFINITY, inter);
    }

    bool bvh_tree::intersect_seg(const vecf &a, const vecf &b) const noexcept
    {
        const auto segment = b - a;
        const auto distance = segment.norm();
        intersection dummy;
        return _branch_intersect_ray(0, a, (1.f / distance) * segment, distance, dummy);
    }

} // namespace Xrender