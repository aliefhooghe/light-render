#ifndef BVH_TREE_H_
#define BVH_TREE_H_

#include <variant>
#include <vector>

#include "face.h"
#include "ray_intersection.h"

namespace Xrender {

    /**
     * \brief axis aligned bounding box
     */
    struct aabb_box {
        vecf ext_min;
        vecf ext_max;
    };

    /**
     * 
     */
    class bvh_tree {

        public:

        using leaf = const face*;

        struct box {
            aabb_box box;
            unsigned int second_child_idx;
        };

        using node = 
            std::variant<box, leaf>;

        bvh_tree(const std::vector<face>& model);
        bvh_tree(const bvh_tree&) = default;
        bvh_tree(bvh_tree&&) = default;

        bool intersect_ray(const vecf& pos, const vecf& dir, intersection& inter) const noexcept;
        bool intersect_seg(const vecf& a, const vecf& b) const noexcept;

    private:
        bool _branch_intersect_ray(
            unsigned int root_index,
            const vecf& pos, const vecf& dir, float distance_max,
            intersection& inter) const noexcept;
        
        std::vector<node> _tree{}; // stored in depth first order
    };


}

#endif