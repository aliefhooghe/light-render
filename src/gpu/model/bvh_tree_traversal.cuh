#ifndef BVH_TREE_TRAVERSAL_CUH_
#define BVH_TREE_TRAVERSAL_CUH_

#include "face_intersection.cuh"
#include "bvh_tree.cuh"

namespace Xrender
{
    struct bvh_stack
    {
        int pointer;
        int data[BVH_MAX_DEPTH];
    };

    struct bvh_traversal_state
    {
        bvh_stack stack;
        int best_index;
        float nearest;
    };

    enum class bvh_traversal_status
    {
        IN_PROGRESS,    // bvh traversal is in progress
        HIT,            // the nearest intersection was found.
        NO_HIT          // traversal finished without finding any intersection
    };

    static __device__ void bvh_traversal_init(bvh_traversal_state& state)
    {
        // Root index (0) on the stack
        state.stack.pointer = 1u;
        state.stack.data[0] = 0;
        state.nearest = CUDART_INF_F;
        state.best_index = -1;
    }

    static __device__ bvh_traversal_status bvh_traversal_step(
        bvh_traversal_state& state,
        const bvh_node *tree,
        const face *model,
        const float3& pos, const float3& dir,
        intersection& inter, material& mtl)
    {
        if (state.stack.pointer > 0)
        {
            // Pop a node
            const auto node_id = state.stack.data[--state.stack.pointer];
            const auto node = tree[node_id];

            if (node.type == bvh_node::BOX)
            {
                float box_distance;
                if (node.node.box.intersect(pos, dir, box_distance) && (box_distance < state.nearest))
                {
                    state.stack.data[state.stack.pointer++] = node.node.second_child_idx; // right child
                    state.stack.data[state.stack.pointer++] = node_id + 1;                // left child visited first
                }
            }
            else
            {
                //  leaf : test a face
                intersection tmp;

                if (intersect_ray_face(model[node.leaf].geo, pos, dir, tmp) && tmp.distance < state.nearest)
                {
                    state.nearest = tmp.distance;
                    state.best_index = node.leaf;
                    inter = tmp;
                }
            }

            return bvh_traversal_status::IN_PROGRESS;
        }
        else
        {
            if (state.best_index & 0x80000000u)
            {
                return bvh_traversal_status::NO_HIT;
            }
            else
            {
                mtl = model[state.best_index].mtl;
                return bvh_traversal_status::HIT;
            }
        }
    }

    static __device__ bool intersect_ray_bvh(
        const bvh_node *tree,
        const face *model,
        const float3& pos, const float3& dir,
        intersection& inter, material& mtl)
    {
        bvh_traversal_status traversal_status;
        bvh_traversal_state traversal_state;
        bvh_traversal_init(traversal_state);
        while ((traversal_status = bvh_traversal_step(traversal_state, tree, model, pos, dir, inter, mtl))
            == bvh_traversal_status::IN_PROGRESS);
        return (traversal_status == bvh_traversal_status::HIT);
    }
}

#endif