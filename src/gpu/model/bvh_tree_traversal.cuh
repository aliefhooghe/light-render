#ifndef BVH_TREE_TRAVERSAL_CUH_
#define BVH_TREE_TRAVERSAL_CUH_

#include "face_intersection.cuh"
#include "face.cuh"
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
        intersection best_intersection;
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
        state.best_intersection.distance = CUDART_INF_F;
        state.best_index = -1;
    }

    static __device__ bvh_traversal_status bvh_traversal_step(
        bvh_traversal_state& state,
        const bvh_node *tree,
        const face *geometry,
        const float3& pos, const float3& dir)
    {
        if (state.stack.pointer > 0)
        {
            // Pop a node
            const auto node_id = state.stack.data[--state.stack.pointer];
            const auto node = tree[node_id];

            if (node.type == bvh_node::BOX)
            {
                float box_distance;
                if (node.node.box.intersect(pos, dir, box_distance) && (box_distance < state.best_intersection.distance))
                {
                    state.stack.data[state.stack.pointer++] = node.node.second_child_idx; // right child
                    state.stack.data[state.stack.pointer++] = node_id + 1;                // left child visited first
                }
            }
            else
            {
                //  leaf : test a face
                intersection tmp;

                if (intersect_ray_face(geometry[node.leaf].geo.points, pos, dir, tmp)
                   && tmp.distance < state.best_intersection.distance)
                {
                    state.best_intersection = tmp;
                    state.best_index = node.leaf;
                }
            }

            return bvh_traversal_status::IN_PROGRESS;
        }
        else
        {
            return state.best_index & 0x80000000u ?
                bvh_traversal_status::NO_HIT :
                bvh_traversal_status::HIT;
        }
    }

    static __device__ bool intersect_ray_bvh(
        const bvh_node *tree,
        const face *model,
        const float3& pos, const float3& dir,
        intersection& best_intersection, int& best_geometry)
    {
        bvh_traversal_status traversal_status;
        bvh_traversal_state traversal_state;

        bvh_traversal_init(traversal_state);
        while ((traversal_status = bvh_traversal_step(traversal_state, tree, model, pos, dir))
            == bvh_traversal_status::IN_PROGRESS);

        if (traversal_status == bvh_traversal_status::HIT)
        {
            best_intersection = traversal_state.best_intersection;
            best_geometry = traversal_state.best_index;
            return true;
        }
        else
        {
            return false;
        }
    }
}

#endif