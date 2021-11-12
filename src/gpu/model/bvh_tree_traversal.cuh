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
        bool hit;
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

        state.hit = false;
        state.nearest = CUDART_INF_F;
    }

    static __device__ bvh_traversal_status bvh_traversal_step(
        bvh_traversal_state& state,
        const bvh_node *tree,
        const face *model,
        const float3& pos, const float3& dir,
        intersection& inter)
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
                const auto leaf_face = model[node.leaf];

                if (intersect_ray_face(leaf_face, pos, dir, tmp) && tmp.distance < state.nearest)
                {
                    state.nearest = tmp.distance;
                    inter = tmp;
                    inter.mtl = leaf_face.mtl;
                    state.hit = true;
                }
            }

            return bvh_traversal_status::IN_PROGRESS;
        }
        else
        {
            // Search is finished
            return state.hit ?
                bvh_traversal_status::HIT :
                bvh_traversal_status::NO_HIT;
        }
    }

    static __device__ bool intersect_ray_bvh(
        const bvh_node *tree,
        const face *model,
        const float3& pos, const float3& dir, intersection& inter)
    {
        //  stack = [0|] (put root on the stack)
        int stack_ptr = 1u;  // pointe au dessus du sommet de la pile;
        int stack[BVH_MAX_DEPTH] = {0};

        float neareset = CUDART_INF_F;
        bool hit = false;

        //  while stack is not empty
        while (stack_ptr > 0) {
            // pop a node
            const auto node_id = stack[--stack_ptr];
            const auto node = tree[node_id];

            if (node.type == bvh_node::BOX) {
                float box_distance;
                if (node.node.box.intersect(pos, dir, box_distance) && (box_distance < neareset))
                {
                    stack[stack_ptr++] = node.node.second_child_idx; // right child
                    stack[stack_ptr++] = node_id + 1;           // left child visited first
                }
            }
            else {
                //  leaf : test a face
                intersection tmp;
                const auto leaf_face = model[node.leaf];

                if (intersect_ray_face(leaf_face, pos, dir, tmp) && tmp.distance < neareset)
                {
                    neareset = tmp.distance;
                    inter = tmp;
                    inter.mtl = leaf_face.mtl;
                    hit = true;
                }
            }
        }

        return hit;
    }
}

#endif