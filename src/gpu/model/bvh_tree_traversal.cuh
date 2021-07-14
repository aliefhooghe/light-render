#ifndef BVH_TREE_TRAVERSAL_CUH_
#define BVH_TREE_TRAVERSAL_CUH_

#include "face_intersection.cuh"
#include "bvh_tree.cuh"

namespace Xrender
{
    static __device__ bool intersect_ray_bvh(const bvh_node *tree, const float3& pos, const float3& dir, intersection& inter)
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

                if (intersect_ray_face(node.leaf, pos, dir, tmp) && tmp.distance < neareset)
                {
                    neareset = tmp.distance;
                    inter = tmp;
                    inter.mtl = node.leaf.mtl;
                    hit = true;
                }
            }
        }

        return hit;
    }
}

#endif