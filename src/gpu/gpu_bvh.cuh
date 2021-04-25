
#ifndef GPU_BVH_
#define GPU_BVH_

#include <math_constants.h>

#include "bvh_tree.h"
#include "gpu_face.cuh"

namespace Xrender {


    struct gpu_aabb_box {
        float3 ext_min;
        float3 ext_max;
        int second_child_idx;

        __device__ bool intersect(const float3& pos, const float3& dir, float& distance) const
        {
            // rely on  IEEE 754 if divide by zero

            const float tx1 = (ext_min.x - pos.x) / dir.x;
            const float tx2 = (ext_max.x - pos.x) / dir.x;

            const float ty1 = (ext_min.y - pos.y) / dir.y;
            const float ty2 = (ext_max.y - pos.y) / dir.y;

            const float tz1 = (ext_min.z - pos.z) / dir.z;
            const float tz2 = (ext_max.z - pos.z) / dir.z;
            
            const float tmin = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
            const float tmax = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));

            distance = tmin;
            return (tmax >= 0 && tmin < tmax);
        }
    
    };

    struct gpu_bvh_parent { 
        gpu_aabb_box box;
        int second_child_idx;
    };

    struct gpu_bvh_node {
        enum {LEAF, BOX} type;
        union {
            gpu_face leaf;    
            gpu_bvh_parent node;
        };
    };

    static __device__ bool gpu_intersect_ray_bvh(const gpu_bvh_node *tree, const float3& pos, const float3& dir, gpu_intersection& inter)
    {
        //  stack = [0|] (put root on the stack)
        int stack_ptr = 1u;  // pointe au dessus du sommet de la pile;
        int stack[64] = {0};

        float neareset = CUDART_INF_F;
        bool hit = false;

        //  while stack is not empty
        while (stack_ptr > 0) {
            // pop a node
            const auto node_id = stack[--stack_ptr];
            const auto node = tree[node_id];

            if (node.type == gpu_bvh_node::BOX) {
                float box_distance;
                if (node.node.box.intersect(pos, dir, box_distance) && (box_distance < neareset))
                {
                    stack[stack_ptr++] = node.node.second_child_idx; // right child
                    stack[stack_ptr++] = node_id + 1;           // left child visited first
                }
            }
            else {
                //  leaf : test a face
                gpu_intersection tmp;

                if (gpu_intersect_ray_face(node.leaf, pos, dir, tmp) && tmp.distance < neareset)
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

    // convertion helper

    static __host__ std::vector<gpu_bvh_node> make_gpu_bvh(const bvh_tree& tree)
    {
        std::vector<gpu_bvh_node> gpu_tree{tree._tree.size()};

        std::transform(
            tree._tree.begin(),
            tree._tree.end(),
            gpu_tree.begin(),
            [](const bvh_tree::node& n)
            {
                gpu_bvh_node ret;
                
                if (std::holds_alternative<bvh_tree::box>(n)) {
                    const auto& box = std::get<bvh_tree::box>(n);
                    ret.type = gpu_bvh_node::BOX;
                    ret.node.box.ext_min.x = box.box.ext_min.x;
                    ret.node.box.ext_min.y = box.box.ext_min.y;
                    ret.node.box.ext_min.z = box.box.ext_min.z;
                    ret.node.box.ext_max.x = box.box.ext_max.x;
                    ret.node.box.ext_max.y = box.box.ext_max.y;
                    ret.node.box.ext_max.z = box.box.ext_max.z;
                    ret.node.second_child_idx = box.second_child_idx;
                }
                else {
                    ret.type = gpu_bvh_node::LEAF;
                    ret.leaf = make_gpu_face(*std::get<const face*>(n));
                }

                return ret;
            });

        return gpu_tree;
    }

}


#endif