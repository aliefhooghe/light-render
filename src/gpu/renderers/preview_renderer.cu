
#include <algorithm>
#include <chrono>
#include <iostream>

#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/curand_helper.cuh"
#include "gpu/utils/image_grid_dim.cuh"
#include "gpu/model/bvh_tree_traversal.cuh"

#include "preview_renderer.cuh"

namespace Xrender
{
    __global__ void preview_integrate_kernel(
        const bvh_node *tree,
        const face *geometry,
        const material *mtl_bank,
        const camera cam,
        curandState_t *rand_pool,
        const int sample_count,
        float3 *image)
    {
        //  Get pixel position in ima
        const auto width = cam.get_image_width();

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;
        const int pixel_index = x + y * width;

        //  Initialize random generator
        if (x < width)
        {
            auto rand_state = rand_pool[pixel_index];

            float3 pos;
            float3 dir;
            intersection inter;
            float3 estimator = {0.f, 0.f, 0.f};

            for (auto i = 0; i < sample_count; i++)
            {
                dir = cam.sample_ray(&rand_state, pos, x, y);

                int best_geo_index;
                if (intersect_ray_bvh(tree, geometry, pos, dir, inter, best_geo_index))
                {
                    const auto mtl_index = geometry[best_geo_index].mtl;
                    const auto normal = interpolate_normal(dir, inter.uv, geometry[best_geo_index].geo.normals);
                    const auto mtl = mtl_bank[mtl_index];
                    estimator += gpu_preview_color(mtl) * -dot(dir, normal);
                }
                else
                {
                    estimator += float3{0.f, 0.f, 1.f};
                }
            }

            image[pixel_index] += estimator;
            rand_pool[pixel_index] = rand_state;
        }
    }

    preview_renderer::preview_renderer(
        const bvh_node *device_tree,
        const face *device_model,
        const material *device_mtl_bank,
        std::size_t thread_per_block)
        : _device_tree{device_tree},
          _device_model{device_model},
          _device_mtl_bank{device_mtl_bank},
          _thread_per_block{thread_per_block}
    {
    }

    void preview_renderer::call_integrate_kernel(
        const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor)
    {
        unsigned int thread_per_block = _thread_per_block;
        const auto grid_dim = image_grid_dim(
            cam.get_image_width(), cam.get_image_height(), thread_per_block);

        preview_integrate_kernel<<<grid_dim, thread_per_block>>>(
            _device_tree, _device_model, _device_mtl_bank, cam, rand_pool, sample_count, sensor);
    }
}