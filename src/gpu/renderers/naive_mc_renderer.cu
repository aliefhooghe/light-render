
#include <algorithm>
#include <chrono>
#include <iostream>

#include "gpu/model/bvh_tree_traversal.cuh"
#include "gpu/model/material_brdf.cuh"
#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/curand_helper.cuh"
#include "gpu/utils/image_grid_dim.cuh"

#include "naive_mc_renderer.cuh"

namespace Xrender {

    __device__ float russian_roulette_prob(int bounce, const float3& bounce_coeff)
    {
        if (bounce < 3)
        {
            return 1.f;
        }
        else
        {
            constexpr auto refl_factor = 1.f;
            // constexpr auto bounce_factor = 1.f/30.f;
            const auto refl = fmaxf(bounce_coeff.x, fmaxf(bounce_coeff.y, bounce_coeff.z));
            return fminf(refl_factor * refl, 1.f);// * expf(-bounce*bounce_factor);
        }
    }

    __global__ void path_sampler_kernel(
        const bvh_node *bvh,
        const camera cam,
        const int sample_count,
        curandState_t *rand_pool,
        float3 *sensor)
    {
        const auto width = cam.get_image_width();

        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;
        const int pixel_index = x + y * width;

        if (x < width)
        {
            int sample_counter = 0;
            intersection inter;
            auto rand_state = rand_pool[pixel_index];
            float3 estimator = {0.f, 0.f, 0.f};
            float3 pos;
            float3 dir;
            float3 brdf_coeff;
            float russian_roulette_factor = 1.f;

            //  Initialize first ray
            int bounce = 0;
            dir = cam.sample_ray(&rand_state, pos, x, y);
            russian_roulette_factor = 1.f;
            brdf_coeff = {1.f, 1.f, 1.f};

            while (sample_counter < sample_count)
            {
                //  cast a ray
                if (intersect_ray_bvh(bvh, pos, dir, inter))
                {
                    float3 next_dir;
                    const float3 bounce_coeff = sample_brdf(&rand_state, inter, inter.normal, dir, next_dir);
                    brdf_coeff *= bounce_coeff;

                    if (gpu_mtl_is_source(inter.mtl))
                    {
                        // record ray contribution
                        estimator += (russian_roulette_factor * brdf_coeff);
                    }
                    else
                    {
                        // Russion roulette : does current ray worht the cost ?
                        const float roulette_prob = russian_roulette_prob(bounce, bounce_coeff);

                        if (curand_uniform(&rand_state) < roulette_prob)
                        {
                            // continue the ray
                            russian_roulette_factor /= roulette_prob;
                            pos = inter.pos;
                            dir = next_dir;
                            bounce++;
                            continue;
                        }
                    }
                }

                // start a new ray
                sample_counter++;
                bounce=0;
                dir = cam.sample_ray(&rand_state, pos, x, y);
                russian_roulette_factor = 1.f;
                brdf_coeff = {1.f, 1.f, 1.f};
            }

            sensor[pixel_index] += estimator;
            rand_pool[pixel_index] = rand_state;
        }
    }

    naive_mc_renderer::naive_mc_renderer(const bvh_node *device_tree, std::size_t thread_per_block)
    :   _device_tree{device_tree},
        _thread_per_block{thread_per_block}
    {
    }

    void naive_mc_renderer::call_integrate_kernel(
        const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor)
    {
        unsigned int thread_per_block = _thread_per_block;
        const auto grid_dim = image_grid_dim(
            cam.get_image_width(), cam.get_image_height(), thread_per_block);

        path_sampler_kernel<<<grid_dim, thread_per_block>>>(
            _device_tree, cam, sample_count, rand_pool, sensor);
    }
}