
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

    static __device__ float russian_roulette_prob(int bounce, const float3& bounce_coeff)
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
            return fminf(refl_factor * refl, 0.9f);// * expf(-bounce*bounce_factor);
        }
    }

    static __device__ inline void fill_mtl_cache(const material mtl_bank[], material mtl_cache[], int mtl_count)
    {
        static_assert(sizeof(material) % sizeof(uint32_t) == 0);

        // Optimize the global memory access pattern

        const auto cache_size = mtl_count * (sizeof(material) / sizeof(uint32_t));
        const uint32_t *bank_ref = reinterpret_cast<const uint32_t*>(mtl_bank);
        uint32_t *cache_ref = reinterpret_cast<uint32_t*>(mtl_cache);

        for (int base = 0; base < cache_size; base += blockDim.x)
        {
            const auto cache_index = base + threadIdx.x;
            if (cache_index < cache_size)
                cache_ref[cache_index] = bank_ref[cache_index];
        }
        __syncthreads();
    }

    __global__
    __launch_bounds__(naive_mc_renderer::max_thread_per_block)
    void path_sampler_kernel(
        const bvh_node *tree, int tree_size,
        const face *geometry,
        const material *mtl_bank, int mtl_count,
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

        // Retrieve all materials into cache
        extern __shared__ material mtl_cache[];
        fill_mtl_cache(mtl_bank, mtl_cache, mtl_count);

        if (x < width)
        {
            int sample_counter = 0;
            bvh_traversal_state traversal_state;
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
            bvh_traversal_init(traversal_state);

            while (sample_counter < sample_count)
            {
                auto traversal_status =
                    bvh_traversal_step(traversal_state, tree, tree_size, geometry, pos, dir);

                if (traversal_status == bvh_traversal_status::IN_PROGRESS)
                {
                    // Continue the bvh traversal
                    continue;
                }
                else
                {
                    const auto best_geometry_index = traversal_state.best_index;
                    const auto best_intersection = traversal_state.best_intersection;

                    // bvh traversal finished: prepare next traversal
                    bvh_traversal_init(traversal_state);

                    if (traversal_status == bvh_traversal_status::HIT)
                    {
                        const auto best_face = geometry[best_geometry_index];
                        const auto mtl = mtl_cache[best_face.mtl];
                        const auto edge = best_face.geo.points[1] - best_face.geo.points[0];
                        const auto normal = interpolate_normal(
                            dir, best_intersection.uv, best_face.geo.normals);

                        float3 next_dir;
                        const float3 bounce_coeff = sample_brdf(&rand_state, mtl, normal, edge, dir, next_dir);
                        brdf_coeff *= bounce_coeff;

                        if (gpu_mtl_is_source(mtl))
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
                                pos = pos + best_intersection.distance * dir;
                                dir = next_dir;
                                bounce++;
                                continue;
                            }
                        }
                    }
                }
                // else: status == NO_HIT

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

    naive_mc_renderer::naive_mc_renderer(
        const bvh_node *device_tree, int tree_size,
        const face *device_model,
        const material *device_mtl_bank, int mtl_count)
    :   _device_tree{device_tree},
        _tree_size{tree_size},
        _device_model{device_model},
        _device_mtl_bank{device_mtl_bank},
        _mtl_count{mtl_count}
    {
    }

    void naive_mc_renderer::call_integrate_kernel(
        const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor)
    {
        unsigned int thread_per_block = max_thread_per_block;
        const auto grid_dim = image_grid_dim(
            cam.get_image_width(), cam.get_image_height(), thread_per_block);
        path_sampler_kernel<<<grid_dim, thread_per_block, _mtl_count * sizeof(material)>>>(
            _device_tree, _tree_size, _device_model, _device_mtl_bank, _mtl_count, cam, sample_count, rand_pool, sensor);
    }
}