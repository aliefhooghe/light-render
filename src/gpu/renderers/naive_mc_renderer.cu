
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

    __global__ void render_develop_to_surface_kernel(
        const float3 *sensor, cudaSurfaceObject_t surface, float factor, const int width)
    {
        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;

        if (x < width) {
            const int pixel_index = x + y * width;

            const auto rgb_value = sensor[pixel_index] * factor;
            const float4 rgba_value = {
                rgb_value.x,
                rgb_value.y,
                rgb_value.z,
                1.f
            };

            surf2Dwrite(rgba_value, surface, x * sizeof(float4), y);
        }
    }

    __device__ float russian_roulette_prob(const float3& brdf_coeff)
    {
        constexpr auto threshold = 10.f / 255.f;
        constexpr auto min_prob = 0.2f;
        constexpr auto max_prob = 2.f;
        constexpr auto a = (max_prob - min_prob) / (1.f - threshold);
        constexpr auto b = max_prob - a;

        const auto estimator_norm = norm(brdf_coeff);
        const float prob = a * estimator_norm + b;
        return prob > 1.f ? 1.f : prob;
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
            dir = cam.sample_ray(&rand_state, pos, x, y);
            russian_roulette_factor = 1.f;
            brdf_coeff = {1.f, 1.f, 1.f};

            while (sample_counter < sample_count)
            {
                // Russion roulette : does current ray worht the cost ?
                const float roulette_prob = russian_roulette_prob(brdf_coeff);
                if (curand_uniform(&rand_state) <= roulette_prob)
                {
                    // keep the ray
                    russian_roulette_factor /= roulette_prob;

                    //  cast a ray
                    if (intersect_ray_bvh(bvh, pos, dir, inter))
                    {
                        float3 next_dir;
                        brdf_coeff *= sample_brdf(&rand_state, inter.mtl, inter.normal, dir, next_dir);

                        if (gpu_mtl_is_source(inter.mtl))
                        {
                            // record ray contribution
                            estimator += (russian_roulette_factor * brdf_coeff);
                        }
                        else
                        {
                            // geo_coeff *= fabs(dot(next_dir, inter.normal));
                            pos = inter.pos;
                            dir = next_dir;
                            continue;
                        }
                    }
                }

                // start a new ray
                sample_counter++;
                dir = cam.sample_ray(&rand_state, pos, x, y);
                russian_roulette_factor = 1.f;
                brdf_coeff = {1.f, 1.f, 1.f};
            }

            sensor[pixel_index] += estimator;
            rand_pool[pixel_index] = rand_state;
        }
    }

    naive_mc_renderer::naive_mc_renderer(const bvh_node *device_tree, camera& cam)
    :   gpu_renderer{cam},
        _device_tree{device_tree}
    {
    }

    __host__ void naive_mc_renderer::_call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor)
    {
        path_sampler_kernel<<<_image_grid_dim(), _image_thread_per_block()>>>(
            _device_tree, _camera, sample_count, rand_pool, sensor);
    }

    __host__ void naive_mc_renderer::_call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture)
    {
        const auto factor = 1.f / get_total_sample_count();
        render_develop_to_surface_kernel<<<_image_grid_dim(), _image_thread_per_block()>>>(
            sensor, texture, factor, _camera.get_image_width());
    }
}