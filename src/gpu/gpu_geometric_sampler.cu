
#include <chrono>
#include <curand_kernel.h>

#include <iostream>

#include "gpu_geometric_sampler.cuh"
#include "rand_operations.cuh"
#include "cuda_exception.cuh"

namespace Xrender {

    __device__ float russian_roulette_prob(float geo_coeff, const float3& brdf_coeff)
    {
        constexpr auto threshold = 10.f / 255.f;
        constexpr auto min_prob = 0.2f;
        constexpr auto max_prob = 2.f;
        constexpr auto a = (max_prob - min_prob) / (1.f - threshold);
        constexpr auto b = max_prob - a;

        const auto estimator_norm = geo_coeff * _norm(brdf_coeff);
        const float prob = a * estimator_norm + b;
        return prob > 1.f ? 1.f : prob;
    }    
    
    __global__ void path_sampler_kernel(
        const gpu_bvh_node *bvh,
        const device_camera cam,
        const int sample_count, 
        float3 *image,
        const int width)
    {
        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;
        const int pixel_index = x + y * width;

        if (x < width)
        { 
            int sample_counter = 0;
            gpu_intersection inter;
            curandState rand_state;
            float3 estimator = {0.f, 0.f, 0.f};
            float3 pos;
            float3 dir;
            float3 brdf_coeff;
            float  geo_coeff;

            //  Initialize random generator
            curand_init(pixel_index, x*pixel_index+1, pixel_index+y, &rand_state);
            
            //  Initialize first ray
            dir = cam.sample_ray(&rand_state, pos, x, y);
            geo_coeff = dir.y;
            brdf_coeff = {1.f, 1.f, 1.f};

            while (sample_counter < sample_count)
            {
                // Russion roulette : does current ray worht the cost ?
                const float roulette_prob = russian_roulette_prob(geo_coeff, brdf_coeff);
                if (curand_uniform(&rand_state) <= roulette_prob)
                {
                    // keep the ray
                    geo_coeff /= roulette_prob;
                
                    //  cast a ray
                    if (gpu_intersect_ray_bvh(bvh, pos, dir, inter))
                    {
                        float3 next_dir;
                        brdf_coeff *= gpu_brdf(&rand_state, inter.mtl, inter.normal, dir, next_dir);

                        if (gpu_mtl_is_source(inter.mtl))
                        {
                            // record ray contribution
                            estimator += (geo_coeff * brdf_coeff);
                        }
                        else
                        {
                            geo_coeff *= fabs(_dot(next_dir, inter.normal));
                            pos = inter.pos;
                            dir = next_dir;
                            continue;
                        }
                    }
                }
                    
                // start a new ray
                sample_counter++;
                dir = cam.sample_ray(&rand_state, pos, x, y);
                geo_coeff = dir.y;
                brdf_coeff = {1.f, 1.f, 1.f};
            }

            image[pixel_index] = estimator * (3.f / sample_count);
        }
    }

    std::vector<float3> gpu_naive_mc(
        const std::vector<gpu_bvh_node>& tree,
        const device_camera& cam,
        const int sample_per_pixel,
        int gpu_thread_per_block)
    {
        const auto width = cam.get_image_width();
        const auto height = cam.get_image_height();
        const auto device_image_size = width * height * sizeof(float3);
        const auto device_bvh_size = tree.size() * sizeof(gpu_bvh_node);

        float3 *device_image = nullptr;
        gpu_bvh_node *device_bvh = nullptr;

        std::cout << "@GPU RENDER " << sample_per_pixel << "SPP" << std::endl;

        //  Allocate memory on device for image and model
        CUDA_CHECK(cudaMalloc(&device_image, device_image_size));
        CUDA_CHECK(cudaMalloc(&device_bvh, device_bvh_size));

        //  Copy model to the device
        CUDA_CHECK(cudaMemcpy(device_bvh, tree.data(), device_bvh_size, cudaMemcpyHostToDevice));

        //  Do the computations
        auto thread_per_block = std::min<int>(gpu_thread_per_block, width);
        const auto horizontal_block_count = static_cast<int>(ceilf((float)width / (float)thread_per_block));
        const auto start = std::chrono::steady_clock::now();
        path_sampler_kernel<<<dim3(horizontal_block_count, height), thread_per_block>>>(
            device_bvh, cam, sample_per_pixel, device_image, width);
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for kernel completion
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto end = std::chrono::steady_clock::now();

        std::cout << "GPU render took " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms " << std::endl;

        //  Retrieve result
        std::vector<float3> result{width * height};
        CUDA_CHECK(cudaMemcpy(result.data(), device_image, device_image_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(device_image));
        CUDA_CHECK(cudaFree(device_bvh));

        return result;
    }

}