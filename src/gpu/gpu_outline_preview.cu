
#include <algorithm>
#include <chrono>
#include <iostream>

#include "gpu_outline_preview.cuh"
#include "vector_operations.cuh"
#include "cuda_exception.cuh"
#include "rand_operations.cuh"
#include "utils/image_grid_dim.cuh"

namespace Xrender {

    __global__ void preview_develop_to_surface_kernel(
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

    __global__ void preview_integrate_kernel(
        const gpu_bvh_node *tree,
        const device_camera cam,
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
        if (x < width) {
            auto rand_state = rand_pool[pixel_index];

            float3 pos;
            float3 dir;
            gpu_intersection inter;
            float3 estimator = {0.f, 0.f, 0.f};

            for (auto i = 0; i < sample_count; i++) {
                dir = cam.sample_ray(&rand_state, pos, x, y);
                if (gpu_intersect_ray_bvh(tree, pos, dir, inter))
                    estimator += gpu_preview_color(inter.mtl) *
                                    fabs(_dot(dir, inter.normal));
                else
                    estimator += float3{0.f, 0.f, 1.f};
            }

            image[pixel_index] += estimator;
            rand_pool[pixel_index] = rand_state;
        }
    }

    __device__ __host__ rgb24 _color_of_float3(const float3& color)
    {
        return {
            static_cast<unsigned char>(color.z * 255.f),
            static_cast<unsigned char>(color.y * 255.f),
            static_cast<unsigned char>(color.x * 255.f)};
    }

    //

    gpu_outline_preview::gpu_outline_preview(
            const gpu_bvh_node *device_tree,
            device_camera& cam)
    :   gpu_renderer{cam},
        _device_tree{device_tree}
    {
    }

    __host__ void gpu_outline_preview::_call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor)
    {
        preview_integrate_kernel<<<_image_grid_dim(), _image_thread_per_block()>>>(
            _device_tree, _camera, rand_pool, sample_count, sensor);
    }

    __host__ void gpu_outline_preview::_call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture)
    {
        preview_develop_to_surface_kernel<<<_image_grid_dim(), _image_thread_per_block()>>>(
            sensor, texture, _develop_factor(), _camera.get_image_width());
    }

    // __host__ std::vector<rgb24> gpu_outline_preview::develop()
    // {
    //     // Allocate host image
    //     const auto width = _camera.get_image_width();
    //     const auto height = _camera.get_image_height();
    //     std::vector<float3> host_sensor{width * height};
    //     std::vector<rgb24> output{width * height};

    //     //  Copy result
    //     CUDA_CHECK(cudaMemcpy(host_sensor.data(), _device_sensor, output.size() * sizeof(float3), cudaMemcpyDeviceToHost));

    //     const auto factor = _develop_factor();
    //     std::transform(
    //         host_sensor.begin(), host_sensor.end(),
    //         output.begin(),
    //         [factor](const float3& val)
    //         {
    //             return _color_of_float3(factor * val);
    //         });

    //     return output;
    // }

    float gpu_outline_preview::_develop_factor()
    {
        return 1.f / static_cast<float>(_sensor_total_sample_count());
    }
}