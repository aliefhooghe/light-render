
#include "gpu/model/float3_operators.cuh"
#include "gpu/utils/image_grid_dim.cuh"

#include "gamma_image_developer.cuh"

namespace Xrender
{
    __global__ void gamma_develop_to_surface_kernel(
        const float3 *sensor, cudaSurfaceObject_t surface, float factor, float gamma, const int width)
    {
        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;

        if (x < width) {
            const int pixel_index = x + y * width;

            const auto rgb_value = sensor[pixel_index] * factor;
            const float4 rgba_value = {
                powf(fmin(rgb_value.x, 1.f), gamma),
                powf(fmin(rgb_value.y, 1.f), gamma),
                powf(fmin(rgb_value.z, 1.f), gamma),
                1.f
            };

            surf2Dwrite(rgba_value, surface, x * sizeof(float4), y);
        }
    }

    gamma_image_developer::gamma_image_developer(float factor, float gamma, std::size_t thread_per_block)
    : _factor{factor},
      _gamma{gamma},
      _thread_per_block{thread_per_block}
    {
    }

    void gamma_image_developer::scale_factor(bool up)
    {
        constexpr auto factor = 1.02;
        if (up)
            _factor *= factor;
        else
            _factor /= factor;
    }

    void gamma_image_developer::scale_gamma(bool up)
    {
        constexpr auto factor = 1.02;
        if (up)
            _gamma *= factor;
        else
            _gamma /= factor;
    }

    void gamma_image_developer::call_develop_to_texture_kernel(
        std::size_t total_sample_count,
        const unsigned int sensor_width, const unsigned int sensor_height,
        const float3 *sensor, cudaSurfaceObject_t texture)
    {
        unsigned int thread_per_block = _thread_per_block;
        const auto grid_dim = image_grid_dim(
            sensor_width, sensor_height, thread_per_block);
        const auto factor = _factor / static_cast<float>(total_sample_count);

        gamma_develop_to_surface_kernel<<<grid_dim, thread_per_block>>>(
            sensor, texture, factor, _gamma, sensor_width);
    }
}