
#include "gpu/model/float3_operators.cuh"
#include "gpu/utils/image_grid_dim.cuh"

#include "average_image_developer.cuh"

namespace Xrender
{
    __global__ void averrage_develop_to_surface_kernel(
        const float3 *sensor, cudaSurfaceObject_t surface, float factor, const int width)
    {
        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;

        if (x < width) {
            const int pixel_index = x + y * width;

            const auto rgb_value = sensor[pixel_index] * factor;
            const float4 rgba_value = {
                fmin(rgb_value.x, 1.f),
                fmin(rgb_value.y, 1.f),
                fmin(rgb_value.z, 1.f),
                1.f
            };

            surf2Dwrite(rgba_value, surface, x * sizeof(float4), y);
        }
    }

    average_image_developer::average_image_developer(float factor, std::size_t thread_per_block)
    : _factor{factor},
      _thread_per_block{thread_per_block}
    {
    }

    void average_image_developer::scale_factor(bool up)
    {
        constexpr auto factor = 1.02;
        if (up)
            _factor *= factor;
        else
            _factor /= factor;
    }

    void average_image_developer::call_develop_to_texture_kernel(
        std::size_t total_sample_count,
        const unsigned int sensor_width, const unsigned int sensor_height,
        const float3 *sensor, cudaSurfaceObject_t texture)
    {
        unsigned int thread_per_block = _thread_per_block;
        const auto grid_dim = image_grid_dim(
            sensor_width, sensor_height, thread_per_block);
        const auto factor = _factor / static_cast<float>(total_sample_count);

        averrage_develop_to_surface_kernel<<<grid_dim, thread_per_block>>>(
            sensor, texture, factor, sensor_width);
    }
}