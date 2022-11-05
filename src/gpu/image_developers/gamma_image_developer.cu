
#include "gpu/model/float3_operators.cuh"
#include "gpu/utils/image_grid_dim.cuh"

#include "gamma_image_developer.cuh"

namespace Xrender
{
    static __device__ inline float3 rgb2yuv(const float3& c)
    {
        return make_float3(
            0.2126f   * c.x + 0.7152f  * c.y + 0.0722f  * c.z,
            -0.09991f * c.x - 0.33609f * c.y + 0.436f   * c.z,
            0.615f    * c.x - 0.55861f * c.y - 0.05639f * c.z
        );
    }

    static __device__ inline float3 yuv2rgb(const float3& c)
    {
        return make_float3(
            c.x                  + 1.28033f * c.z,
            c.x - 0.21482f * c.y - 0.38059f * c.z,
            c.x + 2.12798f * c.y);
    }

    static __device__ inline float3 gamma_correction(const float3& rgb, float gamma)
    {
        auto yuv = rgb2yuv(rgb);
        yuv.x = powf(yuv.x, gamma);
        return yuv2rgb(yuv);
    }

    __global__ void gamma_develop_to_surface_kernel(
        const float3 *sensor, cudaSurfaceObject_t surface, float factor, float gamma, const int width)
    {
        //  Get pixel position in image
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;

        if (x < width) {
            const int pixel_index = x + y * width;
            const auto rgb_val = gamma_correction(factor * sensor[pixel_index], gamma);
            const auto rgba_value =  make_float4(rgb_val.x, rgb_val.y, rgb_val.z, 1.f);
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