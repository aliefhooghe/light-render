
#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/curand_pool.cuh"
#include "gpu/utils/image_grid_dim.cuh"

#include "gpu_renderer.cuh"

namespace Xrender
{
    __host__ gpu_renderer::gpu_renderer(camera &cam)
    :   _camera{cam}
    {
        const auto width = _camera.get_image_width();
        const auto height = _camera.get_image_height();

        // Init device sensor
        CUDA_CHECK(cudaMalloc(&_device_sensor, width * height * sizeof(float3)));
        reset();

        // Init random generator pool
        _rand_pool = create_curand_pool(width * height);

        set_thread_per_block(256);
    }

    __host__ gpu_renderer::~gpu_renderer() noexcept
    {
        CUDA_WARNING(cudaFree(_rand_pool));
        CUDA_WARNING(cudaFree(_device_sensor));
    }

    __host__ void gpu_renderer::set_thread_per_block(unsigned int count) noexcept
    {
        _thread_per_block = count;
        _grid_dim = image_grid_dim(
            _camera.get_image_width(),
            _camera.get_image_height(),
            _thread_per_block);
    }

    __host__ void gpu_renderer::reset()
    {
        const auto width = _camera.get_image_width();
        const auto height = _camera.get_image_height();
        CUDA_CHECK(cudaMemset(_device_sensor, 0u, width * height * sizeof(float3)));
        _total_sample_count = 0u;
    }

    __host__ void gpu_renderer::integrate(std::size_t sample_count)
    {
        _call_integrate_kernel(sample_count, _rand_pool, _device_sensor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        _total_sample_count += sample_count;
    }

    __host__ void gpu_renderer::develop_to_texture(gpu_texture &texture)
    {
        auto mapped_surface = texture.get_mapped_surface();
        _call_develop_to_texture_kernel(_device_sensor, mapped_surface.surface());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // __host__ std::vector<rgb24> gpu_renderer::develop()
    // {
    //     return {};
    // }

}