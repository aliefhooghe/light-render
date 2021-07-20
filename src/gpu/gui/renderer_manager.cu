
#include <stdexcept>
#include <iostream>
#include <cmath>

#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/curand_pool.cuh"

#include "renderer_manager.cuh"

namespace Xrender {

    renderer_manager::renderer_manager(
        const camera& cam,
        std::unique_ptr<abstract_renderer>&& renderer,
        std::unique_ptr<abstract_image_developer>&& developer)
    :   _renderer{std::move(renderer)},
        _developer{std::move(developer)},
        _camera{cam}
    {
        if (!_renderer || !_developer)
            throw std::invalid_argument("Renderer manager : dependency is null");

        const auto width = _camera.get_image_width();
        const auto height = _camera.get_image_height();

        // Init device sensor
        CUDA_CHECK(cudaMalloc(&_device_sensor, width * height * sizeof(float3)));
        reset();

        // Init random generator pool
        _rand_pool = create_curand_pool(width * height);
    }

    renderer_manager::renderer_manager(renderer_manager&& other) noexcept
    :   _renderer{std::move(other._renderer)},
        _developer{std::move(other._developer)},
        _sample_per_step{other._sample_per_step},
        _interval{other._interval},
        _total_sample_count{other._total_sample_count},
        _camera{other._camera},
        _rand_pool{other._rand_pool},
        _device_sensor{other._device_sensor}
    {
        // Avoir multiple desalocation
        other._rand_pool = nullptr;
        other._device_sensor = nullptr;
    }

    renderer_manager::~renderer_manager()
    {
        if (_rand_pool) CUDA_WARNING(cudaFree(_rand_pool));
        if (_device_sensor) CUDA_WARNING(cudaFree(_device_sensor));
    }

    void renderer_manager::reset()
    {
        const auto width = _camera.get_image_width();
        const auto height = _camera.get_image_height();
        CUDA_CHECK(cudaMemset(_device_sensor, 0u, width * height * sizeof(float3)));
        _total_sample_count = 0u;
    }

    void renderer_manager::integrate()
    {
        using namespace std::chrono;
        auto start_time = steady_clock::now();
        _render_integrate_step();
        auto end_time = steady_clock::now();
        auto duration = end_time - start_time;

        // Update sample per step to follow the interval requirement

        const auto recorded_speed = static_cast<float>(_sample_per_step) / static_cast<float>(duration.count());
        const auto interval = duration_cast<decltype(duration)>(_interval);

        std::cout
            << "\rIntegrated " << _sample_per_step << " more samples; "
            << "recorded_speed : " << (100.f * static_cast<float>(interval.count())/static_cast<float>(duration.count()))
            << " % (" << (recorded_speed * 1E9) << " spp/sec; total = " << _total_sample_count << " spp )" << std::flush;

        _sample_per_step = std::max<std::size_t>(1u, std::ceil(recorded_speed * interval.count()));
    }

    void renderer_manager::develop_to_texture(gpu_texture &texture)
    {
        auto mapped_surface = texture.get_mapped_surface();
        _developer->call_develop_to_texture_kernel(
            _total_sample_count,
            _camera.get_image_width(), _camera.get_image_height(),
            _device_sensor, mapped_surface.surface());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float4> develop_to_host()
    {
        throw;
    }

    void renderer_manager::set_interval(std::chrono::milliseconds interval)
    {
        _interval = interval;
        _sample_per_step = 1u; // Let recalibrate
    }

    void renderer_manager::_render_integrate_step()
    {
        _renderer->call_integrate_kernel(_camera, _rand_pool, _sample_per_step, _device_sensor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        _total_sample_count += _sample_per_step;
    }
}