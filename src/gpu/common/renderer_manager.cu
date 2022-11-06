
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/curand_pool.cuh"

#include "renderer_manager.cuh"

namespace Xrender {

    renderer_manager::renderer_manager(
        const camera& cam,
        std::unique_ptr<abstract_renderer>&& renderer)
    :   _renderer{std::move(renderer)},
        _camera{cam}
    {
        if (!_renderer)
            throw std::invalid_argument("Renderer manager : renderer is null");

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
        _status{other._status},
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
        _status.total_integrated_sample = 0u;
    }

    void renderer_manager::integrate_for(const std::chrono::milliseconds& max_duration)
    {
        using namespace std::chrono;

        // estimate sample count to run
        _status.last_sample_count = std::max<std::size_t>(1u, static_cast<std::size_t>(
            _status.spp_per_second * static_cast<float>(max_duration.count()) / 1000.f));

        auto start_time = steady_clock::now();
        _render_integrate_step(_status.last_sample_count);
        auto end_time = steady_clock::now();
        auto duration = end_time - start_time;

        // Update sample per step to follow the interval requirement

        const auto recorded_speed = static_cast<float>(_status.last_sample_count ) / static_cast<float>(duration.count());
        const auto recorded_speed_spp_per_sec = recorded_speed * 1E9f;

        std::cout
            << "\r[ Integrated " << _status.last_sample_count  << " more samples ] - [ "
            << std::fixed << recorded_speed_spp_per_sec << " spp/sec - total = " << _status.total_integrated_sample << " spp ]" << std::flush;

        _status.spp_per_second = recorded_speed_spp_per_sec;
    }

    void renderer_manager::_render_integrate_step(std::size_t sample_count)
    {
        _renderer->call_integrate_kernel(_camera, _rand_pool, sample_count, _device_sensor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        _status.total_integrated_sample += sample_count;
    }
}