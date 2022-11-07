
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

        // Init compute events
        CUDA_CHECK(cudaEventCreate(&_compute_start_event));
        CUDA_CHECK(cudaEventCreate(&_compute_end_event));
    }

    renderer_manager::renderer_manager(renderer_manager&& other) noexcept
    :   _renderer{std::move(other._renderer)},
        _status{other._status},
        _camera{other._camera},
        _ready{other._ready},
        _compute_start_event{other._compute_start_event},
        _compute_end_event{other._compute_end_event},
        _rand_pool{other._rand_pool},
        _device_sensor{other._device_sensor}
    {
        // Avoir multiple desalocation
        other._rand_pool = nullptr;
        other._device_sensor = nullptr;
        other._compute_start_event = nullptr;
        other._compute_end_event = nullptr;
    }

    renderer_manager::~renderer_manager()
    {
        if (_rand_pool) CUDA_WARNING(cudaFree(_rand_pool));
        if (_device_sensor) CUDA_WARNING(cudaFree(_device_sensor));
        if (_compute_start_event) CUDA_WARNING(cudaEventDestroy(_compute_start_event));
        if (_compute_end_event) CUDA_WARNING(cudaEventDestroy(_compute_end_event));
    }

    void renderer_manager::reset()
    {
        _dirty = true;
    }

    void renderer_manager::start_integrate_for(const std::chrono::milliseconds& max_duration)
    {
        if (!_ready)
        {
            std::cout << "Warning: render manager: tried to start integration whereas the manager is not ready" << std::endl;
        }
        _ready = false;

        // Clean the sensors if needed
        if (_dirty)
        {
            const auto width = _camera.get_image_width();
            const auto height = _camera.get_image_height();
            CUDA_CHECK(cudaMemset(_device_sensor, 0u, width * height * sizeof(float3)));
            _status.total_integrated_sample = 0u;
            _dirty = false;
        }

        // estimate sample count to run according to current speed
        _status.frame_sample_count = std::max<std::size_t>(1u, static_cast<std::size_t>(
            _status.spp_per_second * static_cast<float>(max_duration.count()) / 1000.f));

        // start the kernel with the given sample count
        CUDA_CHECK(cudaEventRecord(_compute_start_event));
        _renderer->call_integrate_kernel(_camera, _rand_pool, _status.frame_sample_count, _device_sensor);
        CUDA_CHECK(cudaEventRecord(_compute_end_event));
    }

    bool renderer_manager::is_ready()
    {
        if (!_ready)
        {
            const cudaError_t status = cudaEventQuery(_compute_end_event);
            if (status == cudaSuccess)
            {
                float compute_duration;
                CUDA_CHECK(cudaEventElapsedTime(&compute_duration, _compute_start_event, _compute_end_event));
                _status.spp_per_second = 1000.f * (static_cast<float>(_status.frame_sample_count) / compute_duration);
                _status.total_integrated_sample += _status.frame_sample_count;
                _ready = true;
            }
            else if (status != cudaErrorNotReady)
            {
                throw cuda_exception("renderer_manager::is_ready: event polling error", status);
            }
        }
        return _ready;
    }
}