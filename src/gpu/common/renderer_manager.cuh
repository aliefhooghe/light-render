#ifndef XRENDER_RENDERER_MANAGER_H_
#define XRENDER_RENDERER_MANAGER_H_

#include <chrono>
#include <memory>
#include <vector>

#include "abstract_renderer.cuh"
#include "abstract_image_developer.cuh"
#include "gpu/model/camera.cuh"
#include "gpu/common/gpu_texture.cuh"
#include "host/renderer_frontend/rendering_status.h"

namespace Xrender {

    class renderer_manager
    {

    public:
        renderer_manager(const camera &cam, std::unique_ptr<abstract_renderer>&& renderer);
        renderer_manager(const renderer_manager&) = delete;
        renderer_manager(renderer_manager&&) noexcept;
        ~renderer_manager() noexcept;

        void reset();

        void start_integrate_for(const std::chrono::milliseconds& max_duration);
        bool is_ready();

        const rendering_status& get_status() const noexcept { return _status; }

        const float3 *get_device_sensor() const noexcept { return _device_sensor; }

    private:
        std::unique_ptr<abstract_renderer> _renderer{nullptr};

        rendering_status _status{};
        const camera& _camera;

        // Async working resources
        bool _ready{true};
        bool _dirty{false};
        cudaEvent_t _compute_start_event{nullptr};
        cudaEvent_t _compute_end_event{nullptr};

        // gpu resources
        curandState *_rand_pool{nullptr};
        float3 *_device_sensor{nullptr};
    };
}

#endif