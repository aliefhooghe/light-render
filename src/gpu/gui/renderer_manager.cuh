#ifndef XRENDER_RENDERER_MANAGER_H_
#define XRENDER_RENDERER_MANAGER_H_

#include <chrono>
#include <memory>
#include <vector>

#include "abstract_renderer.cuh"
#include "abstract_image_developer.cuh"
#include "gpu/model/camera.cuh"
#include "gpu/gui/gpu_texture.cuh"

namespace Xrender {

    class renderer_manager
    {

    public:
        renderer_manager(
            const camera &cam,
            std::unique_ptr<abstract_renderer>&& renderer,
            std::unique_ptr<abstract_image_developer>&& developer);
        renderer_manager(const renderer_manager&) = delete;
        renderer_manager(renderer_manager&&) noexcept;
        ~renderer_manager() noexcept;

        void reset();
        void integrate();

        void develop_to_texture(gpu_texture& texture);
        std::vector<float4> develop_to_host();

        void set_interval(std::chrono::milliseconds interval);

    private:
        void _render_integrate_step();

        std::unique_ptr<abstract_renderer> _renderer{nullptr};
        std::unique_ptr<abstract_image_developer> _developer{nullptr};

        std::size_t _sample_per_step{10u};
        std::chrono::milliseconds _interval{100}; // 10 fps
        std::size_t _total_sample_count{0u};
        const camera& _camera;

        // gpu resources
        curandState *_rand_pool{nullptr};
        float3 *_device_sensor{nullptr};
    };
}

#endif