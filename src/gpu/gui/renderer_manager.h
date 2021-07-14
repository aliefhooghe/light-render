#ifndef XRENDER_RENDERER_MANAGER_H_
#define XRENDER_RENDERER_MANAGER_H_

#include <chrono>
#include <memory>

#include "abstract_renderer.h"

namespace Xrender {

    class renderer_manager
    {

    public:
        renderer_manager(const renderer_manager&) = delete;
        renderer_manager(renderer_manager&&) noexcept = default;
        renderer_manager(std::unique_ptr<abstract_renderer>&& renderer);
        ~renderer_manager() noexcept = default;

        void reset();
        void integrate();
        void develop_to_texture(gpu_texture& texture);
        void set_interval(std::chrono::milliseconds interval);

    private:
        std::unique_ptr<abstract_renderer> _renderer{nullptr};
        std::size_t _sample_per_step{10u};
        std::chrono::milliseconds _interval{100}; // 10 fps
    };
}

#endif