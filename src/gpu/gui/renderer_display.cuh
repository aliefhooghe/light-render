#ifndef XRENDER_DISPLAY_H_
#define XRENDER_DISPLAY_H_

#include <memory>
#include <vector>
#include <SDL2/SDL.h>

#include "renderer_manager.cuh"

#include "gpu/model/camera.cuh"
#include "gpu/gui/gpu_texture.cuh"

namespace Xrender {

    class renderer_display {
    public:
        __host__ renderer_display(camera& camera);
        renderer_display(const renderer_display&) = delete;
        renderer_display(renderer_display&&) noexcept = delete;
        __host__ ~renderer_display() noexcept;

        __host__ void execute();
        __host__ void add_view(
            std::unique_ptr<abstract_renderer>&& renderer,
            std::unique_ptr<abstract_image_developer>&& developer);

        static constexpr auto fast_interval = 10000u; // 10 s => 0.1 fps
        static constexpr auto interactive_interval = 60u; // 40 ms => 50 fps

    private:
        __host__ void _update_parameter(float& param, bool up, float factor = 1.01f);
        __host__ void _next_renderer(bool previous = false);
        __host__ void _reset_current_renderer();
        __host__ void _handle_key_down(SDL_Keysym key);
        __host__ void _handle_mouse_wheel(bool up);
        __host__ bool _handle_events();
        __host__ void _draw();
        __host__ void _update_size();
        __host__ void _switch_fast_mode();
        __host__ void _set_interval(std::chrono::milliseconds interval);

        camera& _camera;
        std::unique_ptr<gpu_texture> _texture{nullptr};
        SDL_Window *_window{nullptr};
        SDL_GLContext _gl_context{nullptr};

        bool _fast_mode{false};
        std::chrono::milliseconds _interval{interactive_interval};
        int _current_renderer{0};
        std::vector<renderer_manager> _renderers{};
    };
}

#endif