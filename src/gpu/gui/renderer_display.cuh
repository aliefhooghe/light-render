#ifndef XRENDER_DISPLAY_H_
#define XRENDER_DISPLAY_H_

#include <memory>
#include <vector>
#include <SDL2/SDL.h>

#include "abstract_renderer.h"
#include "renderer_manager.h"

#include "gpu/gpu_camera.cuh"
#include "gpu/gpu_texture.cuh"

namespace Xrender {

    class renderer_display {
    public:
        __host__ renderer_display(device_camera& camera);
        __host__ ~renderer_display() noexcept;

        __host__ void execute();
        __host__ void add_renderer(std::unique_ptr<abstract_renderer>&& renderer);

        template <typename Trenderer, typename ...Targs>
        inline __host__ void add_renderer(Targs&& ...args)
        {
            add_renderer(std::make_unique<Trenderer>(std::forward<Targs>(args)...));
        }

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

        device_camera& _camera;
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