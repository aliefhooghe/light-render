#ifndef XRENDER_APPLICATION_H_
#define XRENDER_APPLICATION_H_

#include <memory>
#include <vector>
#include <SDL2/SDL.h>

#include "host/renderer_frontend/renderer_frontend.h"

namespace Xrender {

    class renderer_application {
    public:
        renderer_application(const renderer_application&) = delete;
        renderer_application(renderer_application&&) noexcept = delete; // todo
         ~renderer_application() noexcept;

        explicit renderer_application(const render_configuration& config);
        void execute();

    private:
        static constexpr auto fast_interval = 10000u; // 10 s => 0.1 fps
        static constexpr auto interactive_interval = 60u; // 40 ms => 50 fps

        void _next_renderer();
        void _next_developer();
        void _handle_key_down(SDL_Keysym key);
        void _handle_mouse_wheel(bool up);
        bool _handle_events();
        void _draw();
        void _update_size();
        void _switch_fast_mode();
        void _save_current_image();

        SDL_Window *_window{nullptr};
        SDL_GLContext _gl_context{nullptr};
        GLuint _texture{0u};
        std::unique_ptr<renderer_frontend> _renderer;
        bool _fast_mode{false};
    };
}

#endif