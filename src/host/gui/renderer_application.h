#ifndef XRENDER_APPLICATION_H_
#define XRENDER_APPLICATION_H_

#include <memory>
#include <vector>
#include <SDL2/SDL.h>

#include "host/renderer_frontend/renderer_frontend.h"
#include "renderer_gui.h"

namespace Xrender {

    class renderer_application {
    public:
        renderer_application(const renderer_application&) = delete;
        renderer_application(renderer_application&&) noexcept = delete; // todo
         ~renderer_application() noexcept;

        explicit renderer_application(const render_configuration& config);
        void execute();

    private:
        void _next_camera_setting();
        void _handle_camera_mouse_wheel(bool up);
        void _handle_camera_mouse_motion(int xrel, int yrel);
        void _handle_key_down(SDL_Keysym key);
        bool _handle_events();
        void _draw();
        void _update_size();

        void _switch_mouse_mode();
        void _save_current_image();

        SDL_Window *_window{nullptr};
        SDL_GLContext _gl_context{nullptr};
        GLuint _texture{0u};
        std::unique_ptr<renderer_frontend> _renderer;
        std::unique_ptr<renderer_gui> _gui;

        enum class mouse_mode
        {
            GUI,
            CAMERA
        };

        mouse_mode _mouse_mode{mouse_mode::GUI};
        renderer_frontend::lens_setting _camera_mouse_wheel_setting{renderer_frontend::lens_setting::FOCAL_LENGTH};
        float _camera_theta{0.f};
        float _camera_phi{0.f};
    };
}

#endif