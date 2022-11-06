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
        void _next_setting();
        void _handle_key_down(SDL_Keysym key);
        void _handle_mouse_wheel(bool up);
        void _handle_mouse_motion(int xrel, int yrel);
        bool _handle_events();
        void _draw();
        void _update_size();

        void _switch_fast_mode();
        void _switch_rotation();
        void _save_current_image();

        SDL_Window *_window{nullptr};
        SDL_GLContext _gl_context{nullptr};
        GLuint _texture{0u};
        std::unique_ptr<renderer_frontend> _renderer;
        std::unique_ptr<renderer_gui> _gui;
        bool _fast_mode{false};

        // control mode

        enum class camera_setting
        {
            SENSOR_LENS_DISTANCE,
            FOCAL_LENGTH,
            DIAPHRAGM_RADIUS
        };

        camera_setting _camera_setting{camera_setting::SENSOR_LENS_DISTANCE};
        std::size_t _control_setting_id{0u};
        bool _freeze_camera_rotation{true};

        float _camera_theta{0.f};
        float _camera_phi{0.f};
    };
}

#endif