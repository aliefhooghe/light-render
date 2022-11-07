#ifndef XRENDER_GUI_H_
#define XRENDER_GUI_H_

#include <array>
#include "host/renderer_frontend/renderer_frontend.h"

namespace Xrender {

    class renderer_gui
    {
        static constexpr auto speed_buffer_size = 64;
    public:
        renderer_gui(renderer_frontend& frontend);

        void draw();
    private:
        void _draw_worker_panel(renderer_frontend::worker_type);
        void _draw_worker_selector(renderer_frontend::worker_type);
        void _draw_camera_setting(renderer_frontend::lens_setting, float speed, float power, float vmin, float vmax, const char* label);
        void _draw_camera_panel();
        void _draw_status_panel();

        renderer_frontend& _frontend;
        std::array<float, speed_buffer_size> _speed_values{};
        std::size_t _speed_offset{0};
        float _last_speed{0.f};
        float _rendering_fps{16.f};
    };

}

#endif
