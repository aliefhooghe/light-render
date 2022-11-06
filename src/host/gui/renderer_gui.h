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
        void _draw_status_panel();

        renderer_frontend& _frontend;
        std::array<float, speed_buffer_size> _speed_values{};
        std::size_t _speed_offset{0};
    };

}

#endif
