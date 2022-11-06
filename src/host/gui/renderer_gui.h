#ifndef XRENDER_GUI_H_
#define XRENDER_GUI_H_

#include "host/renderer_frontend/renderer_frontend.h"

namespace Xrender {

    class renderer_gui
    {
    public:
        renderer_gui(renderer_frontend& frontend);

        void draw();
    private:
        void _draw_worker_panel(renderer_frontend::worker_type);
        void _draw_worker_selector(renderer_frontend::worker_type);

        renderer_frontend& _frontend;
    };

}

#endif
