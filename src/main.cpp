
#include <iostream>
#include "host/gui/renderer_application.h"
#include "host/renderer_frontend/renderer_frontend.h"

static void usage(const char *argv0)
{
    std::cout << "usage : " << argv0 << " <render.conf>" << std::endl;
}

static bool use_gui()
{
    const char* var = std::getenv("XRENDER_USE_GUI");
    return var == nullptr || (0 == strcmp(var, "1"));
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        usage(argv[0]);
        return 1;
    }

    const auto config = Xrender::load_render_configuration(argv[1]);

    if (use_gui())
    {
        std::cout << "Using gui" << std::endl;
        Xrender::renderer_application app{config};
        app.execute();
    }
    else
    {
        using worker_type = Xrender::renderer_frontend::worker_type;

        std::cout << "Using cli" << std::endl;
        throw std::runtime_error("CLI is not supported");
        // auto renderer_frontend = Xrender::renderer_frontend::build_renderer_frontend(config);

        // const auto renderer_id = 1;
        // renderer_frontend->set_current_worker(worker_type::Renderer, renderer_id);

        // const auto& desc = renderer_frontend->get_current_worker_descriptor(worker_type::Renderer);
        // std::cout << "Using renderer " << desc.name() << std::endl;

        // const auto interval = 500;
        // const auto total = 10000;
        // auto sum = 0;
        // while (sum < total)
        // {
        //     renderer_frontend->integrate_for(std::chrono::milliseconds{interval});
        //     sum += interval;
        // }
    }

    std::cout << "\nQuiting..." << std::endl;
    return 0;
}