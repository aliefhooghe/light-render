
#include <iostream>

#include "host/gui/renderer_application.h"

void usage(const char *argv0)
{
    std::cout << "usage : " << argv0 << " <render.conf>" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        usage(argv[0]);
        return 1;
    }

    const auto config = Xrender::load_render_configuration(argv[1]);
    Xrender::renderer_application app{config};

    app.execute();

    std::cout << "\nQuiting..." << std::endl;
    return 0;
}