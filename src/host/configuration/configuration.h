#ifndef XRENDER_CONFIGURATION_H_
#define XRENDER_CONFIGURATION_H_

#include <filesystem>

namespace Xrender
{
    struct camera_configuration
    {
        float focal_length{100E-3};
        float focus_distance{8};
        float diaphragm_radius{1E-3};
        float sensor_width{36E-3};
        unsigned int image_width{360};
        unsigned int image_height{240};
    };

    struct render_configuration
    {
        std::filesystem::path model_path{"model.obj"};
        camera_configuration camera_config{};
    };

    render_configuration load_render_configuration(const std::filesystem::path&);
}

#endif
