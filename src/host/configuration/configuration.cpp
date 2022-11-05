
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "configuration.h"

namespace Xrender
{
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(camera_configuration,
        focal_length,
        focus_distance,
        diaphragm_radius,
        sensor_width,
        image_width,
        image_height)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(render_configuration,
        model_path,
        camera_config)

    render_configuration load_render_configuration(const std::filesystem::path &config_path)
    {
        render_configuration config;
        std::ifstream stream{config_path, std::ios_base::in};

        if (stream.is_open() && stream.good())
        {
            nlohmann::json json;
            stream >> json;
            from_json(json, config);

            // make sure image dimensions are even
            config.camera_config.image_width &= ~1u;
            config.camera_config.image_height &= ~1u;
        }
        else
        {
            throw std::invalid_argument("Unable to open render configuration file");
        }

        std::cout << "Loaded configuration " << config_path.generic_string() << "\n"
                  << "\tModel Path       : " << config.model_path.generic_string() << "\n"
                  << "\tFocal Length     : " << config.camera_config.focal_length << " m\n"
                  << "\tFocus Distance   : " << config.camera_config.focus_distance << " m\n"
                  << "\tDiaphragm Radius : " << config.camera_config.diaphragm_radius << " m\n"
                  << "\tSensor Width     : " << config.camera_config.sensor_width << " m\n"
                  << "\tImage Width      : " << config.camera_config.image_width << " pixels \n"
                  << "\tImage Height     : " << config.camera_config.image_height << " pixels " << std::endl;

        return config;
    }

}