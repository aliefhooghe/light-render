
#include <fstream>
#include <iostream>

#include "configuration.h"

namespace Xrender
{
    static void parse_line(const std::string& line, render_configuration& config)
    {
        char path[256];

        if (std::sscanf(line.c_str(), "model_path=%s\n", path) == 1)
        {
            config.model_path = path;
        }
        else if (std::sscanf(line.c_str(), "focal_length=%f\n", &config.camera_config.focal_length) == 1);
        else if (std::sscanf(line.c_str(), "focus_distance=%f\n", &config.camera_config.focus_distance) == 1);
        else if (std::sscanf(line.c_str(), "diaphragm_radius=%f\n", &config.camera_config.diaphragm_radius) == 1);
        else if (std::sscanf(line.c_str(), "sensor_width=%f\n", &config.camera_config.sensor_width) == 1);
        else if (std::sscanf(line.c_str(), "image_width=%u\n", &config.camera_config.image_width) == 1);
        else if (std::sscanf(line.c_str(), "image_height=%u\n", &config.camera_config.image_height ) == 1);
        else
        {
            // unhandled line
        }
    }

    render_configuration load_render_configuration(const std::filesystem::path &config_path)
    {
        render_configuration config;
        std::ifstream stream{config_path, std::ios_base::in};

        if (stream.is_open() && stream.good())
        {
            std::string line;
            while (stream >> line)
            {
                parse_line(line, config);
            }
        }
        else
        {
            throw std::invalid_argument("Unable to open render configuration file");
        }

        std::cout << "Loaded configuration " << config_path.generic_string() << "\n"
                  << "\tModel Path       : " << config.model_path << "\n"
                  << "\tFocal Length     : " << config.camera_config.focal_length << "\n"
                  << "\tFocus Distance   : " << config.camera_config.focus_distance << "\n"
                  << "\tDiaphragm Radius : " << config.camera_config.diaphragm_radius << "\n"
                  << "\tSensor Width     : " << config.camera_config.sensor_width << "\n"
                  << "\tImage Width      : " << config.camera_config.image_width << "\n"
                  << "\tImage Height     : " << config.camera_config.image_height << std::endl;

        return config;
    }

}