

#include "camera_configuration.cuh"

namespace Xrender
{

    static unsigned int make_even(unsigned int x)
    {
        return (x % 2 == 0) ? x : (x + 1);
    }

    void configure_camera(const camera_configuration& config, camera &cam)
    {
        const auto image_width = make_even(config.image_width);
        const auto image_height = make_even(config.image_height);

        cam._diaphragm_radius = config.diaphragm_radius;
        cam._focal_length = config.focal_length;
        cam._sensor_lens_distance =
            (config.focal_length * config.focus_distance) /
            (config.focus_distance - config.focal_length);
        cam._pixel_size = config.sensor_width / static_cast<float>(image_width);
        cam._image_pixel_half_width = image_width / 2;
        cam._image_pixel_half_height = image_height / 2;
    }
}