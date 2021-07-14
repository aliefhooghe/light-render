

#include "camera_configuration.cuh"

namespace Xrender
{

    static unsigned int make_even(unsigned int x)
    {
        return (x % 2 == 0) ? x : (x + 1);
    }

    static float sensor_lens_dist(float focal_length, float focus_distance)
    {
        return (focal_length * focus_distance) / (focus_distance - focal_length);
    }

    static float current_focus_distance(const camera &cam)
    {
        return
            (cam._focal_length * cam._sensor_lens_distance) /
            (cam._focal_length - cam._sensor_lens_distance);
    }


    void configure_camera(const camera_configuration& config, camera &cam)
    {
        const auto image_width = make_even(config.image_width);
        const auto image_height = make_even(config.image_height);

        cam._diaphragm_radius = config.diaphragm_radius;
        cam._focal_length = config.focal_length;
        cam._sensor_lens_distance = sensor_lens_dist(config.focal_length, config.focus_distance);
        cam._pixel_size = config.sensor_width / static_cast<float>(image_width);
        cam._image_pixel_half_width = image_width / 2;
        cam._image_pixel_half_height = image_height / 2;
    }

    void camera_update_focal_length(camera &cam, float focal_length)
    {
        const auto focus_distance = current_focus_distance(cam);
        cam._focal_length = focal_length;
        cam._sensor_lens_distance = sensor_lens_dist(focal_length, focus_distance);
    }
}