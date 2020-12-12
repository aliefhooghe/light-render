
#include "camera.h"
#include "random_generator.h"

namespace Xrender {


    camera camera::from_sensor_lens_distance(
        unsigned int image_pixel_width,
        unsigned int image_pixel_height,
        float sensor_real_width,
        float sensor_real_height,
        float diaphragm_radius,
        float focal_length,
        float sensor_lens_distance)
    {
        return camera{
            image_pixel_width, image_pixel_height,
            sensor_real_width, sensor_real_height,
            diaphragm_radius, focal_length, sensor_lens_distance
        };
    }

    camera camera::from_focus_distance(
        unsigned int image_pixel_width,
        unsigned int image_pixel_height,
        float sensor_real_width,
        float sensor_real_height,
        float diaphragm_radius,
        float focal_length,
        float focus_distance)
    {
        const auto sensor_lens_distance = (focal_length * focus_distance) / (focus_distance - focal_length);
        return camera{
            image_pixel_width, image_pixel_height,
            sensor_real_width, sensor_real_height,
            diaphragm_radius, focal_length, sensor_lens_distance
        };
    }

    camera::camera(
            unsigned int image_pixel_width,
            unsigned int image_pixel_height,
            float sensor_real_width,
            float sensor_real_height,
            float diaphragm_radius,
            float focal_length,
            float sensor_lens_distance)
    :   _image_pixel_half_width{image_pixel_width / 2u},
        _image_pixel_half_height{image_pixel_height / 2u},
        _pixel_width{sensor_real_width / static_cast<float>(image_pixel_width)}, 
        _pixel_height{sensor_real_height / static_cast<float>(image_pixel_height)}, 
        _diaphragm_radius{diaphragm_radius}, _focal_length{focal_length},
        _sensor_lens_distance{sensor_lens_distance}
    {
    }

    vecf camera::_sample_lens_point() const noexcept
    {
        
        vecf len_pos = {0.f, 0.f, 0.f};
        rand::unit_disc_uniform(len_pos.x, len_pos.z);
        return _diaphragm_radius * len_pos;
    }

    void camera::sample_ray(unsigned int px, unsigned int py, vecf& pos, vecf& dir) const noexcept
    {
        const float signed_px = (static_cast<int>(px) - static_cast<int>(_image_pixel_half_width)) + rand::uniform();
        const float signed_py = (static_cast<int>(py) - static_cast<int>(_image_pixel_half_height)) + rand::uniform();

        // Compute the pixel origin position on the sensor
        const vecf pixel_origin = 
        {
            -signed_px * _pixel_width,
            -_sensor_lens_distance,
            -signed_py * _pixel_height
        }; 

        // Sample an uniform sample on the lens disc
        pos = _sample_lens_point();

        // // Compute the outgoing direction (from camera to scenes)
        dir = (pos * (_focal_length / _sensor_lens_distance - 1.f) - 
                (_focal_length / _sensor_lens_distance) * pixel_origin
            ).normalized();
    }

} 