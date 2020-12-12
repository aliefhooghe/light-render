
#include "camera.h"
#include "random_generator.h"

namespace Xrender {


    camera camera::from_sensor_lens_distance(
        unsigned int sensor_pixel_width, unsigned int sensor_pixel_height,
        float sensor_real_width, float sensor_real_height,
        float diaphragm_radius, float focal_length, float sensor_lens_distance)
    {
        return camera{
            sensor_pixel_width, sensor_pixel_height,
            sensor_real_width, sensor_real_height,
            diaphragm_radius, focal_length, sensor_lens_distance
        };
    }

    camera camera::from_focus_distance(
        unsigned int sensor_pixel_width, unsigned int sensor_pixel_height,
        float sensor_real_width, float sensor_real_height,
        float diaphragm_radius, float focal_length, float focus_distance)
    {
        const auto sensor_lens_distance = (focal_length * focus_distance) / (focus_distance - focal_length);
        return camera{
            sensor_pixel_width, sensor_pixel_height,
            sensor_real_width, sensor_real_height,
            diaphragm_radius, focal_length, sensor_lens_distance
        };
    }

    camera::camera(
            unsigned int sensor_pixel_width, unsigned int sensor_pixel_height,
            float sensor_real_width,float sensor_real_height,
            float diaphragm_radius, float focal_length, float sensor_lens_distance)
    :   _sensor_pixel_half_width{sensor_pixel_width / 2u}, _sensor_pixel_half_height{sensor_pixel_height / 2u},
        _pixel_width{sensor_real_width / static_cast<float>(sensor_pixel_width)}, 
        _pixel_height{sensor_real_height / static_cast<float>(sensor_pixel_height)}, 
        _diaphragm_radius{diaphragm_radius}, _focal_length{focal_length},
        _sensor_lens_distance{sensor_lens_distance}
    {
    }

    vecf camera::_sample_lens_point()
    {
        
        vecf len_pos = {0.f, 0.f, 0.f};
        rand::unit_disc_uniform(len_pos.x, len_pos.z);
        return _diaphragm_radius * len_pos;
    }

    void camera::sample_ray(unsigned int px, unsigned int py, vecf& pos, vecf& dir)
    {
        const int signed_px = static_cast<int>(px) - static_cast<int>(_sensor_pixel_half_width);
        const int signed_py = static_cast<int>(py) - static_cast<int>(_sensor_pixel_half_height);

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
        // dir = (pos * (_focal_length / _sensor_lens_distance - 1.f) - 
        //         (_focal_length / _sensor_lens_distance) * pixel_origin
        //     ).normalized();

        dir = unit_dir(pixel_origin, pos);
    }

} 