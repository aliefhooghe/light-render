#ifndef CAMERA_H_
#define CAMERA_H_

#include "vec.h"

namespace Xrender
{

    /**
     * \brief camera
     * \note Camera position and orientation :
     *      - focal point is located at (0, 0, 0)
     *      - view direction is (0, 1, 0)
     */
    class camera
    {

    public:
        camera() = delete;
        camera(const camera&) = default;
        camera(camera&&) = default;
        
        static camera from_sensor_lens_distance(
            unsigned int sensor_pixel_width, unsigned int sensor_pixel_height,
            float sensor_real_width, float sensor_real_height,
            float diaphragm_radius, float focal_length, float sensor_lens_distance);

        static camera from_focus_distance(
            unsigned int sensor_pixel_width, unsigned int sensor_pixel_height,
            float sensor_real_with, float sensor_real_height,
            float diaphragm_radius, float focal_length, float focus_distance);


        void sample_ray(unsigned int px, unsigned int py, vecf& pos, vecf& dir);

    private:
        camera(
            unsigned int sensor_pixel_width,
            unsigned int sensor_pixel_height,
            float sensor_real_width,
            float sensor_real_height,
            float diaphragm_radius,
            float focal_length,
            float sensor_lens_distance);

        vecf _sample_lens_point();

        unsigned int _sensor_pixel_half_width;
        unsigned int _sensor_pixel_half_height;
        float _pixel_width;             // real with of a sensor pixel
        float _pixel_height;            // real height of a sensor pixel
        float _diaphragm_radius;
        float _focal_length;
        float _sensor_lens_distance;
    };

} // namespace Xrender

#endif