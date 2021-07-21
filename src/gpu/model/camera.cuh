#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

#include "float3_operators.cuh"
#include "gpu/utils/curand_helper.cuh"

namespace Xrender {

    struct camera {

        __device__ float3 sample_ray(curandState *state, float3& pos, int px, int py) const
        {
            const float signed_px = (px - _image_pixel_half_width) + curand_uniform(state);
            const float signed_py = (py - _image_pixel_half_height) + curand_uniform(state);

            // Compute the pixel origin position on the sensor
            const float3 pixel_origin =
            {
                -signed_px * _pixel_size,
                -_sensor_lens_distance,
                -signed_py * _pixel_size
            };

            const auto org_pos = _sample_lens_point(state);

            const auto start_dir = normalized((_focal_length / _sensor_lens_distance - 1.f) * org_pos -
                              (_focal_length / _sensor_lens_distance) * pixel_origin);
            pos = org_pos + _position;
            return start_dir;
        }

        __device__ float3 _sample_lens_point(curandState *state) const
        {
            const float2 pos2d = rand_unit_disc_uniform(state);
            return _diaphragm_radius * float3{pos2d.x, 0.f, pos2d.y};
        }

        __device__ __host__ unsigned int get_image_width() const
        {
            return 2 * _image_pixel_half_width;
        }

        __device__ __host__ unsigned int get_image_height() const
        {
            return 2 * _image_pixel_half_height;
        }

        int _image_pixel_half_width;
        int _image_pixel_half_height;
        float _pixel_size;             // real size of a sensor pixel
        float _focal_length;
        float _sensor_lens_distance;
        float _diaphragm_radius;
        float3 _position{0.f, 0.f, 0.f};
    };

}

#endif