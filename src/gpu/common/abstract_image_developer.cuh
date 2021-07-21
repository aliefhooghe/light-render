#ifndef XRENDER_ABSTRACT_DEVELOPER_H_
#define XRENDER_ABSTRACT_DEVELOPER_H_

#include <cuda.h>

namespace Xrender
{
    class abstract_image_developer
    {
    public:
        virtual ~abstract_image_developer() noexcept = default;
        virtual void call_develop_to_texture_kernel(
            std::size_t total_sample_count,
            const unsigned int sensor_width, const unsigned int sensor_height,
            const float3 *sensor, cudaSurfaceObject_t texture) =0;
    };
}

#endif