#ifndef XRENDER_AVERAGE_IMAGE_DEVELOPPER_H_
#define XRENDER_AVERAGE_IMAGE_DEVELOPPER_H_

#include "gpu/common/abstract_image_developer.cuh"

namespace Xrender
{

    class average_image_developer : public abstract_image_developer
    {
    public:
        average_image_developer(
            float factor = 1.f,
            std::size_t thread_per_block = 256);
        ~average_image_developer() noexcept override = default;

        const float& factor() const noexcept { return _factor; };
        float& factor() noexcept { return _factor; };

        void call_develop_to_texture_kernel(
            std::size_t total_sample_count,
            const unsigned int sensor_width, const unsigned int sensor_height,
            const float3 *sensor, cudaSurfaceObject_t texture) override;

    private:
        float _factor;
        const std::size_t _thread_per_block;
    };

}

#endif /* XRENDER_AVERAGE_IMAGE_DEVELOPPER_H_ */