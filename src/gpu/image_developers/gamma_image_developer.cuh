#ifndef XRENDER_GAMMA_IMAGE_DEVELOPPER_H_
#define XRENDER_GAMMA_IMAGE_DEVELOPPER_H_

#include "gpu/common/abstract_image_developer.cuh"

namespace Xrender
{

    class gamma_image_developer : public abstract_image_developer
    {
    public:
        gamma_image_developer(
            float factor = 1.f, float gamma = 1.f,
            std::size_t thread_per_block = 256);
        ~gamma_image_developer() noexcept override = default;

        void scale_gamma(bool up);
        void scale_factor(bool up);

        void call_develop_to_texture_kernel(
            std::size_t total_sample_count,
            const unsigned int sensor_width, const unsigned int sensor_height,
            const float3 *sensor, cudaSurfaceObject_t texture) override;

    private:
        float _gamma;
        float _factor;
        const std::size_t _thread_per_block;
    };

}

#endif /* XRENDER_AVERAGE_IMAGE_DEVELOPPER_H_ */