
#ifndef XRENDER_ABSTRACT_RENDERER_H_
#define XRENDER_ABSTRACT_RENDERER_H_

#include "gpu/model/camera.cuh"

namespace Xrender
{
    class abstract_renderer
    {
    public:
        virtual ~abstract_renderer() noexcept = default;
        virtual void call_integrate_kernel(
            const camera& cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor) =0;
    };
}

#endif