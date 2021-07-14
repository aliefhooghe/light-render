
#include "bitmap.h"

#include "gpu_bvh.cuh"
#include "gpu_camera.cuh"
#include "gpu_texture.cuh"

#include "gpu_renderer.cuh"

namespace Xrender {


    class gpu_outline_preview : public gpu_renderer {

    public:
        gpu_outline_preview(const gpu_outline_preview&) = delete;
        gpu_outline_preview(gpu_outline_preview&&) noexcept = default;
        __host__ gpu_outline_preview(const gpu_bvh_node *_device_tree, device_camera& cam);
        ~gpu_outline_preview() noexcept override = default;

    protected:
        __host__ void _call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor) override;
        __host__ void _call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture) override;

    private:
        float _develop_factor();

        const gpu_bvh_node *_device_tree{nullptr};
    };
}