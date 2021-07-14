
#ifndef PREVIEW_RENDERER_H_
#define PREVIEW_RENDERER_H_

#include "gpu/gui/gpu_texture.cuh"
#include "gpu/model/bvh_tree.cuh"
#include "gpu/model/camera.cuh"

#include "gpu_renderer.cuh"

namespace Xrender {


    class preview_renderer : public gpu_renderer {

    public:
        preview_renderer(const preview_renderer&) = delete;
        preview_renderer(preview_renderer&&) noexcept = default;
        __host__ preview_renderer(const bvh_node *_device_tree, camera& cam);
        ~preview_renderer() noexcept override = default;

    protected:
        __host__ void _call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor) override;
        __host__ void _call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture) override;

    private:
        float _develop_factor();

        const bvh_node *_device_tree{nullptr};
    };
}

#endif