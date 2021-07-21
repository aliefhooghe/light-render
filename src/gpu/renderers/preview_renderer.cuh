
#ifndef PREVIEW_RENDERER_H_
#define PREVIEW_RENDERER_H_

#include "gpu/common/abstract_renderer.cuh"
#include "gpu/common/gpu_texture.cuh"
#include "gpu/model/bvh_tree.cuh"
#include "gpu/model/camera.cuh"

namespace Xrender
{

    class preview_renderer : public abstract_renderer
    {
    public:
        __host__ preview_renderer(
            const bvh_node *_device_tree,
            std::size_t thread_per_block = 256);

        preview_renderer(const preview_renderer &) = delete;
        preview_renderer(preview_renderer &&) noexcept = default;
        ~preview_renderer() noexcept override = default;

        void call_integrate_kernel(
            const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor) override;

    private:
        const bvh_node *_device_tree{nullptr};
        std::size_t _thread_per_block{};
    };
}

#endif