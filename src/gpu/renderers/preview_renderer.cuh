
#ifndef PREVIEW_RENDERER_H_
#define PREVIEW_RENDERER_H_

#include "gpu/common/abstract_renderer.cuh"
#include "gpu/common/gpu_texture.cuh"
#include "gpu/model/bvh_tree.cuh"
#include "gpu/model/camera.cuh"
#include "gpu/model/material.cuh"

namespace Xrender
{

    class preview_renderer : public abstract_renderer
    {
    public:
        static constexpr auto max_thread_per_block = 64u;

        __host__ preview_renderer(
            const bvh_node *device_tree, int tree_size,
            const face *device_model,
            const material *device_mtl_bank);

        preview_renderer(const preview_renderer &) = delete;
        preview_renderer(preview_renderer &&) noexcept = default;
        ~preview_renderer() noexcept override = default;

        void call_integrate_kernel(
            const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor) override;

    private:
        const bvh_node *_device_tree{nullptr};
        const int _tree_size{0};
        const face *_device_model{nullptr};
        const material *_device_mtl_bank{nullptr};
    };
}

#endif