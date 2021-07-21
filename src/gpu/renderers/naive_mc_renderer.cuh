#ifndef NAIVE_MC_RENDERER_CUH
#define NAIVE_MC_RENDERER_CUH

#include <vector>

#include "gpu/model/bvh_tree.cuh"
#include "gpu/common/abstract_renderer.cuh"

namespace Xrender
{

    class naive_mc_renderer : public abstract_renderer
    {
    public:
        __host__ naive_mc_renderer(
            const bvh_node *_device_tree,
            std::size_t thread_per_block = 256);

        naive_mc_renderer(const naive_mc_renderer &) = delete;
        naive_mc_renderer(naive_mc_renderer &&) noexcept = default;
        ~naive_mc_renderer() noexcept override = default;

        void call_integrate_kernel(
            const camera &cam, curandState_t *rand_pool, std::size_t sample_count, float3 *sensor) override;

    private:
        const bvh_node *_device_tree{nullptr};
        std::size_t _thread_per_block{};
    };

}

#endif