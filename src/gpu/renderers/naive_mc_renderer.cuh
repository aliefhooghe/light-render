#ifndef NAIVE_MC_RENDERER_CUH
#define NAIVE_MC_RENDERER_CUH

#include <vector>

#include "gpu/model/bvh_tree.cuh"

#include "gpu_renderer.cuh"

namespace Xrender {

    class naive_mc_renderer : public gpu_renderer {

    public:
        naive_mc_renderer(const naive_mc_renderer&) = delete;
        naive_mc_renderer(naive_mc_renderer&&) noexcept = default;
        __host__ naive_mc_renderer(const bvh_node *_device_tree, camera& cam);
        ~naive_mc_renderer() noexcept = default;

    protected:
        __host__ void _call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor) override;
        __host__ void _call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture) override;

    private:
        float _develop_factor();

        const bvh_node *_device_tree{nullptr};
    };

}

#endif