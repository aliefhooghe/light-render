#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include <vector>

#include "gpu/model/camera.cuh"
#include "gpu/gui/abstract_renderer.h"
#include "gpu/gui/gpu_texture.cuh"

namespace Xrender {


    class gpu_renderer : public abstract_renderer {

    public:
        gpu_renderer(const gpu_renderer&) = delete;
        gpu_renderer(gpu_renderer&&) noexcept = default;
        __host__ gpu_renderer(camera& cam);
        __host__ ~gpu_renderer() noexcept;

        __host__ void set_thread_per_block(unsigned int) noexcept;

        __host__ void reset() override;
        __host__ void integrate(std::size_t sample_count) override;
        __host__ void develop_to_texture(gpu_texture& texture) override;
        __host__ std::size_t get_total_sample_count() override;

        // __host__ std::vector<rgb24> develop();

    protected:
        virtual void _call_integrate_kernel(std::size_t sample_count, curandState_t *rand_pool, float3 *sensor) =0;
        virtual void _call_develop_to_texture_kernel(const float3 *sensor, cudaSurfaceObject_t texture) =0;

        __host__ inline const auto& _image_grid_dim() const noexcept { return _grid_dim; }
        __host__ inline const auto _image_thread_per_block() const noexcept { return _thread_per_block; }

        camera& _camera;

    private:
        curandState *_rand_pool{nullptr};
        float3 *_device_sensor{nullptr};
        unsigned int _thread_per_block{};
        dim3 _grid_dim{};
        std::size_t _total_sample_count{0u};
    };

}
#endif