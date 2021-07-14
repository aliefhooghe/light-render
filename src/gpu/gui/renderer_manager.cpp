
#include <stdexcept>
#include <iostream>
#include <cmath>

#include "renderer_manager.h"

namespace Xrender {

    renderer_manager::renderer_manager(std::unique_ptr<abstract_renderer>&& renderer)
    :   _renderer{std::move(renderer)}
    {
        if (!_renderer)
            throw std::invalid_argument("Renderer manager : renderer is nullptr");
    }

    void renderer_manager::reset()
    {
        _renderer->reset();
    }

    void renderer_manager::integrate()
    {
        using namespace std::chrono;
        auto start_time = steady_clock::now();
        _renderer->integrate(_sample_per_step);
        auto end_time = steady_clock::now();
        auto duration = end_time - start_time;

        // Update sample per step to follow the interval requirement

        const auto recorded_speed = static_cast<float>(_sample_per_step) / static_cast<float>(duration.count());
        const auto interval = duration_cast<decltype(duration)>(_interval);

        std::cout
            << "Integrated " << _sample_per_step << " more samples\n"
            << "recorded_speed : " << (100.f * static_cast<float>(interval.count())/static_cast<float>(duration.count()))
            << " % (" << (recorded_speed * 1E9) << " spp/sec)" << std::endl;

        _sample_per_step = std::max<std::size_t>(1u, std::ceil(recorded_speed * interval.count()));
    }

    void renderer_manager::develop_to_texture(gpu_texture& texture)
    {
        _renderer->develop_to_texture(texture);
    }

    void renderer_manager::set_interval(std::chrono::milliseconds interval)
    {
        _interval = interval;
        _sample_per_step = 1u; // Let recalibrate
    }
}