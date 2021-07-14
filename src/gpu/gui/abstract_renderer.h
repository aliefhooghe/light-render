
#ifndef XRENDER_ABSTRACT_RENDERER_H_
#define XRENDER_ABSTRACT_RENDERER_H_

#include <cstdint>

namespace Xrender {

    class gpu_texture;

    class abstract_renderer
    {
    public:
        virtual ~abstract_renderer() noexcept = default;

        virtual void reset() = 0;
        virtual void integrate(std::size_t sample_count) = 0;
        virtual void develop_to_texture(gpu_texture& texture) = 0;
        //  TODO develop to vector
    };
}

#endif