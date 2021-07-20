#ifndef XRENDER_GPU_TEXTURE_H_
#define XRENDER_GPU_TEXTURE_H_

#include <GL/glew.h>
#include <GL/gl.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

namespace Xrender {

    class gpu_texture {

    public:
        class mapped_surface {
            friend class gpu_texture;
        public:
            mapped_surface(const mapped_surface&) = delete;
            mapped_surface(mapped_surface&&) noexcept = default;
            __host__ ~mapped_surface() noexcept;

            __host__ cudaSurfaceObject_t surface() const noexcept;
        private:
            __host__ mapped_surface(cudaGraphicsResource *graphic_resource);

            cudaGraphicsResource *_graphic_resource{nullptr};
            cudaSurfaceObject_t _surface_object;
        };

        __host__ gpu_texture(unsigned int width, unsigned int height);
        gpu_texture(const gpu_texture&) = delete;
        gpu_texture(gpu_texture&&) noexcept = delete; // todo
        __host__ ~gpu_texture() noexcept;

        __host__ unsigned int get_width() const noexcept;
        __host__ unsigned int get_height() const noexcept;
        __host__ GLuint get_gl_texture_id() const noexcept;

        __host__ mapped_surface get_mapped_surface();

    private:
        unsigned int _width;
        unsigned int _height;
        cudaGraphicsResource *_graphic_resource{nullptr};
        GLuint _gl_id{0u};
    };

}

#endif