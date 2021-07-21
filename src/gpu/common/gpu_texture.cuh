#ifndef XRENDER_GPU_TEXTURE_H_
#define XRENDER_GPU_TEXTURE_H_

#include <GL/glew.h>
#include <GL/gl.h>
#include <vector>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

namespace Xrender
{

    class registered_texture
    {

    public:
        class mapped_surface
        {
            friend class registered_texture;

        public:
            mapped_surface(const mapped_surface &) = delete;
            mapped_surface(mapped_surface &&) noexcept;
            __host__ ~mapped_surface() noexcept;

            __host__ cudaSurfaceObject_t surface() const noexcept;

        private:
            __host__ mapped_surface(registered_texture &);

            registered_texture *_texture;
            cudaSurfaceObject_t _surface_object;
        };

        friend class mapped_surface;

        __host__ registered_texture(GLuint texture_id, unsigned int width, unsigned int height);
        registered_texture(const registered_texture &) = delete;
        registered_texture(registered_texture &&) noexcept = delete; // todo
        __host__ ~registered_texture() noexcept;

        __host__ unsigned int get_width() const noexcept;
        __host__ unsigned int get_height() const noexcept;
        __host__ GLuint get_gl_texture_id() const noexcept;

        /**
         * \brief Return a handle to a surface object mapped
         * to the openGL texture.
         */
        __host__ mapped_surface get_mapped_surface();

        /**
         * \brief Download the tecture from the device
         */
        __host__ std::vector<float4> retrieve_texture();

    private:
        unsigned int _width;
        unsigned int _height;
        cudaGraphicsResource *_graphic_resource{nullptr};
        GLuint _gl_id{0u};
    };

}

#endif