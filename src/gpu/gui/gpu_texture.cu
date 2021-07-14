
#include "gpu/utils/cuda_exception.cuh"

#include "gpu_texture.cuh"

namespace Xrender
{

    gpu_texture::mapped_surface::mapped_surface(cudaGraphicsResource *graphic_resource)
    :   _graphic_resource{graphic_resource}
    {
        cudaArray_t array;

        CUDA_CHECK(cudaGraphicsMapResources(1, &_graphic_resource, nullptr));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, _graphic_resource, 0, 0));


        cudaResourceDesc resource_desc;
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = array;

        CUDA_CHECK(cudaCreateSurfaceObject(&_surface_object, &resource_desc));
    }

    __host__ gpu_texture::mapped_surface::~mapped_surface() noexcept
    {
        CUDA_WARNING(cudaDestroySurfaceObject(_surface_object));
        CUDA_WARNING(cudaGraphicsUnmapResources(1, &_graphic_resource, NULL));
    }

    __host__ cudaSurfaceObject_t gpu_texture::mapped_surface::surface() const noexcept
    {
        return _surface_object;
    }

    __host__ gpu_texture::gpu_texture(unsigned int width, unsigned int height)
        : _width{width}, _height{height}
    {
        glGenTextures(1, &_gl_id);
        glBindTexture(GL_TEXTURE_2D, _gl_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);

        // glBindTexture(GL_TEXTURE_2D, 0u);

        auto error = glGetError(); // GL_NO_ERROR
        std::cout << "glGetError : "<< (char*)gluErrorString(error) << std::endl;

        CUDA_CHECK(cudaGetLastError())

        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &_graphic_resource, _gl_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    }

    __host__ gpu_texture::~gpu_texture() noexcept
    {
        CUDA_WARNING(cudaGraphicsUnregisterResource(_graphic_resource));
        glDeleteTextures(1, &_gl_id);
    }

    __host__ unsigned int gpu_texture::get_width() const noexcept
    {
        return _width;
    }

    __host__ unsigned int gpu_texture::get_height() const noexcept
    {
        return _height;
    }

    __host__ GLuint gpu_texture::get_gl_texture_id() const noexcept
    {
        return _gl_id;
    }

    __host__ gpu_texture::mapped_surface gpu_texture::get_mapped_surface()
    {
        return mapped_surface{_graphic_resource};
    }
}