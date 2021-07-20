
#include "gpu/utils/cuda_exception.cuh"

#include "gpu_texture.cuh"

namespace Xrender
{

    gpu_texture::mapped_surface::mapped_surface(gpu_texture& texture)
    :   _texture{&texture}
    {
        cudaArray_t array;

        CUDA_CHECK(cudaGraphicsMapResources(1, &(_texture->_graphic_resource), nullptr));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, _texture->_graphic_resource, 0, 0));

        cudaResourceDesc resource_desc;
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = array;

        CUDA_CHECK(cudaCreateSurfaceObject(&_surface_object, &resource_desc));
    }

    gpu_texture::mapped_surface::mapped_surface(gpu_texture::mapped_surface&& other) noexcept
    :   _texture{other._texture},
        _surface_object{other._surface_object}
    {
        other._texture = nullptr;
    }

    __host__ gpu_texture::mapped_surface::~mapped_surface() noexcept
    {
        if (_texture != nullptr) {
            CUDA_WARNING(cudaDestroySurfaceObject(_surface_object));
            CUDA_WARNING(cudaGraphicsUnmapResources(1, &(_texture->_graphic_resource), NULL));
        }
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

        CUDA_CHECK(cudaGetLastError())

        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &_graphic_resource, _gl_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
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
        return mapped_surface{*this};
    }

    __host__ std::vector<float4> gpu_texture::retrieve_texture()
    {
        std::vector<float4> host_texture{_width * _height};
        cudaArray_t array;

        CUDA_CHECK(cudaGraphicsMapResources(1, &_graphic_resource, nullptr));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, _graphic_resource, 0, 0));

        CUDA_CHECK(cudaMemcpy2DFromArray(
            host_texture.data(), _width * sizeof(float4), array,
            0, 0, _width * sizeof(float4), _height, cudaMemcpyDeviceToHost));

        CUDA_WARNING(cudaGraphicsUnmapResources(1, &_graphic_resource, NULL));
        return host_texture;
    }
}