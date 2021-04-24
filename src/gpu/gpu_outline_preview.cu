
#include <algorithm>
#include <chrono>

#include "gpu_outline_preview.cuh"
#include "vector_operations.cuh"
#include "cuda_exception.cuh"
#include "rand_operations.cuh"

namespace Xrender {


    __global__ void preview_kernel(
        const gpu_face *model, 
        const int face_count,
        const device_camera cam,
        const int sample_count, 
        float3 *image)
    {
        //  Get pixel position in image
        const int x = threadIdx.x;
        const int y = blockIdx.x;
        const int width = blockDim.x;
        const int pixel_index = x + y * width;

        //  Initialize random generator
        curandState rand_state;
        curand_init(1984+pixel_index, 0, 0, &rand_state);

        float3 pos;
        float3 dir;     
        gpu_intersection inter;
        float3 estimator = {0.f, 0.f, 0.f};

        for (auto i = 0; i < sample_count; i++) {
            dir = cam.sample_ray(&rand_state, pos, x, y);     
            if (gpu_intersect_ray_model(model, face_count, pos, dir, inter))
                estimator = estimator + fabs(_dot(dir, inter.normal)) * 
                            gpu_preview_color(inter.triangle->mtl);
        }

        image[pixel_index] = (1.f / sample_count) * estimator;
    }       
    
    __device__ __host__ rgb24 _color_of_float3(const float3& color)
    {
        return {
            static_cast<unsigned char>(color.x * 255.f),
            static_cast<unsigned char>(color.y * 255.f),
            static_cast<unsigned char>(color.z * 255.f)};
    }

    std::vector<rgb24> gpu_render_outline_preview(const std::vector<gpu_face>& model, const device_camera& cam, std::size_t sample_count)
    {
        const auto width = cam.get_image_width();
        const auto height = cam.get_image_height();

        // Copy device model to device
        const auto model_size = model.size() * sizeof(gpu_face);
        gpu_face *device_model = nullptr;

        CUDA_CHECK(cudaMalloc(&device_model, model_size));
        CUDA_CHECK(cudaMemcpy(device_model, model.data(), model_size, cudaMemcpyHostToDevice));

        // Init device image
        const auto device_image_size = width * height * sizeof(float3);
        float3 *device_image = nullptr;

        CUDA_CHECK(cudaMalloc(&device_image, device_image_size));

        const auto start = std::chrono::steady_clock::now();
        preview_kernel<<<height, width>>>(device_model, model.size(), cam, sample_count, device_image);
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for kernel completion
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto end = std::chrono::steady_clock::now();

        printf("GPU computation took %d ms\n", 
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        // Allocate host image
        std::vector<float3> output{width * height};
        std::vector<Xrender::rgb24> rgb24_output{width * height};

        //  Copy result
        CUDA_CHECK(cudaMemcpy(output.data(), device_image, device_image_size, cudaMemcpyDeviceToHost));

        std::transform(
            output.begin(), output.end(), rgb24_output.begin(),
            _color_of_float3);

        CUDA_CHECK(cudaFree(device_model));
        CUDA_CHECK(cudaFree(device_image));

        return rgb24_output;
    }

}