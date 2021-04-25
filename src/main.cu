

#include <iostream>
#include <chrono>

#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "camera.h"

#include "outline_preview.h"
#include "gpu/gpu_outline_preview.cuh"
#include "geometric_sampler.h"
#include "gpu/gpu_geometric_sampler.cuh"

// ---

// gpu adapter


auto make_device_cam(const Xrender::camera& cam)
{
    Xrender::device_camera device_cam;

    device_cam._image_pixel_half_width = cam._image_pixel_half_width;
    device_cam._image_pixel_half_height = cam._image_pixel_half_height;
    device_cam._pixel_size = cam._pixel_height;
    device_cam._focal_length = cam._focal_length;
    device_cam._sensor_lens_distance = cam._sensor_lens_distance;
    device_cam._diaphragm_radius = cam._diaphragm_radius;

    return device_cam;
}

void write_bitmap(std::vector<Xrender::vecf> rendered_sensor, const std::string& path, size_t width, size_t height)
{
    std::vector<Xrender::rgb24> rendered_bitmap(width * height);
    std::transform(
        rendered_sensor.begin(),
        rendered_sensor.end(),
        rendered_bitmap.begin(),
        Xrender::rgb24::from_vecf);

    Xrender::bitmap_write(path, rendered_bitmap, width, height);
}

__device__ __host__ Xrender::rgb24 rgb24_of_float3(const float3& color)
{
    return {
        static_cast<unsigned char>(color.z * 255.f),
        static_cast<unsigned char>(color.y * 255.f),
        static_cast<unsigned char>(color.x * 255.f)};
}

void write_bitmap(std::vector<float3> rendered_sensor, const std::string& path, size_t width, size_t height)
{
    std::vector<Xrender::rgb24> rendered_bitmap(width * height);
    std::transform(
        rendered_sensor.begin(),
        rendered_sensor.end(),
        rendered_bitmap.begin(),
        rgb24_of_float3);
    Xrender::bitmap_write(path, rendered_bitmap, width, height);
}

// --

#define ENABLE_CPU_COMPUTE
#define ENABLE_GPU_COMPUTE
#define ENABLE_PREVIEW
#define ENABLE_RENDER

int main()
{
    std::cout << "Loading OBJ" << std::endl;
    auto model = Xrender::wavefront_obj_load("../../untitled.obj");
    std::cout << "Building tree" << std::endl;
    const Xrender::bvh_tree tree{model};
    const auto gpu_tree = Xrender::make_gpu_bvh(tree);

    std::cout << "Bvh max depth is " << tree.depth() << std::endl;

    const auto size_factor = 40;
    const auto width = 36 * size_factor;
    const auto height = 24 * size_factor;

    constexpr auto gpu_thread_per_block = 32;

    std::cout << "Output image size = " << width << "x" << height << std::endl;

    const auto preview_sample_count = 1;
    const auto render_sample_count = 128;

    // Init camera
    auto camera = Xrender::camera::from_focus_distance(width, height, 36E-3f, 24E-3f, 0.1E-3, 50E-3, 3.f);
    auto device_camera = make_device_cam(camera);

    std::chrono::steady_clock::time_point start, end;

#if defined(ENABLE_CPU_COMPUTE) && defined(ENABLE_PREVIEW)
    // CPU PREVIEW
    start = std::chrono::steady_clock::now();
    auto preview = Xrender::render_outline_preview(tree, camera, preview_sample_count);
    end = std::chrono::steady_clock::now();

    Xrender::bitmap_write("preview-cpu.bmp", preview, width, height);
    
    const auto cpu_preview_duration = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout   << "Rendered cpu Preview in " 
                << cpu_preview_duration
                << " ms; Start gpu preview " << std::endl;
#endif
#if defined(ENABLE_GPU_COMPUTE) && defined(ENABLE_PREVIEW)
    // GPU PREVIEW
    start = std::chrono::steady_clock::now();
    const auto gpu_review = Xrender::gpu_render_outline_preview(gpu_tree, device_camera, preview_sample_count, gpu_thread_per_block);
    end = std::chrono::steady_clock::now();

    const auto gpu_preview_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout   << "Rendered gpu Preview in " 
                << gpu_preview_duration
                << " ms" << std::endl;

    Xrender::bitmap_write("preview-gpu.bmp", gpu_review, width, height);

    std::cout << "Rendered Preview.\nRendering lights on cpu" << std::endl;
#endif
#if defined(ENABLE_CPU_COMPUTE) && defined(ENABLE_RENDER)
    //  CPU RENDER
    start = std::chrono::steady_clock::now();
    auto rendered_sensor = Xrender::mc_naive(tree, camera, render_sample_count);
    end = std::chrono::steady_clock::now();

    const auto cpu_render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout
        << "CPU Render took "
        << cpu_render_duration
        << " ms" << std::endl;

    write_bitmap(rendered_sensor, "render-cpu.bmp", width, height);

    std::cout << "Rendered light on cpu.\nRendering lights on gpu" << std::endl;
#endif
#if defined(ENABLE_GPU_COMPUTE) && defined(ENABLE_RENDER)
    //GPU RENDER
    start = std::chrono::steady_clock::now();
    auto gpu_rendered_sensor = Xrender::gpu_naive_mc(gpu_tree, device_camera, render_sample_count, gpu_thread_per_block);
    end = std::chrono::steady_clock::now();
    
    const auto gpu_render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout
        << "GPU Render took "
        << gpu_render_duration
        << " ms" << std::endl;

    write_bitmap(gpu_rendered_sensor, "render-gpu.bmp", width, height);

    std::cout << "Done." << std::endl;
#endif

    std::cout << "\nSummary :\n" <<

#ifdef ENABLE_PREVIEW  
                "  Preview : \n"
#ifdef ENABLE_CPU_COMPUTE
                "    cpu     : " << cpu_preview_duration << " ms\n"
#endif
#ifdef ENABLE_GPU_COMPUTE
                "    gpu     : " << gpu_preview_duration << " ms\n"
#endif
#if defined(ENABLE_CPU_COMPUTE) && defined(ENABLE_GPU_COMPUTE)
                "    speedup : " << ((100 * cpu_preview_duration / gpu_preview_duration)) << " %\n"
#endif

#endif

#ifdef ENABLE_RENDER
                "  Render : \n"
#ifdef ENABLE_CPU_COMPUTE
                "    cpu     : " << cpu_render_duration << " ms\n"
#endif
#ifdef ENABLE_GPU_COMPUTE
                "    gpu     : " << gpu_render_duration << " ms\n"
#endif
#if defined(ENABLE_CPU_COMPUTE) && defined(ENABLE_GPU_COMPUTE)
                "    speedup : " << ((100 * cpu_render_duration / gpu_render_duration)) << " %\n"
#endif

#endif
            << std::endl;

    return 0;
}
