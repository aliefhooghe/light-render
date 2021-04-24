

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

auto make_device_model(const std::vector<Xrender::face>& model)
{
    std::vector<Xrender::gpu_face> device_model(model.size());
    
    std::transform(
        model.begin(), model.end(), device_model.begin(),
        [](const Xrender::face& f)
        {
            Xrender::gpu_face ret;
            for (int i = 0; i < 3; i++)
            {
                ret.points[i].x = f.points[i].x;
                ret.points[i].y = f.points[i].y;
                ret.points[i].z = f.points[i].z;
            }
            for (int i = 0; i < 3; i++)
            {
                ret.normals[i].x = f.normals[i].x;
                ret.normals[i].y = f.normals[i].y;
                ret.normals[i].z = f.normals[i].z;
            }

            ret.normal.x = f.normal.x;
            ret.normal.y = f.normal.y;
            ret.normal.z = f.normal.z;

            if (Xrender::is_source(f.mtl)) {
                ret.mtl = Xrender::gpu_make_source_material();
            }
            else {
                const auto vecf_color = Xrender::material_preview_color(f.mtl);
                const float3 color = {vecf_color.z, vecf_color.y, vecf_color.x}; 
                ret.mtl = Xrender::gpu_make_lambertian_materal(color);
            }

            return ret;
        });

        return device_model;
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
        static_cast<unsigned char>(color.x * 255.f),
        static_cast<unsigned char>(color.y * 255.f),
        static_cast<unsigned char>(color.z * 255.f)};
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
#define DISABLE_CPU_COMPUTE

int main()
{
    auto model = Xrender::wavefront_obj_load("../../untitled.obj");
    auto device_model = make_device_model(model);

    Xrender::bvh_tree tree{model};

    const auto width = 36 * 20;
    const auto height = 24 * 20;

    const auto preview_sample_count = 12;
    const auto render_sample_count = 4096;

    // Init camera
    auto camera = Xrender::camera::from_focus_distance(width, height, 36E-3f, 24E-3f, 10E-3, 50E-3, 3.f);
    auto device_camera = make_device_cam(camera);

    std::chrono::steady_clock::time_point start, end;

#ifndef DISABLE_CPU_COMPUTE
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
    // GPU PREVIEW
    start = std::chrono::steady_clock::now();
    const auto gpu_review = Xrender::gpu_render_outline_preview(device_model, device_camera, preview_sample_count);
    end = std::chrono::steady_clock::now();

    const auto gpu_preview_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout   << "Rendered gpu Preview in " 
                << gpu_preview_duration
                << " ms" << std::endl;

    Xrender::bitmap_write("preview-gpu.bmp", gpu_review, width, height);

    std::cout << "Rendered Preview.\nRendering lights on cpu" << std::endl;

#ifndef DISABLE_CPU_COMPUTE
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
    //GPU RENDER
    start = std::chrono::steady_clock::now();
    auto gpu_rendered_sensor = Xrender::gpu_naive_mc(device_model, device_camera, render_sample_count);
    end = std::chrono::steady_clock::now();
    
    const auto gpu_render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout
        << "GPU Render took "
        << gpu_render_duration
        << " ms" << std::endl;

    write_bitmap(gpu_rendered_sensor, "render-gpu.bmp", width, height);

    std::cout << "Done." << std::endl;

#ifndef DISABLE_CPU_COMPUTE
    std::cout << "\nSummary :\n"
            <<  "  Preview : \n"
                "    cpu     : " << cpu_preview_duration << " ms\n"
                "    gpu     : " << gpu_preview_duration << " ms\n"
                "    speedup : " << ((100 * cpu_preview_duration / gpu_preview_duration)) << " %\n"
                "  Render : \n"
                "    cpu     : " << cpu_render_duration << " ms\n"
                "    gpu     : " << gpu_render_duration << " ms\n"
                "    speedup : " << ((100 * cpu_render_duration / gpu_render_duration)) << " %\n"
            << std::endl;
#else
    std::cout << "\nSummary :\n"
            <<  "  Preview : \n"
                "    gpu     : " << gpu_preview_duration << " ms\n"
                "  Render : \n"
                "    gpu     : " << gpu_render_duration << " ms\n"
            << std::endl;
#endif
    return 0;
}
