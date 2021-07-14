
#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "camera.h"

#include "gpu/utils/gpu_vector_copy.cuh"
#include "gpu/gui/renderer_display.cuh"
#include "gpu/gpu_bvh.cuh"

#include "gpu/gpu_outline_preview.cuh"
#include "gpu/gpu_geometric_sampler.cuh"

static auto make_device_cam(const Xrender::camera& cam)
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

int main(int argc, char **argv)
{
    auto model = Xrender::wavefront_obj_load("../../untitled.obj");
    const auto bvh_tree = Xrender::make_gpu_bvh(Xrender::bvh_tree{model});

    const auto size_factor = 40;
    const auto width = 36 * size_factor;
    const auto height = 24 * size_factor;
    auto camera = Xrender::camera::from_focus_distance(width, height, 36E-3f, 24E-3f, 100E-3, 50E-3, 8.f);
    auto device_camera = make_device_cam(camera);

    auto device_tree = clone_to_device(bvh_tree);

    Xrender::renderer_display display{device_camera};

    display.add_renderer<Xrender::gpu_outline_preview>(device_tree, device_camera);
    display.add_renderer<Xrender::gpu_geometric_sampler>(device_tree, device_camera);
    display.execute();

    return 0;
}