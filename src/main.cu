
#include <iostream>

#include "gpu/gui/renderer_display.cuh"
#include "gpu/image_developers/average_image_developer.cuh"
#include "gpu/renderers/naive_mc_renderer.cuh"
#include "gpu/renderers/preview_renderer.cuh"
#include "gpu/utils/cuda_exception.cuh"
#include "gpu/utils/device_probing.cuh"
#include "gpu/utils/gpu_vector_copy.cuh"
#include "host/bvh_builder/bvh_builder.cuh"
#include "host/camera_handling/camera_configuration.cuh"
#include "host/configuration/configuration.h"
#include "host/model_loader/wavefront_obj.cuh"
#include "host/utils/chronometer.h"

void usage(const char *argv0)
{
    std::cout << "usage : " << argv0 << " <render.conf>" << std::endl;
}

int main(int argc, char **argv)
{
    using namespace Xrender;

    if (argc != 2)
    {
        usage(argv[0]);
        return 1;
    }
    else if (!select_openGL_cuda_device())
    {
        std::cout << "No cuda capable device was found" << std::endl;
        return 1;
    }

    render_configuration config = load_render_configuration(argv[1]);
    chronometer timewatch{};

    std::cout << "Configure camera" << std::endl;
    camera cam{};
    configure_camera(config.camera_config, cam);

    std::cout << "Loading model " << config.model_path.generic_string() << std::endl;
    const auto model = wavefront_obj_load(config.model_path);

    std::cout << "Building bvh tree (" << model.size() << " faces)" << std::endl;
    timewatch.start();
    const auto host_bvh = build_bvh_tree(model);
    const auto bvh_build_duration = timewatch.stop();
    timewatch.start();
    const auto gpu_bvh = host_bvh->to_gpu_bvh();
    const auto bvh_convertion_duration = timewatch.stop();

    std::cout << "Bvh build took " << bvh_build_duration.count() << " ms; Convertion took " << bvh_convertion_duration.count() << " ms\nAllocate and copy resources to gpu" << std::endl;
    auto *device_bvh = clone_to_device(gpu_bvh);

    renderer_display display{cam};

    // Initialize the views :
    display.add_view(
        std::make_unique<preview_renderer>(device_bvh),
        std::make_unique<average_image_developer>());
    display.add_view(
        std::make_unique<naive_mc_renderer>(device_bvh),
        std::make_unique<average_image_developer>());

    std::cout << "Start rendering." << std::endl;
    display.execute();

    CUDA_CHECK(cudaFree(device_bvh));

    return 0;
}