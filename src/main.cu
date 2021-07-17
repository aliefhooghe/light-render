
#include <iostream>

#include "host/configuration/configuration.h"
#include "host/camera_handling/camera_configuration.cuh"
#include "host/model_loader/wavefront_obj.cuh"
#include "host/bvh_builder/bvh_builder.cuh"
#include "host/utils/chronometer.h"
#include "gpu/gui/renderer_display.cuh"
#include "gpu/renderers/preview_renderer.cuh"
#include "gpu/renderers/naive_mc_renderer.cuh"
#include "gpu/utils/gpu_vector_copy.cuh"
#include "gpu/utils/cuda_exception.cuh"

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

    render_configuration config = load_render_configuration(argv[1]);
    chronometer timewatch{};

    std::cout << "Configure camera" << std::endl;
    camera cam{};
    configure_camera(config.camera_config, cam);

    std::cout << "Loading model " << config.model_path.generic_string() << std::endl;
    const auto model = wavefront_obj_load(config.model_path);

    std::cout << "Building bvh tree" << std::endl;
    timewatch.start();
    const auto host_bvh = build_bvh_tree(model);
    const auto gpu_bvh = host_bvh->to_gpu_bvh();
    const auto bvh_build_duration = timewatch.stop();

    std::cout << "Bvh build took " << bvh_build_duration.count() << " ms\nAllocate and copy resources to gpu" << std::endl;
    auto *device_bvh = clone_to_device(gpu_bvh);

    std::cout << "Initialize display" << std::endl;
    renderer_display display{cam};

    display.add_renderer<preview_renderer>(device_bvh, cam);
    display.add_renderer<naive_mc_renderer>(device_bvh, cam);

    std::cout << "Start rendering." << std::endl;
    display.execute();

    CUDA_CHECK(cudaFree(device_bvh));

    return 0;
}