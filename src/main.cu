
#include <iostream>

#include "host/configuration/configuration.h"
#include "host/camera_handling/camera_configuration.cuh"
#include "host/model_loader/wavefront_obj.cuh"
#include "host/bvh_builder/bvh_builder.cuh"
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

    std::cout << "Configure camera" << std::endl;
    camera cam{};
    configure_camera(config.camera_config, cam);

    std::cout << "Loading model " << config.model_path.generic_string() << std::endl;
    const auto model = wavefront_obj_load(config.model_path);

    std::cout << "Building bvh tree" << std::endl;
    const auto bvh = build_bvh_tree(model);

    std::cout << "Allocate and copy resources to gpu" << std::endl;
    auto *device_bvh = clone_to_device(bvh);

    std::cout << "Initialize display" << std::endl;
    renderer_display display{cam};

    display.add_renderer<preview_renderer>(device_bvh, cam);
    display.add_renderer<naive_mc_renderer>(device_bvh, cam);

    display.execute();

    CUDA_CHECK(cudaFree(device_bvh));

    return 0;
}