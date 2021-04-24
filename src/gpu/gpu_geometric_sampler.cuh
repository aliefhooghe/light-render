#ifndef GPU_GEOMETRIC_SAMPLER_CUH
#define GPU_GEOMETRIC_SAMPLER_CUH

#include <vector>

#include "gpu_bvh.cuh"
#include "gpu_camera.cuh"

namespace Xrender {

    std::vector<float3> gpu_naive_mc(
        const std::vector<gpu_bvh_node>& tree,
        const device_camera& camera,
        const int sample_per_pixel = 1);

}

#endif