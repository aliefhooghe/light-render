
#include "bitmap.h"
#include "camera.h"

#include "gpu_bvh.cuh"
#include "gpu_camera.cuh"


namespace Xrender {


    std::vector<rgb24> gpu_render_outline_preview(
        const std::vector<gpu_bvh_node>& tree,
        const device_camera& cam,
        std::size_t sample_count,
        int thread_per_block);

}