
#include "bitmap.h"
#include "camera.h"

#include "gpu_face.cuh"
#include "gpu_camera.cuh"


namespace Xrender {


    std::vector<rgb24> gpu_render_outline_preview(const std::vector<gpu_face>& model, const device_camera& cam, std::size_t sample_count = 1u);

}