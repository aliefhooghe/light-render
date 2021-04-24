
#include "gpu_mtl.cuh"

namespace Xrender {

    gpu_mtl gpu_make_source_material()
    {
        return {-1.f, -1.f, -1.f};
    }

    gpu_mtl gpu_make_lambertian_materal(float3 absorption)
    {
        return absorption;
    }

}