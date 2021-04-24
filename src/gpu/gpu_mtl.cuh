#ifndef GPU_MTL_CUH
#define GPU_MTL_CUH

#include <cuda.h>

namespace Xrender {

    using gpu_mtl = float3;

    // Onmy lambertian / sources

    gpu_mtl gpu_make_source_material();
    gpu_mtl gpu_make_lambertian_materal(float3 absorption);

    static __device__ __host__ bool gpu_mtl_is_source(const gpu_mtl& mtl)
    {
        return mtl.x < 0.f;
    }

    static __device__ __host__ float3 gpu_preview_color(const gpu_mtl& mtl)
    {
        return gpu_mtl_is_source(mtl) ? float3{1.f, 1.f, 1.f} : mtl;
    }

}

#endif