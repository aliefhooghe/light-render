
#include <cuda.h>
#include <iostream>

#include "cuda_exception.cuh"

namespace Xrender
{
    bool select_openGL_cuda_device()
    {
        int device_count;

        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        for (auto id = 0; id < device_count; ++id)
        {
            int compute_mode;
            int is_integrated;
            CUDA_CHECK(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, id));
            CUDA_CHECK(cudaDeviceGetAttribute(&is_integrated, cudaDevAttrIntegrated, id));

            if (compute_mode != cudaComputeModeProhibited)
            {
                cudaDeviceProp device_prop;
                CUDA_CHECK(cudaGetDeviceProperties(&device_prop, id));
                CUDA_CHECK(cudaSetDevice(id));
                std::cout << "Found cuda capable device : " << device_prop.name << std::endl;
                return true;
            }
        }

        return false;
    }
}