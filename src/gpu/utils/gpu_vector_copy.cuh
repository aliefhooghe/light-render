#ifndef GPU_VECTOR_COPY_CUH_
#define GPU_VECTOR_COPY_CUH_

#include <vector>
#include <cuda.h>
#include "gpu/cuda_exception.cuh"

namespace Xrender
{

    template <typename T>
    T *clone_to_device(const std::vector<T>& vec)
    {
        const auto data_size = vec.size() * sizeof(T);
        T *device_ptr = nullptr;

        //  Allocate and copy to device
        CUDA_CHECK(cudaMalloc(&device_ptr, data_size));
        CUDA_CHECK(cudaMemcpy(device_ptr, vec.data(), data_size, cudaMemcpyHostToDevice));

        return device_ptr;
    }

    template <typename T>
    std::vector<T> clone_from_device(const T *device_ptr, std::size_t count)
    {
        const auto data_size = count * sizeof(T);
        std::vector<T> vec(count);
        CUDA_CHECK(cudaMemcpy(vec.data(), device_ptr, data_size, cudaMemcpyDeviceToHost));
        return device_ptr;
    }

}

#endif