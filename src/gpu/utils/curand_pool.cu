
#include <cmath>

#include "curand_pool.cuh"
#include "cuda_exception.cuh"

namespace Xrender
{
    __global__ void curand_pool_init_kernel(curandState *pool, std::size_t count)
    {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < count)
        {
            curand_init(idx, 0, 0, &pool[idx]);
        }
    }

    curandState *create_curand_pool(std::size_t count)
    {
        curandState *pool = nullptr;

        CUDA_CHECK(cudaMalloc(&pool, count * sizeof(curandState)));

        constexpr auto thread_per_block = 1024;
        auto block_count = static_cast<unsigned int>(std::ceil(count / thread_per_block));

        curand_pool_init_kernel<<<block_count, thread_per_block>>>(pool, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return pool;
    }
}
