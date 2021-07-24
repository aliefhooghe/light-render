
#include <algorithm>
#include <cmath>
#include <random>


#include "curand_pool.cuh"
#include "cuda_exception.cuh"
#include "gpu_vector_copy.cuh"

namespace Xrender
{
    __global__ void curand_pool_init_kernel(curandState *pool, const unsigned long long *seeds, std::size_t count)
    {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < count)
        {
            curand_init(seeds[idx], 0, 0, &pool[idx]);
        }
    }

    curandState *create_curand_pool(std::size_t count)
    {
        std::vector<unsigned long long> seeds(count);
        curandState *pool = nullptr;
        std::mt19937 random_generator{};

        // Make sure that to execution give the same numbers
        random_generator.seed(random_generator.default_seed);

        // Generate seeds on host
        std::generate(
            seeds.begin(),
            seeds.end(),
            [&random_generator]() { return random_generator(); });

        // Alocate curand state pool and upload seeds value
        CUDA_CHECK(cudaMalloc(&pool, count * sizeof(curandState)));
        auto *device_seeds = clone_to_device(seeds);

        // Initialize the rando mgenerators pools
        constexpr auto thread_per_block = 1024;
        auto block_count = static_cast<unsigned int>(std::ceil(count / thread_per_block));

        curand_pool_init_kernel<<<block_count, thread_per_block>>>(pool, device_seeds, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free the seeds as they are not anymore useful
        CUDA_CHECK(cudaFree(device_seeds));

        return pool;
    }
}
