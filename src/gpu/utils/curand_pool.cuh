#ifndef XRENDER_CURRAND_POOL_CUH_
#define XRENDER_CURRAND_POOL_CUH_

#include <curand_kernel.h>

namespace Xrender
{

    curandState *create_curand_pool(std::size_t count);

}

#endif