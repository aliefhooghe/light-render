#include <algorithm>
#include <cmath>
#include "image_grid_dim.cuh"

namespace Xrender
{
    dim3 image_grid_dim(unsigned int width, unsigned int height, unsigned int& thread_per_block)
    {
        thread_per_block = std::min<int>(thread_per_block, width);
        const auto horizontal_block_count = static_cast<unsigned int>(std::ceil((float)width / (float)thread_per_block));
        return dim3{horizontal_block_count, height};
    }
}