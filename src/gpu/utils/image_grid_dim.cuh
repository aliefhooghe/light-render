#ifndef XRENDER_IMAGE_GRID_DIM_H_
#define XRENDER_IMAGE_GRID_DIM_H_


namespace Xrender {


    dim3 image_grid_dim(unsigned int width, unsigned int height, unsigned int& thread_per_block);


}

#endif