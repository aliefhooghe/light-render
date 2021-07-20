#ifndef XRENDER_DEVICE_PROBING_H_
#define XRENDER_DEVICE_PROBING_H_

namespace Xrender
{
    /**
     * \brief Look for a cuda capable gpu, able to compute and render
     */
    bool select_openGL_cuda_device();
}

#endif