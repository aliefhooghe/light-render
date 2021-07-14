#ifndef XRENDER_CAMERA_CONFIGURATION_H_
#define XRENDER_CAMERA_CONFIGURATION_H_

#include "host/configuration/configuration.h"
#include "gpu/model/camera.cuh"

namespace Xrender
{

    void configure_camera(const camera_configuration &, camera &);

    /**
     * \brief Update focal length and keep the focus distance
     */
    void camera_update_focal_length(camera &, float focal_length);

}

#endif