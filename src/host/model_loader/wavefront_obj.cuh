#ifndef XRENDER_WAVEFRONT_OBJ_H_
#define XRENDER_WAVEFRONT_OBJ_H_

#include <vector>
#include <filesystem>

#include "model.cuh"

namespace Xrender {

    model wavefront_obj_load(const std::filesystem::path& path);

} /* namespace Xrender */

#endif