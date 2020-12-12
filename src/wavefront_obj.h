#ifndef XRENDER_WAVEFRONT_OBJ_H_
#define XRENDER_WAVEFRONT_OBJ_H_

#include <string>
#include <vector>
#include <filesystem>

#include "face.h"

namespace Xrender {

    std::vector<face> wavefront_obj_load(const std::filesystem::path& path);

} /* namespace Xrender */

#endif