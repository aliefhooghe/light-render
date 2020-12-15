#ifndef XRENDER_OUTLINE_PREVIEW_H_
#define XRENDER_OUTLINE_PREVIEW_H_

#include "bvh_tree.h"
#include "bitmap.h"
#include "camera.h"

namespace Xrender {

    std::vector<rgb24> render_outline_preview(const bvh_tree& tree, const camera& cam, std::size_t sample_count = 1u);

}


#endif /* XRENDER_OUTLINE_PREVIEW_H_ */