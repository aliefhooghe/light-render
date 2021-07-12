#ifndef XRENDER_LIGHT_PATH_H_
#define XRENDER_LIGHT_PATH_H_

#include "bvh_tree.h"
#include "camera.h"

namespace Xrender
{

    /**
     *
     * cast until a source is hitten or the ray is lost or max_bounce_is_reached
     */
    vecf path_sample_geometric(
        const bvh_tree &tree,
        const vecf &start_pos,
        const vecf &start_dir,
        vecf &estimator);

    std::vector<vecf> mc_naive(
        const bvh_tree& tree,
        const camera& cam,
        std::size_t sample_pp_count = 1u);

} // namespace Xrender

#endif /* XRENDER_RAY_NODE_H_ */