#ifndef XRENDER_RENDERING_STATUS_H_
#define XRENDER_RENDERING_STATUS_H_

#include <cstdlib>

namespace Xrender
{
    struct rendering_status
    {
        float spp_per_second{1.f};
        size_t total_integrated_sample{0u};
        size_t last_sample_count{0u};
    };
}
#endif
