
#include "outline_preview.h"
#include "matrix_view.h"

namespace Xrender
{

    std::vector<rgb24> render_outline_preview(bvh_tree &tree, const camera &cam, std::size_t sample_count)
    {
        const auto width = cam.get_image_width();
        const auto height = cam.get_image_height();
        vecf pos, dir;
        intersection inter;

        // init image
        std::vector<Xrender::rgb24> data{width * height, rgb24::from_uint(0u, 0u, 0u)};
        auto view = Xrender::matrix_view{data, width, height};

        for (auto h = 0u; h < height; ++h)
        {
            for (auto w = 0u; w < width; ++w)
            {
                vecf estimator{0.f, 0.f, 0.f};

                for (auto i = 0u; i < sample_count; ++i)
                {
                    cam.sample_ray(w, h, pos, dir);

                    if (tree.intersect_ray(pos, dir, inter))
                        estimator += material_preview_color(inter.triangle->mtl);
                }

                view(w, h) = rgb24::from_vecf(estimator / sample_count);
            }
        }

        (void)sample_count;
        return data;
    }

} // namespace Xrender