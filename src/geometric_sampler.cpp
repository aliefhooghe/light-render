
#include <iostream>
#include <algorithm>

#include "random_generator.h"
#include "geometric_sampler.h"
#include "matrix_view.h"

namespace Xrender
{

    static auto test_prob(float estimator_geometric_coeff, const vecf &estimator_brdf_coeff)
    {
        constexpr auto threshold = 10.f / 255.f;
        constexpr auto min_prob = 0.2f;
        constexpr auto max_prob = 2.f;
        constexpr auto a = (max_prob - min_prob) / (1.f - threshold);
        constexpr auto b = max_prob - a;

        const auto estimator_norm = estimator_geometric_coeff * estimator_brdf_coeff.norm();

        if (estimator_norm < threshold)
            return min_prob;
        else
            return std::clamp(a * estimator_norm + b, 0.f, 1.f);

    }

    vecf path_sample_geometric(
        const bvh_tree &tree,
        const vecf &start_pos,
        const vecf &start_dir,
        std::size_t max_bounce)
    {
        intersection inter;
        vecf pos = start_pos;
        vecf dir = start_dir;
        vecf estimator_brdf_coeff = {1.f, 1.f, 1.f};
        float estimator_geometric_coeff = 1.f;

        for (auto i = 0u; i < max_bounce; )
        {
            const auto prob = test_prob(estimator_geometric_coeff, estimator_brdf_coeff);

            if (rand::uniform() <= prob)
            {
                estimator_geometric_coeff /= prob;
            }
            else
            {
                break;
            }

            if (tree.intersect_ray(pos, dir, inter))
            {
                if (is_source(inter.triangle->mtl))
                {
                    return estimator_brdf_coeff * estimator_geometric_coeff;
                }
                else
                {
                    const auto next_dir = rand::unit_hemisphere_uniform(inter.normal);
                    estimator_geometric_coeff *= dot(next_dir, inter.normal);
                    estimator_brdf_coeff *= brdf(inter.triangle->mtl, inter.normal, dir, next_dir);
                    pos = inter.pos;
                    dir = next_dir;
                }
            }
            else
            {
                break;
            }
        }

        return {0.f, 0.f, 0.f};
    }

    std::vector<vecf> mc_naive(
        const bvh_tree &tree,
        const camera &cam,
        std::size_t sample_pp_count,
        std::size_t max_bounce)
    {
        const auto width = cam.get_image_width();
        const auto height = cam.get_image_height();

        std::vector<vecf> sensor{width * height, {0.f, 0.f, 0.f}};
        auto view = Xrender::matrix_view{sensor, width, height};

        std::cout << "RENDER " << sample_pp_count << "SPP" << std::endl;

// for each pixel
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (auto h = 0u; h < height; ++h)
        {
            for (auto w = 0u; w < width; ++w)
            {
                vecf pixel_estimator{0.f, 0.f, 0.f};
                vecf pos, dir;

                for (auto i = 0; i < sample_pp_count; ++i)
                {
                    cam.sample_ray(w, h, pos, dir);
                    pixel_estimator += path_sample_geometric(tree, pos, dir, max_bounce);
                }

                view(w, h) = 2.f * (pixel_estimator / static_cast<float>(sample_pp_count));
            }
        }

        return sensor;
    }

} // namespace Xrender
