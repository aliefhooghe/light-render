
#include "random_generator.h"

namespace Xrender {

    namespace rand {

        thread_local static std::mt19937 __rand_generator__(std::random_device{}());

        float uniform(float min, float max) noexcept
        {
            std::uniform_real_distribution<float> distribution{min, max};
            return distribution(__rand_generator__);
        }

        int integer(int min, int max)
        {
            std::uniform_int_distribution<int> distribution{min, max};
            return distribution(__rand_generator__);
        }

        vecf unit_sphere_uniform() noexcept
        {
            float u1, u2, s;

            do
            {
                u1 = uniform(-1.0f, 1.0f);
                u2 = uniform(-1.0f, 1.0f);
            } while ((s = (u1 * u1 + u2 * u2)) >= 1.0f);

            const float tmp = 2.0f * std::sqrt(1.0f - s);

            return {u1 * tmp,
                    u2 * tmp,
                    1.0f - (2.0f * s)};
        }

        vecf unit_hemisphere_uniform(const vecf& normal)
        {
            const vecf ret = unit_sphere_uniform();

            if (dot(ret, normal) < 0.0f)
                return -ret;
            else
                return ret;
        }

        void unit_disc_uniform(float& x, float &y)
        {
            do{
                x = uniform(-1.f, 1.f);
                y = uniform(-1.f, 1.f);
            }while(x*x + y*y > 1.f);
        }
    } 

} /* namespace Xrender */

