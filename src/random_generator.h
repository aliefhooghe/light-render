#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_

#include <random>
#include "vec.h"

namespace Xrender {

    namespace rand {

        /**
         * \brief sample a float uniformly in the interval [min, max]
         */
        float uniform(float min = 0.f, float max = 1.f) noexcept;

        /**
         * \brief sample an integer uniformly in the inteval [min, max]
         */
        int integer(int min, int max);

        /**
         * \brief sample a direction uniformly according to the solid angle measure
         */
        vecf unit_sphere_uniform() noexcept;

        /**
         * \brief sample a direction uniformly according to the solid angle measure
         * in the hemisphere oriented by 'normal'
         */
        vecf unit_hemisphere_uniform(const vecf& normal);
        
        /**
         * \brief sample a point on the unit radius disc
         */
        void unit_disc_uniform(float& x, float &y);
    }

} /* namespace Xrender */

#endif