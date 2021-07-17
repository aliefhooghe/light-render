#ifndef XRENDER_CHRONOMETER_H_
#define XRENDER_CHRONOMETER_H_

#include <chrono>

namespace Xrender
{


    class chronometer
    {
    public:
        void start() { _start = std::chrono::steady_clock::now(); }
        std::chrono::milliseconds stop()
        {
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - _start);
        }
    private:
        std::chrono::steady_clock::time_point _start;
    };

}

#endif