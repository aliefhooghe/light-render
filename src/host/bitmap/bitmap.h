#ifndef XRENDER_BITMAP_H_
#define XRENDER_BITMAP_H_

#include <algorithm>
#include <vector>
#include <stdint.h>
#include <filesystem>

namespace Xrender {

    struct rgb24  {

        static rgb24 from_uint(uint8_t red, uint8_t green, uint8_t blue) noexcept
        {
            return {blue, green, red};
        }

        static rgb24 from_float(float red, float green, float blue)
        {
            return from_uint(
                static_cast<uint8_t>(std::clamp(red, 0.f, 1.f) * 255.f),
                static_cast<uint8_t>(std::clamp(green, 0.f, 1.f) * 255.f),
                static_cast<uint8_t>(std::clamp(blue, 0.f, 1.f) * 255.f));
        }

        uint8_t b;
        uint8_t g;
        uint8_t r;
    } __attribute__((__packed__));

    void bitmap_write(
        const std::filesystem::path& path, const std::vector<rgb24>& data,
        unsigned int width, unsigned int height);

}

#endif