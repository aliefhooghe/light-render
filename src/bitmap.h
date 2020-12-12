#ifndef XRENDER_BITMAP_H_
#define XRENDER_BITMAP_H_

#include <vector>
#include <stdint.h>
#include <filesystem>

namespace Xrender {

    struct rgb24  {
        uint8_t b;
        uint8_t g;
        uint8_t r;
    } __attribute__((__packed__));

    inline rgb24 make_rgb24(uint8_t red, uint8_t green, uint8_t blue) noexcept
    {
        return rgb24{blue, green, red};
    }

    inline rgb24 make_rgb24(float red, float green, float blue) noexcept
    {
        return rgb24{
            static_cast<uint8_t>(red * 255.f), 
            static_cast<uint8_t>(green * 255.f),
            static_cast<uint8_t>(red * 255.f)};
    }

    inline rgb24 make_rgb24(uint8_t monochrome) noexcept
    {
        return rgb24{monochrome, monochrome, monochrome};
    };

    inline rgb24 make_rgb24(float monochrome) noexcept
    {
        const auto val = static_cast<uint8_t>(monochrome * 255.f);
        return rgb24{val, val, val};
    };
    
    void bitmap_write(
        const std::filesystem::path& path, const std::vector<rgb24>& data,
        unsigned int width, unsigned int height);

}

#endif