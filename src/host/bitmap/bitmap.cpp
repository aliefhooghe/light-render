
#include <stdexcept>
#include <fstream>

#include "bitmap.h"

namespace Xrender {

    struct bitmap_header {
        const uint8_t magic[2] = {'B', 'M'};
        uint32_t file_size;
        const uint32_t reserved = 0u;
        const uint32_t offset = 12 + 0x28 + 24; // sortie de ton cul ?
    } __attribute__((__packed__));

    struct image_header {
        const uint32_t img_hdr_size = 0x28u;
        uint32_t width;
        uint32_t height;
        const uint16_t nb_plan = 1u;
        const uint16_t depth = 24u;
        const uint32_t compresion = 0u;
        uint32_t img_size;
    } __attribute__((__packed__));


    void bitmap_write(
        const std::filesystem::path& path, const std::vector<rgb24>& data,
        unsigned int width, unsigned int height)
    {
        if (width * height != data.size())
            throw std::invalid_argument("bitmap write : size mismatch");

        std::ofstream stream{path, std::ios::binary};

        if (!stream.is_open())
            throw std::runtime_error("bitmap write : open file");

        bitmap_header bmp_header;
        image_header img_header;
        const auto line_size = sizeof(rgb24) * width;
        const auto pad_size = (4u - (line_size % 4u)) % 4; // line must be padded to be a multiple of 4 byte
        const auto img_size = (line_size + pad_size) * height;

        // Fill headers
        bmp_header.file_size = bmp_header.offset + img_size;
        img_header.width = width;
        img_header.height = height;
        img_header.img_size = img_size;

        // write headers
        stream.write(reinterpret_cast<const char*>(&bmp_header), sizeof(bitmap_header));
        stream.write(reinterpret_cast<const char*>(&img_header), sizeof(image_header));

        // write data : each line + zero padd
        stream.seekp(bmp_header.offset);

        for (auto i = 0u; i < height; ++i)
        {
            stream.write(reinterpret_cast<const char*>(&data[i * width]), line_size);
            if (pad_size != 0u)
            {
                const uint32_t zero = 0u;
                stream.write(reinterpret_cast<const char*>(&zero), pad_size);
            }
        }
    }
}