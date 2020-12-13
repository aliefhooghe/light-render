
#include <chrono>
#include <iostream>

#include "bvh_tree.h"
#include "random_generator.h"
#include "wavefront_obj.h"

using namespace Xrender;

static std::size_t raycast_benchmark(const bvh_tree &tree, std::size_t raycast_count)
{
    constexpr vecf origin = {0.f, 0.f, 0.f};
    constexpr vecf origin_normal = {0.f, 1.f, 0.f};

    intersection inter;
    unsigned int ray_length = 0;
    vecf pos = origin;
    vecf dir = rand::unit_hemisphere_uniform(origin_normal);

    const auto start = std::chrono::steady_clock::now();
    

    for (auto i = 0u; i < raycast_count; ++i)
    {
        if (ray_length <= 4u && tree.intersect_ray(pos, dir, inter))
        {
            ray_length++;
            pos = inter.pos;
            dir = rand::unit_hemisphere_uniform(inter.normal);
        }
        else
        {
            // return to origin
            ray_length = 0u;
            vecf pos = origin;
            vecf dir = rand::unit_hemisphere_uniform(origin_normal);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

static void usage(const char *argv0)
{
    std::cout << "usage : " << argv0 << " <model.obj> <raycast_count>" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        usage(argv[0]);    
    }
    else
    {
        const std::filesystem::path obj_path{argv[1]};
        const std::size_t raycast_count = std::stoi(argv[2]);

        std::cout << "Load " << argv[1] << std::endl;
        auto model = wavefront_obj_load(argv[1]);
        std::cout << "Build BVH tree..." << std::endl;
        bvh_tree tree(model);
        std::cout << "Run benchmark (" << raycast_count << " ray casts)..." << std::endl;
        const auto duration_ms = raycast_benchmark(tree, raycast_count);

        if (duration_ms == 0u)
        {
            std::cout << "Duration is less than 1 ms" << std::endl;
        }
        else 
        {
            const auto ray_per_second = (1000u * raycast_count) / duration_ms; 
            std::cout 
                <<  "# Results #\n" 
                    "duration           : " << duration_ms << " ms\n"
                    "raycast per second : " << ray_per_second << std::endl; 
        }
    }   

    return 0;
}