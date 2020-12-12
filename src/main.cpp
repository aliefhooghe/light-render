

#include <iostream>
#include <chrono>

#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "matrix_view.h"
#include "camera.h"

  // const auto start = std::chrono::steady_clock::now();
    // Xrender::bvh_tree tree{model};
    // const auto end = std::chrono::steady_clock::now();
    // const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // const auto sec_duration = (float)duration / 1000.f;
    // std::cout << "Took " << sec_duration << "seconds" << std::endl;


    
int main()
{   
    auto model = Xrender::wavefront_obj_load("../untitled.obj");
    //auto model = Xrender::wavefront_obj_load("../../../render/Rendus/cube/untitled.obj");
    Xrender::bvh_tree tree{model};
    
    const auto width = 576;
    const auto height = 384;

    // Init camera
    auto camera = Xrender::camera::from_sensor_lens_distance(width, height, 36E-3f, 24E-3f, 0.0, 25E-3f, 25E-3);

    // Init img
    const auto color = Xrender::make_rgb24(0.f);
    std::vector<Xrender::rgb24> data{width*height, color};
    auto view = Xrender::matrix_view{data, width, height};

    Xrender::vecf pos, dir;
    Xrender::intersection inter;

    // For each pixel
    for (auto h = 0u; h < height; ++h) {
        for (auto w = 0u; w < width; ++w) {
            camera.sample_ray(w, h, pos, dir);

            if (tree.intersect_ray(pos, dir, inter))
                view(w, h) = Xrender::make_rgb24(inter.distance / 10.f);
        }
    }

    Xrender::bitmap_write("out.bmp", data, width, height);
    return 0;
}