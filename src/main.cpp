

#include <iostream>
#include <chrono>

#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "outline_preview.h"
#include "camera.h"

  // const auto start = std::chrono::steady_clock::now();
    // Xrender::bvh_tree tree{model};
    // const auto end = std::chrono::steady_clock::now();
    // const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // const auto sec_duration = (float)duration / 1000.f;
    // std::cout << "Took " << sec_duration << "seconds" << std::endl;


    
int main()
{   
    auto model = Xrender::wavefront_obj_load("../../../render/Rendus/Nefertiti/nef.obj");
    //auto model = Xrender::wavefront_obj_load("../../../render/Rendus/cube/untitled.obj");
    Xrender::bvh_tree tree{model};
    
    const auto width = 576;
    const auto height = 384;

    // Init camera
    auto camera = Xrender::camera::from_sensor_lens_distance(width, height, 36E-3f, 24E-3f, 0, 25E-3f, 25E-3);

    // Render preview
    auto data = Xrender::render_outline_preview(tree, camera, 10);


    Xrender::bitmap_write("out.bmp", data, width, height);
    return 0;
}