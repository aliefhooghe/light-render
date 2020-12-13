

#include <iostream>

#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "outline_preview.h"
#include "camera.h"

    
int main()
{   
    //auto model = Xrender::wavefront_obj_load("../../../render/Rendus/Nefertiti/nef.obj");
    auto model = Xrender::wavefront_obj_load("../untitled.obj");
    Xrender::bvh_tree tree{model};
    
    const auto width = 576;
    const auto height = 384;

    // Init camera
    auto camera = Xrender::camera::from_focus_distance(width, height, 36E-3f, 24E-3f, 2E-3, 25E-3, 3.1f); //Xrender::camera::from_sensor_lens_distance(width, height, 36E-3f, 24E-3f, 0, 25E-3f, 25E-3);

    // Render preview
    auto data = Xrender::render_outline_preview(tree, camera, 1u);


    Xrender::bitmap_write("out.bmp", data, width, height);
    return 0;
}