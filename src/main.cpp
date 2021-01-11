

#include <iostream>
#include <chrono>

#include "wavefront_obj.h"
#include "bvh_tree.h"
#include "bitmap.h"
#include "camera.h"

#include "outline_preview.h"
#include "geometric_sampler.h"

int main()
{
    auto model = Xrender::wavefront_obj_load("../../untitled.obj");
    Xrender::bvh_tree tree{model};

    const auto width = 576;
    const auto height = 384;

    // Init camera
    auto camera = Xrender::camera::from_focus_distance(width, height, 36E-3f, 24E-3f, 10E-3, 50E-3, 3.f);

    // Render an outlie preview
    auto preview = Xrender::render_outline_preview(tree, camera, 1);
    Xrender::bitmap_write("preview.bmp", preview, width, height);

    std::cout << "Rendered Preview.\nRendering lights" << std::endl;

    auto start = std::chrono::steady_clock::now();
    auto rendered_sensor = Xrender::mc_naive(tree, camera, 12000);
    auto end = std::chrono::steady_clock::now();

    std::cout
        << "Render took "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " seconds" << std::endl;

    std::vector<Xrender::rgb24> rendered_bitmap(width * height);
    std::transform(
        rendered_sensor.begin(),
        rendered_sensor.end(),
        rendered_bitmap.begin(),
        Xrender::rgb24::from_vecf);

    Xrender::bitmap_write("rendered.bmp", rendered_bitmap, width, height);

    return 0;
}
