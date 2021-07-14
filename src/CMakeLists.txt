cmake_minimum_required (VERSION 3.8)
project(Xrender CXX CUDA)

include(CheckCXXCompilerFlag)
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" ON)

# CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
# if(COMPILER_SUPPORTS_MARCH_NATIVE AND OPTIMIZE_FOR_NATIVE)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# endif()

set(XRENDER_CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/bitmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bitmap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bvh_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bvh_tree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/geometric_sampler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/geometric_sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material.h
    ${CMAKE_CURRENT_SOURCE_DIR}/material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_view.h
    ${CMAKE_CURRENT_SOURCE_DIR}/outline_preview.h
    ${CMAKE_CURRENT_SOURCE_DIR}/outline_preview.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face.h
    ${CMAKE_CURRENT_SOURCE_DIR}/random_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/random_generator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ray_intersection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ray_intersection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vec.h
    ${CMAKE_CURRENT_SOURCE_DIR}/wavefront_obj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wavefront_obj.h

    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_outline_preview.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_outline_preview.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_camera.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_mtl.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_mtl.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_geometric_sampler.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_geometric_sampler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_face.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_renderer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_renderer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/rand_operations.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/vector_operations.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/vector_operations.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda_exception.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_bvh.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_texture.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_texture.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gui/abstract_renderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gui/renderer_manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gui/renderer_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gui/renderer_display.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gui/renderer_display.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/gpu_vector_copy.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/image_grid_dim.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/image_grid_dim.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/curand_pool.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/curand_pool.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(xrender ${XRENDER_CORE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)
add_executable(xrender-gui ${XRENDER_CORE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/gui_main.cu)
#add_executable(ray_bench ${XRENDER_CORE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/raycast_benchmark.cpp)

target_link_libraries(xrender PUBLIC OpenMP::OpenMP_CXX OpenGL::GL OpenGL::GLU GLEW::GLEW)
target_link_libraries(xrender-gui SDL2::SDL2 OpenGL::GL OpenGL::GLU GLEW::GLEW)