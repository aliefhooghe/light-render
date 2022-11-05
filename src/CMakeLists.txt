cmake_minimum_required (VERSION 3.8)
project(Xrender CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(xrender
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/common/abstract_renderer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/common/gpu_texture.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/common/gpu_texture.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/common/renderer_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/common/renderer_manager.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/image_developers/average_image_developer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/image_developers/average_image_developer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/image_developers/gamma_image_developer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/image_developers/gamma_image_developer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/bvh_tree_traversal.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/bvh_tree.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/camera.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/camera.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/face_intersection.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/face.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/float3_operators.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/material_brdf.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/material.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/model/material.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/renderers/naive_mc_renderer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/renderers/naive_mc_renderer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/renderers/preview_renderer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/renderers/preview_renderer.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/cuda_exception.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/curand_helper.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/curand_pool.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/curand_pool.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/device_probing.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/device_probing.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/gpu_vector_copy.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/image_grid_dim.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utils/image_grid_dim.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bitmap/bitmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bitmap/bitmap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/bvh_builder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/bvh_builder.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/host_bvh_tree.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/host_bvh_tree.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/random_generator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/bvh_builder/random_generator.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/camera_handling/camera_configuration.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/camera_handling/camera_configuration.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/configuration/configuration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/configuration/configuration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/face_builder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/face_builder.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/material_builder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/material_builder.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/wavefront_obj.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/wavefront_obj.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/wavefront_obj.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/model_loader/wavefront_obj.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/host/gui/renderer_application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/gui/renderer_application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/host/renderer_frontend/renderer_frontend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/host/renderer_frontend/renderer_frontend.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/host/utils/chronometer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/host/utils/chronometer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(xrender PUBLIC
    SDL2::SDL2
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW
    nlohmann_json::nlohmann_json
    nlohmann_json)
