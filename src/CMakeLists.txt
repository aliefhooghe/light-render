cmake_minimum_required (VERSION 3.1)
project(Xrender CXX)

include(CheckCXXCompilerFlag)
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" ON)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE AND OPTIMIZE_FOR_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(XRENDER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/bitmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bitmap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bvh_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bvh_tree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
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
)


add_executable(xrender ${XRENDER_SRC})
target_include_directories(xrender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(xrender PUBLIC OpenMP::OpenMP_CXX)
