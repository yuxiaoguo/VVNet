cmake_minimum_required(VERSION 3.5.0)
project(TFGraphics)

add_definitions(-DGOOGLE_CUDA)

find_package(CUDA REQUIRED)
include_directories($ENV{TF_INC})
if (CMAKE_HOST_WIN32)
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include")
else ()
    include_directories("/usr/local/cuda/include")
endif()

set(CMAKE_INSTALL_PREFIX ".")
SET(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../..")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../..")
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "../..")

# set (CMAKE_VERBOSE_MAKEFILE  on)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(cu)
# add_library(viewproj sampling_ops.cc)
#
# cuda_add_library(feat_tsdf_op.cu SHARED feat_tsdf_op.cu)
add_library(tfgraphics SHARED
        common_utils.h sampling_ops.h
        tsdf_via_vox_op.cc
        feat_fusion_vox_op.cc feat_fusion_vox_grad_op.cc
        projection_ops.cc
        sampling_ops.cc sampling_grad_ops.cc)
target_link_libraries(tfgraphics view_projection_kernel)

install(TARGETS tfgraphics
        RUNTIME DESTINATION ../../bin
        LIBRARY DESTINATION ../../bin
        ARCHIVE DESTINATION ../../bin)
