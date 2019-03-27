cmake_minimum_required(VERSION 3.5.0)
project(ViewProjection)

include_directories('/usr/local/cuda/include')

set(CMAKE_CXX_STANDARD 11)
set_source_files_properties(projection_ops.cu.cc PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(sampling_ops.cu.cc PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(feat_fusion_op.cu.cc PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "..")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "..")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "..")

cuda_add_library(view_projection_kernel
        kernel_utils.h
        projection_ops.cu.cc
        sampling_ops.cu.cc
        feat_fusion_op.cu.cc SHARED)

install(TARGETS view_projection_kernel
        RUNTIME DESTINATION ../../bin
        LIBRARY DESTINATION ../../bin
        ARCHIVE DESTINATION ../../bin)
