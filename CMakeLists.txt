cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cumsum)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
option(USE_CUDA "Use CUDA" OFF)
option(CMAKE_CUDA_ARCHITECTURES "CUDA arch" 75)

add_library(add SHARED "add.cpp")
target_compile_features(add PRIVATE cxx_std_17)
target_link_libraries(add "${TORCH_LIBRARIES}")
