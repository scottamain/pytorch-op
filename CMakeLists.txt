cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cumsum)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

add_library(cumsum SHARED "cumsum.cpp")
target_compile_features(cumsum PRIVATE cxx_std_17)
target_link_libraries(cumsum "${TORCH_LIBRARIES}")