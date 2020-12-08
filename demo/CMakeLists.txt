cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pd_example)

# We assume libtorch is installed under /opt/.
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(pd pd.cpp)
target_link_libraries(pd "${TORCH_LIBRARIES}")
target_link_libraries(pd Eigen3::Eigen)
set_property(TARGET pd PROPERTY CXX_STANDARD 14)
