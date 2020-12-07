#
# Copyright (c) 2019, New York University and Max Planck Gesellschaft.
#
# License BSD-3 clause
#

#
# set up the project
#
cmake_minimum_required(VERSION 3.10.2)

project(dg_pytorch)

# Using C++17
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# We assume libtorch is installed under /opt/.
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

#
# Dependencies
#

# In order to quiet the find_package(dynamic-graph[-python] REQUIRED)
set(CHECK_MINIMAL_CXX_STANDARD ON)

# depend on ament macros
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(mpi_cmake_modules REQUIRED)
find_package(dynamic-graph REQUIRED)
find_package(dynamic-graph-python REQUIRED)
find_package(real_time_tools REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

#
# Prepare the export
#
# Export the targets.
ament_export_interfaces(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
# Export de dependencies.
ament_export_dependencies(mpi_cmake_modules dynamic-graph dynamic-graph-python
                          Eigen3 Torch real_time_tools)

#
# manage the creation of the libraries and executables #
#
add_subdirectory(src)


#
# Install the package
#
install(DIRECTORY include/ DESTINATION include)

install(
  TARGETS ${all_targets}
  EXPORT ${all_target_exports}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

#
# Building documentation.
#
add_documentation()

#
# Export as an ament_package
#
ament_package()