#
# Creation of dynamic graph plugin/entity.
#

set(lib_name entities)
# then we build the different entities in a dynamic-graph plugin.
set(cpp_source_files
    dg_pytorch.cpp
)

add_library(${lib_name} SHARED ${cpp_source_files})
# Add the include dependencies.
target_include_directories(
  ${lib_name} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>)
# Dependencies.
target_link_libraries(${lib_name} dynamic-graph::dynamic-graph)
target_link_libraries(${lib_name} dynamic-graph-python::dynamic-graph-python)
target_link_libraries(${lib_name} Eigen3::Eigen)
target_link_libraries(${lib_name} real_time_tools::real_time_tools)
target_link_libraries(${lib_name} "${TORCH_LIBRARIES}")

# Install the plugin python bindings.
install_dynamic_graph_plugin_python_bindings(${lib_name})

# Install the plugin.
get_dynamic_graph_plugin_install_path(plugin_install_path)
install(
  TARGETS ${lib_name}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION ${plugin_install_path}
  ARCHIVE DESTINATION ${plugin_install_path}
  RUNTIME DESTINATION ${plugin_install_path}
  INCLUDES
  DESTINATION include)
