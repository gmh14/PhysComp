add_library(pypgo_physcomp MODULE src/python/pyphyscomp/pyphyscomp.cpp)
target_link_libraries(pypgo_physcomp PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_link_libraries(pypgo_physcomp PRIVATE physcomp_core)
target_compile_definitions(pypgo_physcomp PRIVATE VERSION_INFO=${PYPGO_VERSION_INFO})

pybind11_extension(pypgo_physcomp)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
  # Strip unnecessary sections of the binary on Linux/macOS
  pybind11_strip(pypgo_physcomp)
endif()

set_target_properties(pypgo_physcomp PROPERTIES CXX_VISIBILITY_PRESET "hidden"
  CUDA_VISIBILITY_PRESET "hidden")