set(PHYSCOMP_HEADERS
  src/core/physcomp_c.h
)

set(PHYSCOMP_SOURCES
  src/core/physcomp_c.cpp
)

add_library(physcomp_core STATIC ${PHYSCOMP_SOURCES} ${PHYSCOMP_HEADERS})
target_include_directories(physcomp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/core/")

foreach(lib ${PGO_GLOBAL_LIBRARY_TARGETS})
  if(TARGET ${lib})
    target_link_libraries(physcomp_core PRIVATE ${lib})
  endif()
endforeach()

if(PGO_ENABLE_PYTHON)
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
endif()