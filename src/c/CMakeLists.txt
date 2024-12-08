set(PHYSCOMP_HEADERS
  physcomp_c.h
)

set(PHYSCOMP_SOURCES
  physcomp_c.cpp
)

add_library(physcomp_c SHARED ${PHYSCOMP_SOURCES} ${PHYSCOMP_HEADERS})

set_target_properties(physcomp_c PROPERTIES
  VERSION 0.0.2
  OUTPUT_NAME physcomp_c
  SOVERSION 0
  PUBLIC_HEADER "physcomp_c.h"
)

target_include_directories(physcomp_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(physcomp_c PUBLIC
  $<BUILD_INTERFACE:LIBPGO_C_BUILD_DLL>
  $<INSTALL_INTERFACE:LIBPGO_C_LOAD_DLL>
)

target_link_libraries(physcomp_c PUBLIC pgo_c)
foreach(lib ${PGO_GLOBAL_LIBRARY_TARGETS})
  if(TARGET ${lib})
    target_link_libraries(physcomp_c PRIVATE ${lib})
  endif()
endforeach()

add_flag_poxis(physcomp_c PRIVATE -fvisibility=hidden)
add_link_flag(physcomp_c GNU All -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/physcomp_c.version)
add_link_flag(physcomp_c AppleClang All -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/physcomp_c.symbol.txt)

# pog_c install
include(CMakePackageConfigHelpers)

install(TARGETS physcomp_c
  EXPORT physcomp_cTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# create (temporary) export target file during build
export(EXPORT physcomp_cTargets
  NAMESPACE pgo::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/physcomp_cTargets.cmake)

# install export target, config and version files for find_package
install(EXPORT physcomp_cTargets
  NAMESPACE pgo::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pgo/")

if(PGO_ENABLE_PYTHON)
  add_library(physcomp_c_static STATIC ${PHYSCOMP_SOURCES} ${PHYSCOMP_HEADERS})
  target_include_directories(physcomp_c_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(physcomp_c_static PUBLIC pgo_c_static)
endif()