set(PHYSCOMP_HEADERS
 src/physcomp_c.h
)

set(PHYSCOMP_SOURCES
 src/physcomp_c.cpp
)

add_library(physcomp_c STATIC ${PHYSCOMP_SOURCES} ${PHYSCOMP_HEADERS})

target_include_directories(physcomp_c PUBLIC
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../src/c/>
 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(physcomp_c PUBLIC
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
)

foreach(lib ${PGO_GLOBAL_LIBRARY_TARGETS})
 if(TARGET ${lib})
   target_link_libraries(physcomp_c PRIVATE ${lib})
 endif()
endforeach()

# target_link_libraries(pgo_c_static PUBLIC physcomp_c)
