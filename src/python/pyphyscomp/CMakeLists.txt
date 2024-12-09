message(STATUS "building physcomp")

set(PGO_PYTHON_SOURCES "${PGO_PYTHON_SOURCES};pyphyscomp" CACHE INTERNAL "" FORCE)

target_sources(pypgo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pyphyscomp.cpp")
target_link_libraries(pypgo PRIVATE physcomp_c_static)