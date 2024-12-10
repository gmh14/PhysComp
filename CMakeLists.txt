add_subdirectory(src/c)

if(PGO_ENABLE_PYTHON)
  add_subdirectory(src/python/pyphyscomp)
endif()