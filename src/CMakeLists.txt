file(GLOB module_candidates CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*")

set(CUDA_UCHI_MODULE_DIRS "")
foreach(candidate IN LISTS module_candidates)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${candidate}")
    list(APPEND CUDA_UCHI_MODULE_DIRS "${candidate}")
  endif()
endforeach()
list(SORT CUDA_UCHI_MODULE_DIRS)

foreach(module IN LISTS CUDA_UCHI_MODULE_DIRS)
  set(module_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/${module}")
  set(module_binary_dir "${CMAKE_BINARY_DIR}/${module}")
  file(MAKE_DIRECTORY "${module_binary_dir}")

  if(EXISTS "${module_source_dir}/CMakeLists.txt")
    add_subdirectory("${module_source_dir}" "${module_binary_dir}")
  else()
    string(MAKE_C_IDENTIFIER "${module}" module_id)
    add_custom_target("stub_${module_id}"
      COMMENT "Stub for src/${module}: add a CMakeLists.txt and targets here.")
    message(STATUS "src/${module} has no CMakeLists.txt; created build/${module} and a stub target.")
  endif()
endforeach()
