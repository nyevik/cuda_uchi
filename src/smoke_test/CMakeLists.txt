set(SMOKE_TEST_CUDA_TARGETS
  cuda_smoketest
  simp_dev_query
  simtest
  smcount
)

foreach(target IN LISTS SMOKE_TEST_CUDA_TARGETS)
  add_executable(${target} ${target}.cu)
  target_compile_features(${target} PRIVATE cxx_std_23)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
  )
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endforeach()

add_custom_target(smoke_test_all DEPENDS ${SMOKE_TEST_CUDA_TARGETS})
