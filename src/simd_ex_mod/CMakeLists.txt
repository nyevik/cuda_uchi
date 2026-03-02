set(SIMD_EX_MOD_CUDA_TARGETS
  isprime
  isprime_appl_org_comp
  silly_vector_add
  vector_add
)

foreach(target IN LISTS SIMD_EX_MOD_CUDA_TARGETS)
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

add_custom_target(simd_ex_mod_all DEPENDS ${SIMD_EX_MOD_CUDA_TARGETS})
