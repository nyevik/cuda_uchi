add_executable(isprime isprime.cu)
target_compile_features(isprime PRIVATE cxx_std_23)
target_compile_options(isprime PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
)
set_target_properties(isprime PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(simd_ex_mod_all DEPENDS isprime)
