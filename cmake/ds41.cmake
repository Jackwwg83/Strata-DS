# DeepSeek V4.1 kernels are separate from the upstream engine targets.
add_library(strata_ds41 STATIC src/ds41/kernels/fp8_gemv.cu)
target_include_directories(strata_ds41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_ds41 PUBLIC CUDA::cudart)
set_target_properties(strata_ds41 PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
# Keep subnormals and the reference's FP32 division/scale rounding, regardless of parent flags.
target_compile_options(strata_ds41 PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true>")

if(STRATA_BUILD_TESTS)
    add_executable(fp8_gemv_parity src/ds41/kernels/fp8_gemv_parity.cpp)
    target_link_libraries(fp8_gemv_parity PRIVATE strata_ds41)
    set_target_properties(fp8_gemv_parity PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME fp8_gemv_parity COMMAND fp8_gemv_parity --selftest)
    set_tests_properties(fp8_gemv_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()
