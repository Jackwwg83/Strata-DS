# cmake/ds41_tasks.cmake - the open kernel tasks (ds41/tasks/K*.md). Each task is one library built from one
# replaceable source file, plus its fixed acceptance test. Included at the end of cmake/ds41.cmake.
foreach(task k2_fp8_gemm k3_sparse_attn k5_indexer k7_hc k8_router k10_exl3_moe)
  add_library(ds41_${task} STATIC src/ds41/kernels/${task}.cu)
  target_link_libraries(ds41_${task} PUBLIC strata_ds41_engine)
  add_executable(${task}_test src/ds41/tests/${task}_test.cu)
  target_link_libraries(${task}_test PRIVATE ds41_${task} strata_ds41_engine)
  add_test(NAME ${task}_test COMMAND ${task}_test)
  set_tests_properties(${task}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endforeach()

# K1c: the decode GEMV itself (src/ds41/kernels/fp8_gemv.cu, library strata_ds41) is the replaceable file
add_executable(k1c_fp8_gemv_test src/ds41/tests/k1c_fp8_gemv_test.cu)
target_link_libraries(k1c_fp8_gemv_test PRIVATE strata_ds41 strata_ds41_engine)
add_test(NAME k1c_fp8_gemv_test COMMAND k1c_fp8_gemv_test)
set_tests_properties(k1c_fp8_gemv_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
