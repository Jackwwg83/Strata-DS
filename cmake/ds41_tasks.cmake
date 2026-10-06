# cmake/ds41_tasks.cmake - the open kernel tasks (ds41/tasks/K*.md). Each task is one library built from one
# replaceable source file, plus its fixed acceptance test. Included at the end of cmake/ds41.cmake.
foreach(task k2_fp8_gemm k3_sparse_attn k5_indexer k7_hc k8_router k10_exl3_moe k13_sparse_attn_prefill k14_indexer_prefill)
  add_library(ds41_${task} STATIC src/ds41/kernels/${task}.cu)
  target_link_libraries(ds41_${task} PUBLIC strata_ds41_engine)
  add_executable(${task}_test src/ds41/tests/${task}_test.cu)
  target_link_libraries(${task}_test PRIVATE ds41_${task} strata_ds41_engine)
  add_test(NAME ${task}_test COMMAND ${task}_test)
  set_tests_properties(${task}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endforeach()
# the engine calls the task kernels (K2, K3, K5, K7, K8, K10, K13, K14; K1 lives in strata_ds41): a merged winner
# speeds it up
target_link_libraries(strata_ds41_engine PUBLIC ds41_k2_fp8_gemm ds41_k3_sparse_attn ds41_k5_indexer ds41_k7_hc
                      ds41_k8_router ds41_k10_exl3_moe ds41_k13_sparse_attn_prefill ds41_k14_indexer_prefill)

# K11: the CPU expert kernel (third_party/exllamav3_moe, inside strata_ds41_engine) is the replaceable file
add_executable(k11_cpu_moe_test src/ds41/tests/k11_cpu_moe_test.cpp)
target_link_libraries(k11_cpu_moe_test PRIVATE strata_ds41_engine)
add_test(NAME k11_cpu_moe_test COMMAND k11_cpu_moe_test)
set_tests_properties(k11_cpu_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K1c: the decode GEMV itself (src/ds41/kernels/fp8_gemv.cu, library strata_ds41) is the replaceable file
add_executable(k1c_fp8_gemv_test src/ds41/tests/k1c_fp8_gemv_test.cu)
target_link_libraries(k1c_fp8_gemv_test PRIVATE strata_ds41 strata_ds41_engine)
add_test(NAME k1c_fp8_gemv_test COMMAND k1c_fp8_gemv_test)
set_tests_properties(k1c_fp8_gemv_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K12: prefill experts (rows grouped by expert); the test uses K10 as its reference. cuBLAS is available to it.
add_library(ds41_k12_exl3_moe_prefill STATIC src/ds41/kernels/k12_exl3_moe_prefill.cu)
target_link_libraries(ds41_k12_exl3_moe_prefill PUBLIC strata_ds41_engine CUDA::cublas)
add_executable(k12_exl3_moe_prefill_test src/ds41/tests/k12_exl3_moe_prefill_test.cu)
target_link_libraries(k12_exl3_moe_prefill_test PRIVATE ds41_k12_exl3_moe_prefill ds41_k10_exl3_moe strata_ds41_engine)
add_test(NAME k12_exl3_moe_prefill_test COMMAND k12_exl3_moe_prefill_test)
set_tests_properties(k12_exl3_moe_prefill_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
