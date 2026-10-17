# cmake/ds41_tasks.cmake - the open kernel tasks (ds41/tasks/K*.md). Each task is one library built from one
# replaceable source file, plus its fixed acceptance test. Included at the end of cmake/ds41.cmake.
foreach(task k2_fp8_gemm k3_sparse_attn k5_indexer k7_hc k8_router k10_exl3_moe k13_sparse_attn_prefill k14_indexer_prefill
        k15_hc_prefill k16_mxfp4_moe)
  add_library(ds41_${task} STATIC src/ds41/kernels/${task}.cu)
  target_link_libraries(ds41_${task} PUBLIC strata_ds41_engine)
  add_executable(${task}_test src/ds41/tests/${task}_test.cu)
  target_link_libraries(${task}_test PRIVATE ds41_${task} strata_ds41_engine)
  add_test(NAME ${task}_test COMMAND ${task}_test)
  set_tests_properties(${task}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endforeach()
# K7's split variant (the decode graph's coefficients on a side stream) against hc_mixes_pre, bit for bit
add_executable(k7_split_test src/ds41/tests/k7_split_test.cu)
target_link_libraries(k7_split_test PRIVATE ds41_k7_hc strata_ds41_engine)
add_test(NAME k7_split_test COMMAND k7_split_test)
set_tests_properties(k7_split_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# K8's own GPU regression test (outside the fixed acceptance test): graph replays, exact ties, and negative and
# mixed-sign biased scores against the CPU oracle
add_executable(k8_graph_validation src/ds41/kernels/k8/graph_validation.cu)
target_link_libraries(k8_graph_validation PRIVATE ds41_k8_router strata_ds41_engine)
add_test(NAME k8_graph_validation COMMAND k8_graph_validation)
set_tests_properties(k8_graph_validation PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# the engine calls the task kernels (K2, K3, K5, K7, K8, K10, K13, K14, K15, K16; K1 lives in strata_ds41): a merged winner
# speeds it up
target_link_libraries(strata_ds41_engine PUBLIC ds41_k2_fp8_gemm ds41_k3_sparse_attn ds41_k5_indexer ds41_k7_hc
                      ds41_k8_router ds41_k10_exl3_moe ds41_k13_sparse_attn_prefill ds41_k14_indexer_prefill
                      ds41_k15_hc_prefill ds41_k16_mxfp4_moe)

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
target_link_libraries(strata_ds41_engine PUBLIC ds41_k12_exl3_moe_prefill)   # prefill computes its experts with it
add_executable(k12_exl3_moe_prefill_test src/ds41/tests/k12_exl3_moe_prefill_test.cu)
target_link_libraries(k12_exl3_moe_prefill_test PRIVATE ds41_k12_exl3_moe_prefill ds41_k10_exl3_moe strata_ds41_engine)
add_test(NAME k12_exl3_moe_prefill_test COMMAND k12_exl3_moe_prefill_test)
set_tests_properties(k12_exl3_moe_prefill_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K17: prefill experts from MXFP4 weights (rows grouped by expert); the test uses K16 as its reference
add_library(ds41_k17_mxfp4_moe_prefill STATIC src/ds41/kernels/k17_mxfp4_moe_prefill.cu)
target_link_libraries(ds41_k17_mxfp4_moe_prefill PUBLIC strata_ds41_engine CUDA::cublas)
target_link_libraries(strata_ds41_engine PUBLIC ds41_k17_mxfp4_moe_prefill)
add_executable(k17_mxfp4_moe_prefill_test src/ds41/tests/k17_mxfp4_moe_prefill_test.cu)
target_link_libraries(k17_mxfp4_moe_prefill_test PRIVATE ds41_k17_mxfp4_moe_prefill ds41_k16_mxfp4_moe
                      strata_ds41_engine)
add_test(NAME k17_mxfp4_moe_prefill_test COMMAND k17_mxfp4_moe_prefill_test)
set_tests_properties(k17_mxfp4_moe_prefill_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K18: one dense projection from EXL3 weights (an EXL3 checkpoint's attention, shared expert and head): K10's GEMV
# for few rows, K12's reconstruction and cuBLAS for many; the test checks one path against the other. The engine runs an
# EXL3 pack's dense linears with it in prefill (decode sizes: K20, which reads the weights 2-3x faster on GB10).
add_library(ds41_k18_exl3_linear STATIC src/ds41/kernels/k18_exl3_linear.cu)
target_link_libraries(ds41_k18_exl3_linear PUBLIC ds41_k10_exl3_moe ds41_k12_exl3_moe_prefill CUDA::cublas)
target_link_libraries(strata_ds41_engine PUBLIC ds41_k18_exl3_linear)
add_executable(k18_exl3_linear_test src/ds41/tests/k18_exl3_linear_test.cu)
target_link_libraries(k18_exl3_linear_test PRIVATE ds41_k18_exl3_linear strata_ds41_engine)
add_test(NAME k18_exl3_linear_test COMMAND k18_exl3_linear_test)
set_tests_properties(k18_exl3_linear_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K19: routed experts from EXL3 weights with sudoingX's streaming exl3_smoe kernels (third_party/exl3_smoe), the GB10
# path for EXL3 packs; the test checks it against a K18 reference and K10, and times both
add_library(ds41_k19_exl3_smoe STATIC src/ds41/kernels/k19_exl3_smoe.cu)
target_link_libraries(ds41_k19_exl3_smoe PUBLIC strata_ds41_engine)
target_link_libraries(strata_ds41_engine PUBLIC ds41_k19_exl3_smoe)
add_executable(k19_exl3_smoe_test src/ds41/tests/k19_exl3_smoe_test.cu)
target_link_libraries(k19_exl3_smoe_test PRIVATE ds41_k19_exl3_smoe ds41_k18_exl3_linear ds41_k10_exl3_moe strata_ds41_engine)
add_test(NAME k19_exl3_smoe_test COMMAND k19_exl3_smoe_test)
set_tests_properties(k19_exl3_smoe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

# K20: dense EXL3 linears at decode sizes with sudoingX's mrow kernel (third_party/exl3_smoe), several per launch; the
# test checks it against K18 and times both
add_library(ds41_k20_exl3_mrow STATIC src/ds41/kernels/k20_exl3_mrow.cu)
target_link_libraries(ds41_k20_exl3_mrow PUBLIC strata_ds41_engine)
target_link_libraries(strata_ds41_engine PUBLIC ds41_k20_exl3_mrow)
add_executable(k20_exl3_mrow_test src/ds41/tests/k20_exl3_mrow_test.cu)
target_link_libraries(k20_exl3_mrow_test PRIVATE ds41_k20_exl3_mrow ds41_k18_exl3_linear strata_ds41_engine)
add_test(NAME k20_exl3_mrow_test COMMAND k20_exl3_mrow_test)
set_tests_properties(k20_exl3_mrow_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
