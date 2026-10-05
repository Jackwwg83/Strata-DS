# cmake/ds41_tasks.cmake - the open kernel tasks (ds41/tasks/K*.md). Each task is one library built from one
# replaceable source file, plus its fixed acceptance test. Included at the end of cmake/ds41.cmake.
foreach(task k2_fp8_gemm k3_sparse_attn k5_indexer k7_hc k8_router)
  add_library(ds41_${task} STATIC src/ds41/kernels/${task}.cu)
  target_link_libraries(ds41_${task} PUBLIC strata_ds41_engine)
  add_executable(${task}_test src/ds41/tests/${task}_test.cu)
  target_link_libraries(${task}_test PRIVATE ds41_${task} strata_ds41_engine)
  add_test(NAME ${task}_test COMMAND ${task}_test)
  set_tests_properties(${task}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endforeach()
