# cmake/ds41_engine.cmake - the DeepSeek V4.1 Flash engine (M1): pack loader, GPU ops, CPU experts.
# Included at the end of cmake/ds41.cmake, inside the CUDA section of CMakeLists.txt.

add_library(strata_ds41_engine STATIC
  src/ds41/pack.cpp
  src/ds41/ops.cu
  src/ds41/prefill_ops.cu
  src/ds41/doorbell.cu
  src/ds41/expert_staging.cu
  src/ds41/vram_experts.cu
  src/ds41/engram_rows.cpp
  src/ds41/host_experts.cpp
  src/ds41/lookahead.cpp
  src/ds41/expert_stream.cpp
  src/ds41/wo_a_fp8.cu
  src/platform/direct_file.cpp
  src/ds41/engine.cu
  third_party/exllamav3_moe/moe_mul1.cpp)
target_include_directories(strata_ds41_engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/exllamav3_moe
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(strata_ds41_engine PUBLIC cxx_std_17)
# moe_mul1 picks AVX2 / AVX-512 per function with target attributes and dispatches at run time
set_source_files_properties(third_party/exllamav3_moe/moe_mul1.cpp PROPERTIES COMPILE_OPTIONS "-O3")
find_package(Threads REQUIRED)
target_link_libraries(strata_ds41_engine PUBLIC strata_ds41 CUDA::cudart CUDA::cublas Threads::Threads)

add_executable(ds41_generate src/ds41/ds41_generate.cpp)
target_link_libraries(ds41_generate PRIVATE strata_ds41_engine strata_ds41_suffix)

if(STRATA_BUILD_TESTS)
  foreach(test hybrid_decode hybrid_host_table expert_staging)
    add_executable(${test}_test src/ds41/tests/${test}_test.cu)
    target_link_libraries(${test}_test PRIVATE strata_ds41_engine)
    add_test(NAME ${test}_test COMMAND ${test}_test)
    set_tests_properties(${test}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
  endforeach()
  add_executable(doorbell_test src/ds41/tests/doorbell_test.cu)
  target_link_libraries(doorbell_test PRIVATE strata_ds41_engine)
  add_test(NAME doorbell_test COMMAND doorbell_test)
  set_tests_properties(doorbell_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # engine failure paths: needs --pack (skipped with code 77 without it)
  add_executable(engine_failure_test src/ds41/tests/engine_failure_test.cpp)
  target_link_libraries(engine_failure_test PRIVATE strata_ds41_engine)
  add_test(NAME engine_failure_test COMMAND engine_failure_test)
  set_tests_properties(engine_failure_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(wo_a_fp8_test src/ds41/tests/wo_a_fp8_test.cu)
  target_link_libraries(wo_a_fp8_test PRIVATE strata_ds41_engine)
  add_test(NAME wo_a_fp8_test COMMAND wo_a_fp8_test)
  set_tests_properties(wo_a_fp8_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(lookahead_test src/ds41/tests/lookahead_test.cpp)
  target_link_libraries(lookahead_test PRIVATE strata_ds41_engine)
  add_test(NAME lookahead_test COMMAND lookahead_test)
  add_executable(host_experts_test src/ds41/tests/host_experts_test.cpp)
  target_link_libraries(host_experts_test PRIVATE strata_ds41_engine)
  add_test(NAME host_experts_test COMMAND host_experts_test)
  add_executable(moe_set_expert_test src/ds41/tests/moe_set_expert_test.cpp)
  target_link_libraries(moe_set_expert_test PRIVATE strata_ds41_engine)
  add_test(NAME moe_set_expert_test COMMAND moe_set_expert_test)
  add_executable(engram_rows_test src/ds41/tests/engram_rows_test.cpp)
  target_link_libraries(engram_rows_test PRIVATE strata_ds41_engine)
  add_test(NAME engram_rows_test COMMAND engram_rows_test)
  add_executable(prefill_ops_test src/ds41/tests/prefill_ops_test.cu)
  target_link_libraries(prefill_ops_test PRIVATE strata_ds41_engine)
  add_test(NAME prefill_ops_test COMMAND prefill_ops_test)
  set_tests_properties(prefill_ops_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(vram_lend_test src/ds41/tests/vram_lend_test.cu)
  target_link_libraries(vram_lend_test PRIVATE strata_ds41_engine)
  add_test(NAME vram_lend_test COMMAND vram_lend_test)
  set_tests_properties(vram_lend_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(expert_stream_test src/ds41/tests/expert_stream_test.cu)
  target_link_libraries(expert_stream_test PRIVATE strata_ds41_engine)
  add_test(NAME expert_stream_test COMMAND expert_stream_test)
  set_tests_properties(expert_stream_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(vram_experts_test src/ds41/tests/vram_experts_test.cu)
  target_link_libraries(vram_experts_test PRIVATE strata_ds41_engine)
  add_test(NAME vram_experts_test COMMAND vram_experts_test)
endif()

if(STRATA_BUILD_TESTS)
  foreach(test graph_ops graph_index_attn graph_workspace graph_stream_ops)
    add_executable(${test}_test src/ds41/tests/${test}_test.cu)
    target_link_libraries(${test}_test PRIVATE strata_ds41_engine)
    add_test(NAME ${test}_test COMMAND ${test}_test)
    set_tests_properties(${test}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
  endforeach()
endif()

if(STRATA_BUILD_TESTS)
  foreach(test verify_rows_parity verify_commit_prefix verify_ring_rollback verify_compressor verify_engram spec_end_to_end)
    add_executable(${test} src/ds41/tests/verify_engine_test.cpp)
    target_compile_definitions(${test} PRIVATE VERIFY_TEST="${test}")
    target_link_libraries(${test} PRIVATE strata_ds41_engine strata_ds41_suffix)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 7200)
  endforeach()
endif()
