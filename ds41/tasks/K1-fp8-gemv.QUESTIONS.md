# K1 questions and chosen interpretations

1. Does the norm mean L2, and is the reference rounded to BF16 before comparison?
   Chosen: L2; accumulate the reference in double, then round to BF16 with ties to even,
   matching the specified output and `fp8_gemm`. Also print the error against the
   unrounded double result as a diagnostic. Keep the required tolerance at `2e-3`.
2. The interface has no workspace argument. How should temporary activation storage live?
   Chosen: use stream-ordered `cudaMallocAsync` / `cudaFreeAsync` per call, with no global
   scratch or host synchronization. Time the complete public call, including quantization
   and stream-ordered allocation/free. Reuse each weight load across all m tokens.
3. Are NaN/Inf activations, the E4M3 NaN codes, and E8M0 byte 255 part of acceptance?
   Chosen: test finite activation/weight values and scales 0..254. Byte 255 follows the
   specification's `2^(byte - 127)` formula (infinity in FP32), rather than the reserved
   NaN interpretation of PyTorch's E8M0 dtype. Nonfinite arithmetic is outside the
   specification's statement that products are exact in FP32. No tolerance is relaxed.
4. May the task tracking and report files be created despite the source-file allowlist?
   Chosen: yes; the user explicitly requires PROGRESS.md and REPORT.md and permits this
   QUESTIONS.md. No other repository files are created beyond the four listed source/build
   files and these three task documents. Host verification is built from the parity source.
