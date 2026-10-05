# K1 progress

- [x] Read the specification, BF16 kernels/parity, CMake registration, and Python reference.
- [x] Confirm the requested branch and clean starting tree.
- [x] Record assumptions without changing the numerical tolerance.
- [ ] Write the public interface and asynchronous CUDA implementation.
- [ ] Write CPU reference and GPU parity/timing tests for all required cases.
- [ ] Register the library/test with only one top-level CMake include line.
- [ ] Compile and run the separable host checks available on this Mac.
- [ ] Review CUDA indexing, conversion, reduction, stream ordering, and resource lifetime.
- [ ] Record actual check output and separate written/compiled/GPU-tested status in the report.
- [ ] Commit changes on the current branch in commits of at most five files; do not push.
- [ ] Compile CUDA for sm_86, sm_89, sm_120 (unavailable here: no CUDA toolkit).
- [ ] Run GPU parity and measure RTX 4090 acceptance (unavailable here: no NVIDIA GPU).
