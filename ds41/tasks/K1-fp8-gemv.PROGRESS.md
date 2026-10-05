# K1 progress

- [x] Read the specification, BF16 kernels/parity, CMake registration, and Python reference.
- [x] Confirm the requested branch and clean starting tree.
- [x] Record assumptions without changing the numerical tolerance.
- [x] Write the public interface and asynchronous CUDA implementation.
- [x] Write CPU reference and GPU parity/timing tests for all required cases.
- [x] Register the library/test with only one top-level CMake include line.
- [x] Compile and run the separable host checks available on this Mac (64 shape/batch cases pass).
- [x] Review CUDA indexing, conversion, reduction, stream ordering, and resource lifetime.
- [x] Record actual check output and separate written/compiled/GPU-tested status in the report.
- [ ] Commit all changes on the current branch in commits of at most five files: first four-file commit `cf8bec0` succeeded; the next `git add` failed because protected `.git/index.lock` could not be created (Operation not permitted). Remaining changes are intact. No push attempted.
- [ ] Compile CUDA for sm_86, sm_89, sm_120 (unavailable here: no CUDA toolkit).
- [ ] Run GPU parity and measure RTX 4090 acceptance (unavailable here: no NVIDIA GPU).

- [x] UBSan host build and small-case tests pass.
- [ ] ASan execution: built, but both selftest and a timed CLI probe failed to complete; interrupted the selftest. No ASan pass claimed.
- [ ] Direct PyTorch comparison: unavailable because PyTorch is not installed; independent C++ reference was tested instead.
