# K1b assumptions

1. Does “no allocation in the call” also apply to the old API, which has no caller-owned scratch argument?
   Assumption: it applies to the two new entry points. The compatibility wrapper retains stream-ordered
   allocation/free of the dequantized activation and calls the new functions. Capture/reuse callers use the
   new API with caller-owned storage, as specified.
2. May split-K reduction use shared memory within one block instead of global scratch?
   Assumption: yes. Each block completes its output rows with a fixed FP32 reduction and one BF16 rounding;
   no atomics, global partial-output workspace, allocation or synchronization of the host stream is needed.
3. How should new performance numbers be reported without a GPU?
   Assumption: keep the supplied old measurements, mark new measurements pending, and have the parity
   executable print the matching comparison table on the reviewer's GPU. No estimated times are results.
