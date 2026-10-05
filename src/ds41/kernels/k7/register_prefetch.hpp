// A two-step scalar register pipeline shared by the CUDA producer and CPU model.
// It only changes load timing: every accumulator consumes steps 0..Steps-1.
#pragma once

#ifdef __CUDACC__
#define K7_PIPELINE_INLINE __device__ __forceinline__
#else
#define K7_PIPELINE_INLINE inline
#endif

namespace strata::ds41::kernels::k7_detail {

template <int Tokens>
struct Stage {
    float weight;
    float values[Tokens];
};

template <int Tokens, typename Fma>
K7_PIPELINE_INLINE void consume(const Stage<Tokens>& stage, int step, int row,
                                float (&dots)[Tokens], float (&squares)[Tokens], Fma fma) {
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float value = stage.values[token];
        dots[token] = fma(value, stage.weight, dots[token]);
        if (row < 4 && (step & 3) == row)
            squares[token] = fma(value, value, squares[token]);
    }
}

template <int Tokens, int Steps, typename Load, typename Fma>
K7_PIPELINE_INLINE void register_prefetch(Load load, Fma fma, int row,
                                        float (&dots)[Tokens], float (&squares)[Tokens]) {
    static_assert(Tokens >= 1 && Tokens <= 8 && Steps >= 2 && Steps % 2 == 0);
    Stage<Tokens> current0 = load(0);
    Stage<Tokens> current1 = load(1);
#pragma unroll 1
    for (int step = 0; step < Steps - 2; step += 2) {
        // Fetch two future stride-256 positions before consuming this pair.
        // Current and future values occupy registers, not shared memory.
        const Stage<Tokens> next0 = load(step + 2);
        const Stage<Tokens> next1 = load(step + 3);
        consume(current0, step, row, dots, squares, fma);
        consume(current1, step + 1, row, dots, squares, fma);
        current0 = next0;
        current1 = next1;
    }
    // Separate drain: there is no speculative load past the last legal value.
    consume(current0, Steps - 2, row, dots, squares, fma);
    consume(current1, Steps - 1, row, dots, squares, fma);
}

}  // namespace strata::ds41::kernels::k7_detail
#undef K7_PIPELINE_INLINE
