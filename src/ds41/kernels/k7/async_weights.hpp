// Shared CUDA/host model of the exact-order asynchronous weight-tile schedule.
#pragma once

#ifdef __CUDACC__
#define K7_ASYNC_INLINE __device__ __forceinline__
#else
#define K7_ASYNC_INLINE inline
#endif

namespace strata::ds41::kernels::k7_detail {
constexpr int kWeightTileSteps = 16;
constexpr int kWeightTileValues = kWeightTileSteps * 32;
constexpr int kWeightTiles = 80 / kWeightTileSteps;
constexpr int kWeightBuffers = 2;
static_assert(80 % kWeightTileSteps == 0 && kWeightTileSteps % 4 == 0);

K7_ASYNC_INLINE int copy_offset(int lane, int copy) { return lane * 4 + copy * 128; }
K7_ASYNC_INLINE int weight_column(int warp, int tile, int local) {
    return warp * 32 + (tile * kWeightTileSteps + local / 32) * 256 + local % 32;
}

// At most two groups are pending. A tile is published only after each lane
// waits for its own older group AND all 32 lanes reach the publication barrier.
// The second barrier protects every read before any lane reuses that buffer.
// The last tile drains the sole remaining group; it must use wait_all, not
// wait_one. No zero-fill, speculative overread, or persistent completion state.
template <typename Pipeline, typename Consume>
K7_ASYNC_INLINE void async_weight_tiles(Pipeline pipeline, Consume consume) {
    pipeline.issue(0, 0);
    pipeline.issue(1, 1);
#pragma unroll 1
    for (int tile = 0; tile < kWeightTiles; ++tile) {
        if (tile + 1 == kWeightTiles) pipeline.wait_all();
        else pipeline.wait_one();
        pipeline.barrier();
        consume(tile & 1, tile);
        pipeline.barrier();
        if (tile + 2 < kWeightTiles) pipeline.issue(tile & 1, tile + 2);
    }
}
}  // namespace strata::ds41::kernels::k7_detail
#undef K7_ASYNC_INLINE
