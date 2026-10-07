// Host RAM bandwidth: read (XOR reduce) and copy, OpenMP. Prints one JSON line.
// Build: gcc -O3 -march=native -fopenmp membw.c -o membw
// Run:   OMP_NUM_THREADS=16 OMP_PROC_BIND=spread ./membw <GiB> <reps>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t gib = argc > 1 ? strtoull(argv[1], 0, 10) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    size_t n = gib * (1ull << 30) / 8;
    uint64_t *a = aligned_alloc(4096, n * 8);
    uint64_t *b = aligned_alloc(4096, n * 8);
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = i * 2654435761u; b[i] = 0; }

    double best_read = 1e30, best_copy = 1e30;
    uint64_t sink = 0;
    for (int r = 0; r < reps; r++) {
        uint64_t s = 0;
        double t0 = now();
#pragma omp parallel for reduction(^ : s) schedule(static)
        for (size_t i = 0; i < n; i++) s ^= a[i];
        double dt = now() - t0;
        if (dt < best_read) best_read = dt;
        sink ^= s;

        t0 = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) b[i] = a[i];
        dt = now() - t0;
        if (dt < best_copy) best_copy = dt;
        sink ^= b[n / 2];
    }
    printf("{\"threads\": %d, \"gib\": %zu, \"read_gbps\": %.2f, \"copy_gbps_rw\": %.2f, \"sink\": %llu}\n",
           omp_get_max_threads(), gib, n * 8 / best_read / 1e9, 2.0 * n * 8 / best_copy / 1e9,
           (unsigned long long)(sink & 1));
    return 0;
}
