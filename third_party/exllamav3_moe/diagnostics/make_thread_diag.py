#!/usr/bin/env python3
"""Generate an instrumented source COPY; production code/test/header are untouched."""
import argparse
from pathlib import Path


def generate(source, destination):
    s = source.read_text()
    replacements = [
        ('namespace { std::atomic<bool> g_prof_enabled',
         '#include "thread_diag.h"\n\nnamespace { std::atomic<bool> g_prof_enabled'),
        ('        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);',
         '        k11diag::pin(idx, enc, pthread_setaffinity_np(pthread_self(), sizeof(set), &set));'),
        ('inline void run_rows(const MoeCpuMatrix& mat, const PreparedIn& p, float* tout, int tn0, int tn1)\n{',
         'inline void run_rows(const MoeCpuMatrix& mat, const PreparedIn& p, float* tout, int tn0, int tn1)\n{\n'
         '    k11diag::work(mat.k, mat.bits, mat.hb, p.rows, tn0, tn1);'),
        ('    ForwardCtx& c = *static_cast<ForwardCtx*>(vctx);',
         '    ForwardCtx& c = *static_cast<ForwardCtx*>(vctx);\n    k11diag::Scope diagnostic_scope(c.phase, worker);'),
        ('    g_pool.ensure(threads > 0 ? threads : 1);',
         '    k11diag::begin(threads);\n    g_pool.ensure(threads > 0 ? threads : 1);'),
        ('        ctx.phase = phase;\n',
         '        ctx.phase = phase;\n        k11diag::phase_starts[phase] = k11diag::wall();\n'),
        ('            g_pool.run(&forward_phase, &ctx, n_run);\n        }\n    }\n    if (prof',
         '            g_pool.run(&forward_phase, &ctx, n_run);\n        }\n'
         '        k11diag::phases[phase] = k11diag::wall() - k11diag::phase_starts[phase];\n    }\n'
         '    const int diag_active = n_run > 0 ? std::min(n_run,g_pool.num_workers) : g_pool.num_workers;\n'
         '    k11diag::finish(rows, threads, diag_active, nc, int(g_isa), num_phases, g_pool.core_order, out, size_t(rows)*H);\n'
         '    if (prof'),
    ]
    for old, new in replacements:
        count = s.count(old)
        if count != 1:
            raise RuntimeError(f"Instrumentation anchor count={count}, expected 1: {old!r}")
        s = s.replace(old, new)
    destination.write_text(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    generate(a.source, a.destination)
