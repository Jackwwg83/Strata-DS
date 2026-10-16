#!/bin/bash
# KL divergence of the tail skips: 200 x 256-token segments, teacher-forced step() with the engine reset per segment.
# (a) no skip (DS41_ROUTE_TAIL_SKIP=0) twice: the noise floor (an expert's tier decides GPU or CPU rounding);
# (b) route tail skip only (DS41_SSD_TAIL_SKIP=0); (c) defaults. Order a1, c, b, a2.
H=$HOME/strata-ds-dev; G=$H/gen-q; P=$H/pack-sage; PROF=$H/sync/ds41/data/expert-profile.bin; Q=$H/quality
run() { tag=$1; shift; s=$(date +%s)
  env "$@" $G --pack $P --expert-profile $PROF --threads 16 --max-seq 512 --force-list $Q/kld200.list \
    --logprobs-out $Q/lp_$tag.bin > $Q/kld_$tag.log 2> $Q/kld_$tag.err
  echo "$tag exit $? $(( $(date +%s) - s )) s | $(grep -E "force_list sequences|decode_ms" $Q/kld_$tag.log | tr "\n" " ")"; }
run a1 DS41_ROUTE_TAIL_SKIP=0
run c X=1
run b DS41_SSD_TAIL_SKIP=0
run a2 DS41_ROUTE_TAIL_SKIP=0
echo Q1_DONE
