# Stage 7 — AC Precision Study (execution checkpoint)

## Scope and invariants

Implementation-factor diagnostic according to §9 of `stage7_protocol.md` and
ADR-0012. No new steganalysis detector, no frozen-test scoring, no changes to
Stage 6, detector dataset, primary evaluation or its lock. AC implementation
revision and both LM configs are inherited from the frozen p32-k2048 point.

The method-internal `precision` takes 16, 32, 48 bits. All other parameters
remain fixed: method top_k=2048, external temperature=1, top_k=null,
top_p=1, exclude_special_tokens=true, KV-cache=true, 256 sender carrier tokens.
The normalized AC implementation accepts precision from 2 to 52 inclusive;
no change to the arithmetic implementation is required.

## Selection and paired design

Take exactly 16 of the 32 existing Stage-7 validation article groups by ascending
hex SHA-256 of UTF-8 `VKR-BENCHMARK-STAGE7-v1|implementation-factors|seed=741|article_id=<id>`
(tie-break: article_id). This specifies the exact hash domain, which §9 did not
specify. The same group selection is intended for the later RRC margin study.
One replicate r01 per group; retain the existing Stage-7 per-prompt secret ID
and SHAKE256 stream, shared across both LMs and all three AC variants. AC has
no method RNG or key. The p32 canonical configuration and run ID exactly match
the previous validation generation-only plan, but the complete experiment is
executed separately to obtain full metrics, not only generated texts.

16 groups × 3 precisions × 2 LMs = 96 runs. The comparison is paired by article
within LM; this is an implementation sensitivity analysis, not a new external
test or an attempt to optimize the Stage-6 operating point.

The plan verifies prompt text file and individual context hashes, frozen model
config hashes, LM revision and canonical run configuration. It contains only
IDs, hashes and canonical configurations, no prompt or generated text. Commit
it before running GPU experiments.

## Commands

From repository root with the existing environment activated:

```bash
PYTHONPATH=src python scripts/stage7_ac_precision.py prepare
PYTHONPATH=src python scripts/stage7_ac_precision.py prepare --write-plan
pytest -q
```

The plan is `results/stage7/analysis/ac_precision_plan.json`. Existing different
plan bytes are rejected. Commit this file, this document, script and tests
before generation. For each model perform a smoke of the first article at all
three precisions; then resume the full 48-run per-model plan:

```bash
PYTHONPATH=src python scripts/stage7_ac_precision.py run --model-id meta-llama/Llama-3.2-3B --limit 3
PYTHONPATH=src python scripts/stage7_ac_precision.py run --model-id Qwen/Qwen3-4B-Base --limit 3
PYTHONPATH=src python scripts/stage7_ac_precision.py run --model-id meta-llama/Llama-3.2-3B --all --resume
PYTHONPATH=src python scripts/stage7_ac_precision.py run --model-id Qwen/Qwen3-4B-Base --all --resume
```

## Storage and failure handling

All per-run artifacts are isolated in
`data/stage7/cache/ac_precision/execution/runs/<run_id>/`, already gitignored
by `/data/stage7/cache/`. No summary.parquet is updated. Existing directories
are checked before model loading; only verified complete records may be skipped
with --resume. Incomplete or contradictory artifacts stop execution. Do not
delete them without diagnosis. The script uses the canonical unified runner and
normal persistence, preserving the usual dual-direction KL, TVD, capacity,
quality and reliability metrics for successful runs (including soft failures).

A MethodError with captured sender evidence from text transport or decoding is
persisted as a hard failure. It is never converted to a successful roundtrip or
assigned a fabricated BER. To avoid discarding sender distribution metrics, the
sender is replayed in that case. The replay must match the captured 256 token
IDs, payload bits and secret bits read; then a separate
`sender_replay_metrics.json` is written. It retains both KL directions, finite
and infinite step counts, TVD, capacity and PPL. A replay mismatch or unexpected
exception stops execution and is investigated; the original hard-failure
record is not erased. Successful runs are not replayed. Runtime measurements
from replay are not used as canonical timing results.

## Analysis (next checkpoint)

Audit all 96 planned IDs; include all hard/soft/exact counts with denominator
16 per LM/precision. Compare paired sender metrics, keeping the two KL directions
separate, and report infinite KL explicitly (never replace infinity with a
large finite number). Distinguish missing decoder BER from zero BER. Report
paired differences against p32, number of valid pairs, and token transport
mismatch counts. Because different precision can change sampled token paths,
observed trajectory-level effects combine finite precision and resulting changes
of context; they are not a same-context numerical ablation. RRC uses the same
selected article groups later. Aggregate analysis/report generation is a
separate follow-up checkpoint after the per-run results are available.
