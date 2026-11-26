# Stage 7 — AC Precision Analysis

## Provenance and scope

Frozen AC plan SHA-256: `3bba2773c2e418bb1a718d06b48bd6b39ca0a7eb3af0a13839b3351a5e8f7b0e`.
Execution source: 96 runs, 16 validation article groups, two LMs, precision 16/32/48,
top_k=2048, 256 carrier tokens. Same prompt and secret within LM/article.
No frozen-test access, no detector inference, no alteration of Stage 6 or of the
AC plan/execution artifacts. The uploaded archive contains config/result/token IDs
only; it does not contain step traces or secret bits.

## Reproduction

From repo root, in the existing virtual environment:

```bash
pytest -q
PYTHONPATH=src python scripts/stage7_ac_precision_analysis.py
PYTHONPATH=src python scripts/stage7_ac_precision_analysis.py --write-report
```

Outputs in `results/stage7/analysis/`:
- `ac_precision_analysis.json`: group summaries, paired contrasts, source inventory
  (SHA-256 for every input) and hashes of derived files;
- `ac_precision_runs.csv`: 96 rows, no texts or token sequences;
- `ac_precision_kl.svg`, `ac_precision_tvd.svg`: medians of per-run means.

The script requires the exact plan hash, all expected run directories, matching
canonical config and run_id, all three input artifacts and internally consistent
result and token-transport flags. This analysis is specific to the observed
all-status-ok cohort. An unexpected hard failure raises an error rather than
silently dropping it; a new cohort containing such failures would require an
explicit aggregation extension. Existing outputs are never replaced with
different bytes. Read-only mode does not write outputs.

## Statistical conventions and results

Group summaries: arithmetic mean and median of per-run metrics, with no exclusion
of soft failures. Main distortion figure: median of per-run mean KL (stego||ref)
and TVD. Reverse KL (ref||stego) is kept as `inf` and counted, not mapped to a large
finite value. BER is the mean/median of run-level BER, not pooled bit error rate.

Contrasts are p16−p32 and p48−p32 within each LM/article. Mean and median paired
differences, sign counts, and 95% percentile intervals for mean differences are
reported. Exploratory bootstrap: 2000 paired article resamples, NumPy default_rng
seed 731, shared draws across all metrics and LMs. No multiple-comparison
adjustment. This is an additional analysis convention, not a change to the
pre-test detector protocol. Numerical last-bit differences may occur across
Python/NumPy versions; exact source hashes are independent of those versions.

| LM | precision | median run-mean KL (stego||ref) | median run-mean TVD | exact secret |
|---|---:|---:|---:|---:|
| Llama | 16 | 0.034219 | 0.023641 | 15/16 |
| Llama | 32 | 0.010333 | 0.007079 | 16/16 |
| Llama | 48 | 0.008657 | 0.005930 | 16/16 |
| Qwen | 16 | 0.022355 | 0.015297 | 14/16 |
| Qwen | 32 | 0.008131 | 0.005565 | 13/16 |
| Qwen | 48 | 0.007275 | 0.004988 | 16/16 |

All 24,576 steps have infinite reverse KL. All 96 runs have status ok, but
90 exact secret roundtrips and six soft failures. All six coincide with
sender/receiver token sequence differences, directly checked from token_ids.json.
This association does not prove the causal mechanism; direct sender-token decode
would be needed to distinguish numerical and transport effects.

Mean paired KL and TVD differences are positive for p16−p32, with exploratory
intervals excluding zero in both LMs. For p48−p32, all four distortion intervals
include zero. This is not an equivalence test. p32 is a supported practical
operating point, without claiming optimal precision or absence of benefit at p48.
The fixed top_k may contribute residual distortion, but attribution to truncation
versus quantization would require a fixed-context decomposition. The p32/p48
sender sequences have common-prefix medians of 4 tokens (Llama) and 3 (Qwen):
paired prompts and secrets do not imply identical LM contexts.
No inference was repeated and per-step metrics or BER from original secret bits
were not independently recomputed. The source inventory fingerprints the received
artifacts, not an independently captured pre-execution manifest.

## Verification

8 focused unit tests passed. The complete supplied 96-run archive was analyzed
against a reconstructed plan whose exact SHA-256 matched the committed plan.
Two analyses and idempotent writes produced identical bytes in the working
environment (Python 3.13.5, NumPy 2.3.5). The full repo regression suite must be
run in the user's Python 3.12 environment; no GPU run is required.
