# ADR-0015: Stage-2 readiness gate and v0.2 milestone

- Status: Accepted
- Date: 2026-09-05

## Context

ROADMAP defines Stage 2 as creation of the baseline experimental infrastructure for Bins, Huffman and Arithmetic Coding, including a common execution/metric path, automatic encode/decode checks, unified result storage, an example result table, metric documentation and a Typst presentation.

By the end of implementation steps 7.1-7.6 the repository contains the required technical components, but completion should not depend on an informal visual inspection of files or one successful manual command. A reproducible readiness gate is needed before the project starts Stage 3 author-reproducibility work.

## Decision

Introduce `scripts/check_stage2_readiness.py` as the technical acceptance gate for the v0.2 milestone.

The gate validates:

1. presence of the Stage-2 implementation/config/documentation paths;
2. readability and minimum schema of `results/summary.parquet`;
3. successful normalized rows for Bins, Huffman and Arithmetic Coding;
4. unique canonical `run_id` values;
5. zero BER and exact text/token roundtrip in the Stage-2 smoke rows;
6. basic sanity of capacity, entropy, TVD, raw-LM quality and timing fields;
7. `analytic_exact` Q availability for all three baselines;
8. absence of tracked `environment/stage1_reference/hf.txt`;
9. optionally, the complete pytest suite via `--run-tests`.

The gate explicitly permits `D_KL(P_reference || Q_stego) = +inf`. Infinite benchmark-native KL under exact support loss is a valid measured result, not a readiness failure.

The v0.2 label denotes the **technical Stage-2 benchmark milestone**, not the final experimental protocol. Frozen benchmark specification v0.1 remains unchanged; final v1.0 protocol is still planned after Stage-5 pilot sweeps.

A Git tag/release for v0.2 should be created only after the technical gate passes and the mandatory Stage-2 Typst presentation is added. Until then the technical code milestone may be READY while the complete ROADMAP deliverable set is not yet formally closed.

## Consequences

- Stage-2 completion becomes machine-checkable rather than narrative-only.
- The same gate can be re-run after later refactors to detect accidental regression of baseline infrastructure.
- Runtime artifacts remain local/gitignored; the gate consumes `results/summary.parquet` but does not require committing experimental output.
- Stage 3 can begin from a clearly defined normalized baseline while author-compatible runs are kept conceptually separate.
