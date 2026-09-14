# ADR-0016: Stage-3 reproducibility closeout and acceptance gate

- **Status:** Accepted
- **Date:** 2026-09-14
- **Context:** Stage 3 — reproducibility/conformance of Bins, Huffman and Arithmetic Coding

## Context

Stage 3 contains several different kinds of evidence: author-compatible smoke runs, a paper-sentence protocol, a full Figure-3 sweep, a publication-level interpretation, matched author-vs-normalized runs and a conformance/discrepancy analysis. Completion must not be inferred from one successful GPU run or from visual similarity of one curve.

A closeout gate is therefore required that checks both execution integrity and the frozen scientific conclusions without introducing new post-hoc numerical tolerances.

## Decision

Stage 3 is considered complete only when the following invariants hold:

1. the author-compatible Figure-3 matrix contains 23 points and all 5520 scheduled outcomes;
2. 5513 runs terminate at the pinned first-sentence predicate and the 7 declared termination failures remain explicitly reported;
3. Step 3.11 retains the frozen `partial_reproduction` assessment with 8/8 paper claims classified;
4. the exact `4e-8 nats` prose anchor remains `not_reproducible` under the public precision-26 executable rather than being forced to match;
5. the matched comparison contains 32/32 pairs with equal carrier length, identical secret stream and exact normalized token-ID decoding;
6. the core embedding/decoding principle is preserved at all 4 representative matched points;
7. `D_KL(P_reference || Q_stego)=+inf` in the 32/32 matched sparse-support cases is retained as a valid support diagnostic and is not smoothed away;
8. final report, machine-readable comparison outputs and Stage-3 Typst presentation are present;
9. the complete unit-test suite passes.

The final gate is implemented by `scripts/check_stage3_readiness.py` and may write `results/stage3/stage3_readiness.json`.

`results/stage3/comparison.parquet` is a compact machine-readable closeout table for the four matched representative points. It does not replace the full 32-pair evidence in `results/stage3/matched_author_normalized/`.

## Scientific interpretation frozen by this ADR

Stage 3 supports the following conclusion:

> The normalized Bins, Huffman and Arithmetic Coding adapters preserve the defining method behavior on the tested matched points, while observed sequence/numerical differences are attributable to documented normalization boundaries. The characteristic Figure-3 trends are reproduced, but the exact historical unmodulated `4e-8 nats` anchor and exact historical Monte-Carlo orchestration are not numerically/operationally reproduced.

This conclusion is deliberately weaker than claiming exact historical replication and stronger than merely showing that code runs.

## Consequences

- Stage 4 can add modern methods without reopening baseline-conformance questions unless a later regression violates the closeout gate.
- Benchmark specification v0.1 remains frozen.
- The Stage-3 finding about dual KL directions is recorded as an input to future specification v1.0, not applied retroactively through smoothing or metric replacement.
- Raw/large Figure-3 shards remain provenance artifacts; compact summaries and closeout tables provide stable machine-readable evidence for later stages.
