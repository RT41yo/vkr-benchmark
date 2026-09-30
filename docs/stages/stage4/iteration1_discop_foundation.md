# Stage 4 — iteration 1: foundation + Discop

**Date:** 2026-10-01  
**Scope:** S4.0/S4.1 foundation plus the first modern normalized adapter, Discop.

## Completed in this iteration

- frozen Stage-4 reference revisions for ADG, Discop, RRC and DAIRstega;
- updated RRC Stage-4 reference to `dae326259e4fca8bc4fcf460dafdbc0e88a0a71a` while retaining `8b6a86a...` only as preliminary-analysis provenance;
- added `fixed_payload_bits` termination with `target_payload_bits` and mandatory `max_carrier_tokens` safety cap;
- added optional encoder-session `target_payload_bits` contract;
- corrected reliability handling so finalize-only decoders are valid while incremental extra-bit diagnostics remain visible;
- preserved the historical fixed-carrier canonical storage shape so Stage-2/Stage-3 run IDs do not change merely because the new termination mode exists;
- implemented normalized Discop over canonical `P_reference` using the pinned two-queue Huffman decomposition and synchronized per-node PRNG draws;
- represented the author's shared PRNG seed as `method.key` and derive independent sender/receiver `MethodRandomSource` instances from it;
- added an analytic `Q_stego=P_reference` certificate plus an independent deterministic synthetic distribution validation;
- connected Discop to the common method factory, unified experiment runner, metrics and ordinary-text transport;
- added method/experiment example configs and `scripts/check_discop_e2e.py` for real Llama/Qwen smoke runs;
- corrected a pre-existing Stage-3 readiness path mismatch: the validator expected `0016-stage3-reproducibility-closeout.md`, while the repository contains `0016-stage3-reproducibility-closeout_ru.md`.

## Local validation performed in the prepared snapshot

- `pytest -q`: **318 passed**;
- `python -m compileall -q src scripts tests`: completed successfully;
- `python scripts/check_discop_e2e.py --help`: CLI imports and argument parsing completed successfully;
- `ExperimentConfig.from_json(configs/experiments/stage4_discop.example.json)`: parsed successfully.

Real-model GPU smoke tests were **not** run in this environment because the local Llama/Qwen model directories are not part of the supplied repository snapshot. They remain the first user-side verification after overlaying this iteration onto the local branch.

## Not yet implemented

- RRC normalized adapter;
- ADG normalized adapter;
- DAIRstega normalized adapter;
- author-compatible Discop legacy environment/paper-level reproduction run;
- Stage-4 compatibility matrix and final reproducibility report.

## Next iteration after local verification

Proceed to RRC on top of this foundation, using the fixed-payload lifecycle introduced here and the Stage-4 reference revision frozen in ADR-0017.
