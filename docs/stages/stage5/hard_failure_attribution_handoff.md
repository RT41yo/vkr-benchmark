# Stage 5 Decoder Audit — final hard-failure attribution handoff

## Purpose

This is the final causal-attribution step of the Stage-5 Decoder Audit. Earlier gates established:

- Pilot-B direct-token replay: `504/504 direct_exact`;
- independent decoder conformance: all six streaming methods PASS, `3533` checks and `0` mismatches;
- process-successful text transport: all `62/62` silent-inexact Pilot-B runs are attributed to token-sequence changes during `token IDs -> text -> retokenization`.

The only unresolved Pilot-B reliability outcomes are the `38` hard failures:

- Huffman: 20;
- Arithmetic Coding: 3;
- Discop: 2;
- DAIRstega: 13.

## Important evidence limitation

Pilot B did not persist `sender_token_ids` for runs that ended in hard failure. Therefore the original sender token sequences of these 38 runs cannot be compared with a later replay. The final diagnostic must use deterministic re-encoding from the canonical Pilot-B configuration. Any attribution statement must preserve this limitation explicitly.

## Diagnostic

`scripts/audit_stage5_hard_failure_attribution.py`:

1. requires the completed direct-token audit and verifies `direct_exact` for all 38 hard-failure run IDs;
2. deterministically regenerates the sender sequence from the canonical run config;
3. transmits it through the benchmark ordinary-text channel;
4. replays receiver-side decoding;
5. checks whether the original hard-failure error type/message is reproduced at or after the first sender/receiver token mismatch;
6. writes a per-run checkpoint immediately after every completed run.

Strong classification:

`text_transport_reproduced_on_reconstructed_sender`

requires all of:

- prior direct-token gate is exact;
- text transport changes token IDs;
- receiver-side hard failure occurs at or after the first token mismatch;
- replayed error type and message exactly match the original Pilot-B hard failure.

This classification is deliberately phrased as *reproduced on reconstructed sender*, not as literal replay of the lost original sender IDs.

## Resume semantics

Each completed run is atomically stored under:

`results/stage5/decoder_audit/hard_failure_attribution/records/<run_id>.json`

The consolidated JSONL and summary are refreshed after every run. If the process is interrupted, rerun with `--resume`; completed valid checkpoints are skipped. The model is not loaded if every selected run is already checkpointed.

Filters are supported with repeated `--method` and `--run-id` arguments, so one method or one run can be investigated independently.
