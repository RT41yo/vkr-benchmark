# Stage 5 — Pilot B process-successful text-transport analysis handoff

## Scope

This iteration prepares the first text-channel attribution substep without modifying any production steganographic method.

Target population: the 466 Pilot-B runs with `status=ok`, including the 62 runs with `roundtrip_exact=false`.

The analysis is CPU-only and reads existing artifacts:

- `results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl`;
- `results/runs/<run_id>/token_ids.json`;
- `results/runs/<run_id>/result.json`;
- the completed direct-token audit `results/stage5/decoder_audit/direct_token_replay/direct_token_replay.jsonl` and its sibling `summary.json`.

No LM weights, tokenizer, GPU, encoder rerun or decoder rerun are used.

## Diagnostic

`scripts/analyze_stage5_text_transport_successful.py`

For each process-successful run it independently recomputes the sender/receiver token-sequence comparison and verifies the persisted transport diagnostics. It also verifies that `execution.jsonl` and `result.json` agree on `roundtrip_exact`.

For each run the output records:

- sender and receiver token counts;
- first token mismatch;
- common token prefix and suffix lengths;
- size of the changed middle region;
- token-count delta;
- original Pilot-B BER / bit-error diagnostics;
- direct-token audit result;
- a conservative classification.

A silent-inexact run is classified as `text_transport_attributed` only if:

1. Pilot B completed with `roundtrip_exact=false`;
2. the already completed direct-token audit is `direct_exact` for that same run;
3. persisted sender and receiver token sequences differ.

Thus this substep does not infer text-transport causality from token mismatch alone; it links the token evidence to the independent direct-token control.

Exact Pilot-B runs are retained as controls. If some exact runs also have `sender_ids != receiver_ids`, they demonstrate that a tokenization change is not automatically equivalent to a secret-recovery failure.

## Expected local outputs

The script writes new audit artifacts only under `results/stage5/decoder_audit/`:

- `pilot_b_text_transport_successful.jsonl`;
- `pilot_b_text_transport_successful_summary.json`.

It refuses to write inside `results/runs/` and refuses to overwrite an existing output file.

## Boundary of this substep

This analysis can potentially close the causal attribution of the 62 process-successful silent-inexact runs without a new GPU pass.

It does **not** close the 38 original hard failures because their original sender token artifacts were not persisted. Those runs remain a separate follow-up requiring reconstructed replay with the provenance limitation already documented by the Decoder Audit.
