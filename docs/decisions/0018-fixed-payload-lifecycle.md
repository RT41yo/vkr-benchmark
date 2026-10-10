# ADR-0018: Fixed-payload termination and finalize-only decoding

- **Status:** Accepted
- **Date:** 2026-10-01
- **Context:** Stage 4 infrastructure preparation for RRC and other stateful methods

## Context

Stage 2 implemented only `fixed_carrier_tokens`. Benchmark specification v0.1 explicitly allows `fixed_payload_bits` for RRC-like methods when changing them into a fixed-carrier stream would alter the original semantics.

The stateful method contracts already anticipated this case: `EncodeDecision.bits_consumed` may be `None`, and authoritative payload accounting is returned by encoder finalization. Decoder finalization can likewise recover a message only after the complete carrier has been observed.

Two missing runner features remained:

1. the encoder had no explicit fixed-payload target before its first step;
2. reliability logic used incremental decoder output length, so a decoder that returns all bits only in `finalize()` would be incorrectly marked as failed.

## Decision

1. `TerminationConfig` supports two mutually exclusive modes:
   - `fixed_carrier_tokens` with `target_carrier_tokens`;
   - `fixed_payload_bits` with `target_payload_bits` and mandatory `max_carrier_tokens` safety cap.
2. `StegoMethod.create_encoder(...)` receives optional `target_payload_bits`.
3. Fixed-payload execution stops when `encoder.done` becomes true and fails contract validation if the safety cap is reached first.
4. For fixed-payload runs, `EncoderFinalization.payload_bits` must equal the requested target.
5. `DecoderFinalization.recovered_bits` is authoritative payload content. Incremental decoder output remains a diagnostic; its longer raw length is retained so a streaming decoder cannot hide extra emitted bits by truncating during finalization.
6. Baseline fixed-carrier methods receive `target_payload_bits=None`; their semantics do not change.

## Consequences

- RRC can be integrated without pretending that it embeds an independently finalizable bit count at every token.
- Existing Bins/Huffman/Arithmetic behavior remains fixed-carrier.
- A synthetic finalize-only decoder test is required before RRC is implemented.
