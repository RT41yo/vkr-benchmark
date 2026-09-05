# Stage-2 baseline metrics

This document records the exact interpretation used by the normalized Stage-2 runner. It is an implementation-facing companion to benchmark specification v0.1 and the ADRs; it does not modify the frozen v0.1 specification.

## Common notation

At carrier step `t`, `P_reference,t` is the canonical reference distribution produced from the shared LM path, and `Q_stego,t` is the method-induced distribution exposed by the stego implementation. `T` is the number of generated carrier tokens. `B` is the number of **confirmed useful payload bits** embedded in those carrier tokens. Bits merely read as algorithmic look-ahead are not counted as payload.

All normalized Stage-2 methods use the same text-only transport check: sender token IDs are decoded to ordinary text; only that text is passed to the receiver; the receiver retokenizes it and performs stego decoding from the reconstructed token sequence.

## Capacity

### Payload bits

`payload_bits = B`.

For fixed-rate methods this may equal the number of secret bits read. For stateful interval methods it need not: Arithmetic Coding can keep uncommitted look-ahead bits in its working state, so `secret_bits_read` may exceed `payload_bits`.

### Bits per token

`bits_per_token = B / T`.

This is the canonical Stage-2 payload-rate metric. It uses confirmed payload, not working-window or look-ahead bits.

## Reference entropy and entropy utilization

Per-step reference entropy is

`H_t = - sum_x P_reference,t(x) log2 P_reference,t(x)`.

The runner stores:

- `reference_entropy_mean_bits = (1/T) sum_t H_t`;
- `reference_entropy_sum_bits = sum_t H_t`;
- `entropy_utilization = B / sum_t H_t`;
- `entropy_utilization_percent = 100 * entropy_utilization`.

No clipping is applied. A value above 1, should it occur, is retained as a diagnostic rather than silently forced into `[0, 1]`.

## Distribution distortion

### Q availability

`q_mode` states how `Q_stego` was obtained. The Stage-2 baselines currently expose `analytic_exact` distributions. Future methods may require another exact representation, Monte Carlo estimation, or may be unavailable.

### Benchmark-native KL

Specification v0.1 defines

`D_KL(P_reference || Q_stego) = sum_x P_reference(x) log2(P_reference(x) / Q_stego(x))`.

The implementation applies no epsilon smoothing. Therefore if there exists any token with

`P_reference(x) > 0` and `Q_stego(x) = 0`,

the step KL is `+inf`. Run-level fields `kl_mean_bits`, `kl_median_bits`, `kl_p95_bits` and `kl_max_bits` therefore may legitimately be infinite. `kl_infinite_steps` and `kl_finite_steps` make that support behavior explicit.

The current generic persisted `kl_*` names are v0.1 fields and mean **reference → stego**. ADR-0012 additionally requires Stage 3 reproducibility experiments to calculate the author-compatible opposite direction

`D_KL(Q_stego || P_reference)`

without replacing the benchmark-native metric. Directional field names must be used when both are first-class in the result schema.

### Total variation distance

`TVD(P_reference, Q_stego) = 0.5 * sum_x |P_reference(x) - Q_stego(x)|`.

Unlike strict `D_KL(P_reference || Q_stego)`, TVD remains finite under support truncation and therefore provides a useful graded distortion measure for methods whose benchmark-native KL is infinite.

The runner stores mean, median, nearest-rank p95 and maximum TVD across carrier steps.

## Raw-LM quality

For the actually generated carrier token `x_t`, NLL is evaluated using the **raw generating LM distribution**, before `P_reference` masking/truncation and before any stego transformation:

`NLL = -(1/T) sum_t ln P_LM-raw,t(x_t)`  in nats/token.

`PPL = exp(NLL)`.

These metrics answer a different question from TVD: they measure how probable the realized stegotext tokens are under the underlying LM, rather than how far the complete method distribution is from `P_reference`.

## Reliability

### Exact round-trip

`roundtrip_exact` is true only when the recovered useful payload equals the expected payload exactly under the text-only channel.

`token_sequence_roundtrip_exact` is a transport diagnostic: it records whether decoding the sender IDs to text and retokenizing that text reproduces the same carrier-token sequence.

### Bit error rate

`BER = bit_errors / expected_length_bits` when the expected payload length is non-zero.

A missing recovered bit counts as an error. Extra recovered bits do not increase the BER numerator because they have no corresponding expected payload position, but they make `roundtrip_exact = false` and are reported through the recovered-length diagnostics.

The runner also stores expected and recovered lengths, length delta, extra recovered bits and first mismatch positions.

## Computational efficiency

Timing excludes model loading, result persistence, text storage and post-hoc metric computation. A warm-up is performed before the measured run. CUDA measurements synchronize the device at timing boundaries.

Run-level fields include:

- `encode_total_ms`, `decode_total_ms`;
- `encode_ms_per_token`, `decode_ms_per_token`;
- `payload_bits_per_second_encode`, `payload_bits_per_second_decode`.

The total timed work is decomposed into:

- `lm_forward_total_ms` — LM prefill/cached-forward work;
- `distribution_processing_total_ms` — construction of the shared `P_reference` path;
- `stego_algorithm_total_ms` — method-specific encode/decode operations.

The decomposition is intended to separate common LM cost from method overhead. Wall-clock values should be treated as environment-dependent measurements and aggregated over repeated runs in later experimental stages.

## Aggregation rule

Stage 2 currently validates the metric pipeline on individual smoke runs. A single run is **not** a scientific comparison. Later operating points must aggregate across the planned prompts, secrets/keys and repeats. Timing in particular should be summarized over repeated runs rather than interpreted from one 16-token sample.
