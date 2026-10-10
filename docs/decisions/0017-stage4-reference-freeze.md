# ADR-0017: Freeze reference revisions for Stage 4 modern methods

- **Status:** Accepted
- **Date:** 2026-10-01
- **Context:** Stage 4 — integration and reproducibility checks for ADG, Discop, RRC and DAIRstega

## Context

Stage 4 must compare a normalized adapter against a concrete author implementation, not against a moving Git branch. The external repositories remain live and can change after an earlier project review.

The RRC repository is the concrete example: the preliminary project analysis used commit `8b6a86a66e516e3763c907a1ac640c17bcd3641d`, while before Stage 4 its `main` branch advanced by four commits to `dae326259e4fca8bc4fcf460dafdbc0e88a0a71a`. Those commits include correctness-relevant changes: a shared encode/decode core, message-length-dependent Decimal precision, termination replay verification, and explicit token-ID versus text round-trip diagnostics.

## Decision

Stage 4 uses the following immutable author references:

| Method | Repository | Stage-4 reference revision |
|---|---|---|
| ADG | `Mhzzzzz/ADG-steganography` | `b4a7e802a97bc3a3b84d07e63f24627b16c51a10` |
| Discop | `comydream/Discop` | `3c3a10099a242eae405b49cc4d09fba1abb148ad` |
| RRC | `ryehr/RRC_steganography` | `dae326259e4fca8bc4fcf460dafdbc0e88a0a71a` |
| DAIRstega | `WangYH-BUPT/DAIRstega` | `8d85edf98d48c3efa827a125b6d4e90f88141ea2` |

The earlier RRC commit `8b6a86a...` remains provenance for preliminary analysis only. It is not a second supported RRC implementation and is not used as the Stage-4 conformance target.

`reference/reference_sources.json` is the machine-readable source of these revisions.

## Consequences

- Future upstream changes do not silently alter Stage-4 results.
- Reproduction reports can state exactly which author code was used.
- If a later upstream revision is intentionally adopted, this ADR and the reference manifest must be amended explicitly.
