# Stage 4 method contracts

This file freezes implementation boundaries before adapting modern methods. Exact external revisions are in `reference/reference_sources.json` and ADR-0017.

| Method | Secret semantics | Shared randomness | Termination | Decoder in pinned repo | Normalized Q strategy | Stage-4 characteristic check |
|---|---|---|---|---|---|---|
| Discop | streaming, variable bits/token | shared PRNG seed; represented by `method.key` | fixed carrier | yes | analytic `Q=P` certificate + independent synthetic validation | encode/decode fixture, distribution preservation, real-LM roundtrip |
| RRC | fixed-length whole message | fresh shared PRNG offset per carrier token | fixed payload + safety cap | yes | analytic/theoretical `Q=P`, separately validate numerically | fixed-message roundtrip, termination, distribution preservation |
| ADG | streaming variable grouping | no separate sampling RNG after final group except reference implementation's within-group stochastic selection; exact mapping must be frozen before coding | fixed carrier | no public decoder | explicit/analytic Q from grouping where derivable | grouping correspondence + own inverse decoder + author characteristic result |
| DAIRstega | streaming common-prefix payload from allocated interval | secret bitstream drives interval point | fixed carrier | extraction described by paper; public repo generation-focused | explicit Q from integer interval allocation | allocation/inverse tests + alpha/beta characteristic point |

## Common normalized boundaries

All four adapters receive the already constructed canonical `P_reference`. They do not own the LM, tokenizer, common temperature/top-k/top-p policy, KV-cache path, ordinary-text transport, or benchmark metrics. Method-internal candidate restrictions that are part of the published algorithm remain inside the method and therefore affect `Q_stego`.

Direct token-ID decoding is an algorithm-conformance diagnostic. Benchmark reliability is still measured through `token IDs → text → retokenized token IDs → decoder`.
