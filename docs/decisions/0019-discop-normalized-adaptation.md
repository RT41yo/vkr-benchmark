# ADR-0019: Normalized Discop adaptation

- **Status:** Superseded in part by ADR-0020
- **Date:** 2026-10-01
- **Context:** First modern method integrated in Stage 4
- **Author reference:** `comydream/Discop@3c3a10099a242eae405b49cc4d09fba1abb148ad`, `src/stega_cy.pyx`

## Preserved algorithmic core

For every canonical `P_reference`:

1. candidates are arranged in descending probability order;
2. a Huffman tree decomposes the distribution into binary sub-distributions;
3. one shared-PRNG draw is consumed at each visited internal node;
4. two pointers separated by half of the current node mass represent distribution copies 0 and 1;
5. if both copies select the same branch, no payload bit is consumed;
6. if they select different branches, one secret bit selects the copy/branch;
7. the decoder reconstructs the same tree, consumes the same PRNG draws, and infers the bit whenever the two copies diverge.

The pinned author's two-queue Huffman construction is preserved, including its rule that an internal-node queue entry wins an exact weight tie against a leaf queue entry. Benchmark canonical token ordering supplies deterministic leaf ordering for equal-probability tokens.

## Shared randomness / key

The author code calls Python `random.seed(settings.seed)` independently before encoding and decoding. Stage-4 normalized Discop therefore stores this shared seed as `method.key`. The common runner constructs two independent `MethodRandomSource` objects from that key. This preserves the author PRNG algorithm (`random.Random`) without sharing mutable sender/receiver state.

`method.random_seed` is rejected for Discop to avoid two competing sources of PRNG initialization.

## Q_stego

Discop's theoretical construction claims `Q_stego = P_reference` under uniformly distributed secret bits and PRNG output. This was the initial Stage-4 representation.

ADR-0020 supersedes this part of the decision: ordinary benchmark runs now construct an explicit `Q_stego` independently on every step and pass it to the common KL/TVD layer. The explicit construction is benchmark metric instrumentation and is excluded from method-performance timing.

## Boundaries of normalization

The normalized adapter does not own:

- LM/tokenizer loading;
- temperature/top-p filtering;
- text transport or token repair;
- result metrics;
- global RNG state.

Those remain common benchmark infrastructure.
