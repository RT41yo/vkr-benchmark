# Stage 4 — modern methods integration plan

**Branch:** `stage4-modern-methods`  
**Starting point:** `main@85fbebc`  
**Target milestone:** benchmark v0.3

Stage 4 follows the roadmap chain for each modern method:

`publication → pinned author implementation → normalized adapter → conformance/reproducibility check`.

The fixed Stage-4 method set is ADG, Discop, RRC and DAIRstega, in addition to the already integrated Bins, Huffman and Arithmetic Coding baselines.

## Implementation order

1. **Foundation + Discop.** Freeze external revisions, add fixed-payload/finalize-only lifecycle support, integrate normalized Discop, add core and common-runner tests, and provide a real-LM smoke script.
2. **RRC.** Use the current pinned shared `rrc_core.py` semantics, fixed payload termination, rotation history and reverse finalization. Validate direct token-ID and ordinary-text paths separately.
3. **ADG.** Resolve the pinned `near()` grouping ambiguity against the paper before freezing normalized grouping semantics; implement inverse decoding and explicit Q where possible.
4. **DAIRstega.** Map paper alpha/beta notation to pinned implementation parameters, preserve internal candidate restriction after `P_reference`, implement interval allocation and extraction.
5. **Stage-4 conformance closeout.** For each method record publication result, author-compatible result, normalized result and discrepancy explanation; build Llama/Qwen compatibility matrix; prepare Stage-4 report and Typst presentation.

Parameter sweeps and final operating-point grids remain Stage 5 work.

## Completion gate

Stage 4 closes only when all four modern methods have a normalized encoder/decoder, synthetic algorithm tests, ordinary-text round-trip evidence, at least one documented author/paper-level conformance target, compatibility status for Llama-3.2-3B and Qwen3-4B-Base, and a reproducibility report.
