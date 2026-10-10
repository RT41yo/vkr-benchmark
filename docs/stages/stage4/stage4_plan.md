# Stage 4 — modern methods integration plan

**Branch:** `stage4-modern-methods`  
**Starting point:** `main@85fbebc`  
**Target milestone:** benchmark v0.3

Stage 4 follows the roadmap chain for each modern method:

`publication → pinned author implementation → normalized adapter → conformance/reproducibility check`.

The reproducibility evidence is intentionally split into two layers: (1) conformance with the pinned author implementation and (2) reproduction of a characteristic scientific result from the publication. The detailed rule is frozen in `docs/decisions/0023-publication-result-vs-implementation-conformance_ru.md` and summarized in `docs/stages/stage4/reproducibility_methodology_ru.md`.

The fixed Stage-4 method set is ADG, Discop, RRC and DAIRstega, in addition to the already integrated Bins, Huffman and Arithmetic Coding baselines.

## Current status

- Discop: Stage-4 reproducibility/conformance closed.
- RRC: Stage-4 reproducibility/conformance closed; direct token-ID recovery 100/100, ordinary-text recovery 98/100 on the fixed Llama series.
- ADG: Stage-4 reproducibility/conformance closed; Algorithm-1 conformance, direct author `near()` check and Table-1 characteristic probe complete.
- DAIRstega: Stage-4 method-level reproducibility/conformance closed; real-LM smoke, synthetic paper/repository conformance, direct author source/helper conformance and publication-characteristic trade-off probe all PASS.

## Implementation order

1. **Foundation + Discop.** Freeze external revisions, add fixed-payload/finalize-only lifecycle support, integrate normalized Discop, add core and common-runner tests, and provide a real-LM smoke script.
2. **RRC.** Use the current pinned shared `rrc_core.py` semantics, fixed payload termination, rotation history and reverse finalization. Validate direct token-ID and ordinary-text paths separately.
3. **ADG.** Resolve the pinned `near()` grouping ambiguity against the paper before freezing normalized grouping semantics; implement inverse decoding and explicit Q where possible.
4. **DAIRstega.** Paper/code semantics frozen in ADR-0025; normalized interval allocation, inverse extraction and state-aware exact Q implemented. Real-LM smoke, synthetic core conformance, direct author gate and alpha/beta publication-characteristic probe complete.
5. **Stage-4 conformance closeout.** For each method record publication result, author-compatible result, normalized result and discrepancy explanation; build Llama/Qwen compatibility matrix; prepare Stage-4 report and Typst presentation.

Parameter sweeps and final operating-point grids remain Stage 5 work.

## Completion gate

Stage 4 closes only when all four modern methods have a normalized encoder/decoder, synthetic algorithm tests, ordinary-text round-trip evidence, at least one documented author/paper-level conformance target, compatibility status for Llama-3.2-3B and Qwen3-4B-Base, and a reproducibility report.


## ADG closeout

ADG Stage-4 reproducibility is closed. The 100-token Llama characteristic probe produced BPT `4.6300`, mean `D_KL(P_ref || Q) = 0.009936`, mean `D_KL(Q || P_ref) = 0.010232` and mean TVD `0.028920`, with exact token and secret round-trip. This reproduces the Table-1 qualitative pattern (substantial embedding rate with small but non-zero distribution distortion) without claiming numerical reproduction across different LM/tokenization/corpora.

The normalized adapter follows Algorithm 1 of the paper. The pinned public repository's `near()` discrepancy is retained as explicit provenance/conformance evidence rather than silently copied into normalized semantics.


## DAIRstega foundation

DAIRstega is parameterized by method-local `alpha`, `beta` and `top_k`. Internal candidate restriction is applied after canonical `P_reference`. The adapter keeps the overlapping alpha-bit secret look-ahead required by the publication and reconstructs the inverse decoder from Formula 10. Exact Q is state-aware: after prior carrier observations, the current secret window can occupy only a contiguous posterior range, so token probabilities are computed from interval intersections with that range rather than assuming a fresh uniform alpha-bit word at every step.

Paper/code ambiguities (`2^alpha-1` vs `2^alpha`, power-weight normalization, author-only probability-floor filter, singleton-prefix under-consumption) are frozen in ADR-0025 and will be measured explicitly by conformance tooling rather than hidden.


## DAIRstega smoke + synthetic conformance

The first real-Llama point (`alpha=8`, `beta=1`, method-local `top_k=100`) embedded 27 bits into 16 carrier tokens (BPT `1.6875`) with exact token/secret round-trip. The independent state-aware Q path was active. Strict forward KL was infinite on all steps because DAIRstega's method-local candidate restriction removes canonical `P_reference` support; this is expected and not treated as a runtime failure.

Independent synthetic conformance on 134 fixtures produced zero normalized-vs-paper failures. Literal repository allocation differed on 29 fixtures, all in `beta=0.5`/`sqrt` cases because the pinned code rounds `sqrt(p)` to four decimals before re-normalization. `beta=1` allocation matched on every fixture. The pinned `num_same_from_beg` singleton under-consumption remains a separate known repository-vs-paper difference.


## DAIRstega direct author conformance

Pinned `DAIRstega_t.py` direct gate passed: original bit conversion helpers match expected semantics, original prefix helper matches the literal repository transcription, the known singleton interval differs from the paper-defined maximal-prefix rule, all guarded source fragments are present, and the external checkout remains unchanged. This closes implementation-conformance. The publication-characteristic gate has also passed; DAIRstega is now closed at method level for Stage 4.


## DAIRstega publication characteristic + closeout

The 100-token Llama probe reproduced the characteristic capacity–quality/concealment trade-off. `alpha=8,beta=1.0` produced BPT `0.2700`, raw-LM PPL `2.2522`, mean reverse KL `0.867062` and TVD `0.359716`; `alpha=32,beta=0.5` produced BPT `3.3500`, PPL `125.3471`, mean reverse KL `3.609336` and TVD `0.798266`. `beta=0.5` increased BPT at both tested alpha values. All four probe points preserved exact token/text and secret round-trip and used independent state-aware analytic Q.

This reproduces the publication's qualitative family of operating points without claiming numerical Table-3 reproduction. DAIRstega is therefore closed at method level; remaining Stage-4 work is the cross-model compatibility matrix and consolidated v0.3 closeout.
