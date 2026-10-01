# Stage 4 — Discop conformance and reproducibility protocol

## Scope

Discop is checked at three distinct levels so that algorithmic conformance is
not confused with paper-level numerical reproduction.

1. **Pinned-source oracle vs normalized adapter.** A separate Python oracle is a
   literal translation of `cy_encode_step`/`cy_decode_step` from
   `comydream/Discop@3c3a10099a242eae405b49cc4d09fba1abb148ad`.  The oracle
   and normalized adapter receive the same canonical probability vector,
   secret bits and Python PRNG seed.  Token choice, payload consumption,
   recovered bits and PRNG draw count must match exactly over fixed and
   randomized synthetic fixtures.  This is source-level conformance; it does
   not claim that the original Cython binary was executed.

2. **Original Cython step API vs normalized adapter.** When a pinned local
   checkout can be compiled, `stega_cy.encode_step` and `decode_step` are
   executed directly on the same synthetic fixtures.  The checkout HEAD must
   equal the frozen commit and tracked author files must remain unchanged.
   This isolates the algorithmic core from legacy GPT-2/dataset dependencies.

3. **Real-LM characteristic probe.** The normalized adapter is run for 100
   carrier tokens at the paper top-p grid `0.80, 0.92, 0.95, 0.98, 1.00`.
   We record BPT, reference entropy, entropy utilization, both benchmark KL
   directions, TVD and ordinary-text roundtrip.  This probe uses the benchmark
   model and explicit prompt, so it is **not** a numerical reproduction of the
   paper's GPT-2/IMDb Table II values.

## Publication target

Ding et al. (IEEE S&P 2023), Table II reports for recursive Discop on GPT-2:

| top-p | Capacity, bit/token | Entropy, bit/token | Utilization | Ave KLD | Max KLD |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 3.48 | 3.79 | 0.92 | 0 | 0 |
| 0.92 | 4.55 | 4.86 | 0.94 | 0 | 0 |
| 0.95 | 4.84 | 5.18 | 0.94 | 0 | 0 |
| 0.98 | 5.29 | 5.59 | 0.95 | 0 | 0 |
| 1.00 | 5.76 | 6.08 | 0.95 | 0 | 0 |

The paper protocol uses 100 IMDb texts, the first three sentences as context,
and 100 generated tokens per context.  Therefore the short Stage-4 Llama/Qwen
probe is interpreted only as a characteristic check.  A future exact
paper-compatible numerical run must preserve the paper model/data protocol or
be explicitly classified as a near-publication replication.


## Reference build environment

The pinned Discop repository declares `cython~=0.29.28`. The Stage-4
compatibility extra therefore pins `Cython==0.29.37`. This is intentionally
separate from the benchmark runtime implementation: the old Cython dependency
is needed only to build the untouched author `stega_cy.pyx` for direct
step-level conformance. Cython 3.x is not used for this reference build because
it rejects the pinned source during cythonization.

## Q_stego handling

The normalized adapter reports `Q_stego = P_reference` as
`analytic_exact/reference_equality_certificate` with source
`analytic_theory`.  The benchmark never imports the author's hard-coded `kld=0`
field as measured evidence.  Independent synthetic sampling remains a separate
validation of the equality claim, while central benchmark metrics compute both
KL directions and TVD from the certificate.

## Acceptance for the Discop adapter

Discop can move to Stage-4 "integrated and conformant" status when:

- the full unit/regression suite passes;
- source-oracle conformance has zero mismatches;
- the original Cython step-level comparison passes, or an environment/build
  limitation is recorded explicitly;
- Llama normalized ordinary-text roundtrip is exact on the characteristic
  probe and the same probe is attempted on Qwen;
- zero reported KL/TVD is accompanied by the analytic-certificate provenance;
- observed utilization is reported without treating a different-model value as
  a direct numerical reproduction of Table II.
