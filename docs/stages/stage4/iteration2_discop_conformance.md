# Stage 4 — iteration 2: Discop conformance tooling

**Date:** 2026-10-01  
**Scope:** validate the first modern adapter before starting RRC.

## Added

- independent pinned-source Discop step oracle and deterministic fixture suite;
- exact source-oracle vs normalized conformance checker;
- optional direct comparison against the original compiled `stega_cy` Cython
  API from the pinned external checkout;
- real-LM top-p characteristic probe aligned with the five Table-II truncation
  points while clearly separating benchmark-model results from the paper's
  GPT-2/IMDb numerical results;
- frozen Stage-4 Discop conformance configuration and protocol document;
- compatibility dependencies for the pinned Cython step module. In particular,
  `Cython==0.29.37` is used because the author repository declares
  `cython~=0.29.28`; Cython 3.x rejects one of the pinned source expressions
  during cythonization. The author source itself remains unchanged.

## Interpretation boundary

The self-contained source oracle is strong evidence that the normalized
algorithm preserves the pinned step semantics, but it is not a substitute for
executing the original Cython module.  The Cython runner exists specifically to
make that distinction observable.

The characteristic probe likewise does not classify Llama/Qwen BPT or
utilization as a numerical reproduction of Table II because the publication
used GPT-2 and 100 IMDb contexts.  Its immediate purpose is to verify the same
qualitative security characteristic (zero distribution distortion) and measure
capacity/utilization under the benchmark model.


## Build compatibility note

The first iteration of this tooling incorrectly declared `cython>=3.0,<4` for
`discop-reference`. That does not preserve the pinned repository's declared
Cython compatibility range and fails while cythonizing `range(2**capacity)` in
`stega_cy.pyx`. The benchmark extra is therefore pinned to `Cython==0.29.37`,
the latest 0.29 maintenance release, so the author source can be compiled in
the Python 3.12 benchmark environment without editing tracked reference files.
