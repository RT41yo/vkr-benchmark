# Result storage

Runtime artifacts are generated locally and are not committed to Git.

Default layout:

```text
results/
├── runs/
│   └── <run_id>/
│       ├── config.json
│       ├── result.json
│       ├── stegotext.txt
│       ├── token_ids.json
│       └── trace.jsonl.gz
└── summary.parquet
```

`run_id = SHA256(canonical_config_json)[:16]`.

`summary.parquet` requires the optional storage dependency:

```bash
pip install -e ".[storage]"
```

The files under `results/runs/` and `summary.parquet` are ignored by Git; this
README and `runs/.gitkeep` remain tracked.

## Stage 3 Figure-3 full reproduction

`results/stage3/paper_reproduction/figure3_full/` contains the completed Step-3.10 author-compatible sweep: 69 point/replicate JSONL shards (23 points × 3 replicates, 80 contexts each), `run_state.json`, `summary.json`, and `figure3_points.csv`. Step 3.11 adds `interpretation.json`, `claim_assessment.csv`, and `figure3_reproduction.svg`; these are deterministic analysis artifacts generated from the completed sweep and the frozen precision-probe interpretation. Incomplete `*.tmp` and failed diagnostic shards are not successful benchmark artifacts.

## Stage 3 matched author/normalized comparison

Step 3.12 writes paired differential evidence to `results/stage3/matched_author_normalized/`: one `records.jsonl` entry per frozen point/context pair, a compact `paired_comparison.csv`, and `summary.json`. These outputs are generated from committed author Figure-3 records plus normalized reruns; numerical closeness is intentionally interpreted only in Step 3.13.

## Stage 3 conformance analysis

Step 3.13 adds deterministic interpretation artifacts to `results/stage3/matched_author_normalized/`: `conformance_analysis.json` and `conformance_table.csv`. They are derived only from the frozen Step-3.12 matched outputs; no new LM generation is performed and no post-hoc numerical closeness threshold is introduced.


## Stage 3 closeout

Step 3.14 materializes `results/stage3/comparison.csv` and `results/stage3/comparison.parquet` as a compact four-row matched-conformance table, plus `stage3_closeout_summary.json` and `stage3_readiness.json`. These files summarize, but do not replace, the full Figure-3 and 32-pair evidence stored in the subdirectories above. `comparison.parquet` is generated with the project `storage` extra (`pyarrow`).
