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
