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

`results/stage3/paper_reproduction/figure3_full/` is reserved for the Step-3.10 author-compatible sweep. Successful execution produces 69 point/replicate JSONL shards (23 points × 3 replicates, 80 contexts each), plus `run_state.json`, `summary.json`, and `figure3_points.csv`. Incomplete `*.tmp` and failed diagnostic shards are not successful benchmark artifacts.
