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
