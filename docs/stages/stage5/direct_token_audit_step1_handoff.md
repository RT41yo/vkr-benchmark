# Stage 5 Decoder Audit — Step 1 (direct-token replay)

Status: diagnostic tooling only; no decoder/encoder changes and no claims of attribution.

## Scope

`scripts/audit_stage5_direct_tokens.py` checks the Pilot-B manifest/log 504-run join, reads successful runs' original sender IDs from `results/runs/<run_id>/token_ids.json`, regenerates carriers using the canonical config to check replay determinism and reconstruct missing carriers for failed runs, and decodes **sender IDs directly** (no text transport). It uses benchmark strict exactness, including excess incremental recovered bits. Reconstructed hard-failure runs are labelled and must not be reported as proof that the missing original sender tokens were identical.

## Run locally (from repository root)

```bash
python scripts/audit_stage5_direct_tokens.py --inventory-only
python scripts/audit_stage5_direct_tokens.py --run-id 6c030aa632743db9 --output-dir decoder_audit_local_smoke
python scripts/audit_stage5_direct_tokens.py --output-dir decoder_audit_local_full
```

For another location of local persisted run data, specify `--run-root /path/to/results/runs`. Run on the pinned model/runtime described in Stage 5 documentation. A diagnostic `audit_error` or `replay_nonreproducible` is **not** a decoder defect; investigate before drawing conclusions. The script intentionally refuses to overwrite existing output and refuses output inside repository `results/`, `src/`, or `configs/`. Do not use the same output directory for smoke and full runs.

For the full audit, check `selected_count=504`, `status_counts`, and all `reencode_sender_ids_exact` for persisted successful runs. The audit result and its provenance must be reviewed before progressing to independent decoder conformance. This step does **not** establish conformance to the papers, author implementations, or text-channel attribution.

## Tests in this handoff environment

- `PYTHONPATH=src pytest -q tests/unit`: 366 passed.
- `PYTHONPATH=src pytest -q --ignore=tests/integration`: 388 passed.
- `--inventory-only`: 504 entries, 466 successful runs without local sender IDs in ZIP, 38 requiring reconstruction.
- Model-backed 504-run replay: **not executed** because `results/runs/` and LM weights are intentionally absent from snapshot.
