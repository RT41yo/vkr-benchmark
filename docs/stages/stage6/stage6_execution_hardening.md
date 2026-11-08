# Stage 6 — execution hardening before main experiments

## Scope

This change is infrastructure-only. It does **not** change method algorithms,
Stage 6 operating points, generation policy, secret generation, metric
definitions, streaming/fixed-payload semantics, or canonical `run_id`
materialization.

## 1. Crash-safe success completion

`result.json` with `status=ok` is now the final per-run completion marker. Before
rewriting an existing run attempt, the old marker is removed. Success artifacts
are persisted in this order:

1. `config.json`;
2. `stegotext.txt`;
3. `token_ids.json`;
4. `trace.jsonl.gz`;
5. optional `summary.parquet` upsert;
6. `result.json` with `status=ok` **last**.

Therefore interruption before the final step cannot leave a newly written
`status=ok` marker for a partial attempt.

## 2. Resume validation

The sweep runner no longer skips a run merely because `result.json` says
`status=ok`. `--resume` now requires the complete mandatory artifact set and
validates that:

- canonical `config.json` hashes back to the expected `run_id`;
- `result.json` is readable, has `status=ok`, and contains the same `run_id`;
- `stegotext.txt` is readable UTF-8;
- `token_ids.json` contains integer sender/receiver token-id lists and sender
  length matches `carrier_tokens`;
- `trace.jsonl.gz` is readable JSONL through the gzip stream and has exactly one
  row per carrier token.

`summary.parquet` is deliberately not part of the per-run completion predicate,
because the benchmark already supports `--skip-summary-parquet`.

## 3. Hard-failure sender/transport diagnostics

If encoding finishes but ordinary-text transport or decoding raises an
exception, the original exception type and message are preserved. The runner
attaches non-normative diagnostic state to that same exception.
`persist_failed_run()` writes it to `failure_diagnostics.json`.

For a decode failure this includes the exact original `sender_token_ids`, the
transported text, retokenized `receiver_token_ids`, token-sequence mismatch
diagnostics, payload size, and number of secret bits read. This directly closes
the evidence-loss issue observed in Stage 5, where sender token IDs for hard
failures had to be deterministically reconstructed later.

The file is diagnostic only. It does not convert a hard failure into a
successful run and does not replace the normative full text-channel reliability
path.

## 4. Deliberately unchanged

The following issues are **not** changed by this patch and remain separate
decisions:

- benchmark-wide Git revision is not added to canonical `run_id`;
- failure `status` taxonomy remains the existing generic `error` representation;
- there is no token-level checkpoint inside a single run;
- no Stage 6 manifest or method parameter is modified.
