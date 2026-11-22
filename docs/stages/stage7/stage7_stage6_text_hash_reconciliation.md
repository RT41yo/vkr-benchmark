# Stage 7 — read-only audit and conditional repair of legacy Stage-6 text SHA-256

## Symptom and scope

`build_stage7_detector_dataset.py --allow-confirmed-rrc-cutoff --dry-run`
may fail with `ConfigurationError: stego source text corrupt: <sample_id>`
for a frozen Stage-6 **test** sample. For example,
`08a78ee8640b39a4acbc9b8d` corresponds to frozen Stage-6 run
`b48273c421fd14ae` (Qwen, Bins/b1, `s6p0018`, `r02`).

The Stage-7 test index may have been generated **before** the carrier-text
newline preservation hotfix. Older Stage-7 source indexing used
`Path.read_text()`, which normalizes CR (`\r`) and CRLF (`\r\n`) into LF (`\n`).
The resulting `stego_text_sha256` may differ from the hash of the original
UTF-8 text bytes even though the Stage-6 file has not changed. The old hash
was carried into `data/stage7/dataset/all_cover_plan.jsonl`.

A mismatch alone does **not** prove that this benign explanation applies to
the user's data. Do not regenerate or modify Stage-6 runs or generated covers.

## Controlled reconciliation

Use the new command, default **read-only**:

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark
source .venv/bin/activate
PYTHONPATH=src python scripts/reconcile_stage7_stage6_text_hashes.py
```

The script:

1. Re-audits all 1800 frozen Stage-6 source runs using byte-preserving text I/O.
2. Compares all source-index metadata with the original Stage-7 index. Any
   discrepancy other than `stego_text_sha256` is a hard error.
3. Rechecks each differing SHA-256 against the **legacy** `Path.read_text()`
   representation and the **exact raw bytes**. If the old hash cannot be
   reproduced by CR/CRLF normalization, it refuses repair.
4. Checks exactly 5639 planned pairs, including all 1800 Stage-6 test entries,
   unchanged run IDs, group assignments, lengths, seeds, and method metadata.
5. Prints all permitted changes but changes nothing in the default mode.

If, and only if, the read-only check reports only verified legacy newline
normalization, explicitly authorize the metadata-only reconciliation:

```bash
PYTHONPATH=src python scripts/reconcile_stage7_stage6_text_hashes.py --apply
```

The command preserves byte-level backups of the two original derived files in
`data/stage7/cache/stage6_hash_reconciliation/` and writes:

- `data/stage7/dataset/stage6_test_index.jsonl` (corrected Stage-6 test hashes);
- `data/stage7/dataset/all_cover_plan.jsonl` (matching corrected test hashes);
- `results/stage7/stage6_hash_reconciliation.json` (compact audit report,
  including old/new SHA-256 values and identifiers).

**No other Stage-7 plan fields change.** The original Stage-6 source files,
Stage-7 stegotexts, generated covers, frozen Stage-6 manifests and model
settings are not written to. This metadata migration does not require
another GPU generation step.

If a mismatch is not demonstrably the old newline-normalization bug, the
command aborts without modifying metadata. Investigate the underlying text
source separately instead of relaxing the dataset-integrity contract.

## Dataset assembly after successful reconciliation

```bash
PYTHONPATH=src python scripts/build_stage7_detector_dataset.py \
  --allow-confirmed-rrc-cutoff --dry-run

PYTHONPATH=src python scripts/build_stage7_detector_dataset.py \
  --allow-confirmed-rrc-cutoff
```

Expected after the existing documented RRC validation cutoff: **5639 pairs**,
**11278 detector records**, **152 prompt groups**, with exactly **1800** frozen
Stage-6 test pairs. If the dry-run finds a different error, stop and diagnose
it separately; do not skip or silently exclude test examples.
