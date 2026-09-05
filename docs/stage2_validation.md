# Stage-2 validation snapshot

This file records the first end-to-end normalized smoke comparison after capacity, entropy, distortion, raw-LM quality, reliability, performance, and persistent run storage were integrated. It is a pipeline-validation snapshot, not a final benchmark result.

## Conditions

All three runs used the same model revision, prompt, secret id, normalized `P_reference` policy and 16-carrier-token termination target:

- model: `meta-llama/Llama-3.2-3B`;
- revision: `13afe5124825b4f3751f836b40dafda64c1ed062`;
- prompt id: `p000001` (`The history of artificial intelligence began`);
- secret id: `000001`;
- common reference policy: temperature `1`, common top-k disabled, top-p `1`;
- Bins: `block_size=2`;
- Huffman: `bits_per_word=2`;
- Arithmetic Coding: `precision=16`, method-internal `top_k=50000`.

## Observed results

| Method | BPT | Entropy util. (%) | KL ref→stego | TVD mean | Raw-LM NLL | Raw-LM PPL | BER | Encode ms/token | Decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bins | 2.0000 | 64.603 | inf | 0.596192 | 2.879901 | 17.812511 | 0.000000 | 18.604 | 18.345 |
| Huffman | 2.1875 | 52.606 | inf | 0.422220 | 2.205473 | 9.074547 | 0.000000 | 20.900 | 17.161 |
| Arithmetic Coding | 2.4375 | 65.838 | inf | 0.053387 | 1.800315 | 6.051556 | 0.000000 | 60.683 | 63.055 |

All three runs additionally passed the text-only transport checks:

- `token_sequence_roundtrip_exact = true`;
- `roundtrip_exact = true`;
- decoder complete;
- zero bit errors.

## Interpretation limited to this smoke run

Arithmetic Coding simultaneously produced the highest payload rate, the lowest TVD and the lowest raw-LM PPL, but its encode/decode wall-clock cost was roughly three times the two simpler baselines. Huffman added very little method-specific computational overhead and provided intermediate distribution/quality behavior. Bins was the strongest distortion baseline in this particular trajectory.

The benchmark-native KL was infinite at every carrier step for all three methods because each exact `Q_stego` lost support relative to the broad baseline `P_reference`; no smoothing is applied. This does not make TVD or the opposite author-compatible KL redundant. Stage 3 will calculate both KL directions where required for author-result convergence.

These observations must not be generalized beyond pipeline validation: this snapshot is one model × one prompt × one secret × one operating point per method × 16 carrier tokens. Later stages require repeated, aggregated experiments and parameter sweeps.

## Storage validation

The three completed normalized runs were persisted through the canonical Stage-2 storage path. `results/summary.parquet` uses one row per canonical run and contains the common identifiers, operating-point hash, metric vector and status. Re-running the same canonical configuration uses the same `run_id` and upserts rather than appending a duplicate logical run.

To regenerate a compact table from the current local Parquet file:

```bash
python scripts/summarize_results.py results/summary.parquet
```

Optional filtering is available with `--model-id` and `--prompt-id`; `--output <file.md>` writes the same deterministic Markdown table to disk.
