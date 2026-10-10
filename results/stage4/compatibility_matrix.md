# Stage 4 — матрица совместимости методов и моделей

Статусы отражают короткую техническую проверку совместимости, а не сравнительное качество методов.
`PASS_WITH_ADAPTATION` означает, что общая инфраструктура исключила output ID модели без отображения в токенизаторе и перенормировала P_reference.

| Метод | meta-llama/Llama-3.2-3B | Qwen/Qwen3-4B-Base |
|---|---|---|
| Bins | PASS | PASS_WITH_ADAPTATION |
| Huffman | PASS | PASS_WITH_ADAPTATION |
| Arithmetic Coding | PASS | PASS_WITH_ADAPTATION |
| Discop | PASS | PASS_WITH_ADAPTATION |
| RRC | PASS | PASS_WITH_ADAPTATION |
| ADG | PASS | PASS_WITH_ADAPTATION |
| DAIRstega | PASS | PASS_WITH_ADAPTATION |

## Диагностика

| Метод | Модель | direct secret RT | text token RT | text secret RT | carrier | payload | adaptation | error |
|---|---|---:|---:|---:|---:|---:|---|---|
| Bins | meta-llama/Llama-3.2-3B | True | True | True | 16 | 32 | - | - |
| Huffman | meta-llama/Llama-3.2-3B | True | True | True | 16 | 35 | - | - |
| Arithmetic Coding | meta-llama/Llama-3.2-3B | True | True | True | 16 | 39 | - | - |
| Discop | meta-llama/Llama-3.2-3B | True | True | True | 16 | 23 | - | - |
| RRC | meta-llama/Llama-3.2-3B | True | True | True | 12 | 64 | - | - |
| ADG | meta-llama/Llama-3.2-3B | True | True | True | 16 | 71 | - | - |
| DAIRstega | meta-llama/Llama-3.2-3B | True | True | True | 16 | 27 | - | - |
| Bins | Qwen/Qwen3-4B-Base | True | True | True | 16 | 32 | exclude_and_renormalize_267_output_only_ids | - |
| Huffman | Qwen/Qwen3-4B-Base | True | True | True | 16 | 27 | exclude_and_renormalize_267_output_only_ids | - |
| Arithmetic Coding | Qwen/Qwen3-4B-Base | True | True | True | 16 | 46 | exclude_and_renormalize_267_output_only_ids | - |
| Discop | Qwen/Qwen3-4B-Base | True | True | True | 16 | 35 | exclude_and_renormalize_267_output_only_ids | - |
| RRC | Qwen/Qwen3-4B-Base | True | True | True | 26 | 64 | exclude_and_renormalize_267_output_only_ids | - |
| ADG | Qwen/Qwen3-4B-Base | True | True | True | 16 | 33 | exclude_and_renormalize_267_output_only_ids | - |
| DAIRstega | Qwen/Qwen3-4B-Base | True | True | True | 16 | 25 | exclude_and_renormalize_267_output_only_ids | - |
