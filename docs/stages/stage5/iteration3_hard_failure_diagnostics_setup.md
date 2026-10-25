# Stage 5 — Iteration 3: диагностика hard failures Pilot A

## Мотивация

Pilot A завершил 360 planned streaming runs. В sweep log зафиксированы 13 hard failures:

- Huffman — 8;
- DAIRstega — 5.

Отдельная post-hoc проверка 29 successful-but-inexact runs показала, что во всех них `token_sequence_roundtrip_exact=false`. Это указывает на обычный текстовый канал (`token ids -> text -> retokenized token ids`) как на вероятную первопричину reliability failures, но для 13 hard failures обычный `ExperimentExecution` не был сохранён, поскольку decoder выбросил исключение до завершения run.

## Цель диагностики

Не изменяя семантику Huffman или DAIRstega, для каждого из 13 исходных run воспроизвести sender path и выполнить два decoder control path:

1. **direct-token control** — декодировать исходные sender token IDs, без detokenization/retokenization;
2. **ordinary-text replay** — передать carrier через обычный benchmark text channel и декодировать receiver token IDs до исходного hard failure.

Hard failure классифицируется как `text_transport_induced` только если одновременно выполняются три условия:

- direct-token decode восстанавливает секрет точно;
- text transport меняет token sequence;
- transported decoder падает на шаге не раньше первого transport mismatch.

Любой другой случай остаётся `requires_investigation`; diagnostic script не должен автоматически объяснять неясные failures текстовым каналом.

## Артефакты диагностики

Executable script:

`python scripts/diagnose_stage5_hard_failures.py ...`

Он читает canonical Pilot-A manifest и фактический `execution.jsonl`, выбирает только error records Huffman/DAIRstega и загружает Llama один раз.

Для каждого run сохраняются:

- source и replayed error;
- sender/receiver token counts;
- первый transport token mismatch;
- небольшое token-ID окно вокруг mismatch;
- результат direct-token control;
- decoder failure step;
- окно sender/receiver token IDs около failure;
- method-local decoder state на failure step.

Для Huffman дополнительно сохраняются candidate token IDs и проверка присутствия sender/receiver token в текущем Huffman candidate set.

Для DAIRstega сохраняются positive-width interval token IDs и текущее decoder secret-window state.

Ожидаемые runtime outputs:

- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics.jsonl`;
- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics_summary.json`.

## Методологическое ограничение

Диагностика не чинит text channel, не добавляет BPE repair и не меняет method adapters. Ее задача — установить первопричину уже наблюдавшихся hard failures перед отбором operating points для Pilot B.

## Результат

Диагностика выполнена для всех 13 исходных hard failures:

- direct-token decode exact: 13/13;
- text transport изменил token sequence: 13/13;
- исходная hard error воспроизведена: 13/13;
- classification: `text_transport_induced` — 13/13.

Итоговая интерпретация вместе с 29 successful-but-inexact runs зафиксирована в:

`docs/stages/stage5/pilot_a_reliability_analysis.md`.
