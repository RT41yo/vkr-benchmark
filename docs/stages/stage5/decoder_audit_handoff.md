# Stage 5 — handoff к Decoder Audit

## Контрольная точка

Pilot A завершен и диагностирован; Pilot B завершен по плану 504/504 run. Текущий следующий шаг — **Stage 5 Decoder Audit** до окончательной интерпретации Pilot B и до выбора Stage-6 operating points.

Основные файлы контекста:

- `docs/stages/stage5/stage5_plan.md`;
- `docs/stages/stage5/pilot_a_analysis.md`;
- `docs/stages/stage5/pilot_a_reliability_analysis.md`;
- `docs/stages/stage5/pilot_b_operating_point_selection.md`;
- `docs/stages/stage5/pilot_b_checkpoint.md`;
- `docs/stages/stage5/rrc_precision_calibration_report.md`;
- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/execution.jsonl`;
- `results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl`;
- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics.jsonl`;
- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics_summary.json`;
- `src/vkr_benchmark/diagnostics/`;
- `scripts/diagnose_stage5_hard_failures.py`.

## Pilot B checkpoint

- 21 operating points;
- 8 prompts;
- 3 paired replicates;
- 256 carrier tokens;
- 504 planned / 504 recorded;
- 466 process-successful;
- 38 hard failures;
- среди 466 `ok` еще 62 имеют `roundtrip_exact=false`;
- 404/504 exact end-to-end recovery.

По методам exact / silent inexact / hard failure:

- Bins: 51 / 45 / 0;
- Huffman: 67 / 9 / 20;
- Arithmetic Coding: 134 / 7 / 3;
- ADG: 23 / 1 / 0;
- Discop: 22 / 0 / 2;
- DAIRstega: 107 / 0 / 13.

Причина этих 100 non-exact run пока не считается установленной.

## Почему нужен Decoder Audit

Pilot A показал для своих проблемных run, что direct-token decode проходит точно, а проблемы появляются после `token IDs → text → token IDs`. Однако Pilot B добавил новые failure cases у Arithmetic, ADG и Discop. Кроме того, часть decoder logic в benchmark была реализована/нормализована самостоятельно, а не всегда получена как готовый авторский decoder.

Поэтому перед выводом «проблема в text transport» нужно независимо проверить сам decoder layer.

## Минимальный план Decoder Audit

1. Direct-token replay всех 504 Pilot-B run по исходным sender token IDs.
2. Проверка exact recovery без текстового канала для каждого run.
3. Отдельная author/paper conformance проверка decoder logic по каждому методу.
4. Усиленная независимая oracle-проверка для методов без готового авторского decoder / с самостоятельно реализованным inverse.
5. Для text-channel path — сопоставить первую tokenization divergence с первой ошибкой secret/decode.
6. Классифицировать 100 non-exact run только после этих проверок.

## Локальные данные

Полные per-run artifacts Pilot B находятся в локальном рабочем репозитории в `results/runs/<run_id>/` и не включаются в git/ZIP snapshot по политике проекта. Decoder Audit должен использовать эти локальные артефакты; если нужны новые агрегированные диагностические результаты, они сохраняются в `results/stage5/...` и затем могут быть зафиксированы в репозитории.

## RRC

RRC не входит в 504-run Pilot B. Отдельно остается открытым вопрос finite precision при построении probability boundaries: `precision_margin={20,40,60}` дал plateau и 24/24 exact, но часть forward KL была `+inf` при практически нулевых reverse KL/TVD. Окончательный RRC regime не заморожен.
