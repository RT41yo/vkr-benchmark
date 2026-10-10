# Stage 4 — iteration 16: compatibility matrix closeout

Выполнен реальный запуск матрицы совместимости `7 методов × 2 модели` на pinned локальных checkpoints.

Результат:

- Llama-3.2-3B: 7/7 `PASS`;
- Qwen3-4B-Base: 7/7 `PASS_WITH_ADAPTATION`;
- direct secret round-trip: 14/14;
- text token round-trip: 14/14;
- text secret round-trip: 14/14.

Для Qwen подтверждена ожидаемая общая адаптация: 267 output ID модели не имеют отображения в токенизаторе и исключаются до передачи `P_reference` стегометоду с последующей перенормировкой. Это не method-specific workaround.

Metric instrumentation была отключена, поэтому результат относится только к технической совместимости и reliability на фиксированной smoke-точке. BPT/payload, скорость и другие количественные различия между моделями не интерпретируются как сравнительные научные результаты.

Зафиксированы артефакты:

- `results/stage4/compatibility_matrix.json`;
- `results/stage4/compatibility_matrix.md`;
- `docs/stages/stage4/compatibility_matrix_report_ru.md`.

После этого compatibility matrix как обязательный результат Stage 4 закрыта. Остается итоговая Typst-презентация и формальный Stage-4/v0.3 closeout.
