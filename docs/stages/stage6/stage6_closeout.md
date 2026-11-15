# Закрытие Этапа 6: основные сравнительные эксперименты

## 1. Что выполнено

Stage 6 завершил основные сравнительные эксперименты по `benchmark specification v1.0` без продолжения подбора параметров.

Выполнено:

```text
Llama streaming: 1008 / 1008
Llama RRC:         72 / 72
Qwen streaming:   672 / 672
Qwen RRC:           48 / 48
---------------------------
Всего:            1800 / 1800
```

Все четыре frozen Stage 6 manifests исполнены полностью.

## 2. Integrity checkpoint

Финальный аудит:

```text
unique run_ids: 1800
status=ok: 1703
status=error: 97
missing result.json: 0
missing failure_diagnostics.json: 0
successful runs missing dual-KL/TVD fields: 0
FINAL STAGE 6 INTEGRITY AUDIT: OK
```

`summary.parquet` содержит 1800 уникальных Stage 6 строк с теми же статусами.

Основной экспериментальный набор данных формально зафиксирован как **dataset v1.0** в:

```text
results/stage6/dataset_v1.0_manifest.json
```

Manifest является inventory/provenance-артефактом Stage 6: он фиксирует 1800 canonical `run_id`, четыре sweep, 15 operating points, обе модельные ревизии, 24 промпта, status counts, контракт per-run артефактов и SHA-256 ключевых frozen файлов. Он не заменяет canonical `results/runs/<run_id>/`, а однозначно описывает состав завершённого dataset v1.0.

Таким образом, roadmap-результат **«Основной набор экспериментальных данных v1.0»** закрыт отдельным формальным артефактом.

## 3. Надёжность

Нормативная reliability остаётся полным текстовым каналом.

```text
Exact secret round-trip: 1482 / 1800 = 82.33%
Successful decode but non-exact secret: 221
Hard decode failures: 97
```

Все 97 hard failures имеют `token_sequence_roundtrip_exact=false`; hard failure при сохранённой token sequence не обнаружен.

Этот результат согласуется с Decoder Audit Stage 5 и подтверждает, что direct token-ID decoding следует сохранять только как диагностический инструмент.

## 4. Основные численные наблюдения

- Discop и RRC сохраняют обе KL и TVD на уровне порядка `1e-16` на обеих LM.
- ADG имеет небольшое, но конечное распределительное отклонение.
- Arithmetic Coding при росте `k` резко уменьшает reverse KL/TVD, однако forward KL остаётся `+inf` из-за нулевой массы вне candidate set.
- Bins, Huffman и DAIRstega также имеют `+inf` forward KL в successful runs.
- RRC при фиксированном payload 1024 бит даёт средний BPT 4.152 на Llama и 3.193 на Qwen; медианная длина carrier — 246.5 и 317.5 токена соответственно.
- Общая exact reliability близка между моделями (81.67% Llama против 83.33% Qwen), но отдельные рабочие точки меняются существенно.

Полная описательная сводка находится в `docs/stages/stage6/stage6_main_experiments_report.md`.

## 5. Аналитические артефакты

Добавлен воспроизводимый analysis-only скрипт:

```text
scripts/analyze_stage6_main.py
```

Он читает только зафиксированные Stage 6 `execution.jsonl`, не запускает модели и не пересчитывает стегометоды.

Ключевой inventory/provenance-артефакт dataset:

```text
results/stage6/dataset_v1.0_manifest.json
```

Выходы analysis-скрипта:

```text
results/stage6/stage6_execution_summary.json
results/stage6/stage6_operating_point_summary.csv
results/stage6/stage6_operating_point_summary.json
results/stage6/stage6_operating_point_summary.md
results/stage6/stage6_cross_model_summary.csv
```

Численные benchmark-векторы агрегируются по successful runs; exact end-to-end reliability использует все planned runs. Forward KL `+inf` сохраняется и не заменяется finite-only средним.

## 6. Что намеренно не входит в Stage 6

Stage 6 не включает:

- стегоанализ и ROC-AUC — Stage 7;
- окончательные ответы на RQ и Парето-фронты — Stage 8;
- новые parameter sweeps;
- изменение алгоритмов или operating points;
- выбор единственного «лучшего» метода.

## 7. Готовность к следующему этапу

Основной экспериментальный набор Stage 6 сформирован и прошёл integrity audit. Следующий этап может использовать frozen Stage 6 stegotexts/metadata для формирования сопоставленных выборок стегоанализа, не изменяя результаты Stage 6.
