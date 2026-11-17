# ADR-0028: Изолированный протокол Stage 7 — matched cover/stego и эмпирический стегоанализ

**Статус:** принято как реализационное решение Stage 7, на основании обсуждения методологии

**Контекст:** Stage 6 заморожен (`main@5f695f1`), 1800 нормализованных запусков. Benchmark specification v1.0 §12 и §23 оставляет конкретную архитектуру стегодетектора и split для Stage 7. ADR-0012, 0014, 0020, 0026, 0027 и fixed-payload семантика ADR-0018 остаются обязательными.

## Решение

1. **Охват:** все 15 operating points × Llama-3.2-3B / Qwen3-4B-Base; Stage 6 canonical texts — неизменяемый внешний test. У каждого LM независимый pooled RoBERTa-base и простой TF-IDF + Logistic Regression baseline. Положительный класс — равномерная смесь 15 *точек*, а не семи методов.
2. **Train/validation:** 128 новых prompt groups из тех же pinned CNN/DailyMail 3.0.0 test и алгоритма `sha256_rank_v1`, selection ranks 32–159, разбитые группами `article_id` (96/32). Stage 6 ranks 8–31 — test (24 статьи); Stage 5 ranks 0–7 исключены. Данные и split хешируются и замораживаются до обучения.
3. **Matched sampling:** cover на той же LM/revision и том же промпте с тем же `P_reference` и независимо воспроизводимой случайностью. Контрольный сэмплер — inverse CDF по возрастающим token IDs с FP64 накоплением вероятности, FP32 `P_reference` не изменяется. Отдельный cover для каждого `(LM, point, article, replicate)`.
4. **Длины:** streaming: 256 sender carrier tokens в каждом классе. RRC: fixed payload 1024 bits, не меняется; основной cover берёт *фактическую sender-token длину* соответствующего RRC текста. Дополнительная диагностика unmatched-length использует 256-token cover и length-only классификатор, не подменяет основной результат.
5. **Ошибки Stage 6:** `status=ok` — `stegotext.txt`, `status=error` после decode — `failure_diagnostics.json.transport_text`. Полный corpus и число ошибок предварительно проверяются; ни одна неуспешная по round-trip запись не выбрасывается по признаку ошибки декодирования. При отсутствии полного текста конвейер останавливается, не выполняет selective filtering.
6. **Случайность:** split seed 701; Stage-7 secret domain оставлен нормативным SHAKE256, свежий `secret_id` на новую статью/replicate; control seeds детерминированно получаются SHA-256 доменом `VKR-BENCHMARK-STAGE7-v1`; RoBERTa seeds 711/712/713; bootstrap seed 731; factor selection 741. Stage-6 секреты остаются неизменными.
7. **Обнаружимость:** text-only, prompt-unaware, без метрик и внутренних ID; основной ROC-AUC по точке метода, дополнительные accuracy/balanced accuracy/precision/recall/F1/PR-AUC, фиксированный по validation порог и cluster bootstrap 95% CI по `article_id`. Main test никогда не участвует в model selection.
8. **Implementation factors:** AC precision 16/32/48 при top_k=2048, RRC precision_margin 0/10/20/40/60 плюс offline FP64-vs-Decimal, диагностика text-channel tokenization всех 1800 Stage-6 runs. Отдельные артефакты, не новые Stage-6 points.
9. **Изоляция:** никакие Stage-7 команды не записывают в `results/runs/`, `results/summary.parquet`, Stage-6 manifest/config/analyses. Все новые большие тексты и веса по умолчанию gitignored. Подлинность Stage-6 hashes повторно проверяется на closeout.
10. **Границы:** ROC-AUC≈0.5 означает отсутствие обнаруженного сигнала данным detector, не доказательство безопасности. RRC fixed-payload нельзя отождествлять со streaming; Парето-анализ только Stage 8. Время detector/instrumentation не включается в timing стегометода.

## Последствия и ограничения

- Для train/validation требуется новая генерация stego/cover; Stage-6 per-run artifacts обычно отсутствуют в ZIP, поэтому независимый внешний тест нельзя собрать из snapshot без локальных `results/runs/`.
- Pooled detector ограничивает постановку противника; method-specific detectors могут оказаться сильнее. Число независимых test prompt groups = 24; неопределённость может быть большой.
- Stage-7 train использует уникальные secret_id per-prompt, Stage-6 test использовал shared secret_id per-replicate; это отмечается как ограничение сопоставимости, но не изменяет спецификацию генератора секретов.
- Первой реализационной контрольной точкой являются подготовка prompt groups, реестра frozen test samples и direct cover sampling. Обучение и крупные GPU-прогоны выполняются отдельными последующими checkpoint, после GPU smoke и аудита dataset.

**Нормативная детализация:** `docs/stages/stage7/stage7_protocol.md`.
