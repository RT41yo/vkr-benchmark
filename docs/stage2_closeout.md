# Закрытие Этапа 2: базовая экспериментальная инфраструктура

Статус: **техническая часть Этапа 2 готова к закрытию** после прохождения `scripts/check_stage2_readiness.py`. Этот документ сопоставляет фактически реализованное состояние репозитория с ROADMAP. Презентация Typst является отдельным формальным результатом ROADMAP и оформляется после технического readiness-gate на основе `docs/stage2_presentation_outline.md`.

## 1. Сопоставление с составом работ ROADMAP

### Работа 1. Адаптировать Bins, Huffman и Arithmetic Coding к единой инфраструктуре

Статус: **выполнено**.

Все три базовых метода подключены через общий lifecycle `StegoMethod -> EncoderSession / DecoderSession`, используют общий `LMAdapter`, каноническое `P_reference`, изолированные источники случайности и один text-only transport. Decoder не получает sender token IDs напрямую.

Технические evidence-paths:

- `src/vkr_benchmark/methods/bins.py`;
- `src/vkr_benchmark/methods/huffman.py`;
- `src/vkr_benchmark/methods/arithmetic.py`;
- `src/vkr_benchmark/runner/streaming.py`;
- `src/vkr_benchmark/runner/experiment.py`.

### Работа 2. Реализовать единый порядок входов, encode/decode и базовых метрик

Статус: **выполнено для Stage-2 baseline scope**.

Единый runner задает prompt по `prompt_id`, секрет по `secret_id`, method params, common generation policy и termination target. Реализованы:

- payload capacity и BPT;
- reference entropy и entropy utilization;
- `D_KL(P_reference || Q_stego)` и TVD;
- raw-LM NLL/PPL;
- reliability и BER через ordinary-text channel;
- encode/decode timing и decomposition на LM / distribution processing / stego algorithm.

Определения зафиксированы в `docs/metrics.md`. Уточнения, не меняющие frozen specification v0.1, накоплены в ADR и `specification/benchmark_specification_v0.1_notes.md`.

### Работа 3. Обеспечить единый формат параметров и результатов

Статус: **выполнено**.

Canonical run идентифицируется через `run_id = SHA256(canonical_config_json)[:16]`. Успешный run сохраняет `config.json`, `result.json`, `stegotext.txt`, `token_ids.json`, `trace.jsonl.gz`; общая таблица хранится в `results/summary.parquet` с upsert по `run_id`.

## 2. Сопоставление с результатами Этапа 2

### 1. Репозиторий бенчмарка v0.2 с Bins, Huffman и Arithmetic Coding

**Готово как технический milestone.** Scope v0.2 зафиксирован в `docs/releases/v0.2.md`. Git tag/release рекомендуется ставить только после прохождения readiness-gate и оформления Stage-2 Typst presentation.

### 2. Набор автоматических проверок кодирования и декодирования

**Готово.** Unit/integration tests покрывают общий runner, три метода, text transport, metric layers, storage/reporting и Stage-2 readiness. Реальные GPU smoke runs для трех методов дополнительно прошли на общей Llama-конфигурации; ранее normalized E2E также проверялся на Llama/Qwen.

### 3. Пример единой таблицы экспериментальных результатов

**Готово.** `results/summary.parquet` содержит по одной canonical row для Bins/Huffman/Arithmetic Coding; `scripts/summarize_results.py` формирует Markdown-представление. Snapshot условий и интерпретации хранится в `docs/stage2_validation.md`.

Последняя техническая smoke-таблица:

| Method | BPT | Entropy util. (%) | KL ref→stego | TVD mean | Raw-LM PPL | BER | Encode ms/token | Decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arithmetic Coding | 2.4375 | 65.84 | inf | 0.0534 | 6.0516 | 0 | 62.551 | 62.051 |
| Bins | 2.0000 | 64.60 | inf | 0.5962 | 17.8125 | 0 | 18.734 | 18.486 |
| Huffman | 2.1875 | 52.61 | inf | 0.4222 | 9.0745 | 0 | 17.292 | 17.213 |

Это **pipeline-validation snapshot**, а не итоговое сравнение методов.

### 4. Документ с точными определениями и реализацией базовых метрик

**Готово:** `docs/metrics.md` + ADR-0010/0011/0012/0013.

### 5. Презентация Typst с результатами Этапа 2

**Требует оформления после технического readiness-gate.** Содержание и рекомендуемая структура подготовлены в `docs/stage2_presentation_outline.md`. До появления финального `.typ` формально весь ROADMAP deliverable set Этапа 2 нельзя считать полностью закрытым, хотя программная часть уже завершена.

## 3. Что проверяет readiness-gate

Запуск:

```bash
python scripts/check_stage2_readiness.py --run-tests \
  --json-output results/stage2_readiness.json
```

Gate требует:

- наличие ключевых implementation/config/documentation файлов;
- читаемый `results/summary.parquet`;
- `status=ok` для Bins, Huffman и Arithmetic Coding;
- уникальные `run_id`;
- `BER=0`, exact payload roundtrip и exact token text-roundtrip для smoke rows;
- sanity базовых metric vectors;
- `q_mode=analytic_exact` у трех baseline methods;
- отсутствие tracked `environment/stage1_reference/hf.txt`;
- при `--run-tests` успешное завершение полного `pytest -q`.

Readiness-check **не** требует конечного `KL(P_reference || Q_stego)`: `+inf` является допустимым математическим результатом при потере support.

## 4. Ограничения Stage-2 snapshot

Текущие численные значения нельзя использовать как итоговый научный ranking. Они получены на одном model revision, одном technical prompt, одном secret, одном operating point каждого метода и 16 carrier tokens. Timing не агрегирован по повторным runs. Prompt `p000001` не является финальным benchmark corpus.

`D_KL(P_reference || Q_stego)` на smoke-run бесконечен для всех трех методов из-за support truncation. В Stage 3 для авторской воспроизводимости дополнительно считается противоположное направление `D_KL(Q_stego || P_reference)` согласно ADR-0012; benchmark-native направление не заменяется.

Parameter sweeps, финальные диапазоны параметров, окончательное число prompts/secrets/repeats и benchmark specification v1.0 относятся к Этапу 5, а не к Этапу 2.

## 5. Handoff в Этап 3

После readiness-gate и Stage-2 presentation переход к Этапу 3 не требует изменения common infrastructure. Для каждого из Bins/Huffman/Arithmetic Coding следует сформировать author-compatible режим/запуск, зафиксировать точные upstream settings, воспроизвести publication-level characteristic result и отдельно сопоставить normalized vs author-compatible behavior.

Особенно важно не смешивать два режима эксперимента:

- `normalized` — единая инфраструктура benchmark;
- `author-compatible` — режим для проверки сходимости к исходной публикации/репозиторию.

Этап 3 должен вычислять оба направления KL там, где это возможно, и фиксировать direction/unit/aggregation author metric явно.
