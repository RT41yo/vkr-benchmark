# ADR-0014: Канонический run_id, per-run artifacts и summary.parquet

- **Статус:** Accepted
- **Дата:** 2026-09-05
- **Контекст:** Этап 2, шаг 7.5 — единый формат сохранения параметров и результатов запуска

## Контекст

К шагу 7.4 единый experiment runner уже рассчитывает полный базовый набор
метрик для Bins, Huffman и Arithmetic Coding, однако результаты существуют
только в памяти процесса и в текстовом выводе CLI. Для воспроизводимого
benchmark необходимо однозначно идентифицировать конфигурацию запуска,
сохранять все основные результаты в машиночитаемом формате и формировать
общую таблицу «одна строка — один run».

Benchmark specification v0.1 фиксирует правило:

```text
run_id = SHA256(canonical_config_json)[:16]
```

и логический состав per-run artifacts:

```text
config.json
result.json
stegotext.txt
token_ids.json
trace.jsonl.gz
summary.parquet
```

## Решение

### 1. Каноническая конфигурация

`run_id` вычисляется только из конфигурации, не из измеренных результатов.
Канонический JSON:

- кодируется UTF-8;
- сортирует ключи лексикографически;
- не содержит незначащих пробелов;
- запрещает `NaN`/`Inf` в конфигурации.

В persisted config хранится логическая идентичность модели (`id`, `revision`,
`dtype`) и настройки, способные изменить семантику model path
(`prompt_add_special_tokens`, `attn_implementation`). Машинно-зависимые
`local_path` и `device` в `run_id` не входят.

Параметры method RNG/key, если они присутствуют, входят в canonical config и
следовательно в `run_id`.

### 2. `method_params_hash`

Для идентификации рабочей точки метода используется:

```text
method_params_hash = SHA256(canonical_method_params_json)[:16]
```

Хешируется только `method.params`. `secret_id`, key и random seed не входят в
этот хеш, поскольку относятся к повтору/экземпляру запуска, а не к самой
рабочей точке метода.

### 3. Поведение при повторном запуске одинаковой конфигурации

Одинаковая canonical config обязана давать тот же `run_id`. Поэтому повторный
запуск той же конфигурации не создает дубликат: per-run файлы атомарно
перезаписываются, а строка `summary.parquet` обновляется по `run_id`.

Если в будущем нужны независимые repeats одной operating point, различающий
repeat/key/secret должен быть явно представлен в конфигурации.

### 4. Per-run artifacts

Успешный run сохраняется в:

```text
results/runs/<run_id>/
├── config.json
├── result.json
├── stegotext.txt
├── token_ids.json
└── trace.jsonl.gz
```

`config.json` содержит ровно canonical JSON, использованный для вычисления
`run_id`.

`stegotext.txt` содержит точный ordinary text без автоматически добавленного
перевода строки.

`token_ids.json` хранит prompt token IDs, sender carrier token IDs и receiver
IDs после retokenization.

`trace.jsonl.gz` содержит компактную пошаговую трассу scalar diagnostics.
Полные массивы `P_reference`/`Q_stego` в обычном режиме не сохраняются.

### 5. Представление бесконечного KL в JSON

Строгий JSON не имеет числового литерала `Infinity`. При этом
`D_KL(P_reference || Q_stego)` по правилам v0.1 может законно быть `+inf`.

Поэтому в `result.json` и `trace.jsonl.gz` принято явное строковое
представление:

```json
"kl_mean_bits": "inf"
```

Это не означает `unavailable` и не заменяется `null`. Значение сохраняет
семантику математической бесконечности.

В `summary.parquet` то же значение хранится как IEEE-754 `+inf` в колонке
`float64`, то есть остается числовым.

`NaN` в persisted result запрещен.

### 6. `summary.parquet`

`results/summary.parquet` содержит одну строку на `run_id`. Физическая schema
v0.2 реализует минимальные поля specification v0.1 и несколько уже доступных
диагностических полей (например, entropy sum, q_mode, decode throughput).

Для записи используется optional dependency `pyarrow` из extra `storage`.
Backend проверяется до загрузки большой LM, чтобы отсутствие зависимости не
обнаруживалось после дорогостоящего запуска.

### 7. Ошибочные запуски

Если конфигурация уже валидна и run завершается исключением, сохраняются:

```text
config.json
result.json  # status="error", тип и сообщение ошибки
```

и, при включенном Parquet backend, строка `summary.parquet` со `status=error` и
`null` для недоступных метрик.

Успешные artifacts (`stegotext.txt`, `token_ids.json`, `trace.jsonl.gz`) для
последней неуспешной попытки той же конфигурации удаляются, чтобы каталог не
содержал противоречивое состояние.

## Последствия

Положительные:

- один run имеет устойчивую машинно-независимую идентичность;
- результаты можно агрегировать без разбора console output;
- `inf` KL сохраняется без невалидного JSON и без скрытого smoothing;
- повторный запуск не создает дублирующую строку summary;
- failures остаются частью экспериментального журнала;
- storage не входит в измеряемое encode/decode time.

Ограничения:

- `pyarrow` требуется для обновления общей Parquet-таблицы;
- текущий trace хранит scalar diagnostics, уже захваченные runner'ом, но не
  полные P/Q и не все возможные per-step timing/probability diagnostics;
- политика отдельных repeat-id может быть дополнительно уточнена в v1.0.

## Статус для этапа 2

Шаг 7.5 реализует основу пункта 3 этапа 2: единый формат параметров и
результатов каждого запуска и общую таблицу результатов. Следующим шагом
остается выполнить небольшую серию реальных runs и сформировать демонстрационный
Stage-2 result dataset/таблицу.
