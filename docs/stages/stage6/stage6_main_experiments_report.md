# Этап 6 — итоговый отчёт по основным сравнительным экспериментам

## 1. Статус этапа

Основная генерационная часть Stage 6 завершена полностью по зафиксированной `benchmark specification v1.0`.

Выполнены все четыре основные серии:

- Llama-3.2-3B, streaming: **1008/1008** уникальных запусков;
- Llama-3.2-3B, RRC fixed-payload: **72/72**;
- Qwen3-4B-Base, streaming: **672/672**;
- Qwen3-4B-Base, RRC fixed-payload: **48/48**.

Итого выполнено **1800/1800** запланированных canonical runs.

Ни одна рабочая точка, параметр метода, правило генерации, метрика или семантика streaming/fixed-payload в ходе основных экспериментов не менялись. Пилотный подбор параметров не продолжался.

Финальный integrity audit завершён со статусом:

```text
FINAL STAGE 6 INTEGRITY AUDIT: OK
```

Полный журнал аудита сохранён в `results/stage6/stage6_final_audit.txt`.

---

## 2. Экспериментальный дизайн

Для потоковых методов использовалась фиксированная длина:

```text
carrier_tokens = 256
```

Рабочие точки:

```text
Bins:               b1, b4, b6
Huffman:            e2, e4, e8
Arithmetic Coding:  p32-k128, p32-k2048, p32-k50000
ADG:                 default
Discop:              default
DAIRstega:           a8-b1-k100, a16-b075-k100, a32-b05-k100
```

RRC выполнялся отдельно в fixed-payload режиме:

```text
payload_bits = 1024
precision_margin = 40
verify_termination = true
```

Использовались 24 новых Stage 6 промпта `s6p0001..s6p0024`, не пересекающиеся с Pilot Stage 5. Для Llama выполнялись три повтора `r01..r03`, для Qwen — два `r01..r02`.

Общие правила генерации оставались одинаковыми для всех методов внутри модели:

```text
temperature = 1.0
top_k = null
top_p = 1.0
exclude_special_tokens = true
kv_cache = true
```

---

## 3. Правила агрегации Stage 6

Stage 6 не вводит новую агрегированную метрику «качества метода» и не выполняет Парето-анализ — это относится к Stage 8.

Для предварительных таблиц используются только уже зафиксированные benchmark-метрики:

- BPT и entropy utilization;
- обе стороны KL согласно ADR-0012;
- TVD;
- raw-LM NLL/PPL;
- end-to-end reliability;
- encode/decode timings, где `metric_instrumentation` не входит во время стегометода согласно ADR-0020.

Численные benchmark-векторы агрегируются только по runs с `status=ok`, поскольку hard failure не имеет полного набора численных метрик. При этом показатель полного end-to-end exact round-trip использует **все запланированные runs** в знаменателе: hard failure считается неуспешной передачей.

Для forward KL `+inf` сохраняется как содержательный результат. Finite-only среднее не используется как замена основной KL и сохраняется только как диагностическая статистика.

Поскольку `execution.jsonl` хранит PPL и entropy utilization, но не дублирует все поля `summary.parquet`, analysis-скрипт восстанавливает два уже определённых показателя по тождествам `NLL = ln(PPL)` и `reference_entropy_mean_bits = BPT / entropy_utilization`. Это не новые метрики и не изменение методологии.

Воспроизводимая агрегация реализована в:

```text
scripts/analyze_stage6_main.py
```

Основные машинные результаты:

```text
results/stage6/stage6_execution_summary.json
results/stage6/stage6_operating_point_summary.csv
results/stage6/stage6_operating_point_summary.json
results/stage6/stage6_operating_point_summary.md
results/stage6/stage6_cross_model_summary.csv
```

---

## 4. Полнота и целостность результатов

Финальный аудит подтвердил:

```text
all unique run_ids: 1800
status=ok:           1703
status=error:          97
```

Для всех 1800 запусков присутствует `result.json` в локальном полном наборе результатов. Для всех 97 hard failures сохранён `failure_diagnostics.json`. Среди успешных запусков не обнаружено ни одного случая отсутствия dual-KL/TVD полей.

`summary.parquet` также содержит ровно 1800 Stage 6 строк:

```text
ok:    1703
error:   97
```

Таким образом, основной экспериментальный массив не имеет пропущенных canonical configurations.

Формальная идентичность основного экспериментального набора данных Stage 6 зафиксирована отдельным машинно-проверяемым manifest:

```text
results/stage6/dataset_v1.0_manifest.json
```

Этот файл объявляет набор как **основной экспериментальный набор данных v1.0** и фиксирует его состав без дублирования per-run результатов: 1800 canonical `run_id`, четыре исполненных Stage 6 sweep, 15 operating points, две языковые модели с зафиксированными ревизиями, 24 Stage 6 промпта, итоговые status counts, контракт обязательных per-run артефактов и SHA-256 ключевых frozen входов и журналов. Источником истины для конкретного запуска по-прежнему остаются canonical artifacts в `results/runs/<run_id>/`; manifest служит inventory/provenance-описанием полного dataset v1.0.

---

## 5. End-to-end reliability

### 5.1. Общий результат

Нормативная проверка надёжности сохраняет полный текстовый канал:

```text
secret bits
→ sender token IDs
→ text
→ retokenization
→ receiver token IDs
→ decoder
→ recovered secret
```

Итог:

| Scope | Runs | Exact secret round-trip | Exact rate | Hard failures | Successful decode, но secretRT=false |
|---|---:|---:|---:|---:|---:|
| Llama streaming | 1008 | 816 | 80.95% | 58 | 134 |
| Llama RRC | 72 | 66 | 91.67% | 0 | 6 |
| **Llama total** | **1080** | **882** | **81.67%** | **58** | **140** |
| Qwen streaming | 672 | 557 | 82.89% | 39 | 76 |
| Qwen RRC | 48 | 43 | 89.58% | 0 | 5 |
| **Qwen total** | **720** | **600** | **83.33%** | **39** | **81** |
| **Stage 6 total** | **1800** | **1482** | **82.33%** | **97** | **221** |

### 5.2. Атрибуция hard failures

Все **97/97** hard failures имеют одновременно:

```text
failure_phase = decode
token_sequence_roundtrip_exact = false
error_type = MethodError
```

Hard failure при сохранённой sender/receiver token sequence не обнаружен:

```text
token_sequence_roundtrip_exact=True: 0
```

Это подтверждает вывод Decoder Audit Stage 5 на существенно большей выборке: hard failures возникают после изменения токенной последовательности в канале `tokens → text → retokenization`, после чего receiver закономерно оказывается в состоянии, несовместимом с sender-side кодером.

### 5.3. Связь token round-trip и восстановления секрета

Во всём Stage 6 точная token sequence сохранилась в 1481 успешном запуске. Точный секрет восстановлен в 1482 запусках. Разница объясняется единственным случаем `Llama / Bins b1`, где token sequence изменилась, но встроенный секрет всё же восстановился точно.

Всего нарушение token round-trip наблюдалось в 319 случаях:

- 221 successful runs с `roundtrip_exact=false`;
- 97 hard decode failures;
- 1 случай с изменённой token sequence, но точным секретом.

Таким образом, **318 из 319** наблюдавшихся изменений token sequence привели либо к ошибке секрета, либо к hard failure.

Это эмпирически подтверждает решение считать полный текстовый канал нормативной reliability-метрикой, а direct token-ID decoding — диагностикой.

---

## 6. Искажение распределения

### 6.1. Discop и RRC

На обеих моделях Discop и RRC сохранили обе KL и TVD на уровне машинной погрешности.

Средние run-level значения:

| Method | Model | KL(Q‖P) | TVD |
|---|---|---:|---:|
| Discop | Llama | `9.41e-17` | `1.02e-16` |
| Discop | Qwen | `1.13e-16` | `1.06e-16` |
| RRC | Llama | `1.42e-16` | `9.13e-17` |
| RRC | Qwen | `1.50e-16` | `1.00e-16` |

Forward KL также конечна и находится на уровне порядка `1e-16`.

Это подтверждает на основных экспериментах Stage 6 теоретически ожидаемое zero-distortion поведение этих двух методов и показывает, что Decimal-boundary fix RRC сохраняется на полном наборе 120 RRC runs.

### 6.2. ADG

ADG даёт небольшое, но измеримое отклонение:

```text
Llama: KL(Q||P) ≈ 0.00654, TVD ≈ 0.0200
Qwen:  KL(Q||P) ≈ 0.00431, TVD ≈ 0.0135
```

Forward KL конечна. Это соответствует ожидаемому поведению адаптивной группировки: распределение близко к `P_reference`, но не совпадает с ним строго.

### 6.3. Arithmetic Coding

Для всех трёх рабочих точек Arithmetic Coding forward KL равна `+inf` во всех successful runs. Причина — нулевая масса `Q_stego` вне ограниченного candidate set при положительной массе `P_reference`.

При этом reverse KL и TVD быстро уменьшаются при росте `k`:

| Model | Point | KL(Q‖P) mean | TVD mean |
|---|---|---:|---:|
| Llama | k128 | 0.0776 | 0.0490 |
| Llama | k2048 | 0.0128 | 0.00869 |
| Llama | k50000 | 0.000288 | 0.000208 |
| Qwen | k128 | 0.0534 | 0.0341 |
| Qwen | k2048 | 0.00913 | 0.00621 |
| Qwen | k50000 | 0.0000604 | 0.0000471 |

Этот результат показывает, почему сохранение **обоих направлений KL** существенно: по reverse KL точка `k50000` почти совпадает с исходным распределением, но forward KL всё равно фиксирует нарушение support coverage как бесконечное расхождение.

### 6.4. Bins, Huffman и DAIRstega

Для всех successful runs Bins, Huffman и DAIRstega forward KL также равна `+inf` из-за нулей в `Q_stego`.

По reverse KL/TVD видны ожидаемые рабочие компромиссы. В частности, у Huffman увеличение `e` повышает BPT и одновременно уменьшает распределительное отклонение, тогда как у Bins рост block size резко увеличивает ёмкость ценой очень сильного отклонения и ухудшения raw-LM качества.

---

## 7. Ёмкость и использование энтропии

Средние BPT основных рабочих точек:

| Method | Point | Llama BPT | Qwen BPT |
|---|---|---:|---:|
| Bins | b1 | 1.000 | 1.000 |
| Bins | b4 | 4.000 | 4.000 |
| Bins | b6 | 6.000 | 6.000 |
| Huffman | e2 | 1.797 | 1.786 |
| Huffman | e4 | 2.788 | 2.694 |
| Huffman | e8 | 4.074 | 3.785 |
| Arithmetic | k128 | 3.073 | 2.276 |
| Arithmetic | k2048 | 3.691 | 2.814 |
| Arithmetic | k50000 | 4.168 | 2.852 |
| ADG | default | 2.777 | 1.776 |
| Discop | default | 3.763 | 2.910 |
| RRC | payload1024 | 4.152 | 3.193 |
| DAIRstega | a8-b1-k100 | 0.582 | 0.434 |
| DAIRstega | a16-b075-k100 | 1.175 | 0.790 |
| DAIRstega | a32-b05-k100 | 3.012 | 2.276 |

Bins ожидаемо не зависит от энтропии модели по формальной скорости `b` бит/токен. Для большинства адаптивных методов Qwen даёт меньший BPT, чем Llama, что показывает модельную зависимость доступной/используемой ёмкости.

RRC особенно наглядно демонстрирует этот эффект. При фиксированном секрете 1024 бит:

```text
Llama: median carrier length = 246.5 tokens, mean BPT = 4.152
Qwen:  median carrier length = 317.5 tokens, mean BPT = 3.193
```

То есть один и тот же fixed-payload режим требует существенно различного числа токенов на разных LM.

---

## 8. Raw-LM качество текста

NLL/PPL на Stage 6 являются модельными метриками правдоподобия выбранных токенов и не заменяют человеческую или независимую семантическую оценку.

Тем не менее внутренние зависимости выражены отчётливо.

### Bins

Медианная PPL:

```text
              Llama       Qwen
b1              6.71       6.89
b4             86.43      88.99
b6            319.66     311.48
```

Рост фиксированной ёмкости Bins с 1 до 4–6 bit/token сопровождается резким снижением raw-LM вероятности стеготекста.

### DAIRstega

```text
                       Llama       Qwen
a8-b1-k100               3.89       2.63
a16-b075-k100           12.58       5.15
a32-b05-k100           130.81     119.14
```

Здесь также виден выраженный компромисс: точка с минимальной ёмкостью имеет очень низкую PPL и 100% exact reliability на обеих моделях, а `a32-b05-k100` повышает BPT ценой резкого роста PPL и появления hard failures.

### Huffman

Рост `e` повышает BPT, но медианная PPL также растёт:

```text
Llama: 7.64 → 10.73 → 20.67
Qwen:  8.03 → 10.63 → 18.74
```

---

## 9. Вычислительная эффективность

ADR-0020 соблюдается: `metric_instrumentation` не входит в `encode_ms_per_token`, `decode_ms_per_token` и `stego_algorithm_total_ms`.

Медианные encode timings показывают устойчивое различие классов методов:

```text
                       Llama ms/token    Qwen ms/token
Bins                     ~18              ~23
Huffman                  ~18–19           ~23–24
DAIRstega                ~50–52           ~60–62
Arithmetic Coding        ~70–75           ~79–83
RRC                      ~110             ~134
ADG                      ~162             ~172
Discop                   ~302             ~363
```

При интерпретации полного wall-clock времени важно не смешивать его с собственным временем метода. Например, у ADG медианное `metric_instrumentation_total_ms` составляет примерно 838 секунд на Llama и 756 секунд на Qwen, тогда как эти вычисления сознательно исключены из timing стегометода. У Discop измерительный контур также заметно увеличивает wall-clock стоимость.

Сумма `elapsed_wall_seconds` по 1800 уникальным canonical runs составляет примерно **53.7 часа**:

```text
Llama streaming: ~30.35 ч
Llama RRC:       ~1.14 ч
Qwen streaming:  ~20.85 ч
Qwen RRC:        ~1.37 ч
```

Это ниже Stage 5 медианной оценки ~61.1 часа. Значение является суммой per-run wall time, а не отдельным измерением полного календарного времени Stage 6, поэтому не включает все model-loading/resume/administrative overheads.

---

## 10. Предварительное межмодельное сравнение

На уровне всей модели общая exact reliability близка:

```text
Llama: 81.67%
Qwen:  83.33%
```

Однако это сходство скрывает значительные изменения отдельных operating points.

Наиболее крупные сдвиги exact reliability при переходе Llama → Qwen:

```text
Bins b1:        90.28% → 68.75%   (-21.53 п.п.)
Bins b4:        50.00% → 70.83%   (+20.83 п.п.)
Huffman e2:     81.94% → 70.83%   (-11.11 п.п.)
Discop:         87.50% → 95.83%   (+8.33 п.п.)
Huffman e8:     69.44% → 77.08%   (+7.64 п.п.)
```

Одновременно некоторые структурные свойства остаются устойчивыми:

- Discop и RRC сохраняют практически нулевой KL/TVD на обеих моделях;
- Bins сохраняет формальный BPT 1/4/6;
- рост `k` Arithmetic Coding уменьшает reverse-KL/TVD на обеих моделях;
- высокоёмкие Bins и DAIRstega остаются дорогими по PPL на обеих моделях.

Таким образом, Stage 6 уже даёт предварительное подтверждение центральной гипотезы ВКР: **общие свойства класса метода могут сохраняться, но конкретные количественные выводы и относительное положение рабочих точек меняются при смене LM**. Финальная проверка устойчивости ранжирований и Парето-фронтов выполняется на Stage 8.

---

## 11. Предварительные выводы Stage 6

1. **Основной массив данных собран полностью:** 1800/1800 canonical runs, без пропущенных конфигураций.
2. **Zero-distortion свойства Discop и RRC воспроизвелись на обеих современных LM** на полном Stage 6 наборе.
3. **Dual KL необходима для корректной интерпретации.** Arithmetic Coding при большом `k` имеет очень маленькие reverse KL/TVD, но forward KL остаётся `+inf` из-за поддержки распределения.
4. **Надёжность полного текстового канала является самостоятельной и значимой осью.** Общая exact reliability равна 82.33%, несмотря на корректность декодеров при direct token-ID diagnostics Stage 5.
5. **Все 97 hard failures атрибутированы изменению токенной последовательности после text transport**, а не отдельному дефекту декодеров.
6. **Ёмкость адаптивных методов зависит от LM.** Особенно заметно это для RRC, Arithmetic, ADG и Discop.
7. **Компромисс capacity ↔ raw-LM quality выражен сильно** у Bins и DAIRstega и заметен у Huffman.
8. **Вычислительная стоимость методов существенно различается**, причём metric instrumentation необходимо анализировать отдельно от времени самого стегометода.
9. **Межмодельная устойчивость не сводится к одному общему показателю:** суммарная reliability двух моделей близка, но отдельные operating points меняются на десятки процентных пунктов.

Эти выводы являются описательными результатами Stage 6. Финальные ответы на RQ и многокритериальный/Парето-анализ не фиксируются до Stage 8.

---

## 12. Граница Stage 6 и следующие этапы

В Stage 6 **не выполнялись**:

- обучение или применение стегодетектора;
- ROC-AUC;
- окончательный Парето-анализ;
- выбор единственного «лучшего» метода;
- новые parameter sweeps;
- независимая человеческая/LLM-оценка качества текста.

Stage 7 должен сформировать сопоставленные cover/stego выборки и выполнить эмпирический стегоанализ. После Stage 7 Stage 8 объединит текущие метрики с обнаружимостью и выполнит окончательный многокритериальный анализ.

---

## 13. Основные артефакты Stage 6

```text
configs/experiments/stage6_main_streaming_llama.json
configs/experiments/stage6_main_rrc_llama.json
configs/experiments/stage6_main_streaming_qwen.json
configs/experiments/stage6_main_rrc_qwen.json

results/stage6/dataset_v1.0_manifest.json
results/stage6/stage6_final_audit.txt
results/stage6/stage6_execution_summary.json
results/stage6/stage6_operating_point_summary.csv
results/stage6/stage6_operating_point_summary.json
results/stage6/stage6_operating_point_summary.md
results/stage6/stage6_cross_model_summary.csv

scripts/analyze_stage6_main.py
```

Canonical per-run artifacts остаются локально в `results/runs/<run_id>/` и по политике проекта не включаются в Git/ZIP snapshots.
