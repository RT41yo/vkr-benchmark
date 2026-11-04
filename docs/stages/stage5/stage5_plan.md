# Stage 5 — пилотные свипы и окончательная фиксация экспериментального дизайна

## 1. Цель

Stage 5 не является уменьшенной копией Stage 6. Его задача — дешево и воспроизводимо принять решения, необходимые для основного эксперимента:

1. выбрать общий generation regime;
2. определить содержательные диапазоны method-local параметров;
3. исключить технически некорректные и явно неинтересные точки;
4. откалибровать RRC fixed-payload regime;
5. оценить prompt/replicate variance;
6. измерить wall-clock стоимость, отдельно от времени самого стегометода;
7. зафиксировать Stage-6 scale и benchmark specification v1.0.

Методологическое разделение external conditions / method-local points / calibration закреплено ADR-0026.

## 2. Выполненная подготовка

### 2.1. Iteration 1 — sweep foundation

Выполнено:

- переиспользуются восемь frozen CNN/DailyMail pilot contexts Stage 3;
- Stage-5 prompt registry получает стабильные `prompt_id`, exact text хранится локально и воспроизводимо регенерируется;
- введен machine-readable sweep manifest;
- dry-run planner заранее вычисляет полный набор configs и canonical `run_id` без загрузки LM;
- batch runner загружает одну LM один раз на sweep, поддерживает filters и `--resume`;
- зафиксирован paired replicate/randomness protocol;
- выполнен первый common-`top_p` calibration.

### 2.2. Common generation calibration — завершено

Технически проверялись значения:

```text
top_p = 0.95, 0.98, 1.00
```

`top_p=0.98` оказался несовместим с Bins `block_size=2`: после common nucleus truncation в текущем `P_reference` не осталось positive-probability tokens в трех из четырех фиксированных bins. Уменьшение `top_p` не может расширить support, поэтому `0.95` не является feasible common regime.

При `top_p=1.0` выполнены 14 short calibration runs: 7 methods × 2 prompts × 1 paired replicate. Все 14 завершились успешно, с exact round-trip и `BER=0`.

Основной common regime зафиксирован ADR-0027:

```text
temperature = 1.0
common top_k = null
common top_p = 1.0
exclude_special_tokens = true
kv_cache = true
```

Полный отчет: `docs/stages/stage5/top_p_calibration_report.md`.

## 3. Pilot A — screening — завершено

Pilot A использовал:

- Llama-3.2-3B;
- common regime из ADR-0027;
- prompts: `s5p0001..s5p0004`;
- 2 paired replicates;
- **128 carrier tokens** для streaming methods;
- RRC калибруется отдельно в fixed-payload lifecycle.

Streaming executable manifest:

`configs/pilots/stage5/pilot_a_streaming_llama.json`

Сетка:

| Метод | Pilot-A grid |
|---|---|
| Bins | `block_size = 1..8` |
| Huffman | `bits_per_word = 1..8` |
| Arithmetic | `precision={16,32,48}` × `top_k={128,512,2048,8192,50000}` |
| ADG | one normalized point |
| Discop | one normalized point |
| DAIRstega | `alpha={8,16,32,48}` × `beta={1.0,0.75,0.5}`, initially `top_k=100` |

Это 45 method points × 4 prompts × 2 replicates = **360 runs**. Sweep завершен полностью: 347 process-successful runs, 13 hard failures; exact text-channel recovery — 318/360. Все 42 non-exact runs прошли post-hoc reliability diagnostics и связаны с изменением token sequence после обычного text transport.

Итоговые отчеты:

- `docs/stages/stage5/pilot_a_analysis.md`;
- `docs/stages/stage5/pilot_a_reliability_analysis.md`.

Screening после Pilot A завершен. Из 45 method-local points в Pilot B перенесен 21 representative point; 24 точки остаются документированными screening results и повторно не запускаются. Полный retained/rejected table: `docs/stages/stage5/pilot_b_operating_point_selection.md`; machine-readable решение: `results/stage5/pilot_a_operating_point_selection.json`.

DAIRstega candidate-pool sensitivity, если потребуется для Stage 6, выполняется отдельным targeted probe и не расширяет Pilot B неявно.

## 4. RRC calibration — завершена

RRC не переводится в режим с фиксированным числом токенов.

Первая калибровка использовала:

`configs/pilots/stage5/rrc_precision_margin_calibration_llama.json`

```text
precision_margin = {20,40,60}
verify_termination = true
fixed payload = 1024 bits
max_carrier_tokens = 2048
4 prompts × 2 paired replicates
```

Все 24/24 запуска завершились с точным восстановлением секрета. Для Decimal range-state arithmetic значения `20`, `40` и `60` дали одинаковые траектории и метрики. Поэтому `precision_margin=40` сохраняется как консервативная рабочая настройка.

Во время этой калибровки в части запусков наблюдалась `forward KL = +inf`. Отдельная диагностика на 8 запусках `margin40` показала точную причину: очень маленькие положительные вероятности терялись при последовательном накоплении границ в FP64. Во всех 4 запусках с `+inf` была найдена такая потеря поддержки, а в остальных 4 её не было.

После этого построение границ RRC было переведено на Decimal. Проверка исправленной реализации дала:

- 8/8 запусков завершились успешно;
- обе KL конечны в 8/8;
- максимальная `forward KL` = `1.3798816980502983e-16` бит;
- максимальная TVD = `1.0284529972686038e-16`;
- 7/8 полных текстовых циклов восстановили секрет точно.

Единственный неточный полный цикл отдельно проверен. После текстового преобразования число токенов изменилось с 249 до 250, первое расхождение появилось на позиции 41. Прямое декодирование сохраненных `sender_token_ids` восстановило все 1024 бита точно, `BER=0`. Поэтому этот случай относится к текстовому каналу, а не к численной реализации RRC.

Численный вопрос границ RRC считается закрытым. Итоговый отчет:

`docs/stages/stage5/rrc_decimal_boundary_validation_report.md`

Калибровка размера секрета по `{128,256,512,1024}` завершена: выполнены все 32 запланированных запуска. Для Stage 6 выбран `payload_bits=1024`. Этот режим дает медиану 256 токенов и среднее 253 токена, поэтому по масштабу текста он наиболее близок к 256-токенным потоковым экспериментам. Медианный BPT равен примерно 4.00 бит/токен. Численные метрики остаются на уровне нуля. Полный текстовый цикл дал 7/8 точных восстановлений; единственный неточный случай сопровождается изменением токеновой последовательности после повторной токенизации и сохраняется как нормативный результат надежности.

Итоговый отчет: `docs/stages/stage5/rrc_payload_calibration_report.md`. Машиночитаемое решение: `results/stage5/rrc_stage6_payload_selection.json`.

Калибровка RRC отвечает на вопрос «какие технические настройки фиксировать дальше?», а не создает искусственную кривую компромиссов RRC.

## 5. Pilot B — подтверждающий sweep и Decoder Audit завершены

Согласованный streaming Pilot B содержит 21 retained point:

| Метод | Retained points |
|---|---:|
| Bins | 4 |
| Huffman | 4 |
| Arithmetic Coding | 6 |
| ADG | 1 |
| Discop | 1 |
| DAIRstega | 5 |
| **Итого** | **21** |

Режим:

- executable manifest: `configs/pilots/stage5/pilot_b_streaming_llama.json`;
- 8 frozen pilot prompts `s5p0001..s5p0008`;
- 3 paired replicates;
- 256 carrier tokens;
- тот же common generation regime из ADR-0027;
- **504 planned streaming runs**.

Pilot B завершен: execution log содержит все 504 запланированных запуска. Из них 466 завершились без программной ошибки, 38 завершились `hard failure`; среди 466 завершившихся запусков еще 62 имеют `roundtrip_exact=false`. Таким образом, точное восстановление секрета по полному текстовому пути получено в 404/504 случаях. Полный checkpoint: `docs/stages/stage5/pilot_b_checkpoint.md`.

После Pilot B проведен отдельный Decoder Audit. Он дал следующие результаты:

- прямое декодирование токенов отправителя: **504/504 exact**;
- независимая проверка шести декодеров: **3533 проверки, 0 расхождений**;
- все **62/62** случаев с `roundtrip_exact=false` после успешного завершения связаны с изменением токеновой последовательности после `token IDs → text → retokenization`;
- все **38/38** исходных `hard failure` воспроизведены после того же текстового канала на детерминированно восстановленных последовательностях отправителя.

Аудит не выявил дефектов декодеров. Наблюдавшиеся 100 неточных результатов остаются реальными результатами end-to-end надежности Pilot B, потому что нормативный путь бенчмарка включает передачу через обычный текст. Однако их причина теперь локализована в потере исходного разбиения текста на токены при повторной токенизации.

Полный отчет: `docs/stages/stage5/decoder_audit_report.md`.

RRC в эти 504 запуска не входит. Его отдельная калибровка завершена: для Stage 6 зафиксированы Decimal-границы, `precision_margin=40` и `payload_bits=1024`.

Итоговый отбор потоковых точек для Stage 6 завершен. Из 21 точки Pilot B сохраняются 14:

- Bins: `b1`, `b4`, `b6`;
- Huffman: `e2`, `e4`, `e8`;
- Arithmetic Coding: `p32-k128`, `p32-k2048`, `p32-k50000`;
- ADG: `default`;
- Discop: `default`;
- DAIRstega: `a8-b1-k100`, `a16-b075-k100`, `a32-b05-k100`.

Полное обоснование: `docs/stages/stage5/pilot_b_stage6_operating_point_selection.md`. Машиночитаемое решение: `results/stage5/pilot_b_stage6_operating_point_selection.json`.

## 6. Критерии отбора

Каждая точка сохраняет полный benchmark vector; единой агрегированной оценки «лучший метод» нет.

Обязательные оси:

- reliability: decoder completion, `roundtrip_exact`, BER, token-sequence round-trip;
- capacity: BPT, entropy utilization;
- distribution distortion: оба направления KL по ADR-0012 + TVD;
- quality: raw-LM NLL/PPL;
- performance: encode/decode ms/token, payload bit/s, stego algorithm time;
- instrumentation: `metric_instrumentation_total_ms` отдельно от stego time;
- variability: разброс по prompt × replicate.

Точка может быть исключена как технически некорректная или стабильно доминируемая соседней точкой того же метода. При этом для параметризованного метода желательно сохранить security/quality anchor, промежуточную область и capacity-oriented endpoint, если они содержательно различаются.

## 7. Масштаб и вычислительный бюджет Stage 6 — зафиксированы

Итоговый масштаб основных экспериментов:

```text
Stage-6 prompts: 24 held-out contexts, Stage-3 selection_rank 8..31
streaming carrier: 256 tokens
RRC payload: 1024 bits, precision_margin=40

Llama-3.2-3B:
  24 prompts × 3 replicates × 15 regimes = 1080 runs

Qwen3-4B-Base:
  те же 24 prompts × 2 replicates × 15 regimes = 720 runs

Total: 1800 stego runs
```

Промпты Stage 6 не пересекаются с восемью pilot prompts Stage 5. Они детерминированно выбираются из того же зафиксированного CNN/DailyMail источника как следующие 24 записи `selection_rank=8..31`. Машиночитаемый список: `results/stage5/stage6_prompt_manifest.json`.

По фактическим временам Pilot B и RRC один полный набор из 15 режимов занимает около 26.34 минуты по сумме медиан. Для Llama прогноз составляет около 31.6 часа. Для Qwen используется консервативный коэффициент 1.4 относительно Llama на основании измерений скорости Stage 1; прогноз составляет около 29.5 часа. Суммарная медианная оценка — около 61 часа, консервативная оценка — около 71 часа, верхний ориентир с резервом 20% — около 85 часов последовательного GPU-времени.

Полное обоснование: `docs/stages/stage5/stage6_scale_and_budget_report.md`. Машиночитаемое решение: `results/stage5/stage6_scale_decision.json`.

После этого число промптов, повторов, объем генерации, объем проверки на Qwen и вычислительный бюджет Stage 6 считаются зафиксированными.

## 8. Выходные артефакты Stage 5

Минимальный closeout:

- dataset pilot runs;
- analytic pilot report;
- Stage-6 budget report;
- retained/rejected operating-point table с причинами;
- benchmark specification v1.0;
- Stage-5 Typst presentation;
- machine-readable sweep manifests и execution logs;
- `stage5_closeout.md`.

## 9. Статус закрытия Этапа 5

Содержательная экспериментальная часть Этапа 5 завершена.

Подготовлены:

- единый итоговый отчёт: `docs/stages/stage5/stage5_final_report.md`;
- спецификация основных экспериментов: `specification/benchmark_specification_v1.0.md`;
- четыре исполняемые конфигурации Stage 6 в `configs/experiments/stage6_main_*.json`;
- документ закрытия: `docs/stages/stage5/stage5_closeout.md`.

Новые пилотные GPU-запуски перед Stage 6 не требуются.

Остаётся последняя оформительская задача Этапа 5 — итоговая презентация. Она подготавливается отдельно после проверки текстовых документов.
