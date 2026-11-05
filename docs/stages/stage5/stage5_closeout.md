# Закрытие Этапа 5: пилотные эксперименты и фиксация дизайна Stage 6

Статус: **содержательная и техническая часть Этапа 5 завершена**.

## 1. Что сделано

На Этапе 5:

- выбран общий режим генерации;
- выполнены Pilot A и Pilot B;
- проведён отдельный аудит декодеров;
- выбраны финальные рабочие точки шести потоковых методов;
- отдельно откалиброван и проверен RRC;
- для RRC выбран `payload_bits=1024` и `precision_margin=40`;
- выбраны 24 новых промпта для Stage 6;
- зафиксированы 3 повтора для Llama и 2 для Qwen;
- рассчитан вычислительный бюджет;
- подготовлена спецификация бенчмарка v1.0;
- подготовлены исполняемые конфигурации Stage 6.

Единый итоговый отчёт:

`docs/stages/stage5/stage5_final_report.md`

## 2. Финальные режимы методов

```text
Bins:                b1, b4, b6
Huffman:             e2, e4, e8
Arithmetic Coding:   p32-k128, p32-k2048, p32-k50000
ADG:                 default
Discop:              default
DAIRstega:           a8-b1-k100, a16-b075-k100, a32-b05-k100
RRC:                 payload_bits=1024, precision_margin=40
```

Для первых шести методов фиксируется 256 токенов-носителей. RRC сохраняет собственный режим с фиксированной длиной секрета.

## 3. Масштаб Stage 6

```text
Промпты: 24 новых контекста s6p0001..s6p0024

Llama:
  24 промпта × 3 повтора × 15 режимов = 1080 запусков

Qwen:
  24 промпта × 2 повтора × 15 режимов = 720 запусков

Всего: 1800 стегозапусков
```

Верхняя оценка последовательного GPU-времени с резервом составляет около 85 часов.

## 4. Ключевые итоговые документы

- `docs/stages/stage5/stage5_final_report.md` — единый итоговый отчёт;
- `specification/benchmark_specification_v1.0.md` — окончательный протокол основных экспериментов;
- `docs/stages/stage5/pilot_b_stage6_operating_point_selection.md` — выбор потоковых точек;
- `docs/stages/stage5/decoder_audit_report.md` — аудит ошибок Pilot B;
- `docs/stages/stage5/rrc_decimal_boundary_validation_report.md` — численная проверка RRC;
- `docs/stages/stage5/rrc_payload_calibration_report.md` — выбор размера секрета RRC;
- `docs/stages/stage5/stage6_scale_and_budget_report.md` — масштаб и вычислительный бюджет.

## 5. Исполняемые конфигурации Stage 6

```text
configs/experiments/stage6_main_streaming_llama.json
configs/experiments/stage6_main_rrc_llama.json
configs/experiments/stage6_main_streaming_qwen.json
configs/experiments/stage6_main_rrc_qwen.json
```

Тексты 24 промптов восстанавливаются командой:

```bash
python scripts/prepare_stage6_main_prompts.py
```

## 6. Проверка готовности

Исполняемые конфигурации дают ожидаемый масштаб:

```text
Llama streaming = 1008
Llama RRC       = 72
Qwen streaming  = 672
Qwen RRC        = 48
Итого           = 1800
```

На контрольном snapshot пройдены тесты:

```text
399 passed — tests/unit
421 passed — весь набор без tests/integration
```
