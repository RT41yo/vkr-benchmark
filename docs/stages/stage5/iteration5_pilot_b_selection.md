# Stage 5 — Iteration 5: operating-point selection и Pilot B manifest

## Назначение

Итерация фиксирует согласованный screening после Pilot A и подготавливает Pilot B. Алгоритмы методов и metric semantics не меняются.

Добавлены:

- `docs/stages/stage5/pilot_b_operating_point_selection.md` — полный retained/rejected table с причинами;
- `results/stage5/pilot_a_operating_point_selection.json` — machine-readable версия решения по всем 45 Pilot-A points;
- `configs/pilots/stage5/pilot_b_streaming_llama.json` — executable manifest на 21 retained point, 8 prompts, 3 paired replicates и 256 carrier tokens.

Обновлены Stage-5 plan и Pilot-A analysis, чтобы screening считался завершенным и следующий streaming шаг был Pilot B.

## Методологические границы

Новый ADR не создается: отбор является результатом Pilot-A эксперимента, а не изменением benchmark semantics. RRC в Pilot-B streaming manifest не включается и остается отдельным открытым треком до boundary-precision investigation и fixed-payload calibration.

## Размер Pilot B

```text
21 method points × 8 prompts × 3 replicates = 504 runs
```

Pilot B использует тот же common generation regime из ADR-0027 и увеличивает carrier length с 128 до 256 tokens.
