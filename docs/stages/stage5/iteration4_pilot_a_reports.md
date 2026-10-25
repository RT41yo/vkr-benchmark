# Stage 5 — Iteration 4: фиксация результатов Pilot A и RRC calibration

## Назначение

Эта итерация документирует уже полученные результаты. Она не меняет stego algorithms, generation policy, sweep manifests или metric semantics.

Добавлены:

- `docs/stages/stage5/pilot_a_analysis.md` — аналитический отчет по 360 Pilot-A runs;
- `docs/stages/stage5/pilot_a_reliability_analysis.md` — итог 29 silent + 13 hard reliability failures;
- `docs/stages/stage5/rrc_precision_calibration_report.md` — результаты 24-run RRC precision-margin calibration и открытый boundary-precision issue;
- runtime outputs hard-failure diagnostics;
- обновленный статус `docs/stages/stage5/stage5_plan.md`.

## Методологическая граница

Новые ADR в этой итерации не создаются:

- text transport уже является принятой benchmark semantics;
- common `top_p=1.0` уже закреплен ADR-0027;
- по RRC boundary precision окончательное решение пока не принято, поэтому numerical issue фиксируется в отчете как открытый вопрос до отдельного sensitivity probe.

## Следующий шаг

После фиксации этой контрольной точки необходимо сформировать retained/rejected operating-point table по Pilot A и согласовать сокращенную сетку Pilot B. RRC остается отдельным открытым треком до решения boundary-precision вопроса.
