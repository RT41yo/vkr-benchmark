# Stage 4 — iteration 14: DAIRstega closeout

## Цель

Закрыть reproducibility gate DAIRstega после успешных smoke, paper/repository conformance, direct-author conformance и publication-characteristic probe.

## Подтверждённый publication characteristic

100-token normalized Llama probe дал:

- `alpha=8,beta=1.0`: BPT `0.2700`, PPL `2.2522`, reverse KL `0.867062`, TVD `0.359716`;
- `alpha=8,beta=0.5`: BPT `1.6100`, PPL `20.5365`, reverse KL `2.721758`, TVD `0.683234`;
- `alpha=32,beta=1.0`: BPT `0.8000`, PPL `5.0746`, reverse KL `1.515890`, TVD `0.516834`;
- `alpha=32,beta=0.5`: BPT `3.3500`, PPL `125.3471`, reverse KL `3.609336`, TVD `0.798266`.

Все четыре точки дали exact token/text + secret round-trip. Capacity-oriented точка превосходит quality-oriented по BPT и одновременно хуже по всем трём наблюдаемым осям cost: PPL, reverse KL и TVD. `beta=0.5` повысил BPT при обеих alpha. Publication-characteristic gate: PASS.

## Интерпретация

Результат воспроизводит структуру Table-3 trade-off, но не её абсолютные значения. Это ожидаемо из-за другой LM, токенизации, prompting и единицы payload. Особенно высокая PPL capacity-point является диагностическим наблюдением одной Stage-4 точки и не фиксирует будущую рабочую конфигурацию; диапазоны будут выбраны на Stage 5.

## Итог

DAIRstega method-level Stage-4 integration/reproducibility закрыт. Следующая работа — общий Stage-4 closeout и compatibility matrix на основной и резервной LM.
