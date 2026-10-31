# Stage 5 — Pilot B execution checkpoint

## 1. Статус

Pilot B для шести streaming-методов завершен полностью по executable manifest:

`configs/pilots/stage5/pilot_b_streaming_llama.json`.

План: **21 retained operating point × 8 prompts × 3 paired replicates = 504 runs** при фиксированной длине **256 carrier tokens**. RRC в этот streaming sweep не входит и остается на отдельном numerical/fixed-payload track.

Источник сырых результатов:

`results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl`

SHA-256 execution log:

`381853d6bc7bab0fac9c67313d7db7c01024736e951abda23ecd668231674942`

## 2. Runner outcome

Финальная консольная сводка:

```text
completed: 466
failed: 38
skipped_existing_ok: 0
sweep_wall_seconds: 41951.48
```

Wall-clock sweep time: примерно **11 ч 39 мин**.

Все 504 запланированных run представлены в execution log как `ok` или `error`; незаписанных плановых run нет.

## 3. Reliability checkpoint

Статус `ok` не равен успешному восстановлению секрета. В Pilot B:

| Метод | Runs | Exact | OK, но secret mismatch | Hard failure | Exact rate |
|---|---:|---:|---:|---:|---:|
| Bins | 96 | 51 | 45 | 0 | 53.1% |
| Huffman | 96 | 67 | 9 | 20 | 69.8% |
| Arithmetic Coding | 144 | 134 | 7 | 3 | 93.1% |
| ADG | 24 | 23 | 1 | 0 | 95.8% |
| Discop | 24 | 22 | 0 | 2 | 91.7% |
| DAIRstega | 120 | 107 | 0 | 13 | 89.2% |
| **Итого** | **504** | **404** | **62** | **38** | **80.2%** |

Таким образом, на текущем уровне наблюдения:

- 404/504 run восстановили секрет точно;
- 62 run завершились процессно успешно, но `roundtrip_exact=false`;
- 38 run завершились hard failure во время decoding;
- всего 100/504 run не дали exact end-to-end secret recovery.

## 4. Распределение hard failures

Hard failures по методам:

- Huffman — 20;
- DAIRstega — 13;
- Arithmetic Coding — 3;
- Discop — 2;
- Bins — 0;
- ADG — 0.

Наблюдаемые сообщения похожи на нарушения согласованности receiver state с пришедшим token ID: Huffman/Arithmetic сообщают token outside candidate set, DAIRstega — невозможный или zero-width interval/token для текущего состояния, Discop — branch unavailable to both distribution copies.

**Причина этих ошибок на данном checkpoint не считается доказанной.** Pilot-A diagnostics показали text-transport causality для своих 42 non-exact run, но Pilot B содержит новые случаи у Arithmetic, ADG и Discop. Эти результаты нельзя автоматически приписывать text transport без отдельной проверки.

## 5. Следующий обязательный шаг — Stage 5 Decoder Audit

До окончательной интерпретации Pilot B и до выбора Stage-6 operating points проводится отдельный Decoder Audit.

Минимальные цели:

1. Direct-token replay для всех 504 Pilot-B run: декодирование исходных sender token IDs без промежуточного `tokens → text → tokens`.
2. Проверка, дает ли direct-token path exact recovery для каждого run, включая 100 текущих non-exact cases.
3. Отдельная author/paper conformance проверка decoder logic, особенно для методов, где готового авторского decoder не было или normalized inverse реализован самостоятельно.
4. Для text-channel failures — сопоставление первой tokenization divergence с первой ошибкой секрета / первым decoder failure.
5. Только после аудита классифицировать причины Pilot-B reliability failures как decoder defect, text-transport effect или иной класс.

Главный критерий тревоги: если исходные sender token IDs не декодируются точно хотя бы в части run, это требует исправления/валидации decoder до Stage 6.

## 6. Артефакты, необходимые для Decoder Audit

Tracked checkpoint содержит aggregate execution log. Полные per-run artifacts остаются локально в:

`results/runs/<run_id>/`

и сознательно не включаются в git/snapshot из-за объема и политики проекта. Для Decoder Audit локальный рабочий репозиторий, на котором выполнялся Pilot B, должен сохранить эти каталоги (например `result.json`, token IDs, traces — в зависимости от run artifact schema).

ZIP snapshot для handoff содержит код, конфигурации, документы и aggregate Pilot-B execution log, но не тяжелый `results/runs/`.

## 7. Что пока не фиксируется

На этом checkpoint **не** фиксируются:

- окончательная причина 100 non-exact Pilot-B run;
- финальный retained/rejected набор для Stage 6;
- окончательная benchmark specification v1.0;
- окончательный RRC regime.

Эти решения принимаются после Decoder Audit и отдельного закрытия RRC boundary-precision вопроса.
