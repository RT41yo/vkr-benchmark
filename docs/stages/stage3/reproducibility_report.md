# Этап 3 — итоговый отчёт по воспроизводимости Bins, Huffman и Arithmetic Coding

Статус: **Этап 3 завершён.** Характерное поведение исходных методов воспроизведено, normalized adapters сохраняют принципиальный механизм Bins, Huffman и Arithmetic Coding на matched representative points, а оставшиеся расхождения документированы и отделены от ошибок реализации.

## 1. Цель этапа

Этап 3 проверял два независимых вопроса:

1. можно ли воспроизвести опубликованное поведение базовых методов в максимально author-compatible режиме;
2. не изменила ли нормализация Stage 2 принципиальную механику методов при переносе в единый benchmark-контур.

Поэтому использовались три режима анализа:

```text
author-compatible reproduction
        -> сравнение с публикацией / reference code

normalized benchmark mode
        -> единый P_reference, policy, RNG, transport и storage

matched author vs normalized
        -> одинаковые модель, контекст, секретный поток и carrier length
           для локализации эффекта нормализации
```

## 2. Зафиксированное окружение

Reference implementation:

```text
repository: harvardnlp/NeuralSteganography
commit:     14e982564aeaf9a33f7b4de440deda2184d17f12
```

Publication reproduction:

```text
model:     GPT-2 Medium (345M)
revision:  6dcaa7a952f72f9298047fd5137cd6e4f05f41da
context:   CNN/DailyMail, первые три предложения статьи
manifest:  cc25b492b26e102496137bc2f8eb78320c7f7768d83ff7e10af9b2e04a14aebc
```

Выбор `test` split и operational Monte-Carlo design `80 contexts × 3 replicates` являются явными решениями нашей репликации: исходная статья не публикует точный Figure-3 split, batch driver и число Monte-Carlo samples.

## 3. Author-compatible Figure 3 reproduction

Полный sweep включал 23 operating points:

- Bins: `b = 1..5`;
- Huffman: candidate pool `2^1..2^8`;
- Arithmetic: `temperature = 0.4..1.2`, `top-k = 300`;
- special Arithmetic: `temperature = 1.0`, `top-k = 50256`.

Всего было запланировано и сохранено **5520 outcomes**. Из них **5513** достигли реальной first-sentence boundary, а **7** были сохранены как `sentence_termination_failure`, а не скрыты фильтрацией. Общая termination success rate — `0.9987318841`.

### 3.1. Воспроизведённые тренды

| Метод / утверждение | Наш результат | Статус |
| --- | --- | --- |
| Bins остаётся в области высокого KL при росте capacity | `1 -> 5 bits/word`, KL `2.223 -> 3.293 bits` | `trend_reproduction` |
| Huffman уменьшает KL при росте candidate pool/capacity | KL `1.925 -> 0.525 bits` | `trend_reproduction` |
| Arithmetic имеет минимум около 4 bits/word при `tau=1` | `3.752 bits/word`, KL `0.100844 bits` | `trend_reproduction` |
| Arithmetic лучше Huffman/Bins по KL-capacity trade-off | ниже обеих baseline-кривых на общей области | `trend_reproduction` |
| special Arithmetic близок к unmodified LM | KL `0.000665525 bits` | `partial_reproduction` |
| exact prose anchor `4e-8 nats` | `0.000461307 nats` при public `precision=26` | `not_reproducible` |
| exact historical Figure-3 MC orchestration | original driver/sample count не опубликованы | `partial_reproduction` |

Итоговая paper-level классификация: **`partial_reproduction`**. Это означает, что основные сравнительные выводы Figure 3 воспроизводятся, но два исторических численных/оркестрационных аспекта не могут быть подтверждены точно.

### 3.2. Сравнительный порядок методов

Piecewise-linear diagnostic по воспроизведённым средним показывает:

- Arithmetic ниже Huffman на всех 1001 grid points общей области `1.001–4.294 bits/word`; минимальный запас по KL ≈ `0.381 bits`;
- Arithmetic ниже Bins на всех 1001 grid points общей области `1.001–4.712 bits/word`; минимальный запас ≈ `1.351 bits`.

То есть главный сравнительный вывод публикации — преимущество Arithmetic по KL-capacity trade-off — воспроизведён устойчиво.

## 4. Расхождение special Arithmetic point

В public Harvard executable `run_single.py` для Arithmetic используется конечная precision `26`. При этом special point `tau=1, k=50256` в full sentence sweep даёт:

```text
4.651269 bits/word
0.000665525 bits KL
0.000461307 nats KL
```

Это примерно в `11532.7×` выше paper prose anchor `4e-8 nats`.

Отдельный zero-padding-free precision probe локализовал чувствительность:

```text
precision 26 -> 7.38468e-4 nats/token
precision 32 -> 2.87668e-5 nats/token
precision 40 -> 2.92390e-8 nats/token
precision 48 -> 4.00890e-10 nats/token
```

`precision=40` попадает в тот же порядок, что paper anchor, но это **не является доказательством**, что авторы использовали precision 40. Публичный single-run script задаёт 26, а исторический Figure-3 batch driver недоступен. Поэтому расхождение фиксируется как finite-precision / undocumented orchestration discrepancy, а не исправляется post-hoc.

## 5. Надёжность author-compatible reproduction

Редкие edge cases не скрывались:

- `7/5520` sentence termination failures;
- `23` завершённых Arithmetic runs с нулевым confirmed payload — они корректно входят в capacity mean как `0`;
- text-transport prefix recovery: `5343/5490 = 97.32%` для positive-payload samples;
- transport recovery не использовался как Figure-3 selection gate, чтобы не создавать selection bias;
- modern-cache compatibility shim применялся только для восстановления historical 1022-token sliding-cache semantics и логировался отдельно.

Эти diagnostics не меняют paper-level curve means, но являются частью воспроизводимой отчётности.

## 6. Matched author-compatible vs normalized comparison

Для проверки нормализации был зафиксирован небольшой differential grid:

```text
4 representative points × 8 identical contexts = 32 pairs
Bins b=3
Huffman e=3
Arithmetic tau=0.9, k=300, precision=26
Arithmetic tau=1.0, k=50256, precision=26
```

В каждой паре одинаковы GPT-2 Medium revision, context hash, secret stream и carrier length. Normalized decoder восстановил полезную нагрузку точно по token IDs в **32/32** случаях.

| Точка | Author BPT | Normalized BPT | Δ BPT | Author `KL(Q||P)` | Normalized `KL(Q||P_ref)` | Exact sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bins b=3 | 3.0000 | 3.0000 | 0.0000 | 2.69835 | 2.90112 | 0/8 |
| Huffman e=3 | 2.47277 | 2.51381 | +0.04103 | 1.06416 | 1.10439 | 6/8 |
| Arithmetic `tau=.9,k=300` | 2.93381 | 3.05800 | +0.12419 | 0.10320 | 0.04782 | 0/8 |
| Arithmetic `tau=1,k=50256` | 4.67733 | 4.59381 | −0.08351 | 0.000605 | 0.000542 | 0/8 |

Exact token parity намеренно не является универсальным conformance criterion: normalized benchmark меняет общую policy, RNG ownership, masking и numerical boundaries.

## 7. Conformance по методам

### Bins

Ключевой инвариант сохранён **точно 8/8**: `b=3` даёт `3 bits/token` и у author, и у normalized реализации. Почти полное расхождение token sequences ожидаемо, потому что normalized adapter использует `V_allowed` и benchmark-owned RNG вместо author full-vocabulary NumPy partition.

Классификация:

`core_principle_preserved_expected_partition_divergence`.

### Huffman

Huffman показывает наиболее сильную token-level conformance: **6/8** matched pairs совпали token-for-token. Для этих шести пар capacity идентична, а максимальное |Δ reverse KL| < `0.002 bits`.

Две divergent пары расходятся уже на первом token, что локализует различие к initial candidate/tree construction. Без per-step candidate traces нельзя уникально разделить влияние author-only token-628 mask, exact-tie ordering и numerical ordering.

Классификация:

`strong_conformance_with_localized_candidate_tree_divergence`.

### Arithmetic Coding

Arithmetic сохраняет finite-precision interval mechanism и exact decode, но ожидаемо чувствителен к малым изменениям probability boundaries.

Для `tau=.9,k=300` normalized и author reverse KL используют одинаковое направление, но **не одну и ту же reference semantics**: author сравнивает с untempered LM, normalized — с canonical `P_reference` после temperature policy. Поэтому численную разницу KL нельзя читать как чистую implementation error.

Для near-unmodified `tau=1,k=50256` distributional conformance особенно сильна:

```text
author reverse KL      = 0.000605481 bits
normalized reverse KL  = 0.000541695 bits
normalized mean TVD    = 0.000479630
```

Низкая token parity при этом допустима: finite-precision interval state быстро расходится от малых boundary differences даже при почти одинаковом агрегированном распределении.

Классификации:

- `core_principle_preserved_with_expected_reference_and_numeric_divergence`;
- `strong_distributional_conformance_with_sequence_sensitivity`.

## 8. Главный методологический результат: два направления KL

Во всех **32/32** normalized matched runs:

```text
D_KL(P_reference || Q_stego) = +inf
```

Это математически ожидаемо для sparse `Q_stego`: метод назначает нулевую вероятность части токенов, которым `P_reference` оставляет положительную массу.

Поэтому итоговая рекомендация для будущей specification v1.0:

```text
kl_ref_to_stego + infinite_steps
    -> строгий support-mismatch diagnostic, без epsilon smoothing

kl_stego_to_ref
    -> конечная complementary / author-comparable distortion metric

TVD
    -> конечная companion metric, не насыщаемая support mismatch так же, как forward KL
```

Specification v0.1 на Этапе 3 не переписывается задним числом; рекомендация переносится в notes для будущей v1.0.

## 9. Итоговый ответ на вопрос Этапа 3

**Да, нормализованные реализации Bins, Huffman и Arithmetic Coding сохраняют принципиальное поведение исходных методов на проверенных representative matched points.**

Доказательная цепочка:

```text
pinned author code
    -> author-compatible smoke parity
    -> paper-sentence protocol
    -> full Figure-3 sweep
    -> publication-level trend reproduction
    -> matched author/normalized runs
    -> discrepancy attribution
    -> core principle preserved for 4/4 representative points
```

При этом воспроизводимость исходной публикации не объявляется абсолютной. Exact `4e-8 nats` anchor и exact historical Monte-Carlo orchestration не воспроизведены/недоступны и явно остаются ограничениями.

## 10. Ограничения Этапа 3

1. Публикация не содержит exact Figure-3 batch driver, sample count и однозначного dataset split.
2. Matched diagnostic использует только 4 representative points × 8 contexts, а не полный Figure-3 grid в normalized mode.
3. Для двух divergent Huffman pairs отсутствуют per-step candidate-list traces, поэтому непосредственный источник первого tree divergence не идентифицирован уникально.
4. При `tau=.9` author и normalized reverse KL имеют разные reference-policy semantics.
5. Public precision=26 не воспроизводит exact `4e-8 nats` special-point anchor.
6. Text transport recovery в historical implementation не является идеальным; такие случаи сохранены как diagnostics, а не исключены из основной выборки.

## 11. Артефакты воспроизводимости

Основные документы:

```text
docs/stages/stage3/reproducibility_protocol.md
docs/stages/stage3/figure3_full_run.md
docs/stages/stage3/figure3_interpretation.md
docs/stages/stage3/matched_author_normalized.md
docs/stages/stage3/conformance_analysis.md
docs/stages/stage3/reproducibility_report.md
docs/stages/stage3/stage_3.typ
```

Machine-readable evidence:

```text
results/stage3/paper_reproduction/figure3_full/summary.json
results/stage3/paper_reproduction/figure3_full/interpretation.json
results/stage3/matched_author_normalized/summary.json
results/stage3/matched_author_normalized/conformance_analysis.json
results/stage3/comparison.csv
results/stage3/comparison.parquet
results/stage3/stage3_closeout_summary.json
results/stage3/stage3_readiness.json
```

Итоговая автоматическая проверка:

```bash
python scripts/finalize_stage3.py
python scripts/check_stage3_readiness.py --run-tests \
  --json-output results/stage3/stage3_readiness.json
```

Критерий закрытия:

```text
Stage 3 reproducibility readiness: READY
```

После прохождения gate репозиторий готов к **Этапу 4 — подключению современных методов (ADG, Discop, RRC и последующих методов ROADMAP)** без изменения frozen conclusions Stage 3.
