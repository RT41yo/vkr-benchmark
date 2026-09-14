# Этап 3 — Step 3.11: интерпретация полного воспроизведения Figure 3

**Статус:** завершён аналитический разбор полного Step-3.10 sweep.  
**Базовый commit:** `01244d2` (`Complete full Figure-3 author-compatible sweep`).  
**Источник сравнения:** Ziegler, Deng, Rush (2019), *Neural Linguistic Steganography*, EMNLP-IJCNLP 2019, Figure 3, DOI `10.18653/v1/D19-1115`.

## 1. Что именно интерпретируется

Step 3.10 завершил author-compatible paper-sentence sweep:

```text
23 Figure-3 points
80 frozen CNN/DailyMail contexts
3 deterministic secret-stream replicates
5520 scheduled outcomes
5513 successfully terminated sentences
7 sentence_termination_failure outcomes
```

Step 3.11 **не запускает модель заново**. Он читает агрегаты Step 3.10, строит воспроизведённую кривую `bits/word -> KL`, формально проверяет характерные зависимости и классифицирует paper-level claims по шкале из `reproducibility_protocol.md`.

Главные входы:

```text
results/stage3/paper_reproduction/figure3_full/figure3_points.csv
results/stage3/paper_reproduction/figure3_full/summary.json
results/stage3/paper_reproduction/arithmetic_precision_probe/interpretation.json
```

Машиночитаемые результаты Step 3.11:

```text
results/stage3/paper_reproduction/figure3_full/interpretation.json
results/stage3/paper_reproduction/figure3_full/claim_assessment.csv
results/stage3/paper_reproduction/figure3_full/figure3_reproduction.svg
```

## 2. Ограничение сравнения с Figure 3

Статья публикует Figure 3 и сообщает, что каждая точка является средним со standard error по repeated samples при фиксированном tuning parameter. Однако статья не публикует:

- machine-readable таблицу координат Figure 3;
- точное число Monte-Carlo samples Figure 3;
- оригинальный batch/Monte-Carlo driver;
- явную политику редких non-terminating generations;
- точный dataset split Figure 3.

Поэтому Step 3.11 **не оцифровывает график постфактум и не вводит произвольный численный tolerance для кривых**. Для основных кривых оцениваются воспроизведение направления, формы и сравнительного порядка. Единственное точное paper-level число, проверяемое напрямую, — prose-anchor `KL = 4e-8 nats` для unmodulated Arithmetic.

## 3. Наш Figure-3 trade-off

Сводные точки полного sweep:

| Метод / точка | bits/word | KL, bits |
|---|---:|---:|
| Bins b=1 | 1.000 | 2.223212 |
| Bins b=2 | 2.000 | 2.578298 |
| Bins b=3 | 3.000 | 2.886374 |
| Bins b=4 | 4.000 | 3.151111 |
| Bins b=5 | 5.000 | 3.292654 |
| Huffman e=1 | 1.000 | 1.924780 |
| Huffman e=2 | 1.801 | 1.567508 |
| Huffman e=3 | 2.377 | 1.209243 |
| Huffman e=4 | 2.903 | 0.943162 |
| Huffman e=5 | 3.322 | 0.781565 |
| Huffman e=6 | 3.711 | 0.649648 |
| Huffman e=7 | 4.065 | 0.527217 |
| Huffman e=8 | 4.294 | 0.525106 |
| Arithmetic tau=0.4, k=300 | 1.001 | 0.872972 |
| Arithmetic tau=0.5, k=300 | 1.354 | 0.656522 |
| Arithmetic tau=0.6, k=300 | 1.779 | 0.478686 |
| Arithmetic tau=0.7, k=300 | 2.267 | 0.325141 |
| Arithmetic tau=0.8, k=300 | 2.711 | 0.199000 |
| Arithmetic tau=0.9, k=300 | 3.199 | 0.128595 |
| **Arithmetic tau=1.0, k=300** | **3.752** | **0.100844** |
| Arithmetic tau=1.1, k=300 | 4.166 | 0.126378 |
| Arithmetic tau=1.2, k=300 | 4.712 | 0.200870 |
| **Arithmetic tau=1.0, k=50256** | **4.651** | **0.000665525** |

График сохраняется как `figure3_reproduction.svg`. Error bars на нём — run-level standard error из predeclared aggregation Step 3.10; context-clustered SE остаётся в `summary.json`/`figure3_points.csv` как диагностическая альтернативная оценка.

## 4. Bins / Block

Bins воспроизводит ожидаемую область высокого KL. Ёмкость по construction равна `b` bits/word и поэтому проходит ровно через `1,2,3,4,5`. Средний KL растёт от `2.223` до `3.293 bits`.

Это не объявляется точным численным совпадением с нарисованными координатами статьи, потому что бумажная Figure 3 не даёт точной таблицы точек. Но характерный вывод воспроизводится: Bins создаёт значительно большее распределительное искажение, чем Huffman и Arithmetic на сопоставимой ёмкости.

**Статус:** `trend_reproduction`.

## 5. Huffman

По мере роста candidate pool ёмкость увеличивается:

```text
1.000 -> 4.294 bits/word
```

а KL монотонно снижается:

```text
1.924780 -> 0.525106 bits
```

Тем самым воспроизводится характерная форма Huffman-кривой Figure 3: динамический код лучше Bins по распределительному искажению и приближается к LM distribution при увеличении candidate pool, но остаётся заметно выше Arithmetic.

**Статус:** `trend_reproduction`.

## 6. Arithmetic, k=300

Temperature sweep даёт выраженную U-образную зависимость KL. От `tau=0.4` до `tau=1.0` KL последовательно падает:

```text
tau=0.4: KL = 0.872972 bits, bits/word = 1.001
tau=0.5: KL = 0.656522 bits
tau=0.6: KL = 0.478686 bits
tau=0.7: KL = 0.325141 bits
tau=0.8: KL = 0.199000 bits
tau=0.9: KL = 0.128595 bits
tau=1.0: KL = 0.100844 bits, bits/word = 3.752
```

После `tau=1.0` KL снова растёт:

```text
tau=1.1: KL = 0.126378 bits
tau=1.2: KL = 0.200870 bits
```

Статья формулирует минимум Arithmetic как находящийся примерно около `4 bits/word` при `tau=1.0`. Наш минимум находится **при том же `tau=1.0` и на 3.752 bits/word**, то есть характерное положение минимума воспроизводится без подбора параметров после просмотра результатов.

**Статус:** `trend_reproduction`.

## 7. Сравнительный порядок методов

Для дополнительной проверки Step 3.11 сравнивает не отдельные удобные точки, а кусочно-линейную интерполяцию **наших собственных mean curves** на общей области bits/word. Это diagnostic нашего воспроизведения, а не попытка восстановить скрытые координаты paper Figure 3.

На общей области Arithmetic/Huffman `1.001–4.294 bits/word` Arithmetic ниже Huffman во всех `1001` grid points. Минимальный запас:

```text
Huffman KL - Arithmetic KL >= 0.381349 bits
```

На общей области Arithmetic/Bins `1.001–4.712 bits/word` Arithmetic также ниже Bins во всех `1001` grid points. Минимальный запас:

```text
Bins KL - Arithmetic KL >= 1.350678 bits
```

Следовательно, основной сравнительный вывод Figure 3 — Arithmetic даёт лучший статистический trade-off — уверенно сохраняется.

**Статус:** `trend_reproduction`.

## 8. Special unmodulated Arithmetic point

Для `tau=1.0, k=50256, precision=26` полный sentence sweep даёт:

```text
bits/word = 4.651269
KL = 0.000665525 bits
KL = 0.000461307 nats
```

Относительно нашего `tau=1.0, k=300` это уменьшение KL примерно в `151.5x`. То есть **качественный near-zero эффект полностью виден**: снятие top-k truncation резко приближает стего-распределение к исходному LM distribution.

Однако paper prose сообщает `4e-8 nats`. Наш полный результат при pinned executable `precision=26` выше этого anchor примерно в:

```text
0.000461307 / 4e-8 = 11532.7x
```

Поэтому точное paper-level число не воспроизведено.

### 8.1. Почему это расхождение не считается необъяснённым

Step 3.8 precision probe уже локализовал finite-precision effect на zero-padding-free шагах:

```text
precision=26: mean clean KL = 7.38468e-4 nats
precision=40: mean clean KL = 2.92390e-8 nats
```

`precision=40` попадает в тот же порядок величины, что `4e-8 nats`, тогда как public `run_single.py` задаёт `precision=26`. Но из этого **нельзя делать вывод, что авторы Figure 3 использовали precision=40**: оригинальный Figure-3 driver отсутствует, а executable reference указывает на 26.

Поэтому итоговая формулировка:

- near-zero unmodulated behavior — `partial_reproduction`;
- точный anchor `4e-8 nats` на pinned public precision=26 — `not_reproducible`;
- вероятный источник расхождения локализован до finite precision и/или недоступной historical orchestration.

## 9. Diagnostics, обнаруженные полным sweep

### 9.1. Sentence termination

Из `5520` scheduled outcomes:

```text
5513 successfully terminated
7 sentence_termination_failure
99.873% global termination success
```

Все 7 failures относятся к Arithmetic:

```text
tau=0.4, k=300: 5 / 240
tau=0.5, k=300: 1 / 240
tau=1.0, k=300: 1 / 240
```

Для Arithmetic в целом:

```text
2393 / 2400 completed
99.708% termination success
```

Figure-3 means для этих точек условны на successful sentence termination; failure rate хранится отдельно и не маскируется искусственной boundary.

### 9.2. Zero confirmed payload

В `23 / 2393 = 0.961%` успешно завершённых Arithmetic sentences на первой boundary arithmetic interval ещё не подтвердил ни одного payload bit. Такие outcomes корректно входят в capacity mean с `bits/word=0`.

### 9.3. Text transport / BPE recovery

Exact confirmed-prefix recovery среди положительного payload:

```text
Bins:       1150 / 1200 = 95.833%
Huffman:    1856 / 1920 = 96.667%
Arithmetic: 2337 / 2370 = 98.608%
Overall:    5343 / 5490 = 97.322%
```

Это не execution gate Figure 3, потому что paper KL/BPW характеризуют generation-side distribution/capacity. Но это важный результат для нашего будущего normalized benchmark: надёжность text transport нельзя смешивать с distribution distortion; её нужно хранить отдельной robustness-метрикой.

### 9.4. Historical cache compatibility

Из `2400` Arithmetic scheduled runs compatibility trimming понадобился только в:

```text
encode: 14 runs, 57951 trim events
decode: 7 runs, 7074 trim events
```

Это подтверждает, что compatibility shim обслуживает редкий long-context edge case, а не меняет типичное поведение метода.

## 10. Классификация paper claims

| Claim | Статус |
|---|---|
| Paper-level orchestration целиком | `partial_reproduction` |
| Bins high-KL trade-off | `trend_reproduction` |
| Huffman decreasing-KL trade-off | `trend_reproduction` |
| Arithmetic minimum near 4 bits/word at tau=1 | `trend_reproduction` |
| Arithmetic ниже Huffman/Bins по Figure-3 KL trade-off | `trend_reproduction` |
| Unmodulated Arithmetic near-zero behavior | `partial_reproduction` |
| Exact `4e-8 nats` anchor | `not_reproducible` at pinned public precision=26 |
| Exact historical repeated-sample estimator | `partial_reproduction` |

Полная машиночитаемая версия с evidence находится в `claim_assessment.csv`.

## 11. Итог Step 3.11

Общая классификация — **`partial_reproduction`**, но это не означает, что основная Figure-3 картина не воспроизведена. Наоборот:

1. характерные кривые Bins, Huffman и Arithmetic воспроизведены;
2. сравнительный порядок `Arithmetic < Huffman < Bins` по KL на общей области capacity воспроизведён;
3. Arithmetic minimum снова находится при `tau=1.0` примерно около `4 bits/word`;
4. unmodulated Arithmetic снова даёт near-zero KL;
5. единственный точный prose-anchor `4e-8 nats` не воспроизводится на публично зафиксированной `precision=26`, но расхождение локализовано finite-precision экспериментом;
6. точную historical Figure-3 Monte-Carlo orchestration восстановить нельзя из публичных материалов.

Таким образом, **ядро авторского экспериментального вывода воспроизводится по трендам и относительному порядку**, а оставшиеся ограничения явно измерены и документированы.

## 12. Значение для нашего benchmark

Step 3.11 подтверждает, что исторические методы можно переносить в единый benchmark, но нельзя считать paper-level число единственной проверкой корректности. Для normalized режима необходимо отдельно фиксировать:

- направление KL;
- finite precision;
- termination policy;
- confirmed payload semantics;
- transport/extraction reliability;
- длину carrier;
- единый candidate/reference distribution pipeline.

Именно поэтому следующий шаг не меняет author-compatible код, а сравнивает **author-compatible и normalized реализации в matched conditions**.

## 13. Gate

Step 3.11 считается пройденным, если:

```text
scientific claims classified                 PASS
Arithmetic tau=1 minimum found               PASS
Arithmetic mean curve < Huffman/Bins         PASS
special-point discrepancy quantified         PASS
precision explanation preserved              PASS
termination/transport diagnostics reported   PASS
```

Следующее действие:

```text
Step 3.12 — matched author-compatible vs normalized comparison
```

Перед Step 3.12 необходимо убедиться, что normalized metric layer действительно сохраняет **оба** направления KL согласно ADR-0012, чтобы matched comparison не смешивал `D_KL(Q_stego || P_reference)` и `D_KL(P_reference || Q_stego)`.
