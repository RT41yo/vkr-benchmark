# Этап 3 — targeted investigation после GPT-2 Medium pilot

**Статус:** диагностический шаг до полного Figure-3 sweep.  
**Основание:** GPT-2 Medium pilot завершился 32/32 успешно, но выявил два расхождения, которые нельзя скрывать или «подгонять» под статью.

## 1. Что именно обнаружил pilot

Для representative points на 8 frozen CNN/DailyMail contexts получено:

```text
Bins b=3:                    mean KL = 2.904608 bits/token
Huffman exponent=3:          mean KL = 1.414304 bits/token
Arithmetic tau=0.9, k=300:   mean KL = 0.505828 bits/token
Arithmetic tau=1, k=50256:   mean KL = 0.459775 bits/token
```

Общий qualitative ordering `Arithmetic < Huffman < Bins` проявился, однако специальная Arithmetic-точка не воспроизвела paper near-zero claim. Это не payload/transport failure: каждый из 32 runs завершился со статусом `ok`, exact useful payload prefix восстановлен, reference checkout остался clean.

Отдельно 5/32 runs содержат sentence-finish token **до** финального generated token. Это означает, что public `finish_sent=True` допускает переход через первую границу предложения, пока фиксированный payload ещё не исчерпан.

## 2. Почему full sweep пока заблокирован

Paper-level задача должна различать три слоя:

```text
paper claim
published executable code
наша современная compatibility execution
```

Нельзя исправлять author code так, чтобы он дал желаемую цифру. Сначала требуется локализовать, откуда появляется ненулевой KL и какая sentence-stop orchestration соответствует фразе paper «generate an entire sentence given a uniform random message».

## 3. Исторический код проверен отдельно

История `harvardnlp/NeuralSteganography` показывает, что `arithmetic.py` существовал уже в первом публичном commit Zachary Ziegler от 2019-09-03 (`05a3944f...`). Основная finite-precision arithmetic logic уже тогда использовала текущую ширину integer interval, threshold `1 / cur_int_range`, integer rounding и авторское заполнение остатка через сдвиг всей cumulative vector.

Оригинальный `run_single.py` 2019 года также содержит `precision = 26` для cover-text Arithmetic example. Следовательно, Step-3.7 discrepancy нельзя автоматически списать на compatibility update 2025 года.

Это не доказывает, что Figure 3 запускалась тем же exact driver: опубликованный repository не содержит исходный Figure-3 batch/Monte-Carlo runner.

## 4. Что измеряет precision probe

Frozen config:

```text
configs/reproducibility/arithmetic_precision_probe.json
```

Диагностическая точка остаётся:

```text
temperature = 1.0
topk        = 50256
model       = gpt2-medium
contexts    = те же 8 frozen pilot contexts
```

Меняется только arithmetic precision:

```text
26, 32, 40, 48 bits
```

Для каждого context используется один и тот же deterministic 256-bit uniform-looking stream. `finish_sent=false`, поскольку этот probe изолирует именно payload-coding distribution от sentence-completion tail.

256 bits выбраны только для диагностики. Почти все steps имеют полный `precision`-битный lookahead без implicit zero padding; summary отдельно считает только такие steps.

## 5. Почему instrumented mirror допустим

External author checkout не изменяется. Repository-owned runner повторяет executable arithmetic loop только для instrumentation и **до sweep обязан пройти parity sentinel** с pinned `arithmetic.encode_arithmetic`:

```text
same generated token IDs
same average author KL
same words_per_bit
```

Если parity не проходит, весь diagnostic считается недостоверным и прекращается.

## 6. Декомпозиция per-step KL

Для каждого payload step сохраняются ширина interval, effective precision `log2(width)`, threshold, число retained candidates, округлённый integer residual и три распределения.

`author_kl_bits` — именно исполняемая semantics pinned code. После округления author code вычисляет shortfall

```text
residual = cur_int_range - cumulative[-1]
```

и добавляет его **ко всей cumulative vector**. После обратного differencing это эквивалентно добавлению остатка в первый candidate bin.

Дополнительно, только как counterfactual diagnostics, считаются:

```text
truncation_only_kl_bits
    LM distribution, только renormalized на реально retained prefix;

terminal_fill_counterfactual_kl_bits
    те же rounded integer widths, но residual помещён в последний bin.
```

Ни один counterfactual не используется для выбора token и не объявляется исправленным авторским методом. Они нужны только для локализации источника KL.

## 7. Что будет считаться подтверждением гипотезы

Probe не имеет hard requirement «KL обязан падать». Hard gate — только execution, reference parity и неизменность external checkout.

После фактического запуска анализируем:

- меняется ли author KL при `precision 26 -> 32 -> 40 -> 48`;
- совпадают ли крупные KL steps с низкой effective interval precision;
- насколько KL объясняется pure truncation;
- насколько он меняется в terminal-fill counterfactual;
- коррелирует ли rounding residual fraction с author KL.

Если KL быстро стремится к нулю с ростом precision, finite precision локализована как основной источник discrepancy. Если нет, исследование продолжается без изменения frozen Figure-3 matrix.

## 8. Sentence-shape audit

Отдельный deterministic script:

```text
scripts/analyze_stage3_pilot_sentence_shape.py
```

не запускает модель заново. Он читает committed Step-3.7 `result.json` по exact SHA-256 и фиксирует, где первая sentence boundary произошла раньше конца generation.

Текущий audit даёт:

```text
32 total runs
27 runs: ровно одна boundary на последнем token
5 runs:  ранняя boundary присутствует

Arithmetic tau=0.9, k=300: 2
Arithmetic tau=1, k=50256: 2
Huffman exponent=3:         1
Bins b=3:                    0
```

Это достаточно, чтобы не использовать fixed-24-bit + `finish_sent=True` как final Figure-3 sentence orchestration. Следующий paper-like driver должен работать с достаточно длинным uniform bitstream и рассматривать **первую** sentence boundary как candidate stop.

## 9. Команды

Сначала детерминированный audit уже полученного pilot:

```bash
python scripts/analyze_stage3_pilot_sentence_shape.py
```

Затем GPU diagnostic:

```bash
python scripts/run_stage3_arithmetic_precision_probe.py
python scripts/check_stage3_arithmetic_precision_probe.py
```

Full Figure-3 sweep остаётся заблокированным до интерпретации результата этого шага.

## 10. Фактический результат precision probe

Probe завершён успешно: 32/32 runs, sentinel mirror воспроизвёл pinned `arithmetic.py` с точным совпадением generated token IDs, average KL и words/bit; reference worktree остался неизменным.

Критически важно различать две агрегации. Поле `mean_run_author_kl_bits` усредняет весь finite 256-bit run, включая терминальные steps, где look-ahead уже вышел за конец payload и author code дополнил его нулями. Поле `mean_zero_padding_free_author_kl_bits` использует только steps, где весь `precision`-битный look-ahead состоял из реальных secret bits.

Получено:

```text
precision   full-run mean KL      zero-padding-free KL      clean KL in nats
26          5.9635e-2 bits        1.0654e-3 bits             7.3847e-4
32          7.5480e-2 bits        4.1502e-5 bits             2.8767e-5
40          5.1613e-2 bits        4.2183e-8 bits             2.9239e-8
48          4.9012e-2 bits        5.7836e-10 bits            4.0089e-10
```

Следовательно, прежний console label `mean author KL` для clean subset был двусмысленным. Он исправлен: runner/checker теперь явно выводят и `full-run mean author KL`, и `zero-padding-free author KL`. Сам сохранённый `result.json` уже содержал обе величины раздельно, поэтому GPU rerun для исправления подписи не требуется.

## 11. Что именно объяснилось

Наблюдение Step 3.7 `KL ~= 0.46 bits/token` нельзя интерпретировать как steady-state KL unmodulated Arithmetic. Pilot использовал только 24 useful bits при `precision=26`. В public encoder каждый coding step читает окно длиной `precision`; если реальных битов не хватает, хвост окна неявно дополняется нулями. Так как `24 < 26`, уже **первый** Arithmetic step pilot находился в padding-affected режиме, и все последующие steps тоже.

Длинный 256-bit probe позволил отделить этот terminal effect. На `precision=26` zero-padding-free steps дают около `1.07e-3 bits/token`, тогда как padding-affected steps дают на порядки больший вклад. В сумме padding-affected steps отвечают примерно за `98.57%` summed per-step KL при precision=26; для 32/40/48 — более чем за `99.95%`.

Поэтому исходная проблема распалась на два эффекта:

```text
finite precision / shrinking interval
    -> ненулевой steady-state residual даже до конца payload;

finite message + implicit zero look-ahead
    -> сильный terminal distortion и огромный рост KL около конца сообщения.
```

Первый эффект также подтверждён количественно: clean KL падает от `1.065e-3 bits` при precision=26 до `4.218e-8 bits` при precision=40 и `5.784e-10 bits` при precision=48. На clean steps корреляция rounding-residual fraction с author KL положительна и велика; при precision=26 она около `0.85`, при precision=32 около `0.996`.

## 12. Сопоставление с paper near-zero point

Frozen matrix хранит paper prose anchor `4e-8 nats` и отдельно помнит, что ось Figure 3 подписана в bits. Для zero-padding-free precision=40 probe получено:

```text
4.2183e-8 bits/token
= 2.9239e-8 nats/token
```

Это тот же порядок величины, что и paper prose anchor. Однако **это не означает, что Figure 3 доказанно использовала precision=40**. Открытый `run_single.py` для cover Arithmetic показывает precision=26, а оригинальный Figure-3 batch driver отсутствует. Поэтому корректный вывод уже уже:

> При pinned executable precision=26 опубликованная near-zero величина численно не воспроизводится даже на zero-padding-free steps (`~7.38e-4 nats/token`). Повышение precision демонстрирует ожидаемое стремление к нулю, а precision=40 достигает масштаба paper anchor. Точный исторический способ получения special Figure-3 point остаётся недокументированным.

Иными словами, compatibility layer не является источником расхождения; instrumented mirror имеет exact parity с pinned executable code.

## 13. Решение после probe

Precision discrepancy считается **локализованной**, но full Figure-3 sweep всё ещё не разрешён. Остался второй pre-sweep blocker: final orchestration должна соответствовать paper rule `generate one entire sentence given a uniform random message`, а не fixed short payload + greedy completion.

Следующий шаг должен заморозить paper-sentence driver:

```text
sufficiently long deterministic uniform bitstream
        -> author-compatible embedding
        -> STOP на первой sentence boundary
        -> payload = только биты, подтверждённо встроенные до этой boundary
```

До фиксации и pilot-проверки этого driver полный 23-point sweep остаётся заблокированным.

Машиночитаемая интерпретация текущего probe генерируется командой:

```bash
python scripts/analyze_stage3_arithmetic_precision_result.py
```

и сохраняется в:

```text
results/stage3/paper_reproduction/arithmetic_precision_probe/interpretation.json
```
