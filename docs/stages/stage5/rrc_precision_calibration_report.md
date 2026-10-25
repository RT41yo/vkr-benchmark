# Stage 5 — RRC precision calibration: итоговый отчет

## 1. Назначение

RRC использует fixed-payload lifecycle и не входит в 128-token streaming Pilot A. Отдельный calibration проверяет, достаточно ли текущей Decimal precision для range-state arithmetic и какое значение `precision_margin` фиксировать дальше.

Проверяемый параметр является численной настройкой реализации, а не научной operating-point осью RRC.

Manifest:

`configs/pilots/stage5/rrc_precision_margin_calibration_llama.json`

Raw log:

`results/stage5/sweeps/stage5-rrc-precision-margin-calibration-llama-v1/execution.jsonl`

SHA-256:

`6d5c9982514fdc4362c300d9b76b81b6964c6472a1e6bee0672bf348a054ae3e`

## 2. Режим

```text
model = meta-llama/Llama-3.2-3B
common top_p = 1.0
common top_k = null
temperature = 1.0
fixed payload = 1024 bits
precision_margin = {20,40,60}
verify_termination = true
4 prompts × 2 paired replicates
```

1024 bits выбран как stress-case для Decimal precision; это не означает, что 1024 bits уже является финальным Stage-6 payload.

## 3. Результат precision-margin sweep

Все 24/24 runs завершились успешно:

- exact secret recovery: 24/24;
- max BER: 0;
- numerical/termination failure: 0.

| Margin | Exact | Mean BPT | Mean carrier tokens | Carrier range | Entropy util. | Mean KL `Q→P_ref` | Mean TVD | Mean wall/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 8/8 | 3.8469 | 266.875 | 247–284 | 0.9800 | 1.56e-16 | 2.53e-12 | 28.06 s |
| 40 | 8/8 | 3.8469 | 266.875 | 247–284 | 0.9800 | 1.56e-16 | 2.53e-12 | 28.00 s |
| 60 | 8/8 | 3.8469 | 266.875 | 247–284 | 0.9800 | 1.56e-16 | 2.53e-12 | 28.12 s |

Для каждого одинакового `(prompt_id, replicate_id)` значения BPT, carrier length, entropy utilization, raw-LM PPL, KL и TVD совпали между margin 20/40/60. Различается только небольшая wall-time флуктуация.

Среднее `stego_algorithm_total_ms`:

```text
margin20: 17.13 s
margin40: 17.19 s
margin60: 17.36 s
```

## 4. Вывод по `precision_margin`

Для payload 1024 bits диапазон `20..60` находится на наблюдаемом plateau: увеличение Decimal precision не изменило траектории, capacity, reliability или distribution metrics.

Это означает, что `precision_margin=40` остается разумным консервативным кандидатом на fixed setting. Calibration не показывает необходимости увеличивать его до 60.

Низкие margins `{0,10}` пока не проверялись. Они нужны только если исследовательская задача изменится с «выбрать заведомо достаточную precision» на «найти минимальный порог достаточности». Для текущего benchmark такой поиск не обязателен.

## 5. Отдельно обнаруженный вопрос: исчезновение очень малых интервалов

В 12 из 24 runs forward KL `P_ref→Q_RRC` равен `+inf`:

```text
margin20: 4/8
margin40: 4/8
margin60: 4/8
```

При этом одновременно:

- reverse KL находится около `1e-16`;
- TVD находится около `2.5e-12`;
- exact secret recovery = 100%.

Это не зависит от `precision_margin`: для одного и того же `(prompt, replicate)` forward KL имеет одинаковое поведение при 20, 40 и 60.

## 6. Почему этот эффект отделен от `precision_margin`

Текущая normalized RRC implementation имеет два численно разных этапа.

### 6.1. Построение token intervals

В `src/vkr_benchmark/methods/rrc.py` текущие probabilities сначала накапливаются в `np.float64`:

```text
P_reference
  -> deterministic token order
  -> FP64 cumulative boundaries
```

После этого границы переводятся в `Decimal(str(boundary))` для дальнейшего range arithmetic.

Independent `explicit_rrc_q()` намеренно повторяет то же FP64 partition и получает `Q` как ширины соседних cumulative intervals.

### 6.2. Range-state arithmetic

Секретная точка, текущие `[left,right)` и rotation/narrowing работают в Decimal с precision:

```text
ceil(payload_bits * log10(2)) + precision_margin
```

Именно этот второй уровень проверял sweep `{20,40,60}`.

Если два очень близких cumulative boundaries уже совпали в FP64, последующее увеличение Decimal precision не может восстановить потерянную разницу.

## 7. Рабочая интерпретация `forward KL = +inf`

Наиболее согласованное с текущей реализацией объяснение: у крайне маловероятного token может быть

```text
P_ref(token) > 0
```

но после FP64 cumulative partition соседние boundaries совпадают и практическая interval width становится нулевой:

```text
Q_RRC(token) = 0
```

Тогда по определению

```text
KL(P_ref || Q_RRC) = +inf
```

даже если потерянная probability mass чрезвычайно мала. Поэтому `+inf` forward KL совместим с TVD порядка `1e-12`.

Важное ограничение: этот вывод сильно поддерживается фактическим кодом partition и observed metrics, но отдельный high-precision boundary sensitivity experiment еще не выполнен. Поэтому отчет фиксирует его как открытый numerical issue, а не как окончательно закрытую причинную проверку.

## 8. Влияние на benchmark

На текущем этапе нельзя просто приписать practical RRC значение `KL=0` на основании идеального теоретического свойства rotation.

Benchmark должен различать:

1. идеальное свойство RRC при непрерывной/точной арифметике;
2. поведение конкретной normalized implementation с конечной точностью boundaries и PRNG;
3. reliability fixed-payload channel.

Поэтому:

- raw `+inf` forward KL не сглаживается epsilon-клиппингом;
- reverse KL и TVD сохраняются рядом и позволяют видеть, что потерянная mass очень мала;
- `precision_margin=40` можно считать достаточным для Decimal range-state arithmetic на проверенном payload;
- окончательный RRC regime для Stage 6 пока не замораживается до отдельного решения по boundary precision и последующей payload calibration.

## 9. Следующий RRC шаг

Перед финальным Stage-6 freeze требуется отдельный sensitivity probe, который меняет только численный способ построения probability boundaries, не смешивая его с `precision_margin`.

Цель probe:

- проверить, исчезает ли support loss при более точном accumulation;
- сравнить forward/reverse KL и TVD;
- убедиться, что reliability, BPT и carrier length не получают неожиданных изменений;
- после этого решить, что считать основной practical implementation, а что — sensitivity variant.

До такого решения новый ADR по boundary precision не создается.
