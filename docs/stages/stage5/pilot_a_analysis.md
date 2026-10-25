# Stage 5 — Pilot A: итоговый аналитический отчет

## 1. Назначение

Pilot A — широкий screening, а не финальный сравнительный эксперимент. Его задача — пройти по широким method-local сеткам на небольшой, но одинаковой для всех методов выборке и определить:

- где находятся содержательные области параметров;
- где проявляются явные потери качества, надежности или чрезмерная вычислительная стоимость;
- какие области имеет смысл переносить в более дорогой Pilot B;
- какие дополнительные диагностические вопросы необходимо закрыть до Stage 6.

Pilot A не выбирает единственную «лучшую» конфигурацию и не используется для окончательного ранжирования методов.

## 2. Экспериментальный режим

Общий режим:

```text
model = meta-llama/Llama-3.2-3B
temperature = 1.0
common top_k = null
common top_p = 1.0
exclude_special_tokens = true
carrier length = 128 tokens
prompts = s5p0001..s5p0004
paired replicates = r01, r02
```

RRC в этот sweep не входит: для него применяется отдельный fixed-payload lifecycle.

Исполнительный manifest:

`configs/pilots/stage5/pilot_a_streaming_llama.json`

Основной журнал:

`results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/execution.jsonl`

SHA-256 журнала:

`e55675aa8d9466477c2571ea80f8679eabeabf3022dee87d09d56a871f80c5b2`

## 3. Полнота sweep

Запланировано и записано 360 runs:

| Метод | Method points | Runs |
|---|---:|---:|
| Bins | 8 | 64 |
| Huffman | 8 | 64 |
| Arithmetic Coding | 15 | 120 |
| ADG | 1 | 8 |
| Discop | 1 | 8 |
| DAIRstega | 12 | 96 |
| **Итого** | **45** | **360** |

Статусы:

- `ok`: 347;
- hard error: 13;
- сумма `elapsed_wall_seconds` по всем записям: 9703.22 s = 2.70 h.

Сумма per-run wall time используется только как характеристика стоимости этого sweep; это не равняется обязательно внешнему времени от запуска команды до ее завершения.

## 4. Надежность

Для benchmark успешным считается не только завершение процесса, но и точное восстановление секрета после обычного text transport.

| Метод | Runs | Hard errors | `ok`, но secret mismatch | Exact | Exact rate |
|---|---:|---:|---:|---:|---:|
| Bins | 64 | 0 | 18 | 46 | 71.9% |
| Huffman | 64 | 8 | 7 | 49 | 76.6% |
| Arithmetic Coding | 120 | 0 | 4 | 116 | 96.7% |
| ADG | 8 | 0 | 0 | 8 | 100% |
| Discop | 8 | 0 | 0 | 8 | 100% |
| DAIRstega | 96 | 5 | 0 | 91 | 94.8% |
| **Итого** | **360** | **13** | **29** | **318** | **88.3%** |

Все 42 наблюдавшихся нарушения exact recovery исследованы отдельно. Итог: они связаны с изменением token sequence после `token ids -> text -> retokenized token ids`, а не с обнаруженной внутренней асимметрией encoder/decoder при передаче исходных sender token IDs. Подробности: `docs/stages/stage5/pilot_a_reliability_analysis.md`.

## 5. Bins

`block_size=b` напрямую задает `b` bits/token, поэтому Pilot A хорошо показывает цену роста capacity.

| Point | Exact | BPT | Entropy util. | PPL | KL `Q→P_ref`, bits | TVD |
|---|---:|---:|---:|---:|---:|---:|
| b1 | 8/8 | 1.000 | 0.279 | 8.32 | 1.942 | 0.555 |
| b2 | 7/8 | 2.000 | 0.529 | 19.88 | 2.284 | 0.594 |
| b3 | 5/8 | 3.000 | 0.710 | 49.68 | 2.673 | 0.652 |
| b4 | 6/8 | 4.000 | 0.799 | 111.20 | 2.747 | 0.681 |
| b5 | 6/8 | 5.000 | 0.771 | 198.24 | 2.476 | 0.670 |
| b6 | 7/8 | 6.000 | 0.852 | 449.71 | 2.743 | 0.705 |
| b7 | 4/8 | 7.000 | 0.929 | 883.91 | 2.856 | 0.715 |
| b8 | 3/8 | 8.000 | 0.885 | 1405.43 | 2.416 | 0.673 |

Forward KL `P_ref→Q` равен `+inf` во всех 64 successful Bins runs из-за support mismatch, что соответствует конструкции с фиксированными bins.

Основной screening signal: рост `b` дает линейный рост nominal capacity, но одновременно быстро ухудшает raw-LM quality и на больших `b` ухудшает text-channel reliability. Особенно `b7/b8` выглядят как capacity-oriented крайние точки, а не как кандидаты на обычный рабочий режим.

## 6. Huffman

| Point | Exact | Hard errors | Mean BPT* | Entropy util. | PPL | KL `Q→P_ref`, bits | TVD |
|---|---:|---:|---:|---:|---:|---:|---:|
| e1 | 5/8 | 3 | 1.000 | 0.301 | 5.15 | 1.531 | 0.509 |
| e2 | 6/8 | 1 | 1.757 | 0.506 | 7.32 | 1.163 | 0.410 |
| e3 | 8/8 | 0 | 2.343 | 0.636 | 9.81 | 0.891 | 0.353 |
| e4 | 8/8 | 0 | 2.758 | 0.767 | 10.25 | 0.631 | 0.270 |
| e5 | 5/8 | 2 | 3.168 | 0.840 | 14.11 | 0.557 | 0.246 |
| e6 | 6/8 | 1 | 3.427 | 0.899 | 14.62 | 0.440 | 0.208 |
| e7 | 4/8 | 1 | 3.837 | 0.952 | 18.98 | 0.410 | 0.193 |
| e8 | 7/8 | 0 | 3.987 | 0.982 | 20.50 | 0.373 | 0.176 |

`*` Для Huffman фактический BPT переменный; таблица показывает среднее по successful runs.

Forward KL `P_ref→Q` равен `+inf` во всех successful Huffman runs.

На Pilot A особенно выделились `e3` и `e4`: оба дали 8/8 exact recovery, при этом уже обеспечивают существенно большую capacity относительно `e1/e2`. Более агрессивные точки улучшают BPT и уменьшают distribution distortion, но надежность на текущей малой выборке становится нерегулярной. Это не является основанием окончательно отсечь их без Pilot B, но является причиной не переносить всю сетку `e1..e8` дальше без отбора.

## 7. Arithmetic Coding

Arithmetic Coding показал 120/120 process completion и 116/120 exact text-channel recovery. Четыре silent failures единичны и не образуют очевидной зависимости только от `precision` или `top_k`:

- `p16-k128`: 1;
- `p32-k50000`: 1;
- `p48-k128`: 1;
- `p48-k8192`: 1.

Ключевые точки сетки:

| Point | Exact | BPT | PPL | KL `Q→P_ref`, bits | TVD |
|---|---:|---:|---:|---:|---:|
| p16-k128 | 7/8 | 3.076 | 9.08 | 8.18e-2 | 5.23e-2 |
| p16-k2048 | 8/8 | 3.377 | 11.25 | 3.79e-2 | 2.62e-2 |
| p16-k50000 | 8/8 | 3.393 | 11.34 | 3.73e-2 | 2.58e-2 |
| p32-k512 | 8/8 | 3.718 | 14.17 | 4.01e-2 | 2.65e-2 |
| p32-k2048 | 8/8 | 4.014 | 19.36 | 1.33e-2 | 9.06e-3 |
| p32-k8192 | 8/8 | 3.886 | 16.08 | 2.33e-3 | 1.61e-3 |
| p32-k50000 | 7/8 | 4.070 | 19.10 | 2.07e-4 | 1.51e-4 |
| p48-k512 | 8/8 | 3.567 | 13.11 | 3.44e-2 | 2.29e-2 |
| p48-k2048 | 8/8 | 4.016 | 17.30 | 1.09e-2 | 7.48e-3 |
| p48-k8192 | 7/8 | 3.904 | 15.56 | 2.26e-3 | 1.56e-3 |
| p48-k50000 | 8/8 | 3.732 | 16.14 | 2.12e-4 | 1.47e-4 |

Оставшиеся `p16-k512`, `p16-k8192`, `p32-k128`, `p48-k128` сохранены в raw log и учитываются при дальнейшем screening.

Forward KL `P_ref→Q` равен `+inf` во всех 120 runs. При увеличении method-local `top_k` reverse KL и TVD ожидаемо уменьшаются; при этом `k=50000` заметно увеличивает wall time относительно меньших pools. Pilot A показывает, что вся матрица 3×5 избыточна для Pilot B и может быть сокращена до нескольких representative combinations после отдельного отбора.

## 8. ADG и Discop

Эти методы не имеют искусственно введенной Pilot-A method-local сетки и представлены одной normalized point каждый.

| Method | Exact | BPT | Entropy util. | PPL | KL `P_ref→Q`, bits | KL `Q→P_ref`, bits | TVD | Mean wall/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADG | 8/8 | 3.104 | 0.763 | 20.72 | 0.0064 | 0.0066 | 0.0202 | 465.1 s |
| Discop | 8/8 | 3.448 | 0.947 | 19.17 | ~0 | ~0 | ~0 | 170.6 s |

Оба метода дали 8/8 exact recovery. Главный Pilot-A сигнал здесь — вычислительная стоимость.

Среднее время successful run:

- ADG: wall 465.1 s; stego algorithm 35.18 s; metric instrumentation 417.88 s;
- Discop: wall 170.6 s; stego algorithm 69.15 s; metric instrumentation 94.99 s.

Это подтверждает необходимость сохранять `metric_instrumentation_total_ms` отдельно от времени самого метода и учитывать instrumentation при Stage-6 wall-clock budget.

## 9. DAIRstega

| Point | Exact | Hard errors | BPT | PPL | KL `Q→P_ref`, bits | TVD |
|---|---:|---:|---:|---:|---:|---:|
| a8-b1-k100 | 8/8 | 0 | 0.728 | 4.71 | 1.468 | 0.504 |
| a8-b075-k100 | 8/8 | 0 | 0.813 | 6.81 | 1.902 | 0.562 |
| a8-b05-k100 | 8/8 | 0 | 1.525 | 13.45 | 2.193 | 0.621 |
| a16-b1-k100 | 8/8 | 0 | 0.916 | 5.52 | 1.488 | 0.509 |
| a16-b075-k100 | 8/8 | 0 | 1.304 | 12.02 | 2.114 | 0.581 |
| a16-b05-k100 | 6/8 | 2 | 2.898 | 117.81 | 3.869 | 0.799 |
| a32-b1-k100 | 8/8 | 0 | 0.823 | 5.65 | 1.581 | 0.524 |
| a32-b075-k100 | 8/8 | 0 | 1.547 | 13.38 | 2.092 | 0.583 |
| a32-b05-k100 | 7/8 | 1 | 3.352 | 115.98 | 3.409 | 0.770 |
| a48-b1-k100 | 8/8 | 0 | 0.873 | 5.50 | 1.502 | 0.516 |
| a48-b075-k100 | 8/8 | 0 | 1.217 | 11.13 | 2.184 | 0.589 |
| a48-b05-k100 | 6/8 | 2 | 3.258 | 119.94 | 3.526 | 0.780 |

Forward KL `P_ref→Q` равен `+inf` во всех successful DAIRstega runs из-за method-local support restriction.

Главный screening signal: `beta=0.5` при `alpha>=16` создает очень высокий PPL (~116–120), высокий distortion и одновременно text-channel hard failures. `a8-b05-k100` остается заметно мягче и дал 8/8 exact. Поэтому экстремальные `beta=0.5, alpha>=16` точки выглядят сильными кандидатами на исключение из основной Pilot-B сетки, хотя могут быть сохранены как документированные capacity extremes.

## 10. Вычислительная стоимость

Средний wall time успешного run:

| Method | Mean wall/run | Mean stego algorithm | Mean metric instrumentation |
|---|---:|---:|---:|
| Bins | 4.70 s | 0.059 s | 0.164 s |
| Huffman | 4.64 s | 0.066 s | 0.194 s |
| Arithmetic Coding | 23.28 s | 11.20 s | 5.82 s |
| ADG | 465.07 s | 35.18 s | 417.88 s |
| Discop | 170.64 s | 69.15 s | 94.99 s |
| DAIRstega | 13.07 s | 8.10 s | 0.193 s |

Эти числа относятся к Pilot A с 128 carrier tokens и не должны напрямую переноситься на Stage 6 без Pilot-B подтверждения на 256 токенах.

## 11. Что Pilot A установил

1. Широкая сетка действительно необходима: у параметризованных методов наблюдаются сильные trade-off между capacity, quality, distribution distortion и reliability.
2. Не все nominally high-capacity points практически интересны: Bins `b7/b8` и DAIR `beta=0.5, alpha>=16` демонстрируют явную цену экстремального режима.
3. Huffman `e3/e4` и значительная часть Arithmetic grid выглядят устойчивыми кандидатами для дальнейшего подтверждения, но окончательная retained set еще не зафиксирована.
4. ADG и Discop не требуют искусственной локальной сетки, но существенно влияют на общий вычислительный бюджет.
5. Text-channel reliability — отдельная полноценная ось benchmark; 42 нарушения exact recovery были диагностированы отдельно.
6. Pilot A выполнил screening-задачу, но его 4 prompts × 2 replicates недостаточно для окончательного ранжирования или оценки стабильности выводов.

## 12. Screening decision после Pilot A

Retained/rejected selection зафиксирован отдельно в `docs/stages/stage5/pilot_b_operating_point_selection.md`. Из 45 Pilot-A points в Pilot B перенесен 21 representative point; 24 точки остаются в raw Pilot-A dataset и отчете, но не повторяются. Machine-readable selection: `results/stage5/pilot_a_operating_point_selection.json`.

Pilot B использует 8 prompts × 3 paired replicates × 256 carrier tokens, всего 504 streaming runs. Его задача — подтвердить variance, stability of trade-offs и wall-clock cost для выбранных областей.

RRC рассматривается отдельно; результаты его numerical calibration описаны в `docs/stages/stage5/rrc_precision_calibration_report.md`.
