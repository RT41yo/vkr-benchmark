# Stage 5 — Pilot A: отбор operating points для Pilot B

## 1. Назначение

Этот документ фиксирует screening-решение после Pilot A. Это не финальный выбор конфигураций Stage 6 и не ранжирование методов. Цель — сократить широкую Pilot-A сетку до representative points, которые Pilot B проверит на большей выборке и более длинных carrier texts.

Источник данных: `docs/stages/stage5/pilot_a_analysis.md` и `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/execution.jsonl`. Machine-readable версия решения: `results/stage5/pilot_a_operating_point_selection.json`.

## 2. Принцип отбора

Единый composite score не используется. Для параметризованных методов сохраняются, насколько это возможно:

- quality/security anchor;
- одна или несколько промежуточных trade-off областей;
- capacity-oriented endpoint;
- отдельные точки, необходимые для изоляции эффекта конкретного параметра.

Точки исключаются из Pilot B, если они явно избыточны между соседними retained points или одновременно показывают слабую reliability/quality без нового содержательного режима. Исключение из Pilot B не удаляет Pilot-A результат из ВКР.

## 3. Итог

Pilot A содержит 45 method-local points. В Pilot B переносится **21**, еще **24** остаются документированными Pilot-A screening points, но повторно не запускаются. RRC в этот streaming selection не входит и остается на отдельном fixed-payload/numerical track.

| Метод | Pilot A | Retained | Rejected |
|---|---:|---:|---:|
| Bins | 8 | 4 | 4 |
| Huffman | 8 | 4 | 4 |
| Arithmetic Coding | 15 | 6 | 9 |
| ADG | 1 | 1 | 0 |
| Discop | 1 | 1 | 0 |
| DAIRstega | 12 | 5 | 7 |
| **Итого** | **45** | **21** | **24** |

## 4. Bins

| Point | Решение | Роль / причина |
|---|---|---|
| `b1` | retained | **retain** — quality anchor; Retained as the lowest-capacity/quality anchor; 8/8 exact in Pilot A. |
| `b2` | retained | **retain** — low capacity intermediate; Retained as a low-capacity intermediate point between b1 and more distorted regimes. |
| `b3` | rejected | reject — Rejected from Pilot B as redundant between retained b2 and b4; Pilot-A raw result remains available. |
| `b4` | retained | **retain** — mid capacity intermediate; Retained as a mid-capacity point representing the onset of strong quality degradation. |
| `b5` | rejected | reject — Rejected from Pilot B as redundant between retained b4 and b6. |
| `b6` | retained | **retain** — capacity endpoint; Retained as a documented high-capacity endpoint despite poor PPL; useful to preserve the capacity extreme. |
| `b7` | rejected | reject — Rejected from Pilot B: very poor PPL and only 4/8 exact; b6 already preserves the high-capacity regime. |
| `b8` | rejected | reject — Rejected from Pilot B: extreme PPL and only 3/8 exact; adds little beyond the documented Pilot-A endpoint. |

## 5. Huffman

| Point | Решение | Роль / причина |
|---|---|---|
| `e1` | rejected | reject — Rejected from Pilot B: low capacity and only 5/8 exact; e2 provides a more informative low-capacity point. |
| `e2` | retained | **retain** — quality side; Retained as the low-capacity/quality-side representative. |
| `e3` | rejected | reject — Rejected from Pilot B as close to retained e4; e4 has higher BPT with lower measured distortion. |
| `e4` | retained | **retain** — stable intermediate; Retained as the strongest stable intermediate point in Pilot A (8/8 exact). |
| `e5` | rejected | reject — Rejected from Pilot B: reliability weakened and e4/e6 bracket the same region. |
| `e6` | retained | **retain** — higher capacity intermediate; Retained as a higher-capacity intermediate point. |
| `e7` | rejected | reject — Rejected from Pilot B: only 4/8 exact; e8 is a cleaner capacity endpoint. |
| `e8` | retained | **retain** — capacity endpoint; Retained as the capacity-oriented endpoint; 7/8 exact in Pilot A. |

## 6. Arithmetic Coding

| Point | Решение | Роль / причина |
|---|---|---|
| `p16-k128` | rejected | reject — Rejected from Pilot B: precision effect is probed at k=50000, while top_k effect is probed at precision=32. |
| `p16-k512` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p16-k2048` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p16-k8192` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p16-k50000` | retained | **retain** — precision low anchor; Retained as the low-precision anchor at the largest candidate pool. |
| `p32-k128` | retained | **retain** — top k low anchor; Retained as the low-top_k anchor at the central precision. |
| `p32-k512` | rejected | reject — Rejected from Pilot B as redundant between retained k128 and k2048. |
| `p32-k2048` | retained | **retain** — top k intermediate; Retained as an intermediate top_k point at precision=32. |
| `p32-k8192` | retained | **retain** — top k high intermediate; Retained as a high top_k point before the near-full candidate pool. |
| `p32-k50000` | retained | **retain** — top k full pool endpoint; Retained as the near-full candidate-pool endpoint and central precision point. |
| `p48-k128` | rejected | reject — Rejected from Pilot B: precision effect is isolated at k=50000 rather than repeated over the full matrix. |
| `p48-k512` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p48-k2048` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p48-k8192` | rejected | reject — Rejected from Pilot B as redundant outside the two planned AC cross-sections. |
| `p48-k50000` | retained | **retain** — precision high check; Retained as the high-precision check at the largest candidate pool. |

Для Arithmetic Coding Pilot B намеренно образует два пересекающихся среза: при `precision=32` варьируется `top_k={128,2048,8192,50000}`, а при `top_k=50000` сравнивается `precision={16,32,48}`. Это позволяет отдельно проверить эффект candidate-pool size и plateau вычислительной точности без повторения полной матрицы 3×5.

## 7. ADG

| Point | Решение | Роль / причина |
|---|---|---|
| `default` | retained | **retain** — intrinsic point; Retained: ADG exposes no artificial method-local trade-off parameter in the normalized benchmark. |

## 8. Discop

| Point | Решение | Роль / причина |
|---|---|---|
| `default` | retained | **retain** — intrinsic point; Retained: Discop exposes no artificial method-local trade-off parameter in the normalized benchmark. |

## 9. DAIRstega

| Point | Решение | Роль / причина |
|---|---|---|
| `a8-b1-k100` | retained | **retain** — quality anchor; Retained as the low-capacity/quality anchor. |
| `a8-b075-k100` | rejected | reject — Rejected from Pilot B as redundant with retained neighboring moderate points. |
| `a8-b05-k100` | rejected | reject — Rejected from Pilot B: a32-b075 gives similar BPT with slightly better Pilot-A PPL/distortion; raw result remains available. |
| `a16-b1-k100` | retained | **retain** — alpha progression; Retained to preserve the beta=1 alpha progression. |
| `a16-b075-k100` | retained | **retain** — moderate tradeoff; Retained as a moderate trade-off point. |
| `a16-b05-k100` | rejected | reject — Rejected from Pilot B: high PPL, 2/8 hard failures, and dominated by the chosen a32-b05 capacity extreme on key Pilot-A signals. |
| `a32-b1-k100` | rejected | reject — Rejected from Pilot B as redundant within the beta=1 family. |
| `a32-b075-k100` | retained | **retain** — higher capacity tradeoff; Retained as a higher-capacity moderate-beta trade-off point. |
| `a32-b05-k100` | retained | **retain** — capacity extreme; Retained as the documented capacity extreme among aggressive beta=0.5 points. |
| `a48-b1-k100` | rejected | reject — Rejected from Pilot B as redundant within the beta=1 family. |
| `a48-b075-k100` | rejected | reject — Rejected from Pilot B as redundant with retained a16/a32 beta=0.75 points. |
| `a48-b05-k100` | rejected | reject — Rejected from Pilot B: high PPL, 2/8 hard failures, and no clear advantage over retained a32-b05. |

`a32-b05-k100` сохраняется не как рекомендуемый рабочий режим, а как capacity extreme. Pilot A уже показал высокий PPL и reliability penalty для агрессивного `beta=0.5`; Pilot B проверит, стабилен ли этот крайний режим.

## 10. Pilot B matrix

Executable manifest: `configs/pilots/stage5/pilot_b_streaming_llama.json`.

Фиксируется следующий режим:

```text
model = meta-llama/Llama-3.2-3B
temperature = 1.0
common top_k = null
common top_p = 1.0
prompts = s5p0001..s5p0008
paired replicates = r01, r02, r03
carrier length = 256 tokens
method points = 21
planned runs = 21 × 8 × 3 = 504
```

Pilot B не предназначен для добавления новых method-local grids. Его задача — подтвердить устойчивость retained points: reliability, разброс метрик, относительное положение trade-off points и фактическую вычислительную стоимость на 256 carrier tokens.

## 11. Что остается открытым

- RRC не включается в 504 streaming runs: сначала отдельно исследуется boundary-precision issue, затем выполняется fixed-payload calibration.
- DAIRstega candidate-pool sensitivity, если она будет нужна для Stage 6, должна оформляться отдельным targeted probe и не должна молча расширять Pilot B.
- Pilot B может показать, что некоторые retained points нестабильны; окончательный Stage-6 набор будет фиксироваться только после анализа Pilot B.
