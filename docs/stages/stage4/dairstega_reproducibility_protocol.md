# DAIRstega — протокол воспроизводимости Stage 4

## Источники

Публикация: Yihao Wang, Ruiqi Song, Lingxiao Li, Ru Zhang, Jianyi Liu, *Dynamically Allocated Interval-Based Generative Linguistic Steganography with Roulette Wheel*, arXiv:2401.15656v5 / Applied Soft Computing 176 (2025), 113101.

Авторская реализация: `WangYH-BUPT/DAIRstega@8d85edf98d48c3efa827a125b6d4e90f88141ea2`, основной PyTorch-файл `DAIRstega_t.py`.

## Что именно воспроизводится из публикации

Для DAIRstega характерным результатом является не одна абсолютная цифра, а управляемый компромисс между payload и concealment/quality при изменении `alpha` и `beta`. Статья показывает, что более сглаженное выделение интервалов при `beta=0.5` повышает embedding rate относительно `beta=1`, но более ёмкие точки могут иметь худшее качество/обнаружимость.

Stage-4 criterion: на одинаковой современной LM и одинаковом benchmark окружении должны появиться по меньшей мере две рабочие точки, демонстрирующие тот же качественный trade-off. Сравниваются BPT, raw-LM NLL/PPL, reverse KL, TVD и roundtrip. Strict forward KL сохраняется как определено benchmark и может быть `+inf` из-за method-local candidate truncation.

## Что проверяется против авторского кода

Direct author conformance выполняется отдельно от publication reproduction. На синтетических `P_reference` проверяются:

1. alpha-dependent probability-floor candidate restriction;
2. internal `top_k`;
3. `linear` / `sqrt` mapping и связь с `beta=1` / `beta=0.5`;
4. округление ширин, overfill truncation и residual-to-first-token;
5. alpha-bit integer selection;
6. prefix extraction, включая singleton interval как явно известное расхождение paper-vs-repository.

Pinned checkout должен остаться неизменным после запуска.

Поскольку allocation-код публичной реализации находится inline внутри nested encoder и не экспортирован как отдельный core API, direct-author gate исполняет оригинальные top-level bit/prefix helpers из pinned source и source-guards ключевые allocation statements. Отдельный synthetic checker исполняет literal transcription этих statements на fixtures. В отчёте эти два уровня не смешиваются: source/helper conformance подтверждает происхождение semantics, а synthetic comparison измеряет их отличие от paper/normalized rule.

## Normalized full-cycle gate

Для каждой проверяемой рабочей точки:

`secret -> alpha-bit look-ahead -> allocated interval -> token IDs -> text -> retokenized token IDs -> reproduced intervals -> recovered prefix bits`.

PASS требует:

- `token_sequence_roundtrip_exact=True` для основного smoke;
- `roundtrip_exact=True`;
- `BER=0`;
- `decoder complete=True`;
- explicit conditional Q доступно на каждом шаге как `analytic_exact / independent_analytic`.

Как и для RRC, если в дальнейшем появится расхождение после detokenize/retokenize, алгоритмическая проверка по exact token IDs и обычный text transport анализируются отдельно.

## Publication characteristic points

Предварительные Stage-4 точки, не являющиеся финальным Stage-5 sweep:

- `alpha=8, beta=1.0, top_k=100` — quality-oriented reference point;
- `alpha=8, beta=0.5, top_k=100` — влияние beta при той же alpha;
- `alpha=32, beta=1.0, top_k=100` — влияние finer roulette precision;
- `alpha=32, beta=0.5, top_k=100` — capacity-oriented point, близкий по смыслу к одной из опубликованных конфигураций.

Повторы и длина carrier для characteristic probe зафиксированы в `configs/reproducibility/stage4_dairstega_characteristic.json`; probe реализован в `scripts/probe_stage4_dairstega_characteristic.py`.

## Выполненный smoke и synthetic conformance

Real-Llama smoke для `alpha=8, beta=1, top_k=100`, 16 carrier tokens: payload 27 бит, BPT `1.6875`, exact token/secret round-trip; strict forward KL `+inf` на 16/16 шагах, max reverse KL `1.7358786`, max TVD `0.6983451`. Secret-source accounting прочитал 35 бит: 8 initial look-ahead + 27 replenishment bits, что подтверждает заданную lifecycle-семантику.

Synthetic conformance на 134 fixtures дал 0 normalized-vs-paper failures. Literal repository allocation расходился в 29 случаях, все при `beta=0.5`; `beta=1` совпал во всех fixtures. Это связано с четырёхзнаковым округлением `sqrt` в pinned PyTorch source. Отдельно подтверждён singleton-prefix mismatch. Число 29/134 является характеристикой diagnostic fixture suite, а не оценкой реальной частоты на Llama.


## Direct author conformance

Direct author gate на pinned checkout завершён со статусом PASS: bit conversion failures `0`, author-vs-literal prefix mismatches `0`, singleton author-vs-paper mismatches `1`, missing source fragments `0`, checkout unchanged `True`. Это подтверждает, что singleton-prefix divergence действительно принадлежит опубликованному source, а не нашей транскрипции.

## Publication-characteristic gate

После direct-author PASS выполняется четыре normalized точки `alpha/beta` из раздела выше на одной benchmark LM. Характерный результат считается воспроизведённым, если capacity-oriented `alpha=32,beta=0.5` даёт больший BPT, чем quality-oriented `alpha=8,beta=1`, и одновременно наблюдается цена хотя бы по одной оси raw-LM PPL / reverse KL / TVD. Абсолютные Table-3 Bpw/PPL не сопоставляются напрямую с benchmark BPT/PPL из-за различий модели, токенизации и prompting.

## Выполненный publication-characteristic probe

Normalized Llama-3.2-3B probe на четырёх 100-token точках завершён со статусом PASS (по одному run на точку, method-local `top_k=100`, common `top_k=None`, `top_p=1.0`):

- `alpha=8, beta=1.0`: BPT `0.2700`, raw-LM PPL `2.2522`, mean `D_KL(Q||P_ref)=0.867062`, mean TVD `0.359716`;
- `alpha=8, beta=0.5`: BPT `1.6100`, raw-LM PPL `20.5365`, mean `D_KL(Q||P_ref)=2.721758`, mean TVD `0.683234`;
- `alpha=32, beta=1.0`: BPT `0.8000`, raw-LM PPL `5.0746`, mean `D_KL(Q||P_ref)=1.515890`, mean TVD `0.516834`;
- `alpha=32, beta=0.5`: BPT `3.3500`, raw-LM PPL `125.3471`, mean `D_KL(Q||P_ref)=3.609336`, mean TVD `0.798266`.

На всех четырёх точках exact token/text transport и secret round-trip прошли, `Q` имело provenance `analytic_exact / independent_analytic`, а strict `D_KL(P_ref||Q)` было `+inf` на всех 100 шагах из-за method-local support truncation.

Характерный результат публикации воспроизведён качественно: уменьшение `beta` с `1.0` до `0.5` повысило BPT при обеих проверенных `alpha`, а capacity-oriented точка `alpha=32,beta=0.5` одновременно имела существенно большие BPT, raw-LM PPL, reverse KL и TVD, чем quality-oriented `alpha=8,beta=1.0`. Абсолютные числа публикации не считаются численно воспроизведёнными, поскольку исходная работа использует другую модель/токенизацию/prompting и Bpw вместо benchmark BPT.
