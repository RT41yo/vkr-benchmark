# Stage 4 — iteration 13: DAIRstega publication-characteristic probe

## Цель

После успешного normalized smoke, synthetic paper-vs-repository conformance и direct author gate проверить второй слой воспроизводимости: появляется ли на современной benchmark LM характерный для публикации DAIRstega capacity–quality/concealment trade-off.

## Зафиксированная публикационная цель

Table 3 исходной работы показывает, что при `beta=1` payload остаётся около 1.1 Bpw при сравнительно низком PPL, тогда как `beta=0.5` повышает payload, но увеличивает PPL. В частности, в обзоре статьи зафиксированы точки:

- `alpha=8, beta=1`: 1.10 Bpw, PPL 7.686;
- `alpha=32, beta=1`: 1.11 Bpw, PPL 8.184;
- `alpha=8, beta=0.5`: 1.89 Bpw, PPL 13.775;
- `alpha=32, beta=0.5`: 2.56 Bpw, PPL 20.876.

Это не задаёт числовые ожидания для нашего запуска: Bpw статьи и BPT benchmark — разные единицы, а LM/tokenization/prompting отличаются. Воспроизводится направление trade-off.

## Probe

Добавлены:

- `configs/reproducibility/stage4_dairstega_characteristic.json`;
- `scripts/probe_stage4_dairstega_characteristic.py`;
- bootstrap/unit tests для probe logic.

Фиксируются четыре точки при одном method-local `top_k=100`:

1. `alpha=8, beta=1.0` — quality-oriented reference;
2. `alpha=8, beta=0.5` — чистый beta-effect при той же alpha;
3. `alpha=32, beta=1.0` — alpha-effect при beta=1;
4. `alpha=32, beta=0.5` — capacity-oriented point.

Для каждой точки сохраняются BPT, raw-LM NLL/PPL, reverse KL, TVD, число infinite strict-forward-KL steps, Q provenance и full round-trip diagnostics.

## PASS criterion

Publication characteristic считается воспроизведённой, если:

- все четыре normalized точки дают exact text/token and secret round-trip;
- на всех шагах используется `analytic_exact / independent_analytic` Q;
- reverse KL и TVD конечны;
- capacity-oriented point (`alpha=32, beta=0.5`) имеет больший BPT, чем quality-oriented point (`alpha=8, beta=1`);
- одновременно хотя бы одна из осей quality/concealment ухудшается: raw-LM PPL, reverse KL или TVD.

Внутрипарные сравнения `beta=0.5` против `beta=1` при alpha 8 и 32 выводятся как дополнительные diagnostics. Они не подменяют главный pairwise criterion и помогают увидеть, насколько устойчив опубликованный механизм на современной LM.

Strict `D_KL(P_ref || Q)` ожидаемо может быть `+inf` из-за method-local candidate restriction и не используется как единственный критерий concealment для этого probe.
