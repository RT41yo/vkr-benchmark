# Step 3.13 — conformance/discrepancy analysis

## Цель

Step 3.13 отвечает на вопрос, ради которого выполнялся matched experiment Step 3.12:
**сохранила ли normalized adaptation принципиальное поведение Bins, Huffman и Arithmetic Coding, и какие расхождения являются ожидаемыми следствиями нормализации?**

Анализ использует только frozen outputs Step 3.12. Языковая модель повторно не запускается. Никакой post-hoc численный tolerance не вводится: exact token parity не является обязательным признаком conformance, потому что normalized benchmark намеренно меняет общую policy, RNG ownership, masking, numerical pipeline и transport boundaries.

Входы:

```text
32 matched pairs = 4 representative points × 8 frozen contexts
same GPT-2 Medium revision
same context hashes
same secret-stream hashes
equal carrier length within each pair
normalized token-ID decode exact in 32/32
```

## Общий вывод

На всех четырёх representative points **ядро метода сохраняется**. Нормализованные adapters не демонстрируют признаков того, что алгоритмический принцип Bins, Huffman или Arithmetic Coding был заменён другим методом. При этом exact token sequence может существенно меняться, и характер расхождений соответствует уже документированным решениям нормализации.

Итоговая классификация:

```text
Bins b=3
  -> core_principle_preserved_expected_partition_divergence

Huffman e=3
  -> strong_conformance_with_localized_candidate_tree_divergence

Arithmetic tau=0.9, k=300, precision=26
  -> core_principle_preserved_with_expected_reference_and_numeric_divergence

Arithmetic tau=1.0, k=50256, precision=26
  -> strong_distributional_conformance_with_sequence_sensitivity
```

Overall:

```text
normalized_adaptations_preserve_core_method_behavior_with_documented_normalization_divergences
```

## Bins b=3

Фиксированная полезная нагрузка сохранена **точно во всех 8/8 парах**:

```text
author mean     = 3.000 bits/token
normalized mean = 3.000 bits/token
mean delta      = 0.000
max |pair delta|= 0.000
```

При этом exact token sequence не совпала ни в одной паре; mean token agreement лишь `0.0032`. Это ожидаемый результат, а не regression. Bins определяется схемой «`b` secret bits -> один bin -> лучший токен внутри выбранного bin». Normalized adapter сохраняет эту схему, но partition identity намеренно другая:

- author разбивает полный GPT-2 vocabulary после global `np.random.seed(block_size)` / NumPy shuffle;
- normalized adapter разбивает `V_allowed` и использует benchmark-owned `MethodRandomSource`;
- representatives выбираются относительно canonical `P_reference`.

Поэтому при одинаковых битах выбираются другие bins/members и почти сразу другие carrier tokens, тогда как capacity-инвариант `b=3` сохраняется абсолютно.

Direction-matched reverse KL также не обязан совпадать численно, поскольку normalized reference policy отличается. Средние значения:

```text
author D_KL(Q||P)     = 2.69835 bits
normalized D_KL(Q||P) = 2.90112 bits
mean delta            = +0.20277 bits
```

Это расхождение не меняет вывод о сохранении Bins core mechanism.

## Huffman e=3

Huffman показывает наиболее сильную локальную conformance:

```text
exact token sequences = 6/8 pairs
mean token agreement  = 0.7593
```

Для всех шести exact-sequence pairs payload/capacity совпадает точно, а максимальное абсолютное расхождение reverse KL составляет менее `0.002 bits`.

Две оставшиеся пары расходятся уже на первом token. Это локализует источник не в накопленной decoder ошибке, а в **initial candidate/tree construction**. Документированные различия, способные это вызвать:

- normalized candidate set берётся из canonical `P_reference`;
- author отдельно маскирует GPT-2 token id `628`, normalized common policy этого author-only mask не добавляет;
- normalized Huffman явно разрешает exact tree-weight ties через `(weight, min_token_id_in_subtree)`, тогда как author heap не задаёт отдельного tie-break contract.

Step-3.12 artifacts не сохраняют полный per-step candidate list, поэтому для двух divergent contexts нельзя доказать, какой именно из этих факторов был непосредственной причиной первого другого листа. Это фиксируется как **локализованная, но не уникально атрибутированная** discrepancy.

Сводные средние:

```text
author capacity        = 2.47277 bits/token
normalized capacity    = 2.51381 bits/token
mean delta             = +0.04103 bits/token

author reverse KL      = 1.06416 bits
normalized reverse KL  = 1.10439 bits
mean delta             = +0.04023 bits
```

## Arithmetic tau=0.9, k=300

Arithmetic Coding сохраняет finite-precision interval mechanism, exact token-ID decode и близкий operating region по capacity, но token parity низкая:

```text
exact token sequences = 0/8
mean token agreement  = 0.1516

author capacity        = 2.93381 bits/token
normalized capacity    = 3.05800 bits/token
mean delta             = +0.12419 bits/token
```

Это соответствует чувствительности Arithmetic Coding: небольшое изменение probability boundary может выбрать другой integer subinterval; после этого internal interval state и последующая trajectory расходятся, даже если общий статистический режим остаётся близким.

Здесь есть несколько документированных источников различий:

1. author Arithmetic сортирует logits, переводит их в FP64 и затем считает temperature softmax;
2. normalized pipeline сначала строит canonical FP32 `P_reference`, а Arithmetic adapter уже копирует эти вероятности в FP64 для integer-width construction;
3. author маскирует token `628`, normalized canonical special-token policy — нет;
4. temperature в normalized mode является общей generation policy, а не внутренним parameter method adapter.

Особенно важно: reverse KL в этой точке имеет одинаковое **направление**, но не одинаковый exact reference.

```text
author KL:     D_KL(Q_stego || P_LM untempered)
normalized KL: D_KL(Q_stego || P_reference at temperature=0.9)
```

Поэтому наблюдаемое снижение среднего reverse KL с `0.10320` до `0.04782 bits` нельзя трактовать как чистую ошибку/улучшение Arithmetic implementation. Это partly reference-policy effect по определению matched protocol.

## Arithmetic tau=1.0, k=50256

Special near-unmodified point даёт наиболее сильное **distributional** соответствие Arithmetic:

```text
author capacity        = 4.67733 bits/token
normalized capacity    = 4.59381 bits/token
mean delta             = -0.08351 bits/token

author reverse KL      = 0.000605481 bits
normalized reverse KL  = 0.000541695 bits
mean delta             = -0.000063786 bits
normalized mean TVD    = 0.000479630
```

Exact token sequences при этом `0/8`, mean token agreement `0.0797`. Это не противоречие: near-zero aggregate distribution distortion может сосуществовать с низкой token-level parity. При finite-precision Arithmetic Coding очень малые различия probability mass / integer rounding меняют секретную точку относительно cumulative boundaries и переводят trajectory в другой state.

В этой точке `tau=1`, поэтому проблема untempered-vs-tempered reference из предыдущего пункта исчезает. Оставшиеся documented differences — прежде всего FP32 canonical probability path против author FP64 softmax и различие token-628 masking/support policy.

Этот matched result поддерживает вывод Step 3.11: normalized adaptation сохраняет near-unmodified **характер** Arithmetic regime. Он не разрешает отдельное paper discrepancy `4e-8 nats`; эта проблема уже локализована finite precision / undocumented historical orchestration и не должна смешиваться с author-vs-normalized conformance.

## Два направления KL: вывод для benchmark

Во всех 32/32 normalized matched runs:

```text
D_KL(P_reference || Q_stego) = +inf
```

Причина структурная: Bins, Huffman и Arithmetic формируют sparse `Q_stego`, то есть назначают нулевую вероятность части токенов, которым `P_reference` оставляет положительную массу.

Это **не ошибка метрики** и не повод добавлять epsilon smoothing задним числом. Benchmark-native KL сохраняет полезный строгий смысл: он сигнализирует потерю support. Но на этих методах он насыщается значением `+inf` и поэтому сам по себе не способен ранжировать степень distortion между runs.

Рекомендация для specification v1.0:

```text
kl_ref_to_stego + infinite_steps
    -> оставить как строгий support-mismatch diagnostic, без smoothing

kl_stego_to_ref
    -> хранить как конечную complementary distortion metric и author-comparable direction

TVD
    -> хранить как конечную symmetric-support-insensitive companion metric
```

То есть ADR-0012 подтверждается экспериментально: оба KL-направления несут разную информацию и не должны смешиваться.

## Что Step 3.13 доказывает и чего не доказывает

Поддерживается данными:

- core embedding/decoding principle сохранён у всех трёх normalized adapters на representative matched points;
- Bins sequence divergence объясняется intentional partition normalization;
- Huffman в 6/8 contexts совпадает token-for-token, а оставшиеся расхождения локализованы к candidate/tree construction;
- Arithmetic aggregate behavior сохраняется при сильной token-level sensitivity;
- benchmark-native forward KL структурно бесконечен для всех 32 matched runs и должен читаться как support diagnostic.

Не утверждается:

- что четыре representative points исчерпывают все возможные configurations;
- что exact token parity должна быть целью normalized benchmark;
- что две Huffman discrepancies уникально вызваны именно token `628` без per-step candidate traces;
- что tau=0.9 reverse KL author и normalized численно сравнивает одну и ту же reference distribution;
- что Step 3.13 меняет уже frozen benchmark specification v0.1.

## Reproducibility

Детерминированный анализ:

```bash
python scripts/analyze_stage3_conformance.py
python scripts/check_stage3_conformance.py
```

Создаёт:

```text
results/stage3/matched_author_normalized/
├── conformance_analysis.json
└── conformance_table.csv
```

После успешного checker Step 3.13 готов к Step 3.14 — финальному reproducibility report / closeout Stage 3.
