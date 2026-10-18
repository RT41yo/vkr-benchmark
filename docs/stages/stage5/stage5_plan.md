# Stage 5 — пилотные свипы и окончательная фиксация экспериментального дизайна

## 1. Цель

Stage 5 не является уменьшенной копией Stage 6. Его задача — дешево и воспроизводимо принять решения, необходимые для основного эксперимента:

1. выбрать общий generation regime;
2. определить содержательные диапазоны method-local параметров;
3. исключить технически некорректные и явно неинтересные точки;
4. откалибровать RRC fixed-payload regime;
5. оценить prompt/replicate variance;
6. измерить wall-clock стоимость, отдельно от времени самого стегометода;
7. зафиксировать Stage-6 scale и benchmark specification v1.0.

Методологическое разделение external conditions / method-local points / calibration закреплено ADR-0026.

## 2. Последовательность Stage 5

### 2.1. Iteration 1 — sweep foundation

- повторно использовать восемь frozen CNN/DailyMail pilot contexts Stage 3;
- сформировать Stage-5 prompt registry с новыми стабильными `prompt_id`, сохранив exact text и hashes;
- ввести machine-readable sweep manifest;
- реализовать dry-run planner, заранее вычисляющий полный набор configs и canonical `run_id` без загрузки LM;
- реализовать batch runner, загружающий одну LM один раз на sweep;
- зафиксировать paired replicate/randomness protocol;
- подготовить первый common-`top_p` calibration manifest.

### 2.2. Common generation calibration

Первый технический sweep проверяет:

```text
top_p = 0.95, 0.98, 1.00
```

для всех семи методов при одной representative method point каждого метода. Это не финальное сравнение качества методов. Его цель — измерить влияние support size на емкость, надежность и особенно wall-clock/metric instrumentation cost.

Первый calibration intentionally short:

- prompts: `s5p0001`, `s5p0002`;
- replicates: 1;
- streaming methods: 8 carrier tokens;
- RRC: 32 payload bits, max 128 carrier tokens.

После calibration выбирается один основной common `top_p` для Pilot A/B и Stage 6. Решение принимается по данным, а не зашивается заранее.

## 3. Pilot A — screening

После выбора common generation regime выполняется широкая method-local сетка на небольшой выборке:

- 4 prompt;
- 2 paired replicates;
- ориентир 128 carrier tokens для streaming methods;
- RRC запускается в откалиброванном fixed-payload режиме.

Предварительная сетка:

| Метод | Pilot-A grid |
|---|---|
| Bins | `block_size = 1..8` |
| Huffman | `bits_per_word = 1..8` |
| Arithmetic | `precision={16,32,48}` × `top_k={128,512,2048,8192,50000}` |
| ADG | one normalized point |
| Discop | one normalized point |
| RRC | calibrated normalized point |
| DAIRstega | `alpha={8,16,32,48}` × `beta={1.0,0.75,0.5}`, initially `top_k=100`; candidate-pool sensitivity is added only to retained representative points |

Pilot A отвечает на вопрос: какие области вообще содержательны?

## 4. Pilot B — confirmation

После отсева Pilot A оставляет примерно 4–6 representative points для параметризованных методов. Они проверяются тщательнее:

- 8 frozen pilot prompts;
- 3 paired replicates;
- кандидат финальной длины: 256 carrier tokens;
- RRC fixed-payload length выбирается так, чтобы carrier length была сопоставима по порядку.

Pilot B оценивает prompt/secret variance, стабильность rank/order и фактическую стоимость запуска перед окончательной фиксацией Stage 6.

## 5. Критерии отбора

Каждая точка сохраняет полный benchmark vector; единой агрегированной оценки «лучший метод» нет.

Обязательные оси:

- reliability: decoder completion, `roundtrip_exact`, BER, token-sequence round-trip;
- capacity: BPT, entropy utilization;
- distribution distortion: оба направления KL по ADR-0012 + TVD;
- quality: raw-LM NLL/PPL;
- performance: encode/decode ms/token, payload bit/s, stego algorithm time;
- instrumentation: `metric_instrumentation_total_ms` отдельно от stego time;
- variability: разброс по prompt × replicate.

Точка может быть исключена как технически некорректная или стабильно доминируемая соседней точкой того же метода. При этом для параметризованного метода желательно сохранить security/quality anchor, промежуточную область и capacity-oriented endpoint, если они содержательно различаются.

## 6. Вычислительный бюджет Stage 6

Для каждого retained operating point Pilot B дает как минимум median и p90 wall-clock/run. Оцениваются два разных бюджета:

```text
algorithmic cost = LM + P_ref processing + stego algorithm
wall-clock cost  = algorithmic cost + metric instrumentation + storage/I/O
```

Stage-6 estimate вычисляется как сумма фактической стоимости по всем method points × prompts × replicates с дополнительным резервом на warm-up, I/O и reruns.

Qwen используется для ключевых сравнений, а не обязательно для полного Llama grid. Для Qwen сохраняется уже зафиксированная общая политика: 267 output-only IDs исключаются до `P_reference`, затем распределение перенормируется.

## 7. Выходные артефакты Stage 5

Минимальный closeout:

- dataset pilot runs;
- analytic pilot report;
- Stage-6 budget report;
- retained/rejected operating-point table с причинами;
- benchmark specification v1.0;
- Stage-5 Typst presentation;
- machine-readable sweep manifests и execution logs;
- `stage5_closeout.md`.
