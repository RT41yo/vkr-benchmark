# Stage 5 — Iteration 2: freeze common regime и подготовка Pilot A

## Что зафиксировано

По фактическому calibration:

- common `top_p=1.0`;
- common `top_k=null`;
- `temperature=1.0`;
- Pilot A использует 128 carrier tokens, а не сокращенный 64-token вариант;
- 4 frozen pilot prompts;
- 2 paired replicates.

Основание выбора `top_p` описано в ADR-0027 и `top_p_calibration_report.md`.

## Streaming Pilot A

Executable manifest:

`configs/pilots/stage5/pilot_a_streaming_llama.json`

Матрица:

| Method | Grid | Points |
|---|---|---:|
| Bins | `block_size=1..8` | 8 |
| Huffman | `bits_per_word=1..8` | 8 |
| Arithmetic | `precision={16,32,48}` × method-local `top_k={128,512,2048,8192,50000}` | 15 |
| ADG | normalized default | 1 |
| Discop | normalized default | 1 |
| DAIRstega | `alpha={8,16,32,48}` × `beta={1.0,0.75,0.5}`, `top_k=100` | 12 |

Итого 45 method points × 4 prompts × 2 replicates = **360 runs**.

DAIRstega candidate-pool sensitivity (`top_k`) не расширяется на все 12 points. После первичного screening она будет проверена только на нескольких retained representative points, чтобы не превращать Pilot A в Stage 6.

## RRC calibration

RRC не включен в streaming manifest, поскольку его normalized lifecycle — fixed payload.

Первый RRC calibration:

`configs/pilots/stage5/rrc_precision_margin_calibration_llama.json`

Проверяются:

```text
precision_margin = {20, 40, 60}
verify_termination = true
payload = 1024 bits
max_carrier_tokens = 2048
4 prompts × 2 paired replicates
```

Итого **24 runs**.

1024 bits выбран как самый большой из предварительно рассматриваемых payload candidates `{128,256,512,1024}`: margin сначала проверяется в наиболее требовательном из планируемых режимов. После анализа этой calibration выбирается один margin; только затем выполняется payload calibration для выбора сопоставимого Stage-6 RRC regime.

## Почему Pilot A оставлен на 128 tokens

8-token calibration показал, что полный streaming Pilot A при линейной грубой экстраполяции имеет порядок нескольких часов, а не суток. 128 tokens дают более устойчивые trajectory-level оценки и большую вероятность обнаружить редкие failure states, оставаясь приемлемыми по вычислительному бюджету.

64-token вариант поэтому отвергнут как ненужная экономия ценой более шумного screening.

## Что намеренно еще не зафиксировано

- retained points после Pilot A;
- DAIRstega representative `top_k` sensitivity points;
- итоговый RRC payload;
- Pilot-B retained grid;
- окончательное число Stage-6 prompts/replicates;
- benchmark specification v1.0.
