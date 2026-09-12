# Этап 3 — GPT-2 Medium author-compatible pilot

**Статус до запуска:** конфигурация pilot заморожена; фактические результаты должны быть получены локально на 8 уже зафиксированных CNN/DailyMail contexts.
**Модель:** `gpt2-medium` (GPT-2 345M).
**Reference:** `harvardnlp/NeuralSteganography@14e982564aeaf9a33f7b4de440deda2184d17f12`.

## 1. Цель pilot

Шаги 3.2–3.4 подтвердили исполняемость трёх author-compatible методов на GPT-2 Small. Шаги 3.5–3.6 зафиксировали paper-level parameter matrix и воспроизводимый CNN/DailyMail context set. Следующий риск — уже не алгоритмический smoke, а переход к исторической модели статьи GPT-2 345M и многоконтекстному one-sentence режиму.

Pilot не является Figure-3 reproduction. Он отвечает на технические вопросы:

- загружается ли `gpt2-medium` в текущем окружении/GPU;
- работают ли Bins, Huffman и Arithmetic на всех 8 frozen contexts;
- сохраняется ли exact payload prefix после обычного text transport;
- корректно ли работает public-reference `finish_sent=True`;
- собираются ли author KL/BPW в единую машиночитаемую схему;
- остаётся ли внешний reference checkout неизменным.

## 2. Почему pilot использует 24 бита

Pinned Harvard repository не содержит оригинального Figure-3 batch/Monte-Carlo driver и не сообщает exact random-message length этого driver. Поэтому pilot не должен выдавать выбранную нами длину сообщения за авторскую.

До первого GPT-2 Medium запуска фиксируется небольшой технический budget:

```text
payload useful bits = 24
contexts            = 8
replicates           = 1
finish_sent          = true
```

Для каждого context создаётся отдельный детерминированный 24-битный поток `pilot_sha256_bitstream_v1`; один и тот же payload данного context используется всеми четырьмя pilot points. Это обеспечивает paired comparison, не используя Python/NumPy RNG implementation details.

24 бита нужны только для gate. Они **не фиксируют** длину payload для full Figure-3 sweep.

## 3. Четыре representative points

Полный frozen sweep содержит 23 точки, но pilot заранее ограничивается четырьмя:

```text
Bins:       block_size_bits = 3
Huffman:    candidate_pool_exponent = 3
Arithmetic: temperature = 0.9, topk = 300, precision = 26
Arithmetic: temperature = 1.0, topk = 50256, precision = 26
```

Последняя Arithmetic-точка нужна как ранняя sanity-проверка near-unmodified режима, но численное совпадение с paper `~4e-8` не является hard gate для короткого 8-context pilot.

## 4. Что означает `finish_sent=True` в pinned code

Public `run_single.py` прямо отмечает, что при `finish_sent=True` reference **сначала заканчивает embedding заданного message**, а затем добавляет greedy top-1 tokens до sentence-finish; выводимые statistics относятся к не-дополненной payload-carrying части.

Поэтому pilot хранит две разные скорости:

```text
bits_per_word_author_stats_prefix
    = 1 / reference words_per_bit
    = author statistics только по embedding prefix

useful_payload_bits_per_total_generated_token
    = 24 / все generated tokens после sentence completion
    = pilot diagnostic, не Figure-3 author metric
```

Смешивать эти величины нельзя.

Public `finish_sent` также не запрещает punctuation во время embedding prefix. Поэтому runner отдельно отмечает `has_early_sentence_finish`. Это diagnostic, а не hard failure: hard requirement — последний generated token должен удовлетворять author `is_sent_finish`.

## 5. Compatibility path

Для Bins/Huffman сохраняется уже проверенный Stage-3 compatibility bridge:

```text
slow GPT-2 tokenizer
+ LegacyCausalLMAdapter
+ byte-for-byte unchanged block_baseline.py / huffman_baseline.py
```

GPT-2 Medium загружается один раз и используется последовательно для 8 Bins + 8 Huffman runs.

Затем модель освобождается из памяти. Arithmetic загружает GPT-2 Medium заново через raw `utils.get_model` и использует pinned native `DynamicCache` path без legacy adapter. На той же модели последовательно выполняются две Arithmetic points × 8 contexts.

## 6. Результаты pilot

Runner пишет:

```text
results/stage3/paper_reproduction/gpt2_medium_pilot/result.json
results/stage3/paper_reproduction/gpt2_medium_pilot/stegotexts.jsonl
```

`result.json` содержит 32 run records, summary по четырём points, environment/model/reference provenance, exact payload-prefix checks и sentence-completion diagnostics.

Source CNN/DailyMail contexts в result не копируются: сохраняются только `selection_rank`, source identifiers/hashes и generated stegotext.

## 7. Go/no-go

Запуск:

```bash
python scripts/run_stage3_gpt2_medium_pilot.py
python scripts/check_stage3_gpt2_medium_pilot.py
```

Hard gate:

```text
32 / 32 calls complete
exact 24-bit payload prefix recovered in every run
final generated token is sentence-finish in every run
reference worktree unchanged
```

BPE token roundtrip и наличие раннего punctuation сохраняются как diagnostics. Если они возникают, результат не скрывается и разбирается перед full sweep.

Только после успешного pilot фиксируется следующий шаг: full paper-level runner/aggregation policy для frozen 23-point matrix. Pilot metrics сами по себе не включаются в итоговую Figure-3 reproduction curve.

## 8. Фактический результат Step 3.7

Локальный запуск на RTX 5070 Ti завершился успешно:

```text
32/32 records status=ok
exact payload-prefix recovery: 32/32
final generated token sentence-finish: 32/32
reference worktree unchanged: true
```

При этом pilot выполнил свою диагностическую функцию и обнаружил два pre-sweep discrepancy:

```text
Arithmetic tau=1, k=50256, precision=26:
mean author KL = 0.459774989 bits/token

Early sentence-finish before final generated token:
5/32 runs
```

Поэтому `Stage 3 GPT-2 Medium pilot gate: READY` трактуется как **technical execution READY**, а не как разрешение full Figure-3 sweep. Следующий frozen diagnostic описан в `docs/stages/stage3/arithmetic_precision_investigation.md`.
