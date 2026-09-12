# Этап 3 — paper-sentence driver перед полным Figure-3 sweep

**Статус:** Step 3.9 успешно валидирован; 32/32 paper-sentence pilot runs прошли hard gate.
**Цель:** устранить второй blocker, обнаруженный GPT-2 Medium pilot: public `finish_sent=True` не означает «остановиться на первой границе предложения».

## 1. Почему понадобился отдельный driver

Step 3.7 использовал фиксированный 24-битный payload и public author `finish_sent=True`. В pinned Harvard code эта опция работает так:

```text
сначала встроить весь заданный message
        -> затем, если предложение ещё не закончено,
           greedy top-1 продолжение до sentence-finish
```

Поэтому punctuation, встретившаяся **во время embedding**, не останавливает кодер. Фактический audit дал 5/32 runs с sentence-finish token раньше последнего generated token.

Это не соответствует буквальной экспериментальной формулировке paper:

```text
generate one entire sentence given a uniform random message
```

Оригинального Figure-3 batch/MC driver в публичном pinned repository нет. Поэтому точный исторический orchestration восстановить нельзя. Step 3.9 не выдаёт новую логику за неизвестный author driver, а явно фиксирует Stage-3 operationalization, максимально близкую к доступному paper/code evidence.

## 2. Замороженное правило Step 3.9

Config:

```text
configs/reproducibility/paper_sentence_pilot.json
```

Для каждого из восьми frozen CNN/DailyMail pilot contexts создаётся длинный deterministic uniform-looking bitstream:

```text
16384 bits/context
seed = 1234
replicate = 0
один и тот же stream данного context используется всеми методами
```

Далее encoder работает непрерывно и после каждого выбранного carrier token проверяет **публичный author predicate**:

```python
utils.is_sent_finish(token_id, tokenizer)
```

Он считает token sentence-final, если decoded token содержит `.`, `!` или `?`.

Новое правило:

```text
long uniform bitstream
        -> author-compatible token selection
        -> выбран первый token, для которого is_sent_finish == True
        -> этот token включается в carrier sentence
        -> STOP немедленно
```

Greedy completion tail отсутствует.

## 3. Почему используется именно author `is_sent_finish`

Для contexts Stage 3 имеет собственный deterministic sentence splitter, но он решает другую задачу: воспроизводимо получить первые три source sentences из CNN/DailyMail.

Для generated sentence единственный публичный termination predicate самого Harvard reference — `utils.is_sent_finish`. Использовать здесь другой NLP splitter означало бы незаметно заменить author-compatible orchestration нашей новой эвристикой.

Ограничение фиксируется явно: predicate очень простой и может считать точку внутри некоторых сокращений окончанием предложения. Если это проявится в pilot, это будет зафиксировано как limitation public executable evidence, а не скрыто постобработкой.

## 4. Почему нужен длинный secret stream

Step 3.8 показал, что короткий finite payload особенно опасен для Arithmetic. При `precision=26` кодер читает 26-битное look-ahead окно; если настоящих битов не хватает, public function дописывает нули. В Step 3.7 payload был всего 24 бита, поэтому каждый Arithmetic coding step уже был padding-affected.

Paper-sentence driver запрещает это поведение в измеряемой части:

```text
Arithmetic step разрешён только если доступны все precision реальных secret bits.
```

`16384` бит — не paper parameter и не payload одного предложения. Это просто достаточно длинный implementation buffer. Реальный payload определяется **после остановки**: сколько бит author algorithm подтвердил как встроенные к моменту первого sentence boundary.

Если stream неожиданно закончится раньше boundary, run считается ошибкой. Нули автоматически не дописываются.

## 5. Payload accounting

Для Bins и Huffman:

```text
secret_bits_read == payload_bits_confirmed
```

Для Arithmetic эти величины различаются:

```text
secret_bits_read
    = максимум secret look-ahead, который понадобился encoder;

payload_bits_confirmed
    = только биты, уже зафиксированные arithmetic interval renormalization.
```

Paper-compatible capacity Step 3.9:

```text
bits_per_word_author = payload_bits_confirmed / carrier_tokens
```

Название `word` сохраняется ради совместимости с author/paper terminology, хотя executable code фактически работает с GPT-2 tokens.

## 6. Decoder / reliability check

После sender generation:

```text
token IDs -> tokenizer.decode -> обычный текст -> author decoder
```

Hard gate требует, чтобы recovered bitstream начинался с **всех подтверждённых payload bits**.

Для Arithmetic decoder может вернуть дополнительные tail bits из author final-flush semantics. Они не прибавляются к sender-confirmed payload и сохраняются только как diagnostic.

Exact sender-token / retokenized-token equality остаётся diagnostic: Bins/Huffman/Arithmetic reference содержит BPE repair именно потому, что text transport иногда меняет tokenization.

## 7. Repository-owned mirrors и parity

Public encode helpers не позволяют остановить embedding на первой boundary: их `finish_sent` действует только после исчерпания message. Поэтому Step 3.9 содержит repository-owned mirrors трёх encode loops в:

```text
scripts/stage3_paper_sentence_core.py
```

Они **не входят в normalized benchmark** и не заменяют author code.

Перед тем как доверять новой stopping точке, runner обязан запустить mirrors в обычном fixed-message режиме (`stop_at_first_sentence=false`) и сравнить с pinned functions:

```text
block_baseline.encode_block
huffman_baseline.encode_huffman
arithmetic.encode_arithmetic  (tau=0.9,k=300)
arithmetic.encode_arithmetic  (tau=1.0,k=50256)
```

Для всех четырёх sentinel points требуются exact generated token IDs и совпадение NLL/KL/words-per-bit в заданной численной погрешности. Только после этого разрешается first-boundary pilot.

External `NeuralSteganography` checkout остаётся неизменным.

## 8. Pilot grid

Step 3.9 повторяет четыре representative points Step 3.7, чтобы изменение результата объяснялось orchestration, а не новой сеткой параметров:

```text
Bins:       b=3
Huffman:    exponent=3
Arithmetic: tau=0.9, k=300,   precision=26
Arithmetic: tau=1.0, k=50256, precision=26

8 contexts x 4 points = 32 runs
```

Это ещё не full Figure-3 curve.

## 9. Safety cap

`max_generated_tokens = 256` — только защита от runaway generation. Это не способ обрезать предложение.

Если первый boundary не появился за 256 tokens:

```text
run = failed
pilot gate = NOT READY
```

Параметр full sweep в таком случае не подменяется молча — причина сначала разбирается.

## 10. Success gate

Команды:

```bash
python scripts/run_stage3_paper_sentence_pilot.py
python scripts/check_stage3_paper_sentence_pilot.py
```

Pilot считается успешным только если одновременно:

```text
4/4 fixed-message mirror parity checks PASS
32/32 runs complete
каждый run останавливается именно на первой boundary
ни один run не исчерпал secret stream
ни один run не достиг safety cap
payload_bits_confirmed > 0 для каждого run
все confirmed payload prefixes восстановлены author decoder
Arithmetic не использовал implicit zero look-ahead
reference checkout unchanged
```

После этого checker выдаёт:

```text
READY FOR FULL FIGURE-3 RUNNER IMPLEMENTATION
```

Это означает, что sentence orchestration больше не является pre-sweep blocker. Следующим шагом можно реализовать массовый runner для frozen 23-point matrix × 80 contexts × 3 replicates.

## 11. Фактический результат Step 3.9

Локальный GPT-2 Medium pilot успешно прошёл hard gate:

```text
32/32 runs status=ok
4/4 fixed-message mirror parity checks PASS
все runs остановились на первой boundary
Arithmetic implicit zero look-ahead отсутствует
все confirmed payload prefixes восстановлены
reference checkout unchanged
```

Representative averages:

```text
Bins b=3:                    BPW=3.0000   KL=2.698350 bits
Huffman exponent=3:          BPW=2.4728   KL=1.064161 bits
Arithmetic tau=0.9,k=300:    BPW=2.9338   KL=0.103202 bits
Arithmetic tau=1,k=50256:    BPW=4.6773   KL=0.000605 bits
```

Таким образом, sentence orchestration больше не является pre-sweep blocker. Следующий шаг — Step 3.10 full Figure-3 runner на frozen 23-point matrix × 80 contexts × 3 replicates.
