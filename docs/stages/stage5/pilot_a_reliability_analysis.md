# Stage 5 — Pilot A: отчет по надежности text transport

## 1. Вопрос диагностики

Benchmark проверяет не только внутренний token-ID канал, но и обычный текстовый транспорт:

```text
sender token IDs
    -> tokenizer.decode()
text
    -> tokenizer.encode()
receiver token IDs
    -> stego decoder
```

Pilot A показал два вида нарушения exact recovery:

1. `ok`, но восстановленный секрет не совпадает с исходным;
2. hard failure, когда decoder не может продолжить работу.

Цель post-hoc диагностики — установить, связаны ли эти нарушения с изменением последовательности token IDs после text transport или с внутренней несогласованностью encoder/decoder при одинаковых token IDs.

## 2. Исходные данные

Основной sweep:

`results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/execution.jsonl`

Silent-failure diagnostics:

`results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/reliability_diagnostics.csv`

Hard-failure replay:

- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics.jsonl`;
- `results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/hard_failure_diagnostics_summary.json`.

SHA-256:

```text
execution.jsonl                         e55675aa8d9466477c2571ea80f8679eabeabf3022dee87d09d56a871f80c5b2
reliability_diagnostics.csv             d53beaf8c0c61caddd27d820f3798299a60a04e84220f125e38c9ceadc4c5900
hard_failure_diagnostics.jsonl          ed6f052ce5c2e92ed699741abb8c7486a5433cc2b114cecd3c0cbbf3e5ca3005
hard_failure_diagnostics_summary.json   aa1bb65049322dea9ab5ab3fd011090597e887ac16be7543c72084d078d687c0
```

## 3. Silent reliability failures

В 347 process-successful runs было 29 случаев `roundtrip_exact=false`:

- Bins: 18;
- Huffman: 7;
- Arithmetic Coding: 4.

Во всех 29 случаях:

```text
token_sequence_roundtrip_exact = false
```

Не найдено ни одного случая, в котором receiver token sequence совпала бы с sender sequence, но секрет восстановился неверно.

Для Bins связь особенно прозрачна: поскольку `block_size=b` задает фиксированное число секретных бит на token, первый secret-bit mismatch возникает в позиции, согласующейся с первым token mismatch. Это поддерживает причинную цепочку `token mismatch -> wrong extracted bits -> context divergence`.

Распределение 29 silent failures не сосредоточено на одном prompt или replicate:

```text
s5p0001: 8
s5p0002: 8
s5p0003: 6
s5p0004: 7

r01: 15
r02: 14
```

Это не похоже на единичный поврежденный prompt или один конкретный secret stream.

## 4. Hard failures

Pilot A зафиксировал 13 hard failures:

- Huffman: 8;
- DAIRstega: 5.

Для каждого run diagnostic replay выполнил два контрольных пути:

1. `direct-token control`: decoder получает исходные sender token IDs;
2. `ordinary-text replay`: sender IDs проходят через decode-to-text и повторную tokenization.

Результат:

```text
diagnosed_runs = 13
direct_token_decode_exact_count = 13
transport_mismatch_count = 13
text_transport_induced = 13
all_replayed_errors_match_source = true
```

То есть все 13 исходных hard failures воспроизвелись, но direct-token decode для тех же stegotexts восстановил секрет точно.

Относительно первого transport mismatch hard failure возникал:

```text
на том же token step:       10 runs
через 1 token после него:    2 runs
через 2 tokens после него:   1 run
```

Это согласуется с тем, что первый изменившийся token либо сразу оказывается недопустимым для текущего decoder state, либо сначала изменяет LM context и приводит к невозможному observed token на одном из следующих шагов.

## 5. Интерпретация Huffman hard failures

Сообщение вида:

```text
observed token ... is outside the current Huffman candidate set
```

не означает, что Huffman encoder выбрал token вне своего дерева. Диагностика показывает противоположное: direct-token decode того же sender sequence проходит точно.

После text transport receiver может получить другую token sequence. Тогда decoder строит Huffman tree уже в другом состоянии LM или наблюдает token, которого нет среди текущих candidates, и корректно завершает run ошибкой.

## 6. Интерпретация DAIRstega hard failures

Сообщение вида:

```text
observed token ... has no positive-width DAIRstega interval
```

аналогично возникает после transport mismatch. Sender token имеет допустимый positive-width interval в encoder path, но receiver после retokenization наблюдает другой token и/или уже находится в отличающемся контексте.

Direct-token control 5/5 DAIR hard failures восстанавливает секрет точно.

## 7. Итог по 42 нарушениям reliability

В Pilot A наблюдалось:

```text
29 silent failures
13 hard failures
----------------
42 total non-exact runs
```

По всем 42 случаям обнаружено изменение token sequence после обычного text transport.

Для 13 hard failures дополнительно доказано, что direct-token decode exact, а transported replay воспроизводит исходную ошибку.

Поэтому в рамках наблюдавшегося Pilot-A набора нет свидетельства внутренней encoder/decoder асимметрии при одинаковых token IDs. Наблюдавшиеся нарушения exact recovery относятся к практической устойчивости метода к текстовому транспорту.

Формулировка ограничена именно исследованными runs: она не доказывает отсутствие других классов reliability failures в более широком эксперименте.

## 8. Методологическое следствие

Text-channel failures не удаляются из benchmark и не заменяются передачей исходных token IDs decoder'у. Для практической генеративной лингвистической стеганографии carrier передается как текст, поэтому устойчивость цепочки

```text
token IDs -> text -> token IDs
```

остается самостоятельной осью reliability.

Direct-token path используется как диагностический control, а не как основной канал benchmark.

Новый ADR не требуется: Pilot A не меняет ранее принятую семантику text transport, а эмпирически показывает ее влияние. Результат фиксируется как экспериментальный отчет.
