# ADR-0013: NLL/PPL, reliability и временные метрики одиночного запуска

- **Статус:** Accepted
- **Дата:** 2026-09-05
- **Контекст:** Этап 2, шаг 7.4

## Контекст

После реализации capacity/entropy и KL/TVD единый Stage-2 runner должен
рассчитывать оставшиеся базовые метрики одиночного запуска:

- NLL/PPL по исходной генерирующей LM;
- надежность восстановления секрета и BER;
- вычислительную эффективность encode/decode.

`benchmark_specification_v0.1_final.md` задает формулы и основные правила, но
для воспроизводимой реализации требуется дополнительно зафиксировать точные
границы временных измерений и отделить измеряемый алгоритмический путь от
стоимости самой диагностической instrumentation.

## Решение 1. Raw-LM NLL/PPL

Для каждого фактически выбранного sender-side carrier token `x_t` используется
**тот же raw logits tensor**, из которого затем строится `P_reference`, но NLL
вычисляется **до** любых benchmark/stego преобразований:

```text
raw LM logits
    ↓
FP32 logsumexp, temperature = 1
    ↓
-ln P_LM-raw(x_t)
```

Не применяются:

- маска `V_allowed` / special IDs;
- common top-k/top-p;
- method-internal candidate restriction;
- `Q_stego`.

Таким образом, текущая реализация трактует `P_LM-raw` как исходное
распределение по всему output vocabulary модели. Для Qwen это, в частности,
означает, что output-only IDs участвуют в нормировке raw-LM вероятностей, хотя
они запрещены как carrier IDs в `P_reference`.

Run-level:

\[
NLL=-\frac1T\sum_t \ln P_{LM-raw,t}(x_t),
\]

\[
PPL=\exp(NLL).
\]

Поля:

```text
nll_raw_lm_nats_per_token
ppl_raw_lm
```

## Решение 2. Reliability и BER

BER рассчитывается относительно **подтвержденного полезного payload**, а не
всех битов, прочитанных encoder'ом:

```text
expected_bits = payload_secret_bits
```

Для первых `B` ожидаемых позиций:

- несовпавший бит считается ошибкой;
- отсутствующий восстановленный бит считается ошибкой;
- дополнительные восстановленные биты не входят в числитель BER, но
  сохраняются в `recovered_extra_bits` и делают `roundtrip_exact = false`.

\[
BER=\frac{bit\_errors}{B}.
\]

Если `B=0`, BER сохраняется как `unavailable` (`None`), поскольку отношение
математически не определено.

Основные/диагностические поля:

```text
roundtrip_exact
ber
bit_errors
expected_length_bits
recovered_length_bits
length_delta_bits
recovered_extra_bits
first_mismatch_bit
first_decode_failure_token
token_sequence_roundtrip_exact
first_token_roundtrip_mismatch
```

`first_decode_failure_token` для успешных текущих streaming adapters равен
`None`; явное заполнение этого поля для recoverable decoder failures может
быть расширено вместе с общей run-error model.

## Решение 3. Границы временного измерения

Перед измеряемым запуском выполняется отдельный warm-up:

```text
prompt prefill
→ P_reference build
→ один cached LM advance
```

Warm-up не использует secret stream и не входит в timing metrics.

### Encode timing включает

- prompt prefill LM;
- `create_encoder`;
- carrier-side LM path;
- построение канонического `P_reference`;
- `encoder.step`;
- `encoder.finalize`.

### Decode timing включает

- receiver prompt prefill LM;
- `create_decoder`;
- receiver-side LM reconstruction;
- построение `P_reference`;
- `decoder.observe`;
- `decoder.finalize`.

### Не включаются

- загрузка model/tokenizer;
- чтение experiment/prompt config;
- prompt tokenization;
- ordinary-text transport `tokens → text → tokens`;
- расчеты entropy, KL/TVD, NLL/PPL, BER и post-run aggregation;
- сохранение результатов на диск.

Это позволяет не превращать стоимость benchmark instrumentation в стоимость
самого стегометода.

## Решение 4. CUDA synchronization

Если LM adapter использует CUDA, до и после каждого измеряемого участка
выполняется `torch.cuda.synchronize()`. Это предотвращает измерение только
времени постановки асинхронной CUDA-операции в очередь.

## Решение 5. Компоненты и total time

Каждая сторона хранит три непересекающихся measured components:

```text
lm_forward_ms
distribution_processing_ms
stego_algorithm_ms
```

`encode_total_ms` и `decode_total_ms` определяются как сумма соответствующих
компонентов. Поэтому metric instrumentation, выполняемая между компонентами,
не попадает в total time.

Дополнительно рассчитываются:

```text
encode_ms_per_token
decode_ms_per_token
payload_bits_per_second_encode
payload_bits_per_second_decode
```

Для decode `ms/token` используется число реально полученных после текстового
канала token IDs. В успешном token round-trip оно совпадает с числом sender
carrier tokens.

Агрегированные component totals:

```text
lm_forward_total_ms
distribution_processing_total_ms
stego_algorithm_total_ms
```

равны сумме encoder-side и decoder-side значений.

## Последствия

Положительные:

- NLL/PPL не зависят от common или method-specific truncation;
- AC look-ahead не загрязняет BER/throughput полезной нагрузки;
- performance не включает стоимость вычисления benchmark metrics;
- CUDA timing имеет явную синхронизацию;
- Bins/Huffman/AC измеряются одним и тем же timing contract.

Ограничения:

- текущий `encode_total_ms`/`decode_total_ms` — сумма измеренных компонентов, а
  не wall-clock всей CLI-команды;
- text transport измеряется только как reliability path, но не как
  computational-efficiency component;
- в v1.0 следует явно закрепить этот timing scope и трактовку raw-LM output
  support, чтобы исключить неоднозначность между реализациями.
