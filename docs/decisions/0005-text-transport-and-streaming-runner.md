# ADR-0005: Текстовый канал и минимальный streaming runner

## Статус

Принято для этапа 2.

## Контекст

После реализации `LMAdapter`, канонического `P_reference` и normalized Bins необходимо проверить первый полный путь встраивания и извлечения секрета на реальной языковой модели.

Спецификация v0.1 требует оценивать надежность не по внутренним token ID кодера, а через обычный текстовый канал:

```text
secret bits
→ stego encode
→ generated token IDs
→ tokenizer.decode
→ передается только text
→ tokenizer.encode(text, add_special_tokens=False)
→ stego decode
→ recovered bits
```

Передача token ID напрямую в decoder запрещена для нормативной проверки надежности. Также encoder и decoder должны независимо воспроизводить один и тот же LM/KV-cache путь на фактически наблюдаемом ими префиксе.

## Решение

### 1. `TextChannel`

Создан отдельный компонент `transport/TextChannel`.

Он выполняет только:

1. `token_ids → text` через `LMAdapter.decode_tokens()`;
2. `text → receiver_token_ids` через `LMAdapter.encode_text(..., add_special_tokens=False)`;
3. сохраняет диагностику `token_sequence_roundtrip_exact`, `first_token_mismatch` и изменение длины токеновой последовательности.

`TextChannel` не исправляет BPE-расхождения и не содержит method-specific эвристик. Если после обычного текста токенизация изменилась, это является наблюдаемым свойством канала/метода и должно отражаться в reliability-результате.

### 2. Расширение `LMAdapter`

Добавлен общий метод:

```text
encode_text(text, add_special_tokens=...)
```

Он нужен потому, что политика токенизации prompt и политика retokenization переданного stegotext различаются:

- prompt использует явно зафиксированную model-specific policy (`prompt_add_special_tokens`);
- полученный из канала stegotext всегда токенизируется с `add_special_tokens=False`.

Tokenizer по-прежнему принадлежит LM adapter и не передается стегометоду.

### 3. Minimal streaming runner

В `runner/streaming.py` добавлен минимальный orchestration layer для потоковых методов этапа 2.

Encode path:

```text
prompt
→ LM prefill
→ raw logits_t
→ P_reference_t
→ method.step(...)
→ stego token_t
→ cached LM advance
→ ...
```

Decode path строится заново с того же prompt и использует уже receiver-side token IDs:

```text
prompt
→ LM prefill
→ P_reference_t на receiver prefix
→ decoder.observe(observed_token_t)
→ cached LM advance observed_token_t
→ ...
```

Runner не передает encoder KV-cache decoder'у.

Для N carrier tokens выполняется N-1 cached LM forward после prefill: logits первого carrier token уже получены самим prefill. Это соответствует вычислительному пути, проверенному на этапе 1.

### 4. Payload accounting

Добавлен прозрачный `RecordingSecretSource`, который сохраняет именно те секретные биты, которые реально запросил encoder session. После `finalize()` runner проверяет согласованность:

```text
EncoderFinalization.payload_bits == реально прочитанные SecretSource bits
```

Это позволяет проверять reliability без знания внутренней логики конкретного streaming-метода.

### 5. Extra recovered bits

Если после retokenization decoder получил и восстановил больше битов, чем было реально встроено, дополнительные биты сохраняются как `recovered_extra_bits`, а `roundtrip_exact` считается ложным даже если полезный префикс восстановился правильно.

Это соответствует правилу спецификации v0.1 о недопустимости молчаливого игнорирования лишнего декодированного payload.

## Ограничения текущего шага

`runner/streaming.py` пока не является окончательным benchmark runner v1.0 и не считает основной набор метрик. Его задача — проверить границы компонентов и полный encode/text/decode path для потоковых методов этапа 2.

Параметр `method_seed` в smoke test является техническим способом воспроизвести одинаковое разбиение Bins у encoder/decoder. Итоговая универсальная key/seed policy остается TBD до спецификации v1.0.

Реальная GPU-проверка выполняется вручную через `scripts/check_bins_e2e.py`; она не входит в обычный synthetic `pytest` suite.
