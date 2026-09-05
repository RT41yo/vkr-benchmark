# ADR-0008: Разделить прочитанные секретные биты и полезную нагрузку

## Статус

Принято на этапе 2 при end-to-end интеграции Arithmetic Coding.

## Контекст

Первая версия `runner/streaming.py` была проверена на Bins и Huffman. Для этих двух методов количество битов, прочитанных из `SecretSource`, совпадает с количеством реально встроенных полезных битов:

```text
secret bits read == payload bits
```

Это позволило временно использовать `RecordingSecretSource.consumed_bits` как непосредственный эталон payload для проверки декодирования.

Arithmetic Coding показывает, что такое отождествление является слишком узким. Encoder поддерживает перекрывающееся `precision`-битное окно look-ahead. На первом шаге он читает `precision` битов, но полезной нагрузкой становятся только те старшие биты, которые однозначно фиксируются выбранным целочисленным подинтервалом. После подтверждения `k` битов окно сдвигается и читает ещё `k` битов. Поэтому после fixed-carrier запуска типично:

```text
secret_bits_read = precision + payload_bits
```

При этом дополнительные look-ahead биты не являются встроенной полезной нагрузкой и не должны увеличивать BPT или участвовать в сравнении с decoder как обязательный recovered payload.

## Решение

`StreamingEncodeResult` хранит две разные последовательности:

- `read_secret_bits` — все биты, которые метод фактически запросил у `SecretSource`, включая look-ahead;
- `payload_secret_bits` — подтвержденный полезный префикс длины `EncoderFinalization.payload_bits`.

Для совместимости свойство `consumed_secret_bits` сохранено как alias к `payload_secret_bits`, поскольку в предыдущих Bins/Huffman smoke tests под этим именем уже подразумевался именно полезный секрет.

Общий runner больше не требует:

```text
len(read_secret_bits) == payload_bits
```

Вместо этого он проверяет только, что метод не объявляет больше полезных битов, чем прочитал:

```text
payload_bits <= len(read_secret_bits)
```

Если метод предоставляет `bits_consumed` на каждом шаге, runner дополнительно проверяет:

```text
sum(step_bits_consumed) == EncoderFinalization.payload_bits
```

Это сохраняет строгий контроль Bins, Huffman и Arithmetic Coding, но не блокирует будущие методы вроде RRC, для которых per-step `bits_consumed` может быть `None`.

## Reliability roundtrip

После обычного текстового transport decoder сравнивается только с:

```text
payload_secret_bits
```

а не со всем look-ahead, прочитанным encoder'ом.

Таким образом:

```text
secret stream
    ↓
read_secret_bits
    ├── confirmed prefix → payload_secret_bits → reliability/BPT
    └── unresolved look-ahead → не считается payload
```

## Последствия

1. Bins и Huffman сохраняют прежнее поведение: `read_secret_bits == payload_secret_bits`.
2. Arithmetic Coding корректно учитывает look-ahead без искусственного завышения емкости.
3. `EncoderFinalization.payload_bits` окончательно становится авторитетным session-level источником размера полезной нагрузки.
4. Общий runner становится совместимее с будущими stateful/fixed-message методами и не требует method-specific AC orchestration.
5. В будущей спецификации v1.0 следует явно различить «бит прочитан encoder'ом» и «бит подтвержден как полезная встроенная нагрузка» там, где это существенно для алгоритма.
