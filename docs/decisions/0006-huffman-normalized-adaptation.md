# ADR-0006: Нормализованная адаптация Huffman

## Статус

Принято на этапе 2. Алгоритмическое ядро Huffman проверено синтетическими unit-тестами и подключено к существующим `runner/streaming.py` и `TextChannel` без изменений метода или LM-слоя. Реальная GPU/end-to-end проверка выполняется отдельным smoke script на Llama и Qwen.

## Контекст

Алгоритмический reference — `harvardnlp/NeuralSteganography/huffman_baseline.py` и `huffman.py`, commit `14e982564aeaf9a33f7b4de440deda2184d17f12`.

Авторская реализация на каждом шаге сортирует выходы LM по убыванию вероятности, берёт top `2**bits_per_word` кандидатов, строит по их вероятностям дерево Хаффмана и читает секретные биты по ветвям дерева (`0` — влево, `1` — вправо) до достижения листа. Декодер по тому же префиксу заново строит дерево и возвращает код наблюдаемого листа.

Параметр `bits_per_word` в исходном коде задаёт размер множества кандидатов, но не фиксированное число полезных битов на токен. Длина Huffman-кода выбранного листа зависит от текущего распределения, поэтому `bits_consumed` является переменной величиной.

## Решение

В normalized mode Huffman получает только канонический `P_reference`, `StepContext` и `MethodEnvironment`. LM, tokenizer, общие temperature/top-k/top-p, transport и benchmark-метрики остаются вне метода.

На каждом шаге:

1. Из канонического `token_order=(-probability, token_id)` выбираются первые `2**bits_per_word` токенов с положительной вероятностью.
2. По их значениям `P_reference` строится новое дерево Хаффмана.
3. Encoder читает секрет по одному биту и проходит по дереву до листа.
4. `bits_consumed` равен фактической длине выбранного Huffman-кода.
5. Decoder на том же `P_reference` строит то же дерево и возвращает код наблюдаемого токена.
6. Основные метрики не считаются внутри метода.

## Детерминизм дерева

В reference `HeapNode` сравнивается только по `freq`. Для точных равенств вероятностей отдельное правило tie-break не задано. Это может сделать дерево зависимым от деталей heap/insertion order.

В normalized adapter используется явное правило. Узлы сравниваются по:

```text
(weight, min_token_id_in_subtree)
```

Первый извлечённый узел становится левой ветвью (`0`), второй — правой (`1`). Для листа `min_token_id_in_subtree = token_id`, для внутреннего узла — минимальный ID всех его листьев. Так encoder и decoder получают одно и то же дерево при равных вероятностях и одинаковом `P_reference`.

Это нормализующее уточнение относится только к неоднозначному случаю точного равенства весов; author/conformity режим позднее сможет сохранить исходное heap-поведение отдельно.

## `Q_stego`

При равновероятном i.i.d. secret bitstream вероятность достижения листа с Huffman-кодом длины `l` равна:

```text
Q_stego(token) = 2^(-l)
```

для токенов-кандидатов и `0` для остальных токенов. Полное бинарное дерево удовлетворяет равенству Крафта, поэтому сумма `Q_stego` равна 1.

Adapter возвращает explicit `Q_stego` с:

```text
q_mode = analytic_exact
q_source = adapter_exact
```

KL/TVD далее вычисляет независимый metric layer.

## Отличия от author implementation в normalized mode

- исходные raw logits и LM не передаются методу;
- candidate set строится из общего `P_reference`, а не из собственного softmax метода;
- hardcoded GPT-2 exclusions не используются;
- BPE-repair эвристики из author decoder не переносятся в normalized adapter;
- равенства в Huffman heap разрешаются детерминированно;
- NLL/KL/BPT не считаются внутри метода;
- payload учитывается по фактически прочитанным secret bits.

## Ограничения конфигурации

Для каждого шага требуется как минимум `2**bits_per_word` токенов с положительной вероятностью в `P_reference`. Если общая top-k/top-p policy уменьшила support ниже этого значения, конфигурация считается несовместимой и завершается `UnsupportedConfigurationError`, а secret stream до ошибки не продвигается.

## End-to-end интеграция

`HuffmanMethod` использует тот же общий `runner/streaming.py` и `TextChannel`, что и Bins. Отдельная Huffman-ветка runner не создаётся. Это проверяет, что общий lifecycle поддерживает переменное `bits_consumed`: итоговый `payload_bits` равен сумме фактически пройденных Huffman-кодов, а decoder сравнивается именно с реально использованным префиксом secret stream.

GPU smoke test находится в `scripts/check_huffman_e2e.py` и проходит полный путь:

```text
LM -> P_reference -> Huffman -> token IDs -> text -> retokenization
   -> независимый decoder LM/KV-cache path -> recovered bits
```

BPE-repair эвристики author implementation по-прежнему не применяются. Изменение token sequence после обычного текстового transport должно фиксироваться как reliability-результат, а не исправляться внутри метода.

## Следующий шаг

После проверки Huffman end-to-end на основной и резервной моделях следующий базовый метод этапа 2 — Arithmetic Coding.
