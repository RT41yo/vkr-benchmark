# ADR-0004: Нормализованная адаптация метода Bins

## Статус

Принято на этапе 2 перед интеграционным запуском Bins с языковой моделью.

## Контекст

Первым базовым стегометодом в общей инфраструктуре реализуется Bins (block/binning baseline).
Алгоритмическим reference source является:

- репозиторий: `https://github.com/harvardnlp/NeuralSteganography`;
- файл: `block_baseline.py`;
- commit: `14e982564aeaf9a33f7b4de440deda2184d17f12`.

В reference-реализации словарь один раз разбивается на `2^block_size` bins. Исходный
массив token ID перемешивается после глобального `np.random.seed(block_size)`, затем
делится на последовательные части по границам `int(i * vocab_size / num_bins)`.
Очередные `block_size` секретных битов выбирают bin, после чего из него выводится токен
с максимальным logit. При декодировании bin наблюдаемого токена напрямую преобразуется
обратно в фиксированное число битов.

Reference-код одновременно владеет LM forward, GPT-2-специфичными исключениями token ID,
вычислением KL/NLL, глобальным RNG и BPE repair-эвристиками. Такой уровень связывания
компонентов несовместим с normalized benchmark.

## Решение

### 1. Сохраняем ядро алгоритма Bins

Normalized adapter сохраняет основную семантику:

1. создаётся фиксированное разбиение `V_allowed` на `2^b` bins;
2. каждые `b=block_size` секретных битов интерпретируются как номер bin в MSB-first порядке;
3. из выбранного bin выводится токен с максимальной вероятностью в текущем
   `P_reference`;
4. decoder восстанавливает `b` битов по номеру bin наблюдаемого токена.

### 2. Разбиение строится по `V_allowed`, а не по полному output vocabulary

Метод получает `MethodEnvironment`, содержащий:

- `output_vocab_size`;
- канонический набор `allowed_token_ids`.

Special tokens и output-only IDs уже исключены общей инфраструктурой и не должны
повторно обрабатываться внутри Bins.

Перед shuffle `allowed_token_ids` канонизируются по возрастанию. После shuffle bins
формируются по целочисленным floor-границам:

```text
start_i = floor(i * |V_allowed| / num_bins)
stop_i  = floor((i + 1) * |V_allowed| / num_bins)
```

Это сохраняет принцип разбиения reference-кода, но применяет его к нормализованному
допустимому пространству токенов.

### 3. Глобальный NumPy RNG не используется

Reference-реализация делает `np.random.seed(block_size)`. В normalized benchmark
разбиение использует только переданный `MethodRandomSource`.

Encoder и decoder должны получать независимые экземпляры method RNG с одинаковым
исходным состоянием, чтобы воспроизвести одно и то же разбиение. Политика преобразования
benchmark key/seed в `MethodRandomSource` остаётся ответственностью общей инфраструктуры
и окончательно фиксируется позднее в спецификации v1.0.

### 4. Выбор токена выполняется только из канонического `P_reference`

Bins не получает raw logits и не вызывает LM. Представитель каждого bin выбирается как
первый token ID этого bin в общем `token_order`, который уже определён правилом:

```text
(-probability, token_id)
```

Следовательно, exact ties разрешаются меньшим `token_id` одинаково у encoder и decoder
и одинаково для всех методов benchmark.

### 5. Явно строится точное `Q_stego`

При независимом равновероятном secret bitstream каждый из `2^b` bins выбирается с
вероятностью `2^-b`. Для фиксированного `P_reference` encoder детерминированно выбирает
одного представителя каждого bin. Поэтому induced distribution имеет вид:

```text
Q_stego(representative_i) = 2^-b
Q_stego(other token)      = 0
```

Adapter возвращает это как:

- `q_mode = analytic_exact`;
- explicit probability vector;
- `q_source = adapter_exact`.

KL/TVD внутри Bins не вычисляются: их должен считать общий metric layer.

### 6. Агрессивная common truncation может сделать конфигурацию неработоспособной

Если после common generation policy текущий `P_reference` не содержит ни одного
положительно-вероятного токена хотя бы в одном фиксированном bin, Bins не может
реализовать равномерное отображение всех `b`-битных значений. Такой шаг завершается
`UnsupportedConfigurationError` до чтения очередных secret bits.

Baseline normalized policy (`temperature=1`, без top-k, `top-p=1`) этой проблемы не имеет,
если каждый allowed token получает конечный logit.

### 7. Транспорт и BPE repair не являются частью Bins adapter

Reference-код содержит GPT-2-specific BPE repair. В normalized benchmark adapter
работает только с token ID после общего text transport. Проверка
`tokens -> text -> tokens -> stego decode` будет реализована отдельным `transport/`
слоем. Авторские repair-эвристики при необходимости относятся к будущему
`author_conformity` режиму, а не к normalized Bins.

## Последствия

- Bins можно тестировать полностью на синтетическом `P_reference` без GPU и LM.
- Метод не зависит от Llama/Qwen и может работать с любой моделью, для которой общая
  инфраструктура сформировала совместимые `MethodEnvironment` и `P_reference`.
- Partition randomness отделена от secret stream и control RNG.
- Exact `Q_stego` доступно metric layer с первого шага.
- Для реального round-trip ещё необходимы runner и text transport; текущая реализация
  Bins сама по себе не считается завершённым end-to-end benchmark run.
