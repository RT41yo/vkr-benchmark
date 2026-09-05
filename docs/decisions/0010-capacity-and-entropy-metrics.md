# ADR-0010: Емкость и эффективность использования энтропии

## Статус

Принято для этапа 2, шаг 7.2.

## Контекст

После шага 7.1 Bins, Huffman и Arithmetic Coding запускаются через единый
`ExperimentConfig -> run_experiment()` путь. Следующий слой должен рассчитывать
первые общие benchmark-метрики независимо от конкретного стегометода.

Спецификация v0.1 фиксирует:

- `payload_bits = B` — только фактически встроенные полезные секретные биты;
- `carrier_tokens = T` — число сгенерированных токенов-носителей;
- `bits_per_token = B / T`;
- `H_t = -sum_x P_reference,t(x) log2 P_reference,t(x)`;
- `entropy_utilization = B / sum_t H_t`.

Для Arithmetic Coding прочитанные look-ahead биты не являются payload и поэтому
не входят ни в BPT, ни в числитель entropy utilization.

## Решение

### 1. Энтропия считается общим metric layer

`ReferenceDistribution` остается каноническим FP32-распределением. Функция
`reference_entropy_bits()` читает его без модификации и выполняет только
суммирование/log2 в FP64. Нулевые вероятности дают нулевой вклад.

Стегометод не рассчитывает и не сообщает `H_t` самостоятельно.

### 2. Streaming runner сохраняет только скаляр H_t каждого sender-side шага

`StreamingEncodeResult.step_reference_entropy_bits` содержит один скаляр на
каждый реально сгенерированный carrier token. Полные массивы P_reference для
этого шага не сохраняются.

Это позволяет агрегировать энтропию без раздувания памяти и не связывает
метрику с конкретным методом.

### 3. Run-level агрегат

`CapacityEntropyMetrics` содержит:

- `payload_bits`;
- `carrier_tokens`;
- `bits_per_token`;
- `reference_entropy_mean_bits`;
- `reference_entropy_sum_bits`;
- `entropy_utilization`;
- `entropy_utilization_percent`.

Агрегат строится после завершения encode/decode через
`compute_capacity_entropy_metrics()` и доступен как
`ExperimentExecution.capacity_entropy_metrics`.

### 4. Utilization не ограничивается диапазоном [0, 1]

Результат `B / sum H_t` не клиппируется. Значение выше 1 сохраняется как есть,
поскольку это диагностический результат и его нельзя скрывать постобработкой.

### 5. Нулевая суммарная энтропия

Спецификация v0.1 не задает отдельную конвенцию для случая `sum H_t = 0`.
Поэтому реализация не подменяет неопределенность значением 0, NaN или infinity:
такой случай явно завершает расчет `MetricError`. Если он станет практически
релевантным для нормализованных конфигураций, конвенция будет отдельно
зафиксирована в будущей спецификации.

## Следствия

Шаг 7.2 добавляет первые реальные benchmark metrics в единый experiment runner,
но еще не реализует KL/TVD, NLL/PPL, reliability aggregation, timing и storage.
Следующий шаг 7.3 может использовать тот же per-step sender-side контур для
расчета искажения `P_reference` относительно `Q_stego`.
