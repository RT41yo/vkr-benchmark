# Этап 3 — матрица paper-level воспроизводимости

**Статус:** зафиксирована до первых paper-level запусков.  
**Основная цель:** воспроизвести информационно-теоретическую часть Ziegler, Deng, Rush (2019), прежде всего Figure 3, не подбирая параметры постфактум под желаемый график.

## 1. Зачем нужен отдельный шаг

Три технических smoke-run уже доказали, что pinned Harvard implementation исполняется в нашем окружении и корректно восстанавливает payload для Bins, Huffman и Arithmetic Coding. Это ещё не означает воспроизведение статьи: smoke использовал GPT-2 Small, один фиксированный контекст и короткий 24-битный payload.

Paper-level проверка меняет задачу: теперь нас интересует не единичный roundtrip, а опубликованная зависимость **KL ↔ bits/word** на общей исторической модели и новостных контекстах. Поэтому до запуска `gpt2-medium` фиксируется матрица эксперимента.

Машиночитаемая версия находится в:

```text
configs/reproducibility/paper_reproduction_matrix.json
```

## 2. Что прямо зафиксировано в статье

Ziegler et al. используют GPT-2 345M и CNN/DailyMail. Для каждого примера первые три предложения новости служат контекстом, после чего стегометод генерирует одно следующее предложение под равномерным случайным сообщением.

Информационно-теоретическая оценка использует две величины:

```text
Bits/word = encoded message bits / sentence length
KL        = D_KL(q || p_LM)
```

KL оценивается Monte-Carlo по новостным данным. Figure 3 показывает среднее и standard error для повторных выборок при фиксированном параметре метода.

В статье заданы следующие диапазоны:

```text
Block/Bins:  2^1 ... 2^5 bins
Huffman:     truncation length 2^1 ... 2^8
Arithmetic:  temperature 0.4 ... 1.2, top-k = 300
Special AC:  temperature = 1, top-k = 50256
```

Основные claims, которые считаются paper targets:

1. в перекрывающемся диапазоне примерно 1–5 bits/word Arithmetic имеет меньший KL, чем Block и Huffman;
2. Arithmetic при `temperature=1, topk=50256` даёт практически нулевой KL относительно LM;
3. минимум KL Arithmetic находится примерно около 4 bits/word при `temperature≈1.0`.

## 3. Зафиксированный sweep

### 3.1. Bins / Block

Запускаются пять параметров:

```text
block_size_bits = 1, 2, 3, 4, 5
bins            = 2, 4, 8, 16, 32
```

### 3.2. Huffman

Запускаются восемь параметров:

```text
candidate_pool_exponent = 1 ... 8
candidate_count         = 2, 4, 8, 16, 32, 64, 128, 256
```

### 3.3. Arithmetic Coding

Статья указывает диапазон температуры `0.4 ... 1.2` при `k=300`, но текст не перечисляет каждый промежуточный шаг. Figure 3 визуально содержит девять точек Arithmetic. До экспериментов мы фиксируем равномерную сетку с шагом `0.1`:

```text
temperature = 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2
topk        = 300
precision   = 26
```

Это **наша заранее объявленная операционализация опубликованного диапазона**, а не утверждение, что статья текстом перечисляет именно эти девять значений. `precision=26` берётся из pinned Harvard reference defaults.

Дополнительно выполняется отдельная точка:

```text
temperature = 1.0
topk        = 50256
precision   = 26
```

Итого Figure-3 matrix содержит:

```text
5 Bins + 8 Huffman + 9 Arithmetic + 1 unmodulated Arithmetic = 23 parameter points
```

## 4. Важная неоднозначность единиц KL

Figure 3 подписан как:

```text
KL (bits)
```

и pinned Harvard `utils.kl()` делит натурально-логарифмическую сумму на приблизительное `ln(2)`, то есть возвращает bits.

При этом текст статьи для специальной unmodulated Arithmetic точки пишет:

```text
KL = 4e-8 nats
```

Мы **не исправляем эту неоднозначность молча**. Для paper-compatible результатов сохраняется точная единица, реально возвращаемая кодом, а при сравнении с `4e-8` отдельно указывается paper prose unit. Если потребуется численное сравнение в nats, conversion выполняется явно и сохраняется как отдельное поле.

## 5. Что считаем воспроизведением

Для основной Figure-3 проверки обязательным является **trend reproduction**, а не совпадение каждого пикселя графика:

```text
Arithmetic curve ниже Block/Huffman
в сопоставимом диапазоне capacity.
```

Для special point `temperature=1, topk=50256` ставится более сильная цель — **numerical near-zero reproduction**. Мы ожидаем значение, практически неотличимое от нуля по масштабу Figure 3; точное совпадение `4e-8` не объявляется заранее обязательным, потому что доступный pinned code был обновлён под современный PyTorch/Transformers, а точный Figure-3 batch driver и MC sample count в репозитории не опубликованы.

## 6. Агрегация и объём запусков

Paper пишет, что каждая точка Figure 3 показывает mean и standard error по repeated samples, но точное число MC samples для Figure 3 в основном тексте не указано.

Поэтому мы заранее фиксируем собственную воспроизводимую схему:

```text
Phase 1: pilot
  8 contexts × 1 replicate

Phase 2: Figure-3 curve
  80 contexts per parameter point
  × 3 deterministic replicates
```

Число `80` известно из supplementary human evaluation как число generations per source. Мы **не выдаём его за исходный Figure-3 MC count**; здесь это осознанный фиксированный объём нашей репликации. Три replicate нужны, чтобы получить устойчивые mean/SE и отделить тренд от случайности.

## 7. Что пока намеренно не запускается

Human evaluation из Figure 4 не является обязательным gate Этапа 3. В supplementary material для неё известны:

```text
Arithmetic temperatures: 0.4, 0.7, 1.0, 1.2
Huffman exponents:        1, 3, 5
Block exponents:          1, 3, 5
80 generations/source
```

Но точное воспроизведение MTurk не нужно для текущей проверки соответствия author implementation ↔ normalized benchmark. Оно может быть отдельным дополнительным исследованием/заменено позже нашей общей оценкой естественности согласно финальному benchmark protocol.

## 8. Что блокирует немедленный большой sweep

Перед первым `gpt2-medium` pilot необходимо отдельно зафиксировать CNN/DailyMail input artifact:

```text
dataset source
split
revision/checksum
порядок примеров
sentence segmentation
правило выбора первых трёх предложений
```

Статья задаёт dataset и смысловой context rule, но не современный Hugging Face revision identifier. Подставлять удобную версию молча нельзя — иначе расхождение можно ошибочно приписать алгоритму.

Второе ограничение: pinned Harvard repository содержит сами алгоритмы и `run_single.py`, но не исходный batch/MC driver Figure 3. Поэтому наш будущий orchestration layer должен оставлять algorithm files неизменными и отдельно документировать каждое решение вокруг dataset iteration, termination и aggregation.

## 9. Следующий gate

После этого шага матрица считается **frozen**, но full sweep ещё не начинается. Следующий шаг:

```text
3.6 — pin CNN/DailyMail + deterministic 3-sentence context extraction
      + GPT-2 Medium pilot на 8 contexts
```

Только после успешного pilot и проверки result schema запускается полная матрица из 23 parameter points.
