# Этап 3: протокол проверки воспроизводимости базовых методов

**Статус:** рабочий протокол Этапа 3.  
**Исходная точка:** benchmark v0.2, commit `967cc597f07aa7f575e4e08b75da0ba9b8dc1995`.  
**Рабочая ветка:** `stage3-reproducibility` (создаётся локально пользователем).

Назначение документа — до начала численных экспериментов зафиксировать, что именно считается авторским эталоном, какие режимы сравнения используются и какие сведения необходимо сохранять для объяснения расхождений. Этап 3 не заменяет нормализованные реализации Этапа 2 авторским кодом.

## 1. Цель Этапа 3

Для Bins, Huffman и Arithmetic Coding необходимо:

1. запустить исходную или максимально близкую к авторской реализацию;
2. воспроизвести характерные результаты публикаций или, если точная цифра недостижима, характерные зависимости;
3. проверить, что перенос методов в общую инфраструктуру не изменил их принципиальное поведение;
4. локализовать и документировать причины расхождений.

Результат Этапа 3 — не новое ранжирование методов, а **проверка воспроизводимости и соответствия адаптаций**.

## 2. Зафиксированный executable reference

Для всех трёх базовых методов используется один программный эталон:

```text
repository: https://github.com/harvardnlp/NeuralSteganography
commit: 14e982564aeaf9a33f7b4de440deda2184d17f12
role: algorithm_reference
```

Эта ссылка уже зафиксирована в `reference/reference_sources.json`.

Файлы reference implementation:

- Bins: `block_baseline.py`;
- Huffman: `huffman_baseline.py` + `huffman.py`;
- Arithmetic Coding: `arithmetic.py`;
- общая загрузка модели и метрики: `utils.py`;
- минимальный ручной запуск: `run_single.py`.

Историческое происхождение методов при этом различается:

- Bins — Fang et al.;
- Huffman — Yang et al. / RNN-Stega;
- Arithmetic Coding — Ziegler et al.

Для численного сравнения трёх методов в общей GPT-2 среде основным executable reference является именно код Harvard NLP, использованный Ziegler et al. как общая реализационная база.

## 3. Два экспериментальных режима

### 3.1. `normalized`

Это существующий контур benchmark v0.2:

```text
LM -> canonical P_reference -> normalized method adapter -> common metrics/storage
```

Здесь сохраняются все решения Этапа 2: единое допустимое пространство токенов, общий `P_reference`, единый текстовый transport, общая семантика payload и общий metric layer.

### 3.2. `author-compatible`

Этот контур нужен только для воспроизведения reference implementation / paper result. В нём допускаются особенности исходного Harvard-кода, если они являются частью воспроизводимого результата, например:

- GPT-2-специфичная обработка токенов;
- авторские `temperature` и `top-k`;
- исходное разбиение Bins;
- исходное построение Huffman tree;
- конечная точность Arithmetic Coding и его rounding logic;
- author-specific final flush;
- BPE repair;
- авторское направление и агрегирование KL.

Эти особенности **не должны незаметно переноситься** в `src/vkr_benchmark/methods/*` и изменять семантику normalized benchmark.

## 4. Роли языковых моделей

### 4.1. `gpt2` — технический smoke-test

Первый запуск pinned reference code выполняется на GPT-2 Small (`gpt2`). Это лёгкая техническая проверка того, что Harvard-код запускается в нашем современном окружении. Она сама по себе не считается воспроизведением чисел статьи.

### 4.2. `gpt2-medium` — paper-compatible исторический профиль

Для воспроизведения характерных результатов Ziegler et al. используется GPT-2 Medium (345M), поскольку именно medium-модель использовалась в экспериментах статьи.

### 4.3. Llama/Qwen — normalized benchmark

Модели Этапа 2 остаются без изменения:

- Llama-3.2-3B — основная современная модель;
- Qwen3-4B-Base — резервная.

GPT-2 не заменяет их в основном benchmark: она добавляется только как историческая модель author-compatible контура.

## 5. Политика окружения

На первом проходе отдельное legacy-окружение не создаётся. Pinned Harvard reference сначала проверяется в текущем окружении benchmark v0.2:

```text
Python 3.12.x
PyTorch 2.7.1
Transformers 4.52.4
Tokenizers 0.21.x
NumPy 2.0.x
```

Reference commit Harvard NLP уже адаптирован к современному стеку. Для него дополнительно требуется `bitarray==3.4.2`; эта зависимость оформляется отдельным optional extra `reference`, чтобы не смешивать её с обязательными зависимостями normalized benchmark.

Отдельное legacy environment создаётся только если будет показано, что современный стек не позволяет воспроизвести существенное поведение reference code. В таком случае точные версии и причина перехода фиксируются как отдельный источник расхождения.

## 6. Внешний checkout reference code

Исходный Harvard repository не копируется в `src/vkr_benchmark` и не коммитится внутрь нашего репозитория. Рекомендуемое локальное размещение:

```text
vkr-benchmark/
  external/
    NeuralSteganography/
```

Каталог `external/` игнорируется Git. Проверяемый checkout обязан находиться на commit:

```text
14e982564aeaf9a33f7b4de440deda2184d17f12
```

До первого smoke-run не вносятся исправления в reference algorithm: сначала необходимо сохранить исходный факт запуска или ошибку.

## 7. KL на Этапе 3

Согласно ADR-0012 параллельно используются два направления.

### Benchmark-native

```text
D_KL(P_reference || Q_stego)
```

Правила benchmark v0.2 сохраняются: единицы — bits/token, epsilon smoothing отсутствует, support mismatch может законно давать `+inf`.

### Author-compatible

```text
D_KL(Q_stego || P_reference)
```

Именно это направление необходимо для сопоставления с Harvard implementation / Ziegler. Направление, единицы и способ агрегирования всегда записываются явно. Не допускается неоднозначное поле просто `KL` после введения Stage-3 result schema.

## 8. Уровни воспроизводимости

Для каждого проверяемого результата фиксируется один из статусов:

1. **exact implementation reproduction** — совпадают дискретные внутренние решения/токены при одинаковом входе;
2. **numerical reproduction** — метрика совпадает с reference/paper в заранее обоснованной погрешности;
3. **trend reproduction** — точная цифра недоступна, но воспроизводится опубликованная зависимость или относительный порядок;
4. **partial reproduction** — воспроизведена часть результата, оставшееся расхождение локализовано;
5. **not reproducible** — результат не воспроизведён, причина или ограничение явно зафиксированы.

Настройки нельзя подбирать постфактум только ради совпадения с опубликованной цифрой без основания в paper/code.

## 9. Что фиксируется для author-compatible запуска

Минимальный provenance:

- reference repository и commit;
- benchmark commit;
- Python/PyTorch/Transformers/Tokenizers/NumPy/bitarray versions;
- CUDA и GPU;
- model ID и revision, если она доступна;
- dataset/context/preprocessing;
- способ получения секретной последовательности;
- параметры метода;
- `temperature`, `top-k`, `top-p` и candidate truncation;
- precision/rounding для Arithmetic Coding;
- BPE repair и termination/final-flush policy;
- направление и единицы KL;
- способ агрегации;
- target из reference code / paper;
- фактический результат;
- абсолютное/относительное расхождение;
- классификация причины расхождения.

## 10. Первый технический gate

До paper-level воспроизведения необходимо пройти reference smoke gate:

1. checkout `harvardnlp/NeuralSteganography@14e982...`;
2. использовать текущее Stage-2 environment;
3. установить optional dependency `reference`;
4. проверить окружение скриптом `scripts/check_stage3_reference_env.py`;
5. загрузить `gpt2`;
6. выполнить короткий encode/decode smoke для Bins;
7. выполнить короткий encode/decode smoke для Huffman;
8. выполнить короткий encode/decode smoke для Arithmetic Coding;
9. сохранить версии окружения, stdout/stderr и все предупреждения;
10. только после фиксации исходного поведения при необходимости вводить compatibility patches/harness.

Следующий шаг после этого документа — **реальный smoke-run pinned Harvard reference**, начиная с Bins.
