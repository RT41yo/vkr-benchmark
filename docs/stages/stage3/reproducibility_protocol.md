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

## 11. Первый исполняемый smoke: Bins на GPT-2 Small

После успешного environment/provenance gate первый исполняемый author-compatible тест выполняется отдельно для Bins.

Зафиксированная конфигурация находится в:

```text
configs/reproducibility/author_bins_gpt2_smoke.json
```

Запуск выполняется только через внешний pinned checkout:

```text
scripts/run_stage3_author_bins_smoke.py
    -> external/NeuralSteganography@14e982...
    -> utils.get_model(model_name="gpt2")
    -> block_baseline.get_bins
    -> block_baseline.encode_block
    -> text transport
    -> block_baseline.decode_block
```

Reference-файлы при этом не редактируются. Runner до и после исполнения проверяет `git status --porcelain` внешнего checkout и запрещает запись Python bytecode внутрь него.

Для smoke используется `block_size = 3` и 24-битная последовательность:

```text
000 100 010 110 001 101 011 111
```

В реализации Harvard `bits2int` интерпретирует первый бит блока как младший. Поэтому эти восемь трёхбитных групп соответствуют индексам bins `0, 1, 2, 3, 4, 5, 6, 7` и за один короткий прогон упражняют все восемь bins.

Секрет сразу задаётся битовой последовательностью. На этом smoke шаге намеренно не используется предварительное преобразование естественно-языкового сообщения в uniform bits через Arithmetic Coding из `run_single.py`, потому что цель — изолированно проверить исполняемость и encode/decode поведение Bins.

Результаты сохраняются в:

```text
results/stage3/author_smoke/bins_gpt2/
  result.json
  stegotext.txt
  token_ids.json
```

Авторский KL сохраняется только с явным направлением:

```text
kl_q_stego_to_p_lm_bits_author
```

то есть как авторская величина `D_KL(Q_stego || P_LM)` в битах. Она не подменяет benchmark-native `D_KL(P_reference || Q_stego)`.

Успешный технический smoke требует одновременно:

- pinned reference HEAD совпадает с `14e982...`;
- Bins encode завершается без изменения reference code;
- декодирование из обычного текста восстанавливает весь 24-битный payload;
- внешний reference worktree после запуска имеет то же состояние, что до запуска.

Совпадение sender token IDs с повторной токенизацией текста сохраняется как отдельная диагностика. Оно не является самостоятельным критерием ошибки, если author BPE repair корректно восстанавливает payload из текста.


## 12. Runtime compatibility finding: GPT2TokenizerFast

Первый raw author-reference запуск Bins на современном окружении завершился до encode с ошибкой:

```text
AttributeError: GPT2TokenizerFast has no attribute encoder
```

Это не изменение алгоритма Bins и не основание переходить на отдельное legacy-окружение. Причина локализована на границе API tokenizer: pinned Harvard code напрямую использует `enc.encoder` и `enc.decoder`, тогда как `AutoTokenizer.from_pretrained("gpt2")` в Transformers 4.52 по умолчанию возвращает `GPT2TokenizerFast`, не имеющий ожидаемого публичного `encoder`-атрибута.

Для следующей попытки вводится внешний compatibility profile `hf_4_52_legacy_api`, реализованный только в benchmark runner. Он:

- принудительно вызывает `AutoTokenizer.from_pretrained(..., use_fast=False)`, получая slow `GPT2Tokenizer` с историческими `encoder`/`decoder`;
- адаптирует legacy-вызов модели `past=` к современному `past_key_values=`;
- запрашивает `return_dict=False`, чтобы reference code продолжал получать `(logits, past)`;
- не изменяет файлы `external/NeuralSteganography`;
- не меняет logits, Bins partition, выбор токена, payload или BPE repair.

Исходный raw failure сохраняется как отдельное evidence (`raw_reference_failure.json`) перед compatibility rerun. Если после API-моста возникает следующая несовместимость, она также фиксируется до расширения compatibility layer.

## 13. Runtime compatibility finding: legacy GPT-2 cache shape

После устранения tokenizer/API keyword mismatch второй запуск дошёл до полного Bins encode и остановился уже в `decode_block` на проверке:

```text
if past and past[0].shape[3] >= 1023:
```

с ошибкой:

```text
AttributeError: 'tuple' object has no attribute 'shape'
```

Причина — различие представления GPT-2 KV cache. Pinned Harvard decoder ожидает историческое представление каждого слоя как одного stacked tensor с первой осью `key/value`, тогда как Transformers 4.52 возвращает каждый слой как пару `(key, value)`.

Compatibility profile `hf_4_52_legacy_api` поэтому расширяется representation-only мостом:

```text
reference side:
layer cache = Tensor[2, batch, heads, seq, head_dim]

          <->

Transformers 4.52 side:
layer cache = (key[batch, heads, seq, head_dim],
               value[batch, heads, seq, head_dim])
```

Перед современным model call stacked tensor разбирается обратно в `(key, value)`. После model call пара снова складывается в historical representation. Значения key/value не пересчитываются и не изменяются; преобразуется только контейнер/форма представления на границе API.

Это позволяет оставить `external/NeuralSteganography` неизменённым и, в частности, не патчить строку `past[0].shape[3]` в `decode_block`.

Второй runtime failure сохраняется отдельно как:

```text
results/stage3/author_smoke/bins_gpt2/compat_cache_shape_failure.json
```

Первый tokenizer failure продолжает храниться в `raw_reference_failure.json`. Таким образом, успешный последующий run не уничтожает историю обнаруженных compatibility barriers.

## 14. Закрытие Bins smoke и следующий smoke: Huffman

Третий Bins-запуск после введения полного `hf_4_52_legacy_api` compatibility bridge завершился успешно. На GPT-2 Small при `block_size = 3` исходный 24-битный payload был встроен в 8 токенов и полностью восстановлен после обычного текстового канала. В зафиксированном запуске также совпали sender и retokenized token IDs, а внешний checkout `NeuralSteganography@14e982...` остался неизменным.

Успешный Bins result и два предшествующих compatibility failure сохраняются в `results/stage3/author_smoke/bins_gpt2/`. Эти результаты означают, что отдельное legacy Python/PyTorch environment пока не требуется: для Bins достаточно representation/API bridge вокруг неизменённого reference code.

Следующий технический smoke выполняется для Harvard Huffman implementation:

```text
configs/reproducibility/author_huffman_gpt2_smoke.json
scripts/run_stage3_author_huffman_smoke.py
    -> external/NeuralSteganography@14e982...
    -> utils.get_model(model_name="gpt2")
    -> huffman_baseline.encode_huffman
    -> ordinary text transport
    -> huffman_baseline.decode_huffman
```

Для сопоставимости используются те же GPT-2 Small, seed, Washington-context и 24-битный direct binary payload, что и в Bins smoke. Авторский параметр `bits_per_word = 3` в Huffman-коде задаёт не фиксированные 3 payload bits/token, а размер candidate pool:

```text
2^3 = 8 top-probability candidate tokens
```

После получения восьми кандидатов reference implementation строит Huffman tree заново на каждом carrier step. Длины кодов переменные, поэтому фактическое число считанных секретных битов на токен также переменное.

Есть ещё одна важная особенность исходного encoder: если секрет заканчивается до достижения листа последнего Huffman codeword, reference code продолжает идти по левым (`0`) рёбрам до листа. Поэтому короткий smoke может фактически считать несколько несуществующих trailing zero bits. Runner не считает их полезным payload и отдельно сохраняет:

- `author_bits_consumed` — сколько битов фактически прошло через author traversal;
- `implicit_zero_padding_bits` — разницу между этим числом и длиной заданного payload;
- `payload_bits_per_token_smoke` — полезные 24 бита, делённые на число carrier tokens;
- `bits_per_word_author` — обратную величину к author `words_per_bit`, то есть исходную метрику кода, которая для короткого сообщения может включать terminal zero padding.

Критерий успешного Huffman smoke — восстановление **всего исходного 24-битного payload как префикса** декодированной последовательности после текстового канала и неизменность reference worktree. Дополнительные биты после payload сохраняются диагностически и ожидаются нулевыми, если они возникли только из-за завершения последнего codeword.

Как и для Bins, авторский KL сохраняется явно как:

```text
kl_q_stego_to_p_lm_bits_author
```

то есть `D_KL(Q_stego || P_LM)` в битах. Он не подменяет benchmark-native `D_KL(P_reference || Q_stego)`.

## 15. Закрытие Huffman smoke и следующий smoke: Arithmetic Coding

Author-compatible Huffman smoke на GPT-2 Small завершился успешно. При `bits_per_word = 3` (candidate pool из `2^3 = 8` токенов) исходный 24-битный payload был встроен в 8 carrier tokens и полностью восстановлен после обычного текстового канала. В конкретном smoke авторский encoder фактически потребил ровно 24 бита, поэтому terminal zero padding не понадобился. Sender и retokenized token IDs совпали, а внешний checkout `NeuralSteganography@14e982...` остался неизменным.

Зафиксированные значения этого технического запуска:

```text
payload bits:                         24
carrier tokens:                       8
payload bits/token:                   3.0
PPL_author:                           18.035587623700636
KL_author Q_stego || P_LM [bits]:     1.0004528872668743
```

Это не означает, что Huffman имеет фиксированную скорость 3 bit/token: в исходном методе длины кодов зависят от построенного на каждом шаге дерева. Значение 3.0 относится только к данному короткому smoke.

Третий исполняемый reference smoke выполняется для Harvard Arithmetic Coding:

```text
configs/reproducibility/author_arithmetic_gpt2_smoke.json
scripts/run_stage3_author_arithmetic_smoke.py
    -> external/NeuralSteganography@14e982...
    -> utils.get_model(model_name="gpt2")
    -> arithmetic.encode_arithmetic
    -> ordinary text transport
    -> arithmetic.decode_arithmetic
```

Для сопоставимости сохраняются те же GPT-2 Small, seed, Washington-context и 24-битный direct binary payload. Параметры Arithmetic Coding берутся из default-конфигурации pinned `run_single.py`:

```text
temperature = 0.9
precision   = 26
topk        = 300
finish_sent = false
```

В отличие от Bins и Huffman, pinned `arithmetic.py` уже обновлён под современный Transformers API: он импортирует `DynamicCache` и вызывает модель через `past_key_values`. Поэтому Arithmetic smoke намеренно **не** использует `LegacyCausalLMAdapter` и **не** принуждает slow tokenizer. Модель и tokenizer загружаются через `utils.get_model` ровно так, как предусмотрено pinned reference. Это позволяет сначала проверить нативную исполняемость актуального Harvard Arithmetic-кода без дополнительного compatibility вмешательства.

### 15.1. Две разные вероятностные величины внутри author Arithmetic

При `temperature != 1` исходный код использует разные распределения для кодирования и для author NLL/KL:

```text
logits
  |
  +-- softmax(logits / temperature)
  |      -> cutoff по 1 / текущая_ширина_интервала
  |      -> top-k cap
  |      -> integer rounding
  |      -> Q_stego для Arithmetic Coding
  |
  +-- log_softmax(logits)
         -> untempered P_LM
         -> author NLL
         -> author KL(Q_stego || P_LM)
```

Поэтому smoke сохраняет KL под максимально явным именем:

```text
kl_q_stego_to_p_lm_untempered_bits_author
```

Эта величина не является benchmark-native `D_KL(P_reference || Q_stego)` и не должна с ним смешиваться.

Возвращаемая `encode_arithmetic` величина `Hq` также сохраняется отдельно как:

```text
avg_entropy_p_tau_bits_author_helper
```

поскольку reference `utils.entropy` уже переводит натуральные логарифмы в биты. На smoke-этапе сохраняется непосредственно значение, возвращённое helper, без дополнительной конверсии.

### 15.2. Precision lookahead и author final flush

Arithmetic encoder работает с `precision`-битным окном секрета. Если до конца payload остаётся меньше `precision` бит, окно дополняется нулями. При последнем carrier token число подтверждённых общих старших битов может оказаться больше числа оставшихся полезных payload bits. Поэтому runner разделяет:

- `secret_bit_count` — полезные 24 бита;
- `author_bits_consumed` — число битов, которое следует из author `words_per_bit` и числа carrier tokens;
- `implicit_zero_lookahead_bits` — сколько нулей сверх полезного payload было фактически подтверждено из padded lookahead.

Decoder имеет отдельную author-specific termination policy. Для всех промежуточных carrier tokens он выдаёт только уже однозначно зафиксированный prefix, но на **последнем** carrier token выполняет flush полного `precision`-битного нижнего края финального интервала. Поэтому decoded stream может быть длиннее `author_bits_consumed`.

Runner отдельно сохраняет:

- `recovered_lookahead_padding_bits` — часть после payload, но внутри author-consumed prefix;
- `lookahead_padding_is_zero` — проверку ожидаемого zero padding;
- `decoder_flush_extra_bits` — хвост, добавленный именно final flush сверх author-consumed prefix;
- `decoder_flush_extra_bit_count`;
- `decoder_flush_within_precision_bound` — sanity-check, что дополнительный flush не превышает `precision` бит.

Критерий успешного Arithmetic smoke:

1. весь исходный 24-битный payload восстановлен как точный prefix;
2. подтверждённый encoder-ом lookahead сверх payload состоит из ожидаемых нулей;
3. final-flush tail укладывается в `precision`-битную границу;
4. внешний reference worktree не изменён.

Как и для двух предыдущих методов, совпадение sender token IDs с повторной токенизацией текста сохраняется отдельно. Если оно нарушится, это не автоматически означает failure при условии, что author decoder/BPE repair корректно восстановит payload.

## 16. Закрытие технического smoke-gate и фиксация paper-level matrix

Author-compatible Arithmetic Coding smoke на GPT-2 Small завершился успешно. При `temperature=0.9`, `precision=26`, `topk=300` исходный 24-битный payload был встроен в 17 carrier tokens и полностью восстановлен после ordinary-text transport. Sender и retokenized token IDs совпали; внешний checkout `NeuralSteganography@14e982...` остался неизменным.

Arithmetic smoke дополнительно подтвердил две важные source-specific semantics:

```text
useful payload bits:          24
author bits consumed:         48
implicit zero lookahead:      24
decoder final-flush extra:     1
```

Поэтому `bits_per_word_author` и benchmark payload BPT не отождествляются. Первый следует внутреннему author accounting, второй использует только полезный payload.

После этого технический reference smoke gate считается закрытым:

```text
Bins        PASS
Huffman     PASS
Arithmetic  PASS
```

Следующий подэтап переводит работу от единичной исполняемости к paper-level воспроизводимости. До загрузки GPT-2 Medium и массовых запусков фиксируется отдельная матрица:

```text
configs/reproducibility/paper_reproduction_matrix.json
docs/stages/stage3/paper_reproduction_matrix.md
scripts/check_stage3_paper_matrix.py
```

Основной paper target — Figure 3 Ziegler et al. (2019): зависимость `D_KL(q || p_LM)` от bits/word на GPT-2 345M и CNN/DailyMail.

Матрица заранее замораживает 23 информационно-теоретические точки:

```text
Bins:        block exponent 1..5                         = 5
Huffman:     candidate-pool exponent 1..8                = 8
Arithmetic:  temperature 0.4..1.2 by 0.1, topk=300       = 9
Arithmetic:  temperature=1.0, topk=50256 special point   = 1
                                                               --
                                                               23
```

Primary reproduction targets:

1. Arithmetic имеет меньший author-compatible KL, чем Block/Huffman, в перекрывающемся диапазоне примерно 1–5 bits/word;
2. `temperature=1, topk=50256` даёт near-zero KL;
3. минимум KL Arithmetic находится примерно около 4 bits/word при `temperature≈1`.

Для paper-compatible KL сохраняется направление `Q_stego || P_LM`. Отдельно фиксируется unit inconsistency источника: ось Figure 3 подписана `KL (bits)`, pinned Harvard `utils.kl()` возвращает bits, но prose статьи сообщает специальную величину `4e-8 nats`. Ни одна из единиц не заменяется другой неявно.

Paper-level full sweep пока **не запускается**. Перед pilot необходимо зафиксировать конкретный доступный CNN/DailyMail artifact/revision, split/checksum и deterministic sentence segmentation. Это отдельный gate, поскольку статья задаёт dataset и правило «первые три предложения», но не современный идентификатор ревизии набора данных.

После pinning dataset следующий запуск — короткий GPT-2 Medium pilot на 8 contexts. Только после него запускается полная frozen matrix.

## 17. Pin CNN/DailyMail и deterministic context set

Перед GPT-2 Medium pilot разрешается dataset blocker, заранее отмеченный в frozen paper-reproduction matrix. Первичный paper source задаёт CNN/DailyMail и правило «первые три предложения», но не фиксирует современный artifact/revision и не сообщает train/validation/test split. Поэтому exact historical input snapshot не считается известным.

Stage-3 operationalization фиксируется отдельно в:

```text
configs/reproducibility/cnndm_context_source.json
docs/stages/stage3/cnndm_context_pin.md
```

Pinned source:

```text
abisee/cnn_dailymail@3adf6249f0cc8409a97a4d38471529ef5f7dc496
config: 3.0.0
split: test
artifact: 3.0.0/test-00000-of-00001.parquet
SHA-256: 04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e
rows: 11490
```

Выбор `test` является нашей доэкспериментальной operationalization, а не приписывается статье. Frozen Step-3.5 matrix при этом остаётся byte-for-byte неизменной; её SHA-256 проверяется dataset gate.

Для выбора contexts используется repository-owned `sha256_rank_v1`: строки ранжируются по SHA-256 от фиксированной строки с `seed=1234`, после чего принимаются первые 80 записей, для которых splitter `stage3_news_sentence_splitter_v1` находит не менее четырёх предложений. Первые три предложения образуют context; четвёртое используется только как human-next-sentence diagnostic. Первые 8 accepted records образуют pilot subset.

В Git не сохраняются Parquet и тексты новостей. Commit-able manifest содержит row/article identifiers, selection digest и hashes контекста/следующего предложения. Полные тексты регенерируются локально из pinned artifact.

Go/no-go команды:

```text
python scripts/prepare_stage3_cnndm_contexts.py
python scripts/check_stage3_cnndm_contexts.py
```

Только после `Stage 3 CNN/DailyMail context gate: READY` разрешается загрузка/запуск paper-level GPT-2 Medium pilot.

## 18. GPT-2 Medium author-compatible pilot на 8 frozen contexts

После успешного CNN/DailyMail gate разрешается первый paper-level модельный запуск. Он намеренно остаётся **pilot**, а не частью итоговой Figure-3 curve.

Frozen pilot config:

```text
configs/reproducibility/gpt2_medium_pilot.json
```

Входы уже не выбираются во время запуска:

```text
model:       gpt2-medium (GPT-2 345M)
contexts:    selection_rank 0..7 из frozen CNN/DailyMail manifest
payload:     24 deterministic bits/context, одинаковые для всех методов
finish_sent: true
replicates:  1
```

Четыре predeclared points:

```text
Bins b=3
Huffman exponent=3
Arithmetic tau=0.9, k=300, precision=26
Arithmetic tau=1.0, k=50256, precision=26
```

24-bit payload является только техническим pilot budget. Pinned repository не содержит оригинального Figure-3 batch driver и exact random-message length; поэтому этот budget не выдаётся за paper parameter и не переносится автоматически в full sweep.

Public Harvard `finish_sent=True` завершает предложение greedy top-1 continuation **после** окончания payload. Комментарий в `run_single.py` отдельно предупреждает, что statistics относятся к non-finished/payload prefix. Поэтому runner не смешивает:

```text
bits_per_word_author_stats_prefix
useful_payload_bits_per_total_generated_token
```

Второе поле — только pilot diagnostic. Для paper-compatible curves используется author-compatible definition после отдельной фиксации full-sweep orchestration.

Bins/Huffman продолжают использовать scoped slow-tokenizer + legacy-cache compatibility bridge; reference files не редактируются. Arithmetic идёт через raw GPT-2 Medium и native DynamicCache, как в pinned source.

Pilot gate требует 32/32 успешных calls, exact payload-prefix recovery, sentence-finish на финальном generated token и неизменный reference worktree. Exact sender/retokenized IDs и ранние punctuation tokens сохраняются как diagnostics и не скрываются BPE-repair логикой.

Команды:

```text
python scripts/run_stage3_gpt2_medium_pilot.py
python scripts/check_stage3_gpt2_medium_pilot.py
```

## 19. Targeted investigation после GPT-2 Medium pilot

GPT-2 Medium pilot закрывает technical execution gate, но не разрешает full Figure-3 sweep автоматически. Перед full curve разбираются два наблюдения Step 3.7:

```text
Arithmetic tau=1, k=50256, precision=26 -> mean author KL ~= 0.46 bits/token,
хотя paper описывает near-zero unmodulated point;

5/32 fixed-payload runs содержат sentence-finish token раньше финального token.
```

Для Arithmetic discrepancy замораживается отдельный diagnostic config:

```text
configs/reproducibility/arithmetic_precision_probe.json
```

Он не меняет `paper_reproduction_matrix.json`. На тех же 8 contexts и `tau=1, k=50256` сравниваются precision `26, 32, 40, 48` с общим deterministic 256-bit stream/context и `finish_sent=false`. Long payload нужен, чтобы отдельно анализировать steps без implicit zero lookahead.

Repository-owned instrumented mirror обязан сначала воспроизвести pinned executable `arithmetic.encode_arithmetic` на sentinel run по generated token IDs, author KL и words/bit. Только после parity разрешена instrumentation. External checkout не редактируется.

На каждом step записываются current integer interval width/effective precision, threshold, retained support, rounding residual и three-way KL decomposition: exact author distribution, truncation-only distribution и terminal-fill counterfactual. Counterfactuals никогда не используются для generation и не считаются новым методом.

Sentence issue фиксируется отдельным deterministic audit committed Step-3.7 result. Наличие early boundaries показывает, что public `finish_sent=True` означает «исчерпать фиксированный message, затем закончить предложение», а не «остановить embedding на первой sentence boundary». Поэтому final paper-level driver должен быть определён отдельно до full curve.

Go/no-go этого шага:

```text
python scripts/analyze_stage3_pilot_sentence_shape.py
python scripts/run_stage3_arithmetic_precision_probe.py
python scripts/check_stage3_arithmetic_precision_probe.py
```

`READY FOR REVIEW` означает, что diagnostic достоверно выполнен; это **не** означает автоматическое разрешение full Figure-3 sweep. Разрешение даётся только после интерпретации precision probe и фиксации paper-sentence orchestration.

## 20. Интерпретация Arithmetic precision probe

Step 3.8 завершил диагностическую часть по near-zero special point. Instrumented mirror прошёл exact parity с pinned executable Arithmetic, поэтому выводы не объясняются modern compatibility bridge.

В итоговой интерпретации обязательно различаются:

```text
mean_run_author_kl_bits
    = весь finite-message run, включая terminal implicit-zero look-ahead;

mean_zero_padding_free_author_kl_bits
    = только coding steps с полным look-ahead из реальных secret bits.
```

Фактические clean значения:

```text
precision=26 -> 1.0654e-3 bits/token
precision=32 -> 4.1502e-5 bits/token
precision=40 -> 4.2183e-8 bits/token
precision=48 -> 5.7836e-10 bits/token
```

При этом terminal padding-affected steps дают более 98% summed per-step KL при каждой исследованной precision. Step-3.7 pilot использовал `payload=24` при `precision=26`, следовательно все Arithmetic coding steps этого pilot требовали implicit zero look-ahead. Его `~0.46 bits/token` нельзя использовать как steady-state special-point estimate.

Paper prose anchor `4e-8 nats` всё ещё не считается численно воспроизведённым **при pinned executable precision=26**. Clean precision=40 даёт `~2.92e-8 nats/token`, то есть тот же порядок, но нет оснований утверждать, что Figure 3 использовала precision=40: original batch driver недоступен.

Arithmetic precision discrepancy после этого считается локализованной. Full Figure-3 sweep остаётся заблокирован только до фиксации paper-sentence orchestration: длинный uniform bitstream и stop на первой sentence boundary.
