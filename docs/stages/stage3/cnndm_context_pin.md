# Этап 3 — pin CNN/DailyMail для paper-level reproduction

**Статус до локального запуска:** источник и процедура зафиксированы; фактический Parquet и 80 контекстов должны быть проверены и сгенерированы на машине эксперимента до GPT-2 Medium pilot.  
**Неизменяемая матрица шага 3.5:** `configs/reproducibility/paper_reproduction_matrix.json`, SHA-256 `67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b`.  
**Машиночитаемый pin:** `configs/reproducibility/cnndm_context_source.json`.

## 1. Что сообщает первоисточник

Ziegler et al. (2019) сообщают, что в paper-level экспериментах используется GPT-2 345M и CNN/DailyMail; первые три предложения каждой новости служат контекстом, после чего стегометод генерирует одно предложение по равномерному случайному сообщению. Supplementary material уточняет, что human evaluation использует те же данные, что и KL evaluation, но **ни основной текст, ни supplementary material не называют конкретный train/validation/test split и не дают современный revision/checksum dataset artifact**.

Поэтому exact historical dataset snapshot восстановить из статьи невозможно. На Этапе 3 нельзя выдавать выбранный нами современный artifact или split за дословную авторскую конфигурацию.

Paper source:

```text
https://aclanthology.org/D19-1115/
https://aclanthology.org/D19-1115.pdf
```

## 2. Зафиксированная operationalization

Для воспроизводимого современного запуска фиксируется доступный immutable artifact:

```text
repository:  abisee/cnn_dailymail
revision:    3adf6249f0cc8409a97a4d38471529ef5f7dc496
config:      3.0.0
split:       test
rows:        11490
artifact:    3.0.0/test-00000-of-00001.parquet
size:        30000471 bytes
SHA-256:     04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e
```

Hugging Face commit `3adf624...` explicitly adds the Version 3.0.0 Parquet files and records the test artifact as the above SHA-256/size. The dataset card describes Version 3.0.0 as the cased, non-anonymized summarization representation and reports the standard split sizes.

Sources:

```text
https://huggingface.co/datasets/abisee/cnn_dailymail/commit/3adf6249f0cc8409a97a4d38471529ef5f7dc496
https://huggingface.co/datasets/abisee/cnn_dailymail
```

### Почему `test`

Это **наша заранее зафиксированная operationalization**, а не известный факт об оригинальном запуске. Paper-level задача здесь является evaluation/reproduction, а не обучением модели на CNN/DailyMail; поэтому выбирается стандартный test split. Это решение фиксируется до GPT-2 Medium запуска и больше не меняется в ответ на полученные метрики.

Если позднее будет найден первичный артефакт/код, однозначно указывающий другой split, текущий результат сохраняется как отдельный reproduction profile, а не переписывается задним числом.

## 3. Почему не коммитим тексты статей

Полный Parquet и извлечённые тексты контекстов не нужны в Git. Репозиторий сохраняет только:

- точный dataset repository/revision/path/checksum;
- алгоритм выбора строк;
- `row_index` и source `article_id`;
- SHA-256 извлечённого трёхпредложного контекста;
- SHA-256 следующего human-written предложения;
- размеры/диагностику segmentation.

Сам Parquet кешируется локально в:

```text
data/stage3/cnndm/cache/test-3.0.0.parquet
```

а текст 80 извлечённых контекстов — в:

```text
data/stage3/cnndm/generated_contexts.jsonl
```

Оба пути исключены из Git. Любой исследователь может восстановить те же строки из pinned artifact и проверить их по сохранённым hash.

## 4. Детерминированный выбор 80 статей

Матрица шага 3.5 заранее требует 80 contexts для full curve и 8 для pilot. Чтобы выбор не зависел от порядка вызова RNG-библиотеки, применяется `sha256_rank_v1`.

Для каждого zero-based `row_index` вычисляется:

```text
SHA256("stage3-cnndm-test-v1|seed=1234|row=<row_index>")
```

Все 11490 строк сортируются по digest по возрастанию. Затем они просматриваются в этом порядке; принимаются первые 80 статей, из которых deterministic splitter извлекает не менее четырёх предложений. Первые 8 **принятых** статей являются pilot subset.

Минимум в четыре предложения нужен потому, что:

```text
sentences 1..3 -> paper context
sentence 4     -> human next-sentence diagnostic
```

Следующее human-written предложение не подмешивается в generation и сохраняется только через hash для будущей диагностики.

## 5. Sentence segmentation

Статья определяет семантическое правило «первые три предложения», но не называет tokenizer/segmenter. Поэтому вводится собственный frozen profile:

```text
stage3_news_sentence_splitter_v1
```

Он:

1. удаляет leading/trailing whitespace;
2. схлопывает внутренние Unicode whitespace в один ASCII space;
3. сохраняет case и пунктуацию;
4. определяет границы по `.`, `!`, `?` с поддержкой закрывающих кавычек/скобок;
5. защищает зафиксированный набор news abbreviations, инициалы и decimal points;
6. соединяет первые три предложения одним пробелом.

Это **не утверждение**, что авторы использовали тот же sentence tokenizer. Идентификатор splitter нужен как раз для того, чтобы эта неизвестность была локализована и воспроизводима.

## 6. Локальная подготовка

После установки snapshot запускается:

```bash
python scripts/prepare_stage3_cnndm_contexts.py
```

Скрипт обязан:

1. скачать immutable Parquet по pinned revision, если cache отсутствует;
2. проверить размер и SHA-256 **до чтения**;
3. подтвердить 11490 rows;
4. сформировать 80 accepted contexts;
5. записать локальный ignored JSONL с текстом;
6. записать commit-able manifest без article text:

```text
results/stage3/paper_reproduction/cnndm_context_manifest.json
```

Повторный запуск на тех же байтах должен дать byte-identical manifest и локальный JSONL.

Затем gate:

```bash
python scripts/check_stage3_cnndm_contexts.py
```

проверяет source config, неизменность frozen matrix шага 3.5, dataset checksum, manifest, context hashes и 8/80 selection.

## 7. Go / no-go

GPT-2 Medium pilot разрешён только если:

```text
dataset artifact SHA-256 matches             PASS
split row count == 11490                     PASS
frozen Step-3.5 matrix unchanged             PASS
80 context metadata records frozen           PASS
8-record pilot prefix frozen                 PASS
local context text hashes match manifest     PASS
```

После этого dataset blocker из шага 3.5 считается **разрешённым отдельным provenance artifact**, при этом сам frozen `paper_reproduction_matrix.json` не переписывается.

Следующий подэтап — GPT-2 Medium pilot на восьми frozen contexts. Pilot не имеет права менять набор контекстов или parameter matrix по результатам запуска.
