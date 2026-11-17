# Stage 7 — нормативный протокол эмпирического стегоанализа (v1.0)

Статус: **зафиксирован методологически; реализация выполняется по checkpoint**. Данный документ расширяет §12/§23 benchmark specification v1.0 и не меняет Stage 6 или ADR-0012/0018/0020/0026/0027.

## 1. Научные вопросы и рамки

- Можно ли отличить готовые стеготексты от обычных, независимо сэмплированных текстов *той же* LM при согласованных внешних условиях?
- Как обнаружимость различается между 15 Stage-6 operating points на Llama-3.2-3B и Qwen3-4B-Base?
- Как конечная precision и токенизация влияют на наблюдаемые KL/TVD и end-to-end надежность? Отдельно, без пересчёта Stage-6 benchmark metrics.

Постановка противника: **passive text-only, prompt-unaware**. На входе только продолжение в виде транспортируемого текста, без source prompt, token IDs, метрик, ключей, секретов, статуса декодирования и названия метода. Вывод модели — непрерывный stego score. Оцениваются готовые тексты, а не внутренние распределения. Около 0.5 ROC-AUC не является доказательством необнаружимости.

## 2. Нормативные входы

- Frozen Stage 6: `results/stage6/dataset_v1.0_manifest.json`, 1800 runs (1703 ok, 97 hard decode failures), canonical `results/runs/<run_id>/` локально.
- Источник промптов: `abisee/cnn_dailymail`, config `3.0.0`, split `test`, revision `3adf6249f0cc8409a97a4d38471529ef5f7dc496`, pinned Parquet SHA-256 `04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e`.
- Рейтинг/сегментация: `sha256_rank_v1`, `seed=1234`, `stage3_news_sentence_splitter_v1`, первые 3 предложения, не менее 4 предложений. Первые 80 ранее отобранных контекстов проверяются побайтно через идентификаторы и хеши frozen manifest.
- Статус рабочих точек: 14 streaming с `carrier_tokens=256` и RRC отдельно `payload_bits=1024`, `precision_margin=40`, `verify_termination=true`. `temperature=1.0`, common top_k отсутствует, common top_p=1, exclude special tokens, KV-cache. Для Qwen исключаются output-only IDs как в benchmark v1.0.

## 3. Prompt groups и split

Отдельные article groups: Stage5 ranks 0..7 (нигде не используются); Stage6 external test ranks 8..31, 24 групп; Stage7 ranks 32..159, 128 групп. Для назначения Stage7 split вычисляется SHA-256 строки `VKR-BENCHMARK-STAGE7-v1|split|seed=701|article_id=<id>`, сортировка по hex digest, затем по article_id; первые 96 — train, следующие 32 — validation. Stage 6 — весь test, не участвует в выборе гиперпараметров.

Все LM, methods, operating points, repeats и controls одной исходной статьи находятся в одной partition. Из train/val/test исключаются точные дубликаты `article_id`, `context_sha256`, normalized prompt text; near-duplicate audit проводится до генерации. Изменения split после первого обучения запрещены без новой версии протокола. Промпты Stage 7 сохраняются локально без включения полного текста в Git; в Git — только источник, ids, хеши, split assignments.

## 4. Матчинг cover/stego

Положительный класс: stegotext (включая сохранённый `transport_text` hard failures); отрицательный: **новая независимая** контрольная генерация. Каждой stego-записи сопоставляется один cover той же LM/revision, того же prompt, того же числа **sender carrier tokens** и того же алгоритма обработки исходных logits в FP32 `P_reference`.

Контрольный токен выбирается как `x_t ~ P_reference,t` без стегометода/секретов. Режим: CDF по возрастанию token ID, накопление FP64; `u = ControlRandomSource(seed).random()`, `x = min{i: CDF_i > u*sum(P_ref)}`. Случайность control отделена от randomness метода и секретного SHAKE256. Seed = первые 8 байт SHA-256 (`big endian`) строки `VKR-BENCHMARK-STAGE7-v1|role=cover|model=<model_id>|article=<article_id>|point=<method_id>/<point_id>|replicate=<replicate_id>`.

Фиксированное завершение может обрывать предложение. **Не дописывать** ни cover, ни stego. Сохранять raw text после точного `decode_tokens(..., clean_up_tokenization_spaces=False)` и диагностический `TextChannel` roundtrip без исправлений.

RRC main: cover length соответствует длине соответствующей RRC отправленной последовательности; `unmatched-length` дополнительный эксперимент использует отдельно сгенерированный 256-token cover. Его AUC зависит от принятой политики длины и не подменяет основную метрику Stage 8.

## 5. Секреты и повторы

Для новых Stage7 train/val запусков `secret_id = stage7-det-<prompt_id>-r01`. Все методы и обе LM для данного prompt получают одинаковую последовательность SHAKE256 (домен не изменяется). Внутренний method RNG/key сохраняет существующие normalized adapter semantics, но материал должен быть воспроизводимо получен из `stage7` домена и идентификаторов prompt/method/replicate. Test retains original Stage6 secret IDs/keys and number of replicates (Llama 3, Qwen 2). Контрольный seed никогда не выводится из secret bits/keys.

## 6. Размер и баланс

New train/val: 128 groups × 15 points × 2 LM = 3840 stego и 3840 cover. Frozen test: 1080 stego Llama + 720 stego Qwen и столько же новых cover. Итого 5640 stego + 5640 cover = **11280** records; 152 independent article groups across train/validation/test. Требование строгого баланса 1:1 в каждой `(LM, point, partition)`.

При невозможности извлечь stage6 error transport_text или корректную sender carrier length сборка теста завершается ошибкой — не исключать hard failures молча. После проверки наличие/хеши текстовых файлов подтверждаются sample manifest.

## 7. Детекторы и обучение

**Основной** `FacebookAI/roberta-base`, 2 класса, отдельный checkpoint на LM, text-only input, max_length=512 RoBERTa tokens, truncate-right и dynamic padding. До GPU запуска закрепить точные revision/hash model weights+tokenizer. AdamW, learning rate 2e-5, weight_decay 0.01, max 4 epochs, 10% warmup, linear scheduler, effective batch=32 (microbatch 4 × grad accum 8), cross-entropy, seeds 711/712/713. Выбор одного основного checkpoint по pooled *validation* ROC-AUC, tie-break — более ранняя эпоха, затем меньший seed. Early stopping patience 2. Прочие seeds — sensitivity, без ensemble. Параметры не изменять по Stage6 test.

**Baseline:** TF-IDF char ngrams 3–5, min_df=2, max_features=100000, sublinear_tf=true; LogisticRegression(C=1.0, max_iter=2000, class_weight=None). TF-IDF fit только train. Никакие `run_id`, PPL, raw token IDs или метрики не поступают на вход обоих детекторов.

## 8. Оценка и неопределённость

Основной результат: per-point ROC-AUC на external test с одной и той же score-функцией pooled RoBERTa для каждой LM (30 rows). Pooled ROC-AUC — вторичный. Значения ROC-AUC и prediction scores сохраняются с точными идентификаторами sample, detector revision и manifest hash.

Cluster bootstrap percentile CI 95%, 2000 resamples, seed 731; кластер = исходная статья (`article_id`), и все пары/повторы данной статьи ресэмплируются совместно. При отсутствии обоих классов в bootstrap replicate — повторить выборку по заранее фиксированному алгоритму. Отдельно сообщать разброс от training seeds, не смешивать его с тестовой неопределённостью.

Дополнительные metrics: accuracy, balanced accuracy, precision, recall, F1, PR-AUC, confusion matrix и ROC curves. Один classification threshold на LM подбирается по pooled validation balanced accuracy, не по test и не отдельно для точек. Аномалии ROC-AUC ниже 0.5 диагностируются без post-hoc инвертирования score.

Quality/artefact controls: length-only classifier, shuffled-label baseline (только train), окончания `.?!`, длина в словах/символах, RoBERTa token lengths, доля truncation по классам. Unmatched-length RRC — только диагностический результат, отдельно от Stage8 main.

## 9. Implementation-factor protocol

Выбрать 16 валидационных prompt groups детерминированно с seed 741 до просмотра результатов; повторять одинаковые секреты и промпты между вариантами. AC precision={16,32,48} при top_k=2048, 256 токенов, обе LM. RRC precision_margin={0,10,20,40,60} при fixed-payload 1024 бит, обе LM. FP64 vs Decimal границы — offline diagnostic, не новый normalized RRC. Tokenization study — read-only анализ всех 1800 frozen runs, как минимум точность `token IDs → text → retokenization → IDs`, первая позиция рассинхронизации и её связь с exact/soft/hard decode. Ошибки измерений/декодирования не исключать молча. KL_ref_to_stego и KL_stego_to_ref хранить раздельно, не подменяя Stage6 значения.

## 10. Изоляция, артефакты и критерии готовности

Не записывать в Stage6 canonical `results/runs/` и `results/summary.parquet` (их SHA заморожен), конфигурации и manifests. Новые Stage7 per-run artifacts: `results/stage7/stego_runs/`, `results/stage7/cover_runs/`, локальная dataset text в `data/stage7/dataset/`; модельные веса в `models/detectors/`, все gitignored. Commit только конфигурации, split/manifests без текста, агрегаты, отчёты, скрипты и тесты. `README.md` остаётся пустым.

Сначала пройти: pinned dataset hash, совпадение Stage3 frozen prefix, пересечения article groups, 1800/1800 Stage6 texts включая 97 failures, сбалансированность, smoke control sampling. Затем training/evaluation. При финализации: 30 ROC-AUC + CI, сохранённые checkpoints, implementation-factor reports, presentation Typst, reproducibility manifests, `pytest -q`, unchanged Stage6 hashes. **Парето только Stage 8**.
