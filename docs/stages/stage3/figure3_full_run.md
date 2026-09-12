# Этап 3 — Step 3.10: полный author-compatible Figure-3 sweep

**Статус:** runner и протокол заморожены до первого полного запуска.  
**Предусловие:** Step 3.9 прошёл локальный gate `READY FOR FULL FIGURE-3 RUNNER IMPLEMENTATION`.

## 1. Зачем нужен этот шаг

Предыдущие шаги доказали, что pinned Harvard implementation запускается на GPT-2 Medium, а также устранили два pre-sweep blocker'а:

1. короткий finite payload для Arithmetic создавал сильное terminal-искажение из-за implicit zero look-ahead;
2. public `finish_sent=True` не означает остановку на первой границе предложения.

Step 3.9 зафиксировал Stage-3 operationalization paper-фразы `generate one entire sentence given a uniform random message`: длинный равномерно выглядящий битовый поток, author-compatible embedding и немедленный STOP после первого token, удовлетворяющего pinned `utils.is_sent_finish`.

Step 3.10 переносит **ровно эту проверенную orchestration** на всю заранее замороженную Figure-3 сетку.

## 2. Замороженная сетка

Источник сетки не меняется:

```text
configs/reproducibility/paper_reproduction_matrix.json
SHA-256 = 67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b
```

Полный набор содержит 23 точки:

```text
Bins / Block:
    b = 1, 2, 3, 4, 5                         -> 5 точек

Huffman:
    exponent = 1..8                           -> 8 точек

Arithmetic:
    temperature = 0.4, 0.5, ..., 1.2
    top-k = 300
    precision = 26                            -> 9 точек

Special Arithmetic:
    temperature = 1.0
    top-k = 50256
    precision = 26                            -> 1 точка

Итого                                           23 точки
```

Никакая точка не добавляется и не удаляется после просмотра результатов.

## 3. Число запусков

Для каждой точки используются все 80 frozen CNN/DailyMail contexts и три deterministic replicate secret streams:

```text
23 points × 80 contexts × 3 replicates = 5520 runs
```

`80 × 3` — наша заранее зафиксированная Stage-3 operationalization. Основная статья сообщает repeated Monte-Carlo samples, но не публикует точное число Figure-3 samples. Поэтому `5520` **не является утверждением**, что авторы Ziegler et al. выполнили именно столько генераций.

## 4. Secret stream и sentence stopping

Для одного `(selection_rank, replicate)` используется один и тот же 16384-битный deterministic uniform-looking stream для **всех 23 точек**:

```text
stage3-paper-sentence|
seed=1234|
replicate=<0,1,2>|
selection_rank=<0..79>|
counter=<counter>
```

Это позволяет сравнивать методы и tuning points на парных секретных входах.

Каждая генерация работает так:

```text
long real-bit stream
      -> author-compatible token selection
      -> первый token, где pinned utils.is_sent_finish == True
      -> token включается в sentence
      -> STOP
```

Для Arithmetic на каждом измеряемом шаге обязательно должно существовать полное `precision=26` look-ahead окно из **реальных** secret bits. Implicit zero padding запрещён.

В Step 3.9 safety cap `256 carrier tokens` был только защитой pilot от runaway generation и ни разу не сработал. В полном sweep один `bins_b4` sample дошёл до 256 tokens без `.`/`!`/`?`, поэтому использовать pilot-cap как фактическую границу означало бы искусственно цензурировать длину предложения. Для full run emergency guard увеличен до `1024 carrier tokens`; генерация по-прежнему обязана закончиться **только реальной первой boundary**. Если boundary не появится и к 1024 tokens, run/shard остаётся failure и эксперимент снова требует разбора — sample не обрезается и не включается как предложение.

## 5. Почему full runner использует тот же mirror, что Step 3.9

Файл

```text
scripts/stage3_paper_sentence_core.py
```

зафиксирован exact SHA-256:

```text
b0b52366aece6e5ec406336c699c5a0a9d3d66dae1943e232e9ad367c82ff6c7
```

Full runner отказывается работать, если этот файл изменился. Перед sweep также снова выполняются fixed-message parity sentinels против pinned Harvard functions для Bins, Huffman и двух representative Arithmetic points.

После полного sweep дополнительно проверяется **32-run continuity** с Step 3.9: replicate 0, первые 8 contexts и четыре representative points обязаны воспроизвести generated token IDs, confirmed payload и author metrics Step 3.9.

Таким образом, массовый runner не вводит новую algorithmic semantics после успешного pilot.

## 6. GPT-2 Medium provenance

Author helper по-прежнему вызывает `gpt2-medium` без явного `revision=`. До дорогостоящего sweep runner считывает resolved Hugging Face commit из реально загруженных model/tokenizer objects.

Hard requirement:

```text
resolved model revision существует;
legacy Bins/Huffman phase и raw Arithmetic phase
разрешились в один и тот же revision.
```

Resolved revision сохраняется в `run_state.json`. Это не меняет Harvard helper и не подменяет модель; оно закрывает provenance gap перед массовым экспериментом.

## 7. Resume и shard storage

5520 runs не записываются одним гигантским JSON. Единица checkpoint — одна точка × один replicate:

```text
results/stage3/paper_reproduction/figure3_full/
  run_state.json
  summary.json
  figure3_points.csv
  shards/
    bins_b1/
      replicate_0.jsonl       # 80 contexts
      replicate_1.jsonl
      replicate_2.jsonl
    ...
```

Всего ожидается:

```text
23 × 3 = 69 completed shards
```

Shard пишется во временный файл и атомарно переименовывается только после 80 runs и structural validation. При повторном запуске корректные completed shards пропускаются. Поэтому после прерывания достаточно снова выполнить ту же команду — теряется максимум незавершённый shard, а не весь sweep.

Файлы `*.tmp` и `*.failed.jsonl` считаются локальными диагностическими артефактами и не должны коммититься как успешный результат.

## 8. Hard execution gate для каждого run

Каждый run должен одновременно удовлетворять:

```text
terminal_reason == sentence_boundary
первая sentence boundary является последним generated token
для Bins/Huffman: payload_bits_confirmed > 0
для Arithmetic: payload_bits_confirmed >= 0
Arithmetic implicit zero look-ahead == false
```

Восстановление payload через обычный текстовый transport **сохраняется, но не является hard gate Figure-3 run**. Это отдельная reliability-диагностика. Public Bins/Huffman decoders используют эвристический BPE repair и на полном sweep могут не восстановить отдельные transported samples даже тогда, когда generation-side выбор токенов, KL и payload accounting корректны.

Это различие принципиально: Figure 3 оценивает information-theoretic quantities `KL` и `bits/word` для сгенерированных предложений. Если выбрасывать samples только потому, что эвристический author decoder не смог починить BPE после text transport, итоговая кривая станет условной на decoder success и получит selection bias. Поэтому такие случаи остаются в Figure-3 aggregation, а поля `recovery.*` отдельно показывают exact-prefix rate, BPE warnings и decoder exceptions.

Step 3.9 при этом **не пересматривается**: его 32-run pilot был strict gate и доказал, что выбранный first-boundary orchestration в representative points может пройти end-to-end recovery. Поправка относится только к массовому Figure-3 estimator после обнаружения редких author transport failures.

Exact token roundtrip также сохраняется как diagnostic, а не как отдельный hard failure.

Pinned `external/NeuralSteganography` обязан остаться clean и на commit `14e982...`.

### 8.2. Поправка после достижения 256-token pilot cap

После принятия первых resumable shards полный sweep дошёл до `bins_b4 / replicate_0`, где `selection_rank=78` не встретил token, удовлетворяющий pinned `utils.is_sent_finish`, за первые 256 carrier tokens. Secret stream при Bins b=4 при этом был далеко не исчерпан: 256 tokens требуют только 1024 из 16384 доступных bits. Значит причиной был именно engineering safety cap, а не payload exhaustion.

Замороженная научная семантика говорит «остановиться на **первой** author boundary». Поэтому нельзя ни считать token 256 концом предложения, ни выбрасывать такой sample только из-за его длины: оба варианта изменяют estimator в зависимости от sentence length. Full-run guard увеличен до 1024 tokens. Уже сохранённые shards остаются совместимыми, потому что каждый принятый record в них завершился настоящей boundary **до** старого cap; их deterministic token sequence не меняется. Step 3.9 и его 32-run gate не пересматриваются.

Это operational amendment, а не подбор параметра статьи. Frozen 23-point matrix, GPT-2 Medium, contexts, three replicates, secret-stream prefixes, author token-selection logic, boundary predicate, KL и payload accounting не изменяются. Итоговая aggregation отдельно считает, сколько предложений оказалось длиннее 256 tokens и какова максимальная carrier length.


### 8.3. Поправка после zero-payload Arithmetic samples

После завершения всех legacy shards первый Arithmetic shard `arithmetic_t0.4_k300 / replicate_0` содержал два предложения, которые корректно остановились на первой author boundary, но к этому моменту арифметический интервал ещё не зафиксировал ни одного общего префиксного бита: `payload_bits_confirmed = 0`. Это допустимое состояние арифметического кодера: carrier token уже выбран по текущему interval/look-ahead, но подтверждённый secret prefix может оставаться пустым.

Такие samples нельзя отбрасывать из Figure-3 Monte-Carlo estimator. Их исключение означало бы условие `payload > 0` и систематически завышало бы средний `bits/word`. Поэтому для Arithmetic нулевой confirmed payload считается валидным результатом sentence-level run и даёт `bits_per_word_author = 0`. Отрицательный payload по-прежнему невозможен и считается ошибкой; для Bins/Huffman нулевой payload остаётся ошибкой, потому что каждый сгенерированный carrier token непосредственно кодирует хотя бы один бит.

Transport recovery для пустого confirmed prefix не имеет содержательного смысла и поэтому такие Arithmetic samples исключаются из знаменателя exact-prefix recovery rate и отдельно считаются как `zero_payload_diagnostic`. KL/NLL и длина предложения остаются полностью определёнными и включаются в агрегацию. Step 3.9 не пересматривается: representative pilot просто не встретил zero-payload case.

Frozen matrix, модель, contexts, replicates, secret streams, token selection, boundary predicate, KL semantics и payload accounting не изменяются. Поправка меняет только ошибочное предположение full-run validator о том, что каждый Arithmetic sentence обязан подтвердить хотя бы один бит.

### 8.1. Поправка после первого full-run attempt

Первый Step-3.10 запуск остановился **до принятия первого completed shard** на `bins_b1 / replicate_0`: 80 generation samples были получены, но два record не прошли exact-prefix recovery после сообщений public decoder `Unable to fix BPE error`. Генерационный Figure-3 инвариант при этом не был причиной ошибки.

Поэтому до принятия первого full-run shard протокол уточнён:

```text
generation validity     -> hard gate
transport recovery      -> recorded reliability diagnostic
paper matrix / contexts / model / secret streams / sentence stop / KL semantics -> без изменений
```

Неудачный `*.failed.jsonl` остаётся диагностикой предыдущей попытки до тех пор, пока соответствующий shard не будет успешно пересоздан; после успешной atomic write runner удаляет этот failed-файл автоматически.

## 9. Агрегация Figure-3 points

На каждой точке планируется 240 scheduled runs. Обычно все 240 завершаются first-boundary sentence; если Arithmetic исчерпывает финальный 8192-token guard без boundary, такой scheduled run сохраняется как `sentence_termination_failure` и не считается успешно завершённым предложением.

Primary Stage-3 estimator заранее фиксируется как:

```text
mean(metric) = обычное среднее по successfully terminated sentence runs
SE_runs      = sample_std(metric, ddof=1) / sqrt(n_success)
```

Для контроля зависимости трёх replicates от одного context дополнительно хранится diagnostic:

```text
для каждого context сначала mean по успешно завершившимся replicates
затем SE_context = sample_std(context means, ddof=1) / sqrt(number of contexts with >=1 success)
```

Оба SE сохраняются. Primary Figure-3 reproduction использует `SE_runs`; context-clustered вариант нужен для прозрачности нашего operationalized sampling design.

Основные агрегируемые значения:

```text
bits_per_word_author
kl_q_stego_to_p_lm_bits_author
avg_nll_nats_author
carrier_tokens
payload_bits_confirmed
```

`bits_per_word_author` следует executable author terminology: знаменателем является число GPT-2 carrier tokens.

## 10. Очень важное разделение Step 3.10 и Step 3.11

Step 3.10 проверяет **целостность исполнения**, но не заставляет результаты совпадать со статьёй.

То есть такие утверждения, как

```text
Arithmetic имеет меньший KL, чем Bins/Huffman;
минимум находится около 4 bits/word;
special point имеет near-zero KL;
```

**не являются hard gate runner'а**.

Если все 5520 scheduled outcomes корректно классифицированы как completed sentence или документированный `sentence_termination_failure`, но опубликованный тренд не воспроизводится, Step 3.10 всё равно считается технически успешным. Научная интерпретация, сравнение с Figure 3 и классификация `trend / partial / not reproducible` выполняются только на **Step 3.11**.

## 11. Команды

До GPU run:

```bash
python scripts/check_stage3_figure3_full.py --preflight
```

Ожидается:

```text
points: 23
shards: 69
runs per point: 240
total runs: 5520
Step-3.9 pilot gate: READY
paper-sentence core hash: PINNED
reference checkout: PINNED + CLEAN
Stage 3 Figure-3 preflight: READY TO RUN
```

Полный запуск:

```bash
python scripts/run_stage3_figure3_full.py
```

Команда является resumable. После interruption запускается та же команда ещё раз.

После завершения:

```bash
python scripts/check_stage3_figure3_full.py
python scripts/aggregate_stage3_figure3.py
```

Итоговый execution gate:

```text
Stage 3 Figure-3 full-run gate: READY FOR STEP 3.11 INTERPRETATION
```

Только после этого Step 3.11 строит численное сопоставление с публикацией.

## 11. Long-context Arithmetic cache compatibility amendment

После увеличения full-run emergency sentence guard до 1024 токенов первый Arithmetic sweep обнаружил отдельную runtime-проблему длинного контекста: CUDA завершился assertion `indexSelectSmallIndex: srcIndex < srcSelectDimSize` внутри GPT-2 forward.

Причина локализована в modern-cache ветке pinned Harvard `utils.limit_past`. Исторический код работал со stacked cache layout

```text
[2, batch, heads, sequence, head_dim]
```

и поэтому срез `[:, :, :, -1022:]` ограничивал именно sequence dimension. В современном tuple/DynamicCache каждый `key` / `value` имеет layout

```text
[batch, heads, sequence, head_dim]
```

но публичная tuple-ветка сохраняет тот же срез `key[:, :, :, -1022:]`. В таком layout он ограничивает `head_dim`, а не sequence length. На коротких Step-3.4/3.7/3.9 запусках это незаметно; длинная генерация позволяет KV cache превысить GPT-2 Medium learned position table из 1024 позиций, после чего CUDA embedding lookup падает.

Step 3.10e не меняет Arithmetic distribution или coding rule. Для full-run author-compatible Arithmetic вводится только compatibility shim:

```text
modern key/value [B,H,T,D]
        -> оставить последние 1022 значения по T
        -> [B,H,min(T,1022),D]
```

Идентичный shim временно подставляется в public Arithmetic decoder только при вычислении не-гейтящей transport/recovery диагностики. `scripts/stage3_paper_sentence_core.py` Step 3.9 остаётся byte-for-byte неизменным и по-прежнему pinned по SHA-256. Normalized benchmark также не меняется.

До достижения 1022 cached positions shim является identity по значениям и форме, поэтому short-run fixed-message parity остаётся прежней. Для каждого Arithmetic sample отдельно сохраняются `encode_trim_events`, `decode_trim_events` и максимальная увиденная sequence length. Это позволяет в Step 3.11 явно показать, сколько Figure-3 samples фактически потребовали long-context compatibility.


## 12. Adaptive Arithmetic sentence-guard amendment

После исправления long-context cache полный sweep продолжился на `arithmetic_t0.4_k300 / replicate_1`. Три из 80 deterministic trajectories дошли до full-run guard `1024` tokens без token, для которого pinned `utils.is_sent_finish` вернул бы `True`. Это не cache failure и не secret-stream exhaustion: причина — очень длинная low-temperature trajectory до первой punctuation boundary.

`1024` никогда не был paper parameter или определением предложения; это только инженерный runaway guard. Поэтому считать token 1024 концом предложения или выбрасывать такие samples нельзя: оба решения цензурировали бы Monte-Carlo estimator по длине carrier sentence.

Для Arithmetic full-run вводится узкая adaptive policy: первый запуск остаётся с guard `1024`; если termination reason ровно `max_generated_tokens`, тот же sample детерминированно запускается заново с тем же context и тем же secret-stream prefix при guards `2048`, затем `4096`, затем `8192`. До прежнего cap token sequence обязана быть той же, поскольку кодирование deterministic. Принимается только реальная первая boundary. Если именно финальный guard `8192` исчерпан без boundary, sample классифицируется как `sentence_termination_failure`: он сохраняется как scheduled outcome, исключается из sentence-level Figure-3 means и учитывается в termination failure rate. Bitstream exhaustion, implicit zero look-ahead и любые другие неожиданные non-boundary/runtime failures остаются hard failure.

Bins/Huffman уже завершены и не пересчитываются. Все ранее принятые shards остаются валидными, поскольку каждый их record уже завершился реальной boundary. Отдельно сохраняются число Arithmetic samples, потребовавших escalation, число повторных attempts и максимальный фактически использованный guard. Это diagnostic исполнения, а не новая метрика метода.


## 13. Final sentence-termination failure policy (Step 3.10g)

Повторный `arithmetic_t0.4_k300 / replicate 1` показал, что три trajectories (`selection_rank` 49, 50, 58) не достигают pinned `utils.is_sent_finish` даже после последовательных guards `1024 -> 2048 -> 4096 -> 8192`. Анализ сохранённого failed shard показал не обычные длинные предложения, а degeneration/repetition loops (`World`, `The`, `Trump`) без `.`, `!`, `?`. Поэтому дальнейшее увеличение guard было бы постфактум tuning и не восстанавливало бы неизвестный original Figure-3 driver.

Окончательная политика Step 3.10:

```text
реальная first boundary до 8192
    -> status = ok
    -> sentence входит в Figure-3 KL/BPW/NLL means

финальный 8192-token guard исчерпан без boundary
    -> status = sentence_termination_failure
    -> boundary не подменяется
    -> author decoder / text-transport recovery не запускается
    -> partial trajectory НЕ входит в sentence-level KL/BPW/NLL means
    -> scheduled outcome остаётся в shard и termination failure rate

любая другая ошибка
    -> hard failure, shard не принимается
```

Для каждой точки summary обязан хранить `scheduled_run_count`, `terminated_sentence_run_count`, `termination_failure_count`, `termination_success_rate`. Если failures существуют, Figure-3 mean/SE являются **conditional estimates over successfully terminated sentences**; это условие обязательно переносится в Step 3.11 и итоговый reproducibility report.

Эта поправка не меняет frozen 23-point matrix, GPT-2 Medium, contexts, replicates, secret streams, Arithmetic token selection, boundary predicate, KL semantics или normalized benchmark. Она фиксирует недокументированный крайний случай публичной author-compatible orchestration вместо попытки скрыть его более высоким лимитом.
