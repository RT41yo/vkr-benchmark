# VKR Benchmark

Экспериментальная инфраструктура для воспроизводимой многокритериальной оценки методов генеративной лингвистической стеганографии.

Текущий этап: **Этап 3 — проверка воспроизводимости и соответствия Bins, Huffman, Arithmetic Coding**.

## Статус Этапа 3

Технический author-compatible smoke gate пройден для Bins, Huffman и Arithmetic Coding на pinned Harvard reference. Paper-level matrix и CNN/DailyMail context set зафиксированы. Precision discrepancy локализована, а Step 3.9 paper-sentence pilot прошёл 32/32: генерация останавливается на первой boundary, payload восстанавливается, Arithmetic zero look-ahead отсутствует. Step 3.10 фиксирует resumable full Figure-3 runner: 23 frozen points × 80 contexts × 3 replicates = 5520 запланированных author-compatible outcomes с атомарными shards. Transport recovery через эвристический author BPE repair отделён от Figure-3 execution gate. Full-run initial emergency guard равен 1024, а Arithmetic cap-hit samples детерминированно перезапускаются с guards 2048/4096/8192. Анализ `arithmetic_t0.4_k300 / replicate 1` показал, что ranks 49/50/58 даже после 8192 токенов остаются в дегенеративных repetition loops без `.`, `!` или `?`; это теперь фиксируется как `sentence_termination_failure`, а не маскируется дальнейшим увеличением guard. Такие scheduled outcomes сохраняются и входят в failure-rate, но исключаются из sentence-level Figure-3 KL/BPW/NLL means. Scientific claims статьи не являются execution gate и будут интерпретироваться только после завершения sweep на Step 3.11.
> Step 3.10 full Figure-3 sweep keeps transport recovery as a diagnostic, uses a 1024-token initial emergency sentence guard with deterministic Arithmetic-only escalation to 2048/4096/8192, treats zero-confirmed-payload completed Arithmetic sentences as valid (`bits/word = 0`), and classifies final-8192 non-termination as an explicit reproducibility outcome rather than inventing a sentence boundary. Full execution also records the narrow Arithmetic long-context cache compatibility shim; Step-3.9 core and normalized methods remain unchanged.


## Принципы архитектуры

- LM и токенизатор изолированы в `lm/`.
- Каноническое `P_reference` строится в `distributions/` общей инфраструктурой.
- Стегометоды находятся в `methods/` и не владеют LM, токенизатором или общими generation policy.
- Реализация Method adapter сразу допускает stateful encoder/decoder sessions, чтобы не создавать архитектурный долг перед подключением RRC.
- Источники случайности разделены в `randomness/`.
- Проверка надежности проходит через обычный текстовый канал в `transport/`.
- Метрики считаются внешним `metrics/`, а не самими методами.
- Оркестрация запуска находится в `runner/`, трассировка — в `trace/`, сохранение результатов — в `storage/`.

Спецификация v0.1 остается неизменной. Выявленные уточнения для будущей v1.0 накапливаются в `specification/benchmark_specification_v0.1_notes.md`.

## Локальные модели

Репозиторий рассчитан на использование уже скачанных моделей без повторной загрузки. Локальные веса можно оставить в существующей структуре:

```text
models/
├── llama-3.2-3b/
└── qwen3-4b-base/
```

`models/` исключена из Git. В `configs/models/` хранятся относительный `local_path`, логический Hugging Face model id и зафиксированный revision. Это позволяет загружать модель с диска и одновременно сохранять воспроизводимое происхождение модели в run config.

Модели могут физически оставаться в соседнем `lm_probe/models/` и подключаться в этот репозиторий символическими ссылками `models/llama-3.2-3b` и `models/qwen3-4b-base`. Технический snapshot окружения этапа 1 (`python_version.txt`, `requirements_initial.txt`, `system_info.txt`, `nvidia_smi.txt`, `model_revisions.txt`) сохраняется отдельно в `environment/stage1_reference/` и не перезаписывается. Файлы с access token/credentials (например, `hf.txt`) в Git не сохраняются.

## Реализовано на текущем шаге

Базовое ядро и общий LM/distribution слой:

- immutable `ReferenceDistribution` (FP32) и `StepContext`;
- детерминированный `token_order = (-probability, token_id)`;
- tagged `DistributionInfo` для `Q_stego`;
- lifecycle `StegoMethod -> EncoderSession / DecoderSession`;
- SHAKE256 secret stream и изолированные method/control RNG streams;
- `LMAdapter`, `LMState`, `TokenSpace`;
- `HFCausalLMAdapter` для локальных Hugging Face causal LM с тем же `DynamicCache + attention_mask + cache_position + logits_to_keep=1` путем, который был проверен в `lm_probe` v0.2;
- `ReferenceDistributionBuilder` с masking → temperature → FP32 softmax → top-k → renormalize → top-p → renormalize;
- программное исключение special-token и output-only ID без hardcoded model IDs; наличие tokenizer ID определяется по фактическому `tokenizer.get_vocab()`;
- явная политика `prompt_add_special_tokens=true` для Llama и Qwen, согласованная с `lm_probe` v0.2;
- отдельный GPU smoke test `scripts/check_lm_adapter.py`, включая точное сравнение логитов adapter path с независимым Stage-1-style путем до и после одного cached step;
- `MethodEnvironment` с фиксированным `V_allowed`, не раскрывающий методу LM/tokenizer;
- normalized `BinsMethod`: фиксированное разбиение `V_allowed`, streaming encoder/decoder sessions, изолированный method RNG и exact explicit `Q_stego`;
- `LMAdapter.encode_text()` и `TextChannel` для обязательного обычного текстового transport `tokens → text → retokenize` без BPE-repair эвристик;
- минимальный `runner/streaming.py`, который независимо строит encoder- и decoder-side LM/KV-cache пути;
- `RecordingSecretSource` для фиксации всех битов, реально прочитанных методом из secret stream; runner отдельно хранит `read_secret_bits` и подтвержденный `payload_secret_bits`;
- end-to-end smoke script `scripts/check_bins_e2e.py` для реальной Llama/Qwen;
- normalized `HuffmanMethod`: per-step top `2**bits_per_word`, детерминированное дерево, переменный `bits_consumed`, decoder и exact explicit `Q_stego`;
- synthetic unit tests Huffman, включая tie-break, variable-length payload, exact Q и roundtrip без GPU;
- Huffman подключён к тому же `runner/streaming.py` и `TextChannel` без отдельного method-specific runner;
- `scripts/check_huffman_e2e.py` для реальной Llama/Qwen с выводом per-step variable payload;
- normalized `ArithmeticMethod`: stateful finite-precision interval, method-internal candidate cutoff/top-k, overlapping secret look-ahead, per-step confirmed payload и exact explicit `Q_stego` из integer interval widths;
- synthetic unit tests Arithmetic Coding, включая integer rounding, candidate cutoff, state persistence, encode/decode symmetry, exact Q и отдельный учет look-ahead против полезного payload;
- общий streaming runner обобщён для методов с look-ahead: `secret_bits_read` больше не отождествляется с полезным `payload_bits`;
- `scripts/check_arithmetic_e2e.py` для реальной Llama/Qwen через тот же `TextChannel`.

- `PromptRegistry` и `data/prompts.jsonl` для загрузки точного prompt по стабильному `prompt_id`;
- единый `ExperimentConfig`, который задает model config, prompt, method params, generation policy, `secret_id` и termination policy;
- общий method factory для Bins/Huffman/Arithmetic Coding и воспроизводимое создание sender/receiver method RNG;
- высокоуровневый `runner/experiment.py`, который запускает любой из трех методов через один и тот же `run_experiment()`;
- `scripts/run_experiment.py` — первый единый launcher вместо ручного задания входов в `check_*_e2e.py`; диагностические smoke scripts при этом сохранены.
- общий `metrics/capacity_entropy.py`: BPT, reference entropy и entropy utilization по фактической sender-side траектории;
- общий `metrics/distribution_distortion.py`: exact/available `Q_stego` → KL(`P_reference || Q_stego`) и TVD без epsilon smoothing, с run-level aggregation и явным учетом `inf`.

LM/reference-distribution слой локально проверен на Llama и Qwen. Bins, Huffman и Arithmetic Coding прошли core/unit и end-to-end проверки на обеих реальных моделях. Пункт 1 этапа 2 (адаптация трех базовых методов) завершен.

## Bins end-to-end smoke test

После обычного `pytest -q` реальную GPU-проверку следует запускать отдельно:

```bash
python scripts/check_bins_e2e.py \
  configs/models/llama-3.2-3b.local.json \
  --block-size 2 \
  --carrier-tokens 16
```

Smoke test не является основным экспериментом benchmark. Он проверяет, что единая инфраструктура проходит полный путь от LM до восстановления секрета после обычного текстового канала. `roundtrip_exact=False` при изменении токенизации не маскируется и само по себе не означает ошибку инфраструктуры — это диагностируемый reliability-результат метода/канала.


## Huffman end-to-end smoke test

После `pytest -q` Huffman проверяется на реальной модели отдельно:

```bash
python scripts/check_huffman_e2e.py \
  configs/models/llama-3.2-3b.local.json \
  --bits-per-word 2 \
  --carrier-tokens 16
```

`bits_per_word` задаёт `2**bits_per_word` кандидатов, а не фиксированный BPT. Скрипт поэтому дополнительно выводит `step bits consumed` и фактический средний `payload_bits / carrier_tokens`. Полный roundtrip, как и для Bins, проходит только через ordinary-text transport.


## Arithmetic Coding end-to-end smoke test

После `pytest -q` AC проверяется на реальной модели отдельно:

```bash
python scripts/check_arithmetic_e2e.py \
  configs/models/llama-3.2-3b.local.json \
  --precision 16 \
  --top-k 50000 \
  --carrier-tokens 16
```

Для AC скрипт отдельно выводит `payload bits` и `secret bits read`. Последнее число включает `precision`-битное look-ahead окно и поэтому в нормальном fixed-carrier run ожидается больше полезного payload. В BPT учитываются только подтвержденные `payload bits`. Полный decode выполняется только после `tokens → text → retokenize`; никакие внутренние sender token IDs decoder'у не передаются.


## Единый experiment runner (этап 2, шаг 7.1)

После завершения адаптации трех базовых методов основной технический запуск можно задавать одним JSON config. Например:

```bash
python scripts/run_experiment.py configs/experiments/stage2_bins.example.json
python scripts/run_experiment.py configs/experiments/stage2_huffman.example.json
python scripts/run_experiment.py configs/experiments/stage2_arithmetic.example.json
```

Во всех трех случаях prompt загружается из `data/prompts.jsonl` по `prompt_id`, секрет воспроизводится по `secret_id`, а общая generation policy применяется через один `ReferenceDistributionBuilder`. Текущий `p000001` — только технический Stage-2 smoke prompt, а не финальный корпус benchmark v1.0. На шаге 7.1 runner еще не рассчитывает полный набор benchmark metrics и не сохраняет `run_id/result.json/trace/summary.parquet`; эти слои добавляются следующими шагами этапа 2.

## Capacity + entropy metrics (этап 2, шаг 7.2)

Единый `run_experiment()` теперь рассчитывает первый общий блок benchmark-метрик
для Bins, Huffman и Arithmetic Coding. На каждом sender-side шаге из того же
канонического `P_reference` вычисляется Shannon entropy в битах, после чего
агрегируются:

```text
payload_bits
carrier_tokens
bits_per_token
reference_entropy_mean_bits
reference_entropy_sum_bits
entropy_utilization
entropy_utilization_percent
```

BPT и entropy utilization используют только подтвержденный полезный payload.
Для Arithmetic Coding `secret_bits_read` включает look-ahead и поэтому может
быть больше `payload_bits`, но эти дополнительные биты в метрики емкости не
попадают. `entropy_utilization` реализован ровно как `B / sum_t H_t` и не
клиппируется до 1.

Единый launcher выводит эти значения сразу после генерации:

```bash
python scripts/run_experiment.py configs/experiments/stage2_bins.example.json
python scripts/run_experiment.py configs/experiments/stage2_huffman.example.json
python scripts/run_experiment.py configs/experiments/stage2_arithmetic.example.json
```

NLL/PPL, агрегированная reliability/performance и постоянное хранение
результатов в шаг 7.2 намеренно не входят.

## KL/TVD distribution distortion (этап 2, шаг 7.3)

На sender-side каждом carrier-шаге общий metric layer теперь сравнивает
каноническое `P_reference` с `Q_stego`, возвращенным adapter'ом метода.
Реализованы фиксированные спецификацией v0.1 определения:

```text
KL = D_KL(P_reference || Q_stego), log base 2
TVD = 0.5 * sum_x |P_reference(x) - Q_stego(x)|
```

Скрытое epsilon-сглаживание запрещено: если `P_reference(x) > 0`, а
`Q_stego(x) = 0`, KL данного шага равна `inf`. Поэтому для Bins/Huffman и
некоторых конфигураций Arithmetic Coding бесконечная KL является ожидаемым
структурным результатом ограниченного support `Q_stego`, а не ошибкой. TVD
при этом остается конечной в `[0, 1]`.

Run-level агрегируются:

```text
q_mode
kl_mean_bits
kl_median_bits
kl_p95_bits
kl_max_bits
kl_infinite_steps
kl_finite_steps
tvd_mean
tvd_median
tvd_p95
tvd_max
```

Если хотя бы один шаг имеет бесконечную KL, `kl_mean_bits = inf`. Для p95
реализация v0.2 использует empirical nearest-rank percentile, чтобы не
интерполировать между конечным значением и `+inf`; это уточнение отдельно
зафиксировано в `benchmark_specification_v0.1_notes.md` для рассмотрения в v1.0.
Полные `P/Q` по всем шагам не сохраняются: после вычисления step KL/TVD runner
оставляет только scalar diagnostics, чтобы не раздувать память и будущий trace.

Единый launcher выводит KL/TVD вместе с capacity/entropy:

```bash
python scripts/run_experiment.py configs/experiments/stage2_bins.example.json
python scripts/run_experiment.py configs/experiments/stage2_huffman.example.json
python scripts/run_experiment.py configs/experiments/stage2_arithmetic.example.json
```

NLL/PPL, агрегированная reliability/performance и постоянное хранение
результатов добавляются следующими шагами этапа 2.

## NLL/PPL + reliability + performance (этап 2, шаг 7.4)

Unified runner дополняет предыдущие метрики тремя группами. Raw-LM quality
рассчитывается по исходным logits генерирующей модели при `temperature=1`, без
`V_allowed` mask, top-k/top-p и stego-модификаций:

```text
nll_raw_lm_nats_per_token
ppl_raw_lm
```

Reliability агрегируется после обязательного ordinary-text round-trip:

```text
roundtrip_exact
ber
bit_errors
expected_length_bits
recovered_length_bits
length_delta_bits
recovered_extra_bits
token_sequence_roundtrip_exact
```

Для AC BER и throughput используют только подтвержденный `payload_bits`, а не
`secret_bits_read` с look-ahead.

Перед timing выполняется отдельный warm-up. При CUDA каждый measured component
обрамляется `torch.cuda.synchronize()`. `encode_total_ms`/`decode_total_ms`
составляются из LM forward, canonical distribution processing и stego
algorithm components; model loading, text transport, metric calculations и
storage не включаются.

Единый launcher остается тем же:

```bash
python scripts/run_experiment.py configs/experiments/stage2_bins.example.json
python scripts/run_experiment.py configs/experiments/stage2_huffman.example.json
python scripts/run_experiment.py configs/experiments/stage2_arithmetic.example.json
```

Шаг 7.5 добавляет воспроизводимый `run_id`, per-run artifacts и общую `summary.parquet`.


## Persistent run storage (этап 2, шаг 7.5)

Каждый unified run по умолчанию теперь сохраняется в `results/`. Идентификатор
вычисляется по фиксированному правилу specification v0.1:

```text
run_id = SHA256(canonical_config_json)[:16]
```

Физическая структура одного успешного запуска:

```text
results/runs/<run_id>/
├── config.json
├── result.json
├── stegotext.txt
├── token_ids.json
└── trace.jsonl.gz
```

`results/summary.parquet` содержит одну строку на run и обновляется по `run_id`,
поэтому повтор той же canonical config не создает дубликат. Для Parquet нужен
optional dependency:

```bash
pip install -e ".[storage]"
```

Строгий JSON не поддерживает numeric Infinity, поэтому законный benchmark-native
KL `+inf` записывается в JSON как строка `"inf"`; в Parquet остается числовым
`+inf`. `null` означает именно недоступное значение, а не бесконечность.

Обычный запуск сохраняет результаты автоматически:

```bash
python scripts/run_experiment.py configs/experiments/stage2_bins.example.json
```

Для диагностического запуска без storage доступен `--no-save`; для сохранения
per-run файлов без обновления общей таблицы — `--skip-summary-parquet`. Storage и
формирование trace выполняются после измеряемых encode/decode sections и не
входят в performance metrics.
