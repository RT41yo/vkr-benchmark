# VKR Benchmark

Экспериментальная инфраструктура для воспроизводимой многокритериальной оценки методов генеративной лингвистической стеганографии.

Текущий этап: **Этап 2 — базовая инфраструктура и адаптация Bins, Huffman, Arithmetic Coding**.

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
- 42 synthetic unit tests для контрактов, token space, `P_reference` и Bins.

LM/reference-distribution слой локально проверен на Llama и Qwen. Bins пока проверен синтетически без GPU; следующий шаг — интеграционный encode/decode с реальной LM и затем обязательный text transport `tokens → text → tokens`.
