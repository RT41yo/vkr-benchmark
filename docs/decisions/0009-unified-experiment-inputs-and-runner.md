# ADR-0009: Единые входы эксперимента и общий experiment runner

## Статус

Принято на этапе 2, шаг 7.1.

## Контекст

После адаптации Bins, Huffman и Arithmetic Coding каждый метод уже мог пройти
общий низкоуровневый pipeline, но реальные smoke-проверки запускались отдельными
`scripts/check_*_e2e.py`. В них вручную задавались prompt, secret_id, параметры
метода и правило остановки. Это было допустимо для проверки адаптеров, но не для
воспроизводимого benchmark run.

Спецификация v0.1 требует стабильных `prompt_id`, детерминированного секретного
потока по `secret_id`, одинаковой общей generation policy и явного разделения
внешних условий от параметров стегометода.

## Решение

Ввести один исполняемый `ExperimentConfig` и один высокоуровневый
`run_experiment()`.

`ExperimentConfig` задает:

- benchmark version и run kind;
- ссылку на локальную model config;
- `prompt_id`;
- `method.id`, `method.params`, а при необходимости seed/key;
- общую generation policy;
- `secret_id`;
- termination policy.

Точный текст prompt загружается только через `PromptRegistry` из
`data/prompts.jsonl`. Секрет создается общим `create_secret_source()` и использует
уже зафиксированный SHAKE256 domain спецификации v0.1.

Высокоуровневый runner разрешает метод по `method.id`, создает
`P_reference` builder, секрет и раздельные sender/receiver RNG streams, после
чего вызывает уже проверенный `run_streaming_text_roundtrip()`.

Для Bins `method.random_seed` обязателен: sender и receiver получают два
независимых RNG объекта с одинаковым seed и поэтому воспроизводят одну и ту же
фиксированную partition. Huffman и Arithmetic Coding в текущих normalized
адаптациях RNG не требуют.

На шаге 7.1 поддерживается только `fixed_carrier_tokens`, потому что все три
базовых метода сейчас проверяются в этом режиме. Другие termination modes будут
добавляться вместе с методами, которым они действительно нужны.

## Что намеренно не входит в шаг 7.1

- расчет и агрегация benchmark metrics;
- `run_id`;
- `result.json`, trace и `summary.parquet`;
- batch sweeps.

Это следующие шаги этапа 2. `check_bins_e2e.py`, `check_huffman_e2e.py` и
`check_arithmetic_e2e.py` сохраняются как диагностические smoke scripts и не
являются основным experiment runner.

## Технический prompt

`p000001` в текущем `data/prompts.jsonl` является только Stage-2 technical smoke
prompt. Он не объявляется финальным корпусом benchmark v1.0.
