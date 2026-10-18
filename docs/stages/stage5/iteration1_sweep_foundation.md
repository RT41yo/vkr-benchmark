# Stage 5 — Iteration 1: sweep foundation

## Что добавлено

1. ADR-0026 фиксирует различие method-local operating points, common external conditions и RRC calibration.
2. `scripts/prepare_stage5_pilot_prompts.py` переиспользует ровно восемь frozen Stage-3 CNN/DailyMail pilot contexts. Он не скачивает новые данные, проверяет `context_sha256` и генерирует:
   - локальный `data/stage5/pilot_prompts.jsonl`;
   - commit-able `results/stage5/pilot_prompt_manifest.json` без текста статей.
3. `SweepManifest` (`stage5.sweep.v1`) разворачивает компактную матрицу в обычные `ExperimentConfig` и заранее вычисляет canonical `run_id`.
4. Paired replicate identity автоматически дает:
   - одинаковый `secret_id` для всех методов/точек внутри replicate;
   - воспроизводимый `random_seed` для Bins/ADG;
   - воспроизводимый `key` для Discop/RRC;
   - отсутствие фиктивного RNG у детерминированных Huffman/Arithmetic/DAIRstega.
5. `scripts/plan_stage5_sweep.py` проверяет и печатает матрицу без LM/GPU.
6. `scripts/run_stage5_sweep.py` выполняет отфильтрованную часть матрицы с одной загрузкой модели, поддерживает `--resume`, сохраняет обычные canonical run artifacts и отдельный sweep log.
7. Первый executable manifest `top_p_calibration_llama.json` содержит `3 top_p × 7 methods × 2 prompts × 1 replicate = 42` технических runs.

## Почему calibration короткая

8 carrier tokens (или 32 payload bits для RRC) недостаточны для научного ранжирования, но достаточны для первой проверки поддержки, корректности, exact-Q instrumentation cost и порядка wall-clock стоимости. Эти runs не входят в будущий основной dataset Stage 6.

## Что намеренно НЕ сделано

- не выбран финальный `top_p`;
- не зафиксирован Pilot-A method grid как executable main manifest;
- не выбран окончательный RRC payload;
- не зафиксировано число Stage-6 prompts/replicates;
- benchmark specification v1.0 еще не создана;
- GPU pilot не запускался в этой итерации.

Следующая контрольная точка — реальный `top_p` calibration на локальной Llama-3.2-3B, анализ timing/support/metrics и согласование одного main generation regime перед Pilot A.
