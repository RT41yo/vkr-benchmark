# Stage 7 — Checkpoint 2: train/validation stego generation, matched cover plan, paired dataset

**Статус:** реализованы plan + CPU-контур + generation-only + сборка и валидация dataset. Реальные Llama/Qwen generation runs в среде подготовки архива НЕ выполнялись: отсутствуют большие веса моделей и Stage-6 `results/runs/`. Checkpoint 1, Stage 6 и старые алгоритмы не модифицируются.

## Смысл

- По согласованному immutable split (96/32) и 15 frozen Stage-6 operating points строится **3840** новых plan rows (Llama=1920, Qwen=1920). Один секретный SHAKE256 поток на article/replicate, одинаковый для методов и LM. Randomness/key материала для Bins/ADG/Discop/RRC детерминированно получаются из Stage-7 домена на уровне article+method+replicate и одинаковы между operating points одного метода.
- `generate_stage7_stegos.py`: те же существующие encoder sessions и `P_reference`; не вызывает metric-only hooks и decoder. Генерирует текст при 256 carrier tokens либо RRC 1024 bits, сохраняет полный sender/transport/retokenization trace в отдельном `results/stage7/stego_runs/`. Никакого влияния на frozen Stage6. На test остаются только исходные Stage6 stegotexts, включая hard failures.
- Флаг `--verify-encoder` на 1 запуске выполняет тот же run *вторично* через instrumented canonical encoder и требует совпадения `sender_token_ids`, `payload_bits`, `secret_bits_read` до сохранения результатов. Требует реальной GPU проверочной команды на пользовательской машине.
- После 3840 генераций собирается cover plan из **5640** matched pairs. Длина RRC берётся из реальных sender IDs каждой стегогенерации. Планы Stage6 test сохраняют ранее сформированные cover seed и поэтому уже сгенерированные GPU smoke samples можно `--resume`.
- После 5640 ordinary control generations создаётся text-only `detector_records.jsonl`, 11280 строк, 1:1 cover/stego в каждом split/LM/method point. Full text хранится локально, Git игнорирует его; компактный `results/stage7/dataset_manifest.json` сохраняет хеши. При отсутствии/порче хоть одного источника — ошибка, никакого молчаливого отбора успешных декодирований.
- Near-duplicate preflight audit сравнивает 5-словные shingles между разными split (Jaccard >=0.85) и **только сообщает** подозрительные пары/останавливается; он не перераспределяет и не удаляет статьи.

## Проверка и последовательность команд

После сохранения архива в `~/Downloads`:

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark

git branch --show-current
git status

unzip -o ~/Downloads/vkr_stage7_checkpoint2_overlay.zip -d ..

git status
source .venv/bin/activate

pytest -q tests/unit
pytest -q --ignore=tests/integration
pytest -q
```

Продолжить только в `stage7-steganalysis`; не выполнять Git commit/push/merge до проверки.

Шаг А: plan (без GPU):

```bash
PYTHONPATH=src python scripts/plan_stage7_trainval.py --dry-run
PYTHONPATH=src python scripts/plan_stage7_trainval.py
PYTHONPATH=src python scripts/audit_stage7_prompt_overlap.py
```

Ожидаемо: `3840 runs (train=2880 validation=960)` и `prompt overlap audit: ok (candidates=0)`. Если проверка пересечений сообщает `review_required`, массовую генерацию НЕ запускать; нужны анализ и согласованное решение по конкретным парам.

Шаг B: обязательные GPU smoke/equivalence (один Bins и один RRC для **каждой** LM):

```bash
PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/llama-3.2-3b.local.json \
  --method bins --method-point b1 --limit 1 --verify-encoder --resume

PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/llama-3.2-3b.local.json \
  --method rrc --method-point payload1024-margin40 --limit 1 --verify-encoder --resume

PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/qwen3-4b-base.local.json \
  --method bins --method-point b1 --limit 1 --verify-encoder --resume

PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/qwen3-4b-base.local.json \
  --method rrc --method-point payload1024-margin40 --limit 1 --verify-encoder --resume
```

Для всех четырёх ожидается `Stage-7 canonical sender equivalence: PASS` и `STEGO OK` либо, если smoke уже был запущен, `skipped` при `--resume` (повторить equivalence можно только для нового run). Убедиться в отсутствии ошибок и корректной длине RRC.

Шаг C: **длительная** новая stego train/validation генерация (~1920 запусков на LM). Выполнять только после успешного smoke, можно прерывать/возобновлять `--resume`:

```bash
PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/llama-3.2-3b.local.json --all --resume

PYTHONPATH=src python scripts/generate_stage7_stegos.py \
  --model-config configs/models/qwen3-4b-base.local.json --all --resume
```

Шаг D: plan контрольной генерации (требует всех 3840 stego и Stage6 test audit):

```bash
PYTHONPATH=src python scripts/assemble_stage7_cover_plan.py --dry-run
PYTHONPATH=src python scripts/assemble_stage7_cover_plan.py
```

Ожидается `5640 pairs, 152 prompt groups`; создаются локальные `data/stage7/dataset/all_cover_plan.jsonl` и `all_prompts.jsonl`.

Шаг E: обычная генерация (два запуска, автоматически возобновит уже созданные cover smoke Stage6):

```bash
PYTHONPATH=src python scripts/generate_stage7_covers.py \
  --model-config configs/models/llama-3.2-3b.local.json \
  --prompts-file data/stage7/dataset/all_prompts.jsonl \
  --plan data/stage7/dataset/all_cover_plan.jsonl --all --resume

PYTHONPATH=src python scripts/generate_stage7_covers.py \
  --model-config configs/models/qwen3-4b-base.local.json \
  --prompts-file data/stage7/dataset/all_prompts.jsonl \
  --plan data/stage7/dataset/all_cover_plan.jsonl --all --resume
```

Шаг F: окончательная dataset assembly (строгая проверка 1800 Stage6 + 3840 Stage7, всех 5640 cover):

```bash
PYTHONPATH=src python scripts/build_stage7_detector_dataset.py --dry-run
PYTHONPATH=src python scripts/build_stage7_detector_dataset.py

pytest -q

git status
```

Ожидается `sample_count=11280`, `pair_count=5640`, `class_counts.cover=5640`, `class_counts.stego=5640` и `stage6_test_sources=1800`. Ни один текст не подставляется из других LM, prompt или рабочей точки. `README.md` остаётся пустым.

**Не запускать RoBERTa training до получения полного dataset и фиксации точной base revision.** Следующий checkpoint: detector training, TF-IDF baseline, validation model selection и held-out evaluation protocol. Stage 8 Pareto не входит в Stage 7.

## Примечание к тестам

Тесты этой ветки выполнены на CPU/mock LM и synthetic manifests, включая sender-equivalence для Bins/Huffman/ADG/Discop и RRC fixed-payload, а также integrity checks. Тесты не заменяют реальные GPU smoke и фактические часы генерации на целевом оборудовании.
