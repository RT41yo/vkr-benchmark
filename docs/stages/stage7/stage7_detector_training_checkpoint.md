# Stage 7 — Detector Training checkpoint (implementation overlay)

Статус: инфраструктура обучения готова к локальным smoke/preflight; реальные результаты
на полном Stage-7 корпусе и GPU должны быть получены локально. Этот checkpoint не
открывает внешний test и не пересчитывает Stage 6.

## Нормативный источник

- Базовый коммит Dataset Freeze: `80116f4` в `stage7-steganalysis`.
- `data/stage7/dataset/detector_records.jsonl` (локальный, gitignored), SHA-256
  `43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9`.
- `results/stage7/dataset_manifest.json` (в Git); 5639 matched pairs / 11278 rows,
  1 документированный RRC cutoff `0b0e22af25613edb`.
- `docs/stages/stage7/stage7_protocol.md`, разделы 7–8; ADR-0028.

## Состав

- `src/vkr_benchmark/steganalysis/detectors.py`: проверка SHA-256, metadata,
  покрытия групп и matched pairs; **возвращает только train/validation**.
- `configs/stage7/detector_training.json`: фиксированные параметры baseline и
  RoBERTa. `roberta.base_revision=null` **преднамеренно**, пока не определён
  конкретный immutable commit snapshot.
- `scripts/train_stage7_baseline.py`: TF-IDF `analyzer=char`, 3–5, `min_df=2`,
  `max_features=100000`, `sublinear_tf=true`; LogisticRegression `C=1.0`,
  `max_iter=2000`, `class_weight=None`. Один детектор на исходную LM.
- `scripts/prepare_stage7_roberta_base.py`: фиксирует SHA-256 весов/токенизатора,
  требует full 40-hex revision, проверяет snapshot.
- `scripts/train_stage7_roberta.py`: text-only RoBERTa-base, max length 512,
  truncate-right, dynamic padding, AdamW, 2e-5, decay 0.01, до 4 эпох,
  linear warmup 10%, 4×8=32 effective batch, seeds 711/712/713,
  patience 2, выбор лучшей эпохи по pooled validation ROC-AUC.
- `scripts/select_stage7_roberta_checkpoint.py`: выбор **одного** из трёх
  обученных seeds по validation ROC-AUC; при равенстве — более ранняя эпоха,
  затем меньший seed.
- `tests/unit/test_stage7_detector_training.py`: проверки freeze, seal test,
  реальный CPU fit baseline на синтетическом корпусе, pinning и выбора checkpoint.

Во всех training CLI внешний `test` **не используется**, включая подбор порога
classification: порог определяется только на pooled validation (maximum balanced
accuracy, детерминированный lower-threshold tie-break). Направление stego score
никогда не инвертируется post-hoc.

## Установка и локальная проверка

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark
source .venv/bin/activate
unzip -o ~/Downloads/vkr_stage7_detector_training_checkpoint_overlay.zip -d ..
python -m pip install -e '.[steganalysis]'
PYTHONPATH=src pytest -q tests/unit/test_stage7_detector_training.py
pytest -q
```

Проверка входов перед обучением (не выполняет fit и не создаёт checkpoint):

```bash
PYTHONPATH=src python scripts/train_stage7_baseline.py \
  --model-id meta-llama/Llama-3.2-3B --dry-run
PYTHONPATH=src python scripts/train_stage7_baseline.py \
  --model-id Qwen/Qwen3-4B-Base --dry-run
```

Ожидается `status=frozen_dataset_verified`, `test_access=none`;
`train`/`validation` по исходным LM — соответственно `2880/960` для Llama
и `2880/958` для Qwen.

### CPU baseline

После двух успешных preflight (по отдельности):

```bash
PYTHONPATH=src python scripts/train_stage7_baseline.py \
  --model-id meta-llama/Llama-3.2-3B
PYTHONPATH=src python scripts/train_stage7_baseline.py \
  --model-id Qwen/Qwen3-4B-Base
```

`results/stage7/checkpoints/baseline/<lm>/summary.json` содержит validation ROC-AUC,
threshold и balanced accuracy. Модели `.joblib` находятся в
`models/detectors/stage7/<lm>/`. Эти пути gitignored. **Не изменять test**.

### RoBERTa: обязательно pin до GPU

`FacebookAI/roberta-base` в исходном протоколе имеет `base_revision=null`.
До запуска необходимо получить актуальный immutable SHA через Hugging Face
(с подключением к сети) и согласованно зафиксировать его в конфигурации.

Для получения SHA:

```bash
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info('FacebookAI/roberta-base').sha)
PY
```

Затем подставить **полный** полученный SHA:

```bash
PYTHONPATH=src python scripts/prepare_stage7_roberta_base.py \
  --revision '<40-hex-commit-sha>' --download
```

Команда выведет `snapshot_path` в HF cache, создаст JSON-отчёт с точными
SHA-256 локальных assets. В `configs/stage7/detector_training.json` заменить
`roberta.base_revision` с `null` на тот же SHA и зафиксировать эту версию
конфигурации до полноразмерного обучения (после проверки пользователем).

Только после pin выполнить dry-run (без GPU):

```bash
PYTHONPATH=src python scripts/train_stage7_roberta.py \
  --model-id meta-llama/Llama-3.2-3B --seed 711 \
  --snapshot '<snapshot_path>' --dry-run
```

Потом тот же CLI без `--dry-run` обучает seed 711. Аналогично 712/713 и
Qwen. Каждый seed сохраняет локальный checkpoint и validation predictions.
Реальные training GPU-процессы в среде подготовки overlay не запускались.

После получения **всех трёх** RoBERTa seeds одной LM:

```bash
PYTHONPATH=src python scripts/select_stage7_roberta_checkpoint.py \
  --model-id meta-llama/Llama-3.2-3B
```

Этот checkpoint намеренно **не содержит external test evaluation** и
cluster-bootstrap CI: они должны быть отдельным последующим checkpoint после
получения и проверки validation результатов, чтобы не допустить test leakage.

## Правила сохранности и контроля

- Не менять frozen Stage 6, готовый detector dataset или его SHA.
- Не добавлять полные carrier texts, `.joblib`, веса/предсказания в Git.
- `README.md` не изменять; не выполнять merge в `main` до закрытия Stage 7.
- Выводы smoke и контрольные показатели validation передать на ревью до
  массового GPU обучения.
- Все training artifacts создаются с **no overwrite** при существующем output.
