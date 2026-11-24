# Stage 7 — аудит основных детекторов и исследование горизонта обучения 8 эпох

Статус: **предэкспериментальный протокол**. Основная серия RoBERTa-base по 4 эпохи
уже выполнена локально для 2 исходных LM × seeds 711/712/713. Этот checkpoint
**не меняет** результаты Stage 6, corpus Dataset Freeze и существующие основные
веса/валидационные предсказания. Основной исходный коммит инфраструктуры:
`9edb254`; следующий коммит pinning revision — согласно локальному Git пользователя.

## Неподвижные входы

- Stage-7 dataset: `43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9`;
  train 2880 для каждой LM; validation 960 Llama / 958 Qwen; Stage 6 test закрыт.
- RoBERTa-base revision: `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- SHA-256 локального `roberta_base_snapshot.json`:
  `9acbccb24045fc2697633978cf80aed0ae52bbc3824834f9ed7b1f32e6e6d6f3`.
- SHA-256 **закреплённого пользовательского** `configs/stage7/detector_training.json`:
  `2eb26ae60e768b0e55f041c53b2221a87bbb433f357e7523a5665a96a60b17ac`.
- Основная конфигурация: 4 эпохи, patience=2, AdamW 2e-5,
  weight_decay=0.01, batch 4 × gradient accumulation 8, warmup 10%, линейный
  scheduler на 4 эпохи. **Остаётся неизменной.**
- Основной критерий выбора детектора — pooled validation ROC-AUC; при равенстве
  раньше эпоха, затем меньший seed. Не использовать test для выбора.

## 1. Read-only аудит шести основных обучений

`audit_stage7_roberta_primary.py` сверяет manifest датасета и SHA-256, свойства
train/validation, все 6 summary, 6 файлов validation predictions (ID пар, метки,
значения score, полноту), пересчитывает pooled validation ROC-AUC, оптимальный
validation threshold и balanced accuracy. Проверяет наличие и считает SHA-256
весов и токенизатора каждого из шести checkpoint. **Содержимое файлов весов
не загружается, test-тексты не возвращаются и не используются.**

Из корня `vkr-benchmark`:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/audit_stage7_roberta_primary.py
```

Ожидается `status=verified`, `detector_count=6`, `test_access=none`, selected
seed 713 для каждой LM. При несогласованных сохранённых результатах аудит
прервётся без изменений данных.

После успешного read-only выполнения фиксируем компактный отчёт:

```bash
PYTHONPATH=src python scripts/audit_stage7_roberta_primary.py --write-report
```

Будет создан `results/stage7/roberta_primary_audit.json` — только hashes,
метрики и идентификаторы (никаких секретов или текстов); SHA-256 весов не
подменяет повторного исполнения модели, но фиксирует текущие bytes checkpoint.

Штатный **основной** выбор одного seed на LM:

```bash
PYTHONPATH=src python scripts/select_stage7_roberta_checkpoint.py \
  --model-id meta-llama/Llama-3.2-3B
PYTHONPATH=src python scripts/select_stage7_roberta_checkpoint.py \
  --model-id Qwen/Qwen3-4B-Base
```

Ожидаемые на текущих validation-результатах: seed 713 для обеих моделей.
Исходная серия остаётся основной для будущего frozen test; её артефакты не
перезаписываются.

## 2. Новый, исследовательский эксперимент: max 8 эпох

**Гипотеза:** более длинный горизонт оптимизации может повысить validation
ROC-AUC, поскольку все шесть основных запусков достигли максимума к 4-й эпохе.
Это не доказательство потенциала улучшения на external test.

Новый запуск **с нуля от той же pinned RoBERTa-base**, без продолжения
основного 4-эпохового checkpoint. Все прочие параметры остаются как в основной
серии, кроме общего числа шагов scheduler: при 2880 обучающих строках
`steps_per_epoch = ceil(ceil(2880 / 4) / 8) = 90`, а для 8 эпох
`total_scheduled_steps = 720`, `warmup_steps = 72` (для 4 эпох: 360/36).

**Ограничение интерпретации:** график LR на эпохах 1–4 при 8-эпоховом горизонте
отличается от основного. Разницу нельзя приписывать исключительно «эпохам
5–8»; это сравнение горизонта обучения **совместно с расписанием LR**.

Зафиксированные параметры ablation:
`configs/stage7/roberta_epochs8_ablation.json`; seeds 711/712/713,
patience 2, best epoch по pooled validation ROC-AUC. Дополнительные запуски
**не участвуют в выборе основного primary checkpoint** и не дают оснований
изменить замороженный Stage 7 main post hoc.

После успешно записанного основного аудита dry-run (без GPU):

```bash
PYTHONPATH=src python scripts/train_stage7_roberta_epochs8.py \
  --model-id meta-llama/Llama-3.2-3B --seed 711 \
  --snapshot '/home/matmul/.cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b' \
  --dry-run
```

Ожидается `status=ablation_preflight_ok`, `max_epochs=8`, `test_access=none`,
неизменные hashes. Обучение (после проверки):

```bash
mkdir -p data/stage7/cache
set -o pipefail
PYTHONPATH=src python scripts/train_stage7_roberta_epochs8.py \
  --model-id meta-llama/Llama-3.2-3B --seed 711 \
  --snapshot '/home/matmul/.cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b' \
  2>&1 | tee data/stage7/cache/roberta_e8_llama_seed711.log
```

После smoke на одном seed и его проверки запустить остальные 5 запусков
**последовательно** с тем же скриптом, `--seed` и `--model-id`.

Все дополнительные артефакты в **отдельных** каталогах:

- `results/stage7/checkpoints/roberta_epochs8/<slug>/seed-<N>/` — summary, validation predictions;
- `models/detectors/stage7_epochs8/<slug>/roberta/seed-<N>/` — модель/токенизатор;
- `data/stage7/cache/roberta_e8_*.log` — локальные логи.

Код отказывается перезаписывать существующий output и запрещает направить
8-эпоховые веса/отчёты в каталоги основного эксперимента. Если процесс
прервётся после сохранения весов, не удалять артефакты автоматически:
сначала исследовать причину и сохранить диагностическую копию.

## 3. Сравнение двух серий без test

Только после 6 успешных дополнительных обучений:

```bash
PYTHONPATH=src python scripts/compare_stage7_roberta_epochs8.py
PYTHONPATH=src python scripts/compare_stage7_roberta_epochs8.py --write-report
```

Сравнение проверяет 6 × (primary vs E8), совпадение хешей, количество эпох,
best epoch, разность validation ROC-AUC по каждому seed и средние значения по
LM. Отчёт `results/stage7/roberta_epochs8_comparison.json` — диагностический,
не является выбором primary и **не обращается к test**.

## 4. Контроль сохранности и Git

- Нельзя изменять frozen Stage6, `detector_records.jsonl`, `dataset_manifest.json`,
  primary `detector_training.json`, primary scripts, primary weights/predictions.
- В Git добавлять только новые конфиги, скрипты, тесты, документацию и после
  ревью **компактный** `roberta_primary_audit.json`; большие весовые файлы и
  validation predictions остаются gitignored.
- Основной эксперимент Stage 7 сохраняет 4 эпохи, seeds 711–713 и первичный
  выбор по validation. Внешний test остаётся закрытым до отдельного audit/test
  checkpoint. Не использовать результаты ablation для переопределения primary
  задним числом.
- Скрипт аудитора выполняет проверку checkpoint без прогонов инференса весов;
  GPU smoke может быть выполнен пользователем после успешного preflight.
