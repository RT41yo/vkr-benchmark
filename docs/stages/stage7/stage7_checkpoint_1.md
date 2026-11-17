# Stage 7 — Checkpoint 1: протокол, prompt split, external-test audit, control sampling

**Статус:** реализован CPU-контур; реальные GPU продолжения и обучение детектора НЕ выполнены в snapshot-среде (нет local model weights и canonical `results/runs`). Следующий checkpoint: Stage-7 stego train/val generation + общий paired dataset + detector training/evaluation.

## Новое

- ADR-0028 и `stage7_protocol.md` без изменений benchmark v1.0 / Stage6.
- Pinned extension ranks 32..159 с контролем Stage3 prefix 0..79 и Stage6 test ranks 8..31.
- Grouped article split train=96, validation=32, external Stage6 test=24. Тексты приватны/локальны, метаданные допускают коммит.
- Read-only inventory всех 1800 Stage6 canonical texts с включением `failure_diagnostics.json.transport_text` для 97 ошибок; без локальных run artifacts проверка завершается ошибкой.
- Sampling обычной LM по каноническому `P_reference`, отдельный `ControlRandomSource` для каждой пары; полная sender/transport/retokenize диагностика. `--resume` и отдельный `results/stage7/cover_runs/` (нет обращений к `results/summary.parquet`).
- Unit-тесты на статистический sampler, замороженную политику split и включение error-text.

## Воспроизведение на целевой машине

Перейти в свой репозиторий, убедиться, что ветка Stage7 и clean state; распаковать выданный overlay ZIP, создать/активировать уже установленную `.venv`.

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark
git branch --show-current
git status
unzip -o ~/Downloads/vkr_stage7_checkpoint1_overlay.zip -d ..
git status
source .venv/bin/activate
```

```bash
pytest -q tests/unit
pytest -q --ignore=tests/integration
pytest -q
```

Проверить/построить split. Понадобится pinned Parquet; при наличии локального Stage3 cache он будет использован без скачивания. Если cache отсутствует, скрипт загружает ровно закреплённый Parquet и проверяет SHA-256:

```bash
PYTHONPATH=src python scripts/prepare_stage7_prompts.py --dry-run
PYTHONPATH=src python scripts/prepare_stage7_prompts.py
```

Аудит **локальных** Stage6 source artifacts (в ZIP их нет):

```bash
PYTHONPATH=src python scripts/audit_stage7_stage6_sources.py --dry-run
PYTHONPATH=src python scripts/audit_stage7_stage6_sources.py
```

Должно быть `run_count=1800`, `ok=1703`, `error=97`. Иначе не продолжать detector dataset construction.

После проверки источников и наличия Llama весов выполнить **один** cover smoke (реальная GPU проверка). Это ничего не пишет в Stage6:

```bash
PYTHONPATH=src python scripts/generate_stage7_covers.py \
  --model-config configs/models/llama-3.2-3b.local.json \
  --prompts-file data/stage6/main_prompts.jsonl \
  --plan data/stage7/dataset/stage6_test_index.jsonl \
  --limit 1 --resume
```

Для Qwen аналогично меняется `--model-config configs/models/qwen3-4b-base.local.json`, а для полных контрольных запусков вместо `--limit 1` используется `--all --resume` **после подтверждения smoke**. План Stage6 test содержит обе модели, фильтрация происходит по `model_id`.

## Не запускать пока

- Обучение RoBERTa: exact pretrained revision ещё не разрешена/закреплена, stage7 train/validation stego пока не созданы.
- `--all` без CPU/GPU smoke.
- Любые варианты записи Stage7 в старый `results/runs/` или `results/summary.parquet`.
- Презентация и Stage7 closeout: готовятся после экспериментов.

## Проверяемые инварианты

`README.md` пустой; Stage-6 frozen configs/manifests и metrics не изменены; все новые данные Stage7 изолированы. При отсутствии локальных Stage6 runs нельзя делать вид, что 1800 текстов были проаудированы. Нет claim о новых ROC-AUC/quality/latency до реальных экспериментов.
