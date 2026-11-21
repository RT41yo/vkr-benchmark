# Stage 7 · RRC nontermination: replay и строго ограниченное исключение

## Исходные наблюдения (получены на машине экспериментатора)

Новые train/validation stego: Llama 1920/1920 successful, Qwen 1919/1920 successful. Исключение: `run_id=0b0e22af25613edb`, `sample_id=79775985a169f917bfe94a05`, Qwen3-4B-Base, prompt `s7p0082`, partition `validation`, method `rrc/payload1024-margin40`. Из `results/stage7/stego_failures/0b0e22af25613edb.json`: 1024 payload bits, 2048 carrier limit, 2048 generated, 0 termination candidates, 0 verification rejections, `is_usable_stegotext=false`. Данная запись — подтверждённый факт попытки, **не подтверждение первопричины**.

Успешных образцов Stage7 train/val на двух LM: **3839**. Изначально запланированных Stage7 train/val: **3840**. Замороженных Stage6 test: **1800** (среди них 97 hard decode failures с извлекаемым `transport_text`, не исключать).

## 1. GPU replay прежде любых исключений

Ветвь: `stage7-steganalysis`. Репозиторий и Stage6 manifests не переписываются. Установить текущий overlay поверх предыдущих Checkpoint1/2 + RRC + SHA hotfix.

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark
source .venv/bin/activate

pytest -q

PYTHONPATH=src python scripts/diagnose_stage7_rrc_nontermination.py \
  --model-config configs/models/qwen3-4b-base.local.json \
  --replays 1
```

Replay запускает неизменённый canonical normalized RRC encoder, тот же секрет и ключ из frozen Stage7 plan, `P_reference`, bfloat16 модель Qwen с закреплённой revision; до загрузки весов проверяет соответствие единственного JSON failure и allowlist. Токены не записываются как текст: сохраняется SHA-256 бинарно упакованной последовательности token IDs и контрольные `interval_log2_width` на шагах 1, 64, 128, 256, 512, 1024, 1536, 2048 в `results/stage7/rrc_nontermination_replay.json`. Никаких raw secret bits, key или partial stegotext в report.

Ожидаемый результат при подтверждении cutoff:

```text
RRC replay 1/1: terminated=False, carriers=2048, candidates=0, rejections=0
RRC cutoff replay: reproduced; diagnostics=...
```

В отчёте итоговый `interval_log2_width` должен быть строго положителен (ширина > 1). Тогда отсутствие кандидатов согласуется с недостаточным сужением интервала. Это не доказывает, что именно низкая энтропия токенов была первопричиной, и не исключает всех прочих численных пограничных случаев.

Если `status=mismatch`, encoder завершился, ширина достигла <=1, кандидатов было >0, или replay выбросил иное исключение — **остановиться**: не запускать matched cover/dataset, не подменять результат/коды/secret, сообщить подробности для отдельного исследования. Повторный replay можно осуществить с `--replays 2` после сознательного архивирования/удаления локального JSON replay вручную; по умолчанию утилита не перезаписывает существующий отчёт.

## 2. Явный методологический допуск

После проверки `status=reproduced` и содержимого отчёта применяется **только одно** исключение из `configs/stage7/rrc_nontermination_exception.json`. Без флага `--allow-confirmed-rrc-cutoff` конвейер остаётся прежним, fail-closed: требует 5640 полных пар.

```bash
PYTHONPATH=src python scripts/assemble_stage7_cover_plan.py \
  --allow-confirmed-rrc-cutoff --dry-run

PYTHONPATH=src python scripts/assemble_stage7_cover_plan.py \
  --allow-confirmed-rrc-cutoff
```

Ожидаемый план: **5639 пар**, по partition `train=2880`, `validation=959`, `test=1800`, по LM `Llama=3000`, `Qwen=2639`. Отсутствует исключительно `sample_id=79775985a169f917bfe94a05`. Source-plan имеет 3840 + 1800 записей; другие пропуски, изменённые хеши, вторые failure-records или пересечения Stage6 вызывают ошибку. Подмена или перегенерация RRC не допускается.

## 3. Новые ordinary covers после принятия решения

После успешного dry-run и утверждения решения в рамках Stage7:

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

Каждый `cover` использует совпадающую sender-token длину своего успешно построенного stegotext (для streaming — 256, для RRC — фактическую) и независимую случайность из канонического `P_reference`. Для failed Qwen RRC cover **не создаётся**: у него отсутствует завершённый stegotext для длинового матчинга.

После генерации всех 5639 covers:

```bash
PYTHONPATH=src python scripts/build_stage7_detector_dataset.py \
  --allow-confirmed-rrc-cutoff --dry-run

PYTHONPATH=src python scripts/build_stage7_detector_dataset.py \
  --allow-confirmed-rrc-cutoff

pytest -q
```

Expected: **5639 pairs / 11278 texts**, `cover=5639`, `stego=5639`, `group_count=152`, `stage6_test_sources=1800`. `results/stage7/dataset_manifest.json` хранит перечисленные `excluded_nontermination_run_ids`, численность и SHA-256 exception+replay report; тексты остаются в gitignored `data/stage7/dataset/`.

## 4. Важное ограничение анализа

Обнаружимость RRC оценивается условно на завершённых встраиваниях, а частота cutoff сообщается независимо: 1 / 256 Stage7 train/val RRC attempts, 1 / 128 для Qwen RRC, и 1 / 32 для Qwen RRC validation. Разница между 31 и 32 validation examples одной точки не балансируется post hoc и не компенсируется другой статьёй. `article_id` split не меняется, train/test не затрагиваются. Проверка root cause не заменяет факторное исследование RRC precision_margin из §9 исходного протокола. Сверка с pinned author implementation требует отдельного исследования.
