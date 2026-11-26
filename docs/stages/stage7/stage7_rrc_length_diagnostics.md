# Stage 7 — RRC unmatched-length diagnostic

Отдельный post-test эксперимент из `stage7_protocol.md`, не замена основной оценки для Stage 8. Базовые результаты Stage 6, frozen dataset, Pre-test Lock, исходный отчёт и raw predictions остаются неизменными. Используются 120 исходных RRC test pairs (72 Llama, 48 Qwen, 24 статьи), без исключений по результату декодирования. Новые cover генерируются существующим `generate_stage7_covers.py` с `P_reference`, 256 sender tokens, тем же prompt и исходным cover-seed пары, но в отдельном каталоге. Это common-random-numbers сравнение политики остановки, не независимый повтор с новым seed; выбранный seed фиксируется здесь до генерации. Проверяется совпадение общей части sender-token последовательностей с matched cover; несовпадения сохраняются в отчёте как ограничение интерпретации. Даже при совпадении префикса текстовое декодирование границы может отличаться. Влияние длины оценивается вместе с эффектом выбора границы фрагмента, а не как доказанная причинная связь.

## Команды (из корня репозитория)

```bash
source .venv/bin/activate
pytest -q
PYTHONPATH=src python scripts/stage7_rrc_length_diagnostics.py prepare
PYTHONPATH=src python scripts/stage7_rrc_length_diagnostics.py prepare --write-plan
```

Prepare читает исходные зафиксированные метаданные и сохранённые scores, проверяет SHA-256 исходных результатов, lock, cover plan и split; проверяет 24 текста промптов по context hash и 120 RRC-пар по сохранённым scores. План сохраняется в `data/stage7/cache/rrc_length/cover_plan.jsonl`, повторная запись другого содержимого запрещена. Он не содержит текстов. До генерации закоммитить этот документ, скрипт и unit tests; локальный план не коммитить.

```bash
for model in llama-3.2-3b qwen3-4b-base; do
  PYTHONPATH=src python scripts/generate_stage7_covers.py \
    --model-config "configs/models/${model}.local.json" \
    --prompts-file data/stage6/main_prompts.jsonl \
    --plan data/stage7/cache/rrc_length/cover_plan.jsonl \
    --output-root data/stage7/cache/rrc_length/cover_runs \
    --limit 1 || break
done
```

После успешных smoke запусков заменить `--limit 1` на `--all --resume` в той же команде. Не использовать основной `results/stage7/cover_runs` как output root. Основные matched cover читаются только при диагностике префиксов.

```bash
PYTHONPATH=src python scripts/stage7_rrc_length_diagnostics.py evaluate
```

Evaluate требует закоммиченного диагностического скрипта и вызывает неизменённый основной evaluator с `--dry-run`, чтобы проверить байты lock и выбранных весов. Проверяет план, metadata, revision, sender length и SHA-256 новых cover, а также общие префиксы matched/unmatched. Для новых cover выполняется scoring исходными TF-IDF и RoBERTa E4 (seed 713), неизменные пороги validation, RoBERTa max length 512. Scores исходных stego переиспользуются из проверенного raw predictions, без нового inference для stego. 4 результата (2 LM × 2 detectors) содержат matched AUC, unmatched AUC, delta = unmatched − matched, 95% percentile CI delta по парным shared article-cluster draws (2000, seed 731), метрики unmatched и диагностические показатели длины. Stego length diagnostics перенесены из основного отчёта, а не пересчитаны. Delta может быть положительной или отрицательной; post-test diagnostic не используется для выбора детектора/порогов. Сравнение не измеряет универсальную защищённость RRC и не заменяет основную метрику. Отдельная length-only диагностика и факторы реализации не входят в этот checkpoint.

Отчёт `results/stage7/analysis/rrc_unmatched_length.json` — commit-safe. Новые 240 cover scores `data/stage7/cache/rrc_length/unmatched_predictions.jsonl`, план, carrier text и IDs — локальные, игнорируются Git. Перезапись выходов запрещена; при частичной записи сохранять артефакты и расследовать сбой, не удалять автоматически.
