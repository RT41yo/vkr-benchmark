# Stage 7 — pre-test lock и финальная оценка обнаруживаемости

**Точка отсчёта:** `stage7-steganalysis@7966d31` (аудит E4 и эксперимент E8 уже зафиксированы). Этот checkpoint добавляет **код и предварительную фиксацию** внешней оценки. До сохранения/коммита `results/stage7/final_evaluation_lock.json` тестовый корпус нельзя открывать для вычисления score.

## Неподвижные нормативные входы

- Датасет: SHA-256 `43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9`, 5639 пар (11278 текстов), 152 article groups, включая 1800 frozen Stage 6 test пар и один документированный Qwen/RRC/validation cutoff.
- `configs/stage7/detector_training.json`: SHA-256 `2eb26ae60e768b0e55f041c53b2221a87bbb433f357e7523a5665a96a60b17ac`. Если фактический SHA другой, **не** менять контрольную сумму ради прохождения проверки: сначала выяснить причину.
- `results/stage7/roberta_primary_audit.json`: SHA-256 `bdcdd4443418e2d6b529d1cdea334ba23c08523c6e5e12614fb2aea6e54d37f5`.
- Предобученная RoBERTa-base revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`; **основные E4 seed 713, epoch 4** для обеих LM. Все шесть E4 остаются проверенными, но только один checkpoint на LM поступает на frozen test. E8 не используется в test и не служит для пересмотра выбора.
- Baseline: ранее обученные TF-IDF char 3–5 + Logistic Regression. Конкретные веса (`joblib`), validation predictions и summaries хешируются и фиксируются в lock. Исторический baseline config SHA отличается от позже закреплённого RoBERTa config SHA; это допустимо и явно отражено в lock.
- `configs/stage7/final_evaluation.json` содержит все остальные зафиксированные настройки: 15 operating points, одинаковый pooled validation threshold для всех точек LM, positive-class score для stego, test support Llama 72 пар/точку и Qwen 48 пар/точку.

## Метрики

1. Главные: **30 per-point ROC-AUC** (15 методов × 2 LM) для основного RoBERTa E4. Вторичные: **30 аналогичных результатов baseline**, а также 4 pooled ROC-AUC (два детектора × две LM). Никакого выбора порога, инверсии score или seed по результатам test.
2. 95% percentile article-cluster bootstrap, 2000 ресэмплов, seed 731. 24 статьи одновременно принадлежат обеим LM; общий вектор bootstrap-кратностей применяется ко всем методам, двум LM и обоим детекторам. Cover/stego и все повторы одной статьи ресэмплируются **совместно**. Для точного ROC-AUC используем pairwise positive-negative comparisons с половинным вкладом ties; такой расчёт эквивалентен обычной ROC-AUC с кратностями кластеров. На frozen test каждый кластер содержит оба класса, поэтому нулевые классы не появляются. Квантили — NumPy `quantile` с обычной линейной интерполяцией.
3. Дополнительные показатели для каждой точки/pooled: accuracy, balanced accuracy, precision, recall, F1, confusion matrix (TN/FP/FN/TP), trapezoidal PR-AUC и average precision (AP; **не** смешивать две метрики). Positive label = stego. Threshold берётся только из pooled validation выбранного экземпляра и неизменен по точкам.
4. Контроль длины: средняя длина в символах и словах, доля `.?!` в конце, доля превышения 512 RoBERTa tokens — отдельно для cover/stego, **диагностически**, без изменения тестового порога и без отдельного классификатора длины. Самостоятельный length-only classifier и shuffled-label baseline предусмотрены как дальнейшие quality controls.
5. Тестовые сообщения никогда не передаются для обучения или выбора модели. Test scoring использует только transported text, а не prompt, secret, токены/идентификаторы методов или PPL. Metadata используются после scoring исключительно для разметки результатов по точкам и bootstrap.

## I. Инсталляция и тесты (до открытия test)

Распаковать overlay из корня родительского каталога (`-d ..`). После этого:

```bash
cd /home/matmul/Documents/ITMO/VKR/projects/vkr-benchmark
source .venv/bin/activate
pytest -q
```

## II. Read-only preflight (не открывает test)

```bash
PYTHONPATH=src python scripts/prepare_stage7_final_evaluation.py
```

Ожидаем `status=sealed_pretest`, `test_access=none`. Проверяются **только** конфигурация и манифест датасета, зафиксированный аудит, шесть сведений о RoBERTa, SHA-256 файлов выбранных весов, baseline модель и validation predictions/threshold. Тексты frozen test не читаются; GPU не используется. Если файлы отсутствуют или hash не совпадает — исправить причину, ничего автоматически не переписывать.

После успешного preflight:

```bash
PYTHONPATH=src python scripts/prepare_stage7_final_evaluation.py --write-lock
```

Будет создан `results/stage7/final_evaluation_lock.json` без текстов, секретов и локальных абсолютных путей. Запомни `lock_sha256` из вывода.

## III. Сначала commit/push lock вместе с кодом

**Не открывать frozen test до следующего коммита!** Убедись, что в staging только код, конфиг, протокол, тесты и lock; длинные тексты, модели, validation/test predictions, `.venv` игнорируются. Рекомендуемый commit: `Lock Stage 7 final steganalysis evaluation protocol before test`.

Проверочный dry-run уже после commit (проверяет точное совпадение содержимого lock, protocol и evaluation runner с `HEAD`):

```bash
PYTHONPATH=src python scripts/evaluate_stage7_final.py --dry-run
```

Ожидаем `status=committed_lock_verified`, `test_access=none`. Если `HEAD` не содержит exact lock + код — откажет до обращения к test.

## IV. Только после принятого lock и отдельного запуска — тестовая оценка

Нужно явно передать зафиксированную в commit SHA-256:

```bash
PYTHONPATH=src python scripts/evaluate_stage7_final.py \
  --open-frozen-test \
  --lock-sha256 'ВСТАВЬ_ТОЧНЫЙ_lock_sha256_ИЗ_PREFLIGHT'
```

Ожидается 60 per-point rows для двух детекторов и 4 pooled rows; все 1800 пар из frozen Stage 6, 24 группы статей. GPU используется только для RoBERTa inference; обучение не выполняется. Наиболее чувствительные артефакты остаются **локально**:

- `results/stage7/predictions/raw/final_test_predictions.jsonl` — raw per-record scores с идентификаторами, но **без carrier text**; gitignored.
- `results/stage7/final_evaluation_report.json` — 64 компактные агрегированные строки и контрольный SHA-256 raw predictions; можно внести в будущий коммит после изучения результатов.

Если выходные файлы уже существуют, скрипт не перезапишет их. Веса и frozen Stage 6 artifacts неизменны.

## Ограничения

- Confidence interval покрывает неопределённость **24 article groups test**, а не разброс training seeds и не неопределённость от выбора детектора на validation. Три E4 seeds и результаты E8 остаются отдельными отчетами.
- Разница E4–E8 на validation не является основанием для ретроспективного выбора E8 в test.
- Оценка условна на сформированных текстах; единственный Stage 7 validation RRC nontermination сообщается отдельно. Test Stage 6 включает все 97 hard decode failures по уже утверждённому правилу.
- Перебор новых гиперпараметров или инверсия ROC-AUC после получения test результатов запрещены без явно новой, помеченной exploratory версии протокола.
