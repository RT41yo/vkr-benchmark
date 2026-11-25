# Stage 7 — post-lock исправление формата JSONL в финальной оценке

**Основание:** в ветке `stage7-steganalysis` после фиксации исходного pre-test lock коммитом `e03cdd6` первый разрешённый запуск `evaluate_stage7_final.py --open-frozen-test` остановился с `JSONDecodeError: Unterminated string starting at: line 1 column 330 (char 329)`.

**Замороженные входы (не изменяются):**

- Dataset SHA-256: `43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9`.
- Pre-test lock SHA-256: `ba88ee49f92d7c12b64c7baad7329e10f6af463ab8b0857d2f27e900ed2654ce`.
- RoBERTa E4 seed 713 для обеих языковых моделей; baseline TF-IDF + Logistic Regression, validation thresholds и все остальные настройки `configs/stage7/final_evaluation.json` остаются прежними.

## Обнаруженный дефект

`load_frozen_test()` раньше разделял корректный UTF-8 JSONL через `str.splitlines()`. Эта функция принимает за разделители не только LF (`\n`), но и символы `U+0085`, `U+2028`, `U+2029`. При `json.dumps(..., ensure_ascii=False)` такие символы могут находиться **внутри** легального JSON-значения `text` и не должны разбивать записи. На синтетическом примере это воспроизведено; точный вызвавший ошибку символ в локальном корпусе следует подтвердить отдельно, если необходимо.

Второй дефект был обнаружен статической проверкой до записи итоговых предсказаний: runner формировал файл с расширением `.jsonl` вызовом `canonical_bytes()` для каждого объекта; `canonical_bytes()` форматирует JSON с `indent=2`, то есть делает его многострочным. Это не корректный формат JSONL.

## Исправление (формат/десериализация, без влияния на scoring)

1. `src/vkr_benchmark/steganalysis/final_evaluation.py`: записи frozen dataset разделяются **только** по фактическому символу LF (`split('\n')`). Содержимое `text` возвращается побайтово эквивалентным декодированному UTF-8 входу; требования к SHA-256 датасета и текстов сохраняются. Добавлен отдельный однострочный JSONL-сериализатор для raw predictions.
2. `scripts/evaluate_stage7_final.py`: raw predictions сохраняются как настоящий JSONL — один объект на строку. Сами значения score, порядок и состав полей не меняются.
3. `tests/unit/test_stage7_final_evaluation.py`: добавлены две регрессионные проверки с символами Unicode-разделителей строк и roundtrip выходного JSONL.

Оригинальная функция `canonical_bytes()` для lock и итогового агрегированного JSON-отчёта **не меняется**. Не меняются набор test, checkpoint, bootstrap, пороги, RoBERTa preprocessing, численные метрики или имена выходных файлов. Ошибка чтения происходила внутри `load_frozen_test()` **до** `_evaluate()` и, следовательно, до расчёта тестовых score; исходный runner записывает результаты только после возврата `_evaluate()`. Тем не менее попытка открыть frozen test уже произошла — это исправление документируется как *post-lock I/O erratum*, а не как первоначальный pre-test lock.

## Порядок применения

1. Проверить, что `results/stage7/final_evaluation_report.json` и `results/stage7/predictions/raw/final_test_predictions.jsonl` не появились. **Ничего не удалять и не перезаписывать**, если они существуют.
2. Применить overlay на текущую ветку `stage7-steganalysis` (родительский каталог архива `vkr-benchmark/`).
3. Запустить `pytest -q`. На присланном актуальном source-only snapshot: **544 passed** (исходно 542).
4. Посмотреть `git diff --check` и staged diff. Коммитить **только** исправленные скрипт, модуль, unit-тест и этот документ в ту же ветку. Выполнить push. Git-коммит ОБЯЗАТЕЛЕН **до повторного test scoring**, поскольку runner сравнивает точные байты своих файлов с `HEAD`.
5. Проверить `PYTHONPATH=src python scripts/evaluate_stage7_final.py --dry-run`: ожидается тот же `lock_sha256`, статус `committed_lock_verified`, `test_access=none`.
6. Только после этого разрешён повторный запуск с `--open-frozen-test --lock-sha256 ba88ee49f92d7c12b64c7baad7329e10f6af463ab8b0857d2f27e900ed2654ce`.

Если SHA-256 lock или датасета изменился, появились частичные выходные файлы или возникла новая ошибка — остановиться и разбирать отдельно, не ослабляя защиты и не меняя параметры оценки.
