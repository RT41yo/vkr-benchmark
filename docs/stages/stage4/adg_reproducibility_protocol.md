# Этап 4 — протокол воспроизводимости ADG

## 1. Источники

Публикация: Siyu Zhang, Zhongliang Yang, Jinshuai Yang, Yongfeng Huang, *Provably Secure Generative Linguistic Steganography*, Findings of ACL-IJCNLP 2021.

Авторская реализация: `Mhzzzzz/ADG-steganography@b4a7e802a97bc3a3b84d07e63f24627b16c51a10`.

Нормализованная семантика зафиксирована в ADR-0024.

## 2. Два независимых слоя проверки

### Слой A — соответствие реализации

Проверяются алгоритмические инварианты, не зависящие от исторической LSTM:

- выбор числа групп из текущего `p_max`;
- субоптимальное заполнение групп;
- little-endian отображение секретных битов в индекс группы;
- рекурсивное продолжение только внутри выбранной группы;
- нормализация и стохастический выбор внутри терминальной группы;
- обратное восстановление битов нашим декодером.

Особый пункт — `near()`. Статья требует ближайшую к `epsilon` вероятность, тогда как публичный код содержит сравнение с индексом `up`. Поэтому conformance должен отдельно показать совпадающие случаи и случаи расхождения literal-repository vs paper semantics.

### Слой B — воспроизведение характерного результата публикации

Основной внешний ориентир — Table 1. В статье ADG имеет ER порядка 5.1–5.7 bpp и `KLD1 = D_KL(p_LM || q)` порядка 0.027–0.048 на трёх корпусах.

Для normalized Llama-режима не заявляется численное совпадение с Table 1. Проверяемый характерный результат:

> ADG должен обеспечивать высокую полезную нагрузку при малом, но обычно ненулевом `D_KL(P_reference || Q_stego)`.

Для этого benchmark независимо строит полный `Q_stego`, считает оба направления KL и TVD и параллельно измеряет BPT/эффективность использования энтропии.

## 3. Первый интеграционный gate — пройден

Normalized ADG прошёл:

1. synthetic encode/decode round-trip;
2. synthetic explicit-Q sanity check с сохранением support;
3. real-LM smoke на Llama-3.2-3B;
4. unified-run через `scripts/run_experiment.py`;
5. `BER = 0`, `roundtrip_exact = True`, `decoder complete = True`;
6. явный `q_mode = exact_enumeration`, без equality certificate.

В real-LM smoke на 16 токенах встроен 71 бит (`4.4375` BPT), mean `D_KL(P_ref || Q) = 0.009492` бит/токен, mean TVD `0.026408`. Это интеграционная проверка, а не прямое численное воспроизведение Table 1.

Paper-vs-repository conformance, прямой запуск pinned `near()` и publication-characteristic probe пройдены. Stage-4 reproducibility gate для ADG закрыт.

## 4. Граница `P_reference`

ADG получает уже готовый канонический `P_reference`. Общие `temperature`, `top-k`, `top-p` и исключение special tokens не входят в ADG. Внутреннего candidate-pool параметра у normalized ADG нет.

## 5. Случайность

`method.random_seed` управляет только терминальным сэмплированием токена внутри выбранной группы и нужен для воспроизводимости запуска. Декодирование определяется наблюдаемым токеном и текущим `P_reference`; общий ключ не требуется.

## 6. Производительность

Время кодирования ADG включает построение групп по фактически выбранному секретному пути и терминальный сэмплинг. Полный обход всех ветвей для explicit-`Q` является metric instrumentation и учитывается отдельно.


## 7. Paper-vs-repository conformance — пройден

Synthetic suite (132 распределения): normalized ADG совпал с независимой Algorithm-1 реализацией без failures (`max |Q_normalized-Q_paper| = 5.551e-17`), тогда как literal repository semantics дала отличающийся `Q` в 109/132 fixtures; max TVD составил `0.114661`. Это conformance-диагностика, а не частота расхождения на реальных LM-шагах.

Прямой запуск pinned `adg-stega.py` на 5 диагностических `near()` cases дал 0 расхождений с literal transcription и 1 расхождение с paper-rule. Little-endian helpers прошли без failures; author checkout остался неизменным. Таким образом, обнаруженное отличие подтверждено непосредственно публичным авторским файлом.

## 8. Publication-characteristic probe

Отдельный `probe_stage4_adg_characteristic.py` проверяет на pinned Llama характерный результат Table 1: существенную embedding rate при малом, но обычно ненулевом `D_KL(P_ref || Q)`. Абсолютные числа Table 1 не являются pass/fail порогами из-за другой LM, токенизации, корпуса и контекста. Probe по умолчанию использует 100 токенов и сохраняет полный JSON с dual-KL/TVD, BPT, entropy utilization, exact-Q provenance и round-trip.


## 9. Publication-characteristic result — пройден

100-токенный probe на pinned Llama-3.2-3B дал:

- BPT `4.6300`;
- mean `D_KL(P_ref || Q) = 0.009936` бит/токен;
- mean `D_KL(Q || P_ref) = 0.010232` бит/токен;
- mean TVD `0.028920`;
- exact token round-trip;
- exact secret round-trip;
- ненулевое forward distortion наблюдается.

Table 1 статьи сообщает ER `5.147/5.650/5.411` и KLD1 `0.033/0.027/0.048` для Movie/News/Tweet. Числа не сравниваются как строгая репликация из-за другой LM, токенизации, корпусов и контекста. Воспроизведен характерный результат: существенная embedding rate при малом, но ненулевом распределительном искажении. Один 100-токенный run не используется для утверждений о среднем качестве или оптимальной рабочей точке.

Подробный closeout: `adg_reproducibility_report_ru.md`.
