# Этап 4 — итерация 9: ADG publication-characteristic probe

## 1. Что уже подтверждено до этого шага

После real-LM smoke и двух conformance-проверок цепочка источников разделена явно.

Synthetic paper-vs-repository conformance на 132 распределениях:

- normalized ADG vs независимая реализация Algorithm 1: 0 failures;
- max `|Q_normalized - Q_paper| = 5.551e-17`;
- literal repository semantics дала отличающийся `Q` в 109/132 fixtures;
- max TVD literal-repository vs paper = `0.114661`;
- witness: при `target=0.16` paper-rule выбирает `0.18`, repository `near()` — `0.11`.

Прямой запуск pinned авторского `adg-stega.py` подтвердил:

- 5 диагностических `near()` cases;
- author vs literal-repository mismatches = 0;
- author vs paper mismatches = 1;
- little-endian bit conversion failures = 0;
- author checkout после проверки не изменён;
- итог `PASS`.

Следовательно, расхождение не является артефактом нашей транскрипции: pinned публичный файл действительно исполняет literal `near()` semantics, тогда как normalized ADG сознательно следует Algorithm 1 публикации.

## 2. Следующая публикационная цель

После implementation conformance выполняется отдельный publication-characteristic probe относительно Table 1 Zhang et al. (2021).

В статье для ADG сообщаются:

- Movie: ER `5.147` бит/токен, KLD1 `0.033` бит/токен;
- News: ER `5.650`, KLD1 `0.027`;
- Tweet: ER `5.411`, KLD1 `0.048`.

В benchmark **не заявляется численное воспроизведение этих шести чисел**, потому что экспериментальная среда принципиально другая: современная Llama вместо отдельно обученных word-level LSTM, subword tokenization вместо слов, benchmark prompt вместо Movie/News/Tweet corpora и канонический `P_reference`.

Проверяется характерное научное поведение:

> ADG встраивает существенную полезную нагрузку, при этом `D_KL(P_reference || Q_stego)` остаётся малым, но обычно ненулевым.

`Q_stego` при этом строится независимо полным обходом дерева групп, а не подменяется теоретическим утверждением или значением из авторского кода.

## 3. Probe

Скрипт:

```bash
python scripts/probe_stage4_adg_characteristic.py \
  configs/models/llama-3.2-3b.local.json \
  --carrier-tokens 100 \
  --repeats 1 \
  --secret-id stage4-adg-characteristic \
  --method-seed 42 \
  --json-output results/stage4/adg/characteristic_llama.json
```

Один 100-токенный run выбран как Stage-4 characteristic probe, а не как основной статистический эксперимент. Полные свипы и устойчивость выводов по множеству промптов/секретов относятся к следующим этапам roadmap.

Из-за дорогого независимого explicit-`Q` этот probe существенно медленнее обычного алгоритмического запуска; стоимость полного перечисления ветвей остаётся в `metric_instrumentation_ms` и не включается в `stego_algorithm` performance.

## 4. Что считать успешным результатом

Технический `PASS` требует:

- exact token/text round-trip;
- exact secret round-trip и complete decoder;
- конечные dual-KL/TVD;
- `q_mode = exact_enumeration`;
- `q_source = independent_enumeration`.

Сам publication-characteristic интерпретируется после получения чисел. Абсолютное попадание ER/KLD1 в диапазоны Table 1 не является критерием `PASS`; сравнение с таблицей носит качественный характер из-за несовпадающей LM/токенизации/корпусов.


## 5. Полученный результат

Фиксированный 100-токенный запуск завершился `PASS`:

- BPT = `4.630000`;
- mean `D_KL(P_ref || Q) = 0.009936`;
- mean `D_KL(Q || P_ref) = 0.010232`;
- mean TVD = `0.028920`;
- `nonzero forward distortion observed = True`;
- token round-trip = exact;
- secret round-trip = exact.

Поэтому публикационный gate закрыт как качественное воспроизведение паттерна Table 1: ADG сохраняет существенную емкость при малом, но ненулевом KL. Полученные абсолютные значения не объявляются прямым численным воспроизведением Table 1 и не используются для вывода об оптимальности метода.
