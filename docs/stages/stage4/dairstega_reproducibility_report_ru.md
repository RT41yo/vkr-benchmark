# DAIRstega — отчёт о воспроизводимости Stage 4

## 1. Итог

DAIRstega прошёл Stage-4 reproducibility gate в нормализованном benchmark-пайплайне. Проверены оба независимых слоя:

1. соответствие нашей адаптации алгоритму публикации и зафиксированной публичной реализации;
2. воспроизведение характерного научного результата публикации — управляемого компромисса между ёмкостью и качеством/необнаружимостью при изменении `alpha` и `beta`.

Полный цикл `secret -> stegotext -> recovered secret` работает с exact token/text transport и нулевой ошибкой восстановления на выполненных real-LM smoke/characteristic запусках.

## 2. Зафиксированные источники

Публикация: Yihao Wang, Ruiqi Song, Lingxiao Li, Ru Zhang, Jianyi Liu, *Dynamically Allocated Interval-Based Generative Linguistic Steganography with Roulette Wheel*, Applied Soft Computing 176 (2025), 113101 / arXiv:2401.15656v5.

Авторский код: `WangYH-BUPT/DAIRstega@8d85edf98d48c3efa827a125b6d4e90f88141ea2`, основной PyTorch-файл `DAIRstega_t.py`.

Normalized benchmark semantics зафиксированы в ADR-0025.

## 3. Normalized adaptation

Метод получает canonical `P_reference` после общих benchmark-настроек генерации. После этого DAIRstega применяет собственные method-local параметры:

- `alpha` — размер alpha-bit look-ahead окна;
- `beta` — форма распределения интервалов;
- `top_k` — внутреннее candidate restriction метода.

Secret lifecycle реализован как перекрывающиеся alpha-битные окна: метод смотрит вперёд на `alpha` бит, но продвигает secret stream только на длину максимального общего префикса выбранного интервала. Decoder восстанавливает тот же префикс по наблюдаемому токену и воспроизводимому распределению.

Explicit `Q_stego` не подменяется `P_reference`. Оно рассчитывается независимым state-aware analytic evaluator: после предыдущих наблюдений текущий secret window может лежать только в условном диапазоне допустимых alpha-битных значений, поэтому вероятность токена определяется пересечением этого диапазона с его кодовым интервалом. Provenance: `analytic_exact / independent_analytic`.

## 4. Real-LM smoke

Llama-3.2-3B, `alpha=8`, `beta=1.0`, method-local `top_k=100`, 16 carrier tokens:

- payload: `27` бит;
- BPT: `1.6875`;
- secret bits read: `35` = 8 initial look-ahead + 27 replenishment bits;
- token sequence round-trip: exact;
- secret round-trip: exact;
- decoder complete: `True`;
- strict `D_KL(P_ref||Q)`: `+inf` на 16/16 шагах;
- max `D_KL(Q||P_ref)`: `1.7358786` bits/token;
- max TVD: `0.6983451`.

Infinite strict forward KL ожидаем: method-local `top_k=100` зануляет часть поддержки canonical `P_reference`. Это является свойством выбранной DAIRstega-конфигурации, а не ошибкой общей метрики.

## 5. Paper-vs-repository conformance

Synthetic suite: 134 fixtures.

Результаты:

- normalized-vs-paper failures: `0`;
- literal repository allocation divergences: `29/134` (`21.6%`);
- divergences при `beta=1.0 / linear`: `0`;
- divergences при `beta=0.5 / sqrt`: `29`;
- selection divergences на diagnostic points: `16`;
- max boundary displacement: `5273` codes;
- repository prefix mismatches: `1`;
- status: PASS.

Основная причина `sqrt`-расхождений — дополнительное `round(sqrt(prob), decimals=4)` в pinned PyTorch-коде перед повторной нормировкой. Поэтому normalized adapter следует математической формуле публикации, а literal repository semantics сохраняются в отдельном conformance checker.

Число `29/134` относится только к диагностическому набору fixtures и не является оценкой частоты расхождения на реальных LLM-распределениях.

## 6. Direct author conformance

Прямой gate на неизменённом pinned checkout завершён со статусом PASS:

- bit conversion failures: `0`;
- author-vs-literal prefix mismatches: `0`;
- singleton author-vs-paper mismatches: `1`;
- missing source fragments: `0`;
- author checkout unchanged: `True`.

Это подтверждает, что известное singleton-prefix under-consumption принадлежит публичному авторскому helper `num_same_from_beg`, а не нашей транскрипции. Normalized adapter использует paper-defined максимальный общий префикс.

## 7. Publication-characteristic reproduction

Публикация характеризует DAIRstega не одной рабочей точкой, а trade-off-кривой. В исходной Table 3 при `beta=1` reported payload остаётся около `1.10–1.13 Bpw` при сравнительно низком PPL, а `beta=0.5` повышает reported payload до `1.89–2.58 Bpw`, но ухудшает PPL.

На Llama-3.2-3B получены четыре normalized точки по 100 carrier tokens:

| alpha | beta | BPT | raw-LM PPL | mean KL(Q||P_ref), bits/token | mean TVD |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.0 | 0.2700 | 2.2522 | 0.867062 | 0.359716 |
| 8 | 0.5 | 1.6100 | 20.5365 | 2.721758 | 0.683234 |
| 32 | 1.0 | 0.8000 | 5.0746 | 1.515890 | 0.516834 |
| 32 | 0.5 | 3.3500 | 125.3471 | 3.609336 | 0.798266 |

На всех точках:

- token/text round-trip exact;
- secret round-trip exact;
- decoder complete;
- independent exact analytic Q active;
- strict `D_KL(P_ref||Q)=+inf` на 100/100 шагах из-за internal support truncation.

Характерный результат публикации воспроизведён:

- `beta=0.5` повышает BPT относительно `beta=1.0` и при `alpha=8`, и при `alpha=32`;
- capacity-oriented `alpha=32,beta=0.5` имеет BPT `3.35` против `0.27` у quality-oriented `alpha=8,beta=1.0`;
- одновременно растут raw-LM PPL, reverse KL и TVD.

Таким образом, normalized DAIRstega действительно образует семейство рабочих точек с явным компромиссом `capacity <-> quality/concealment`.

## 8. Что не утверждается

Полученные числа не являются численным воспроизведением Table 3. Исходная работа и benchmark различаются по LM, токенизации, prompting, определению единицы payload (`Bpw` против benchmark `BPT`) и условиям формирования распределения. Поэтому научно корректный вывод — воспроизведение направления и структуры trade-off, а не совпадение абсолютных значений.

Также высокая PPL точки `alpha=32,beta=0.5` (`125.3471`) не означает автоматически, что метод в целом имеет такое качество: это одна Stage-4 diagnostic точка на одном prompt/secret и одном run. Окончательные рабочие диапазоны параметров определяются на Stage 5 пилотными sweep-экспериментами.

## 9. Вывод для Stage 4

DAIRstega считается закрытым на уровне интеграции и воспроизводимости Stage 4:

- encoder/decoder интегрированы;
- secret lifecycle проверен;
- explicit Q считается независимо;
- real-LM round-trip успешен;
- paper-vs-repository conformance задокументирован;
- direct author gate успешен;
- характерный capacity-concealment trade-off публикации воспроизведён.

Следующий шаг относится уже не к DAIRstega как отдельному методу, а к общему Stage-4 closeout: compatibility matrix для Llama/Qwen, сводный отчёт по современным методам и проверка готовности benchmark v0.3 перед Stage 5.
