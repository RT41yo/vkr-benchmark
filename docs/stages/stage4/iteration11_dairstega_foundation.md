# Stage 4 — iteration 11: DAIRstega foundation

## Цель

Зафиксировать paper/code semantics DAIRstega до масштабных прогонов, реализовать normalized encoder/decoder и exact state-aware Q, затем подготовить короткий real-LM smoke.

## Paper/code audit

Публикация задаёт top-k candidate pool, alpha-bit roulette, `n_j ∝ p_j^beta`, выбор токена по интервалу и восстановление общего prefix. Pinned `DAIRstega_t.py` уточняет целочисленную часть: `2^PRECISION` кодовых точек, alpha-dependent probability floor, power mapping с последующей нормировкой, integer rounding, truncation при overfill и перенос остатка первому токену.

Зафиксированы четыре потенциально важные зоны conformance:

1. `2^alpha-1` в формуле статьи против `2^alpha` реально адресуемых alpha-битных значений и author code;
2. Formula 9 против фактической нормировки `p^beta / sum(p^beta)`;
3. дополнительный probability-floor filter в коде, отсутствующий в Algorithm 1;
4. `num_same_from_beg()` возвращает `alpha-1` для singleton interval, тогда как максимальный общий prefix равен `alpha`.

Нормализованные решения закреплены ADR-0025.

## Реализация

Добавлены:

- `src/vkr_benchmark/methods/dair.py`;
- `src/vkr_benchmark/distributions/dair_q.py`;
- registration в unified runner;
- `configs/experiments/stage4_dairstega.example.json`;
- `scripts/check_dairstega_e2e.py`;
- unit/integration/bootstrap tests.

### Secret lifecycle

Adapter хранит alpha-битный look-ahead buffer. После выбранного токена реально продвигаются только `k` prefix-бит; оставшиеся `alpha-k` переиспользуются, справа дочитываются ровно `k` новых бит. Это позволяет общему runner корректно отличать прочитанный look-ahead от authoritative payload.

### Exact Q

В отличие от простого `interval_width / 2^alpha`, используется exact conditional distribution с учётом posterior-range текущего перекрывающегося alpha-битного окна. Independent distribution evaluator заново строит интервалы и считает длины их пересечений с этим диапазоном.

## Следующий gate

1. Запустить полный regression suite.
2. Запустить real-Llama smoke для `alpha=8, beta=1, top_k=100`.
3. После успешного smoke реализовать direct author core conformance и paper-vs-code discrepancy fixtures.
4. Затем выполнить publication-characteristic probe по нескольким alpha/beta points.
