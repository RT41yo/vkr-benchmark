# Этап 4 — итерация 10: ADG closeout

## Статус

Stage-4 reproducibility gate для ADG закрыт.

Подтверждены четыре независимых аспекта:

1. **Интеграция:** normalized ADG работает через общий runner и единый `P_reference`.
2. **Надёжность:** real-LM smoke дал `BER = 0`, точный token round-trip и точное восстановление секрета.
3. **Соответствие публикации:** normalized grouping совпал с независимой реализацией Algorithm 1 на 132 синтетических распределениях, 0 failures, max `|Q_normalized-Q_paper| = 5.551e-17`.
4. **Характерный результат публикации:** 100-токенный Llama probe дал BPT `4.6300`, mean `D_KL(P_ref || Q) = 0.009936`, mean `D_KL(Q || P_ref) = 0.010232`, mean TVD `0.028920`, exact token/secret round-trip.

## Расхождение статьи и публичного кода

Pinned `adg-stega.py` напрямую подтвердил literal-поведение `near()`:

- 0 mismatches author vs literal repository semantics;
- 1 mismatch author vs paper semantics на 5 диагностических cases;
- little-endian conversion failures = 0;
- author checkout unchanged = True.

На synthetic conformance suite literal repository semantics изменила итоговый `Q` в 109/132 fixtures, max TVD `0.114661`. Эти числа используются только как conformance-диагностика и не являются частотой расхождения на реальных LM-шагах.

Normalized ADG следует Algorithm 1 публикации; публичный literal-вариант сохраняется как отдельный provenance/reference path.

## Publication-characteristic interpretation

Table 1 статьи сообщает ER `5.147/5.650/5.411` и KLD1 `0.033/0.027/0.048` для Movie/News/Tweet. На Llama-3.2-3B получены BPT `4.63` и mean forward KL `0.009936`.

Это считается воспроизведением **характера** результата — существенной embedding rate при малом, но ненулевом искажении — а не численным воспроизведением Table 1. Модели, токенизация, корпуса и контексты различаются, поэтому прямое ранжирование по абсолютным числам некорректно.

## Решение

ADG считается готовым для последующих Stage-5/6 сравнительных экспериментов. Массовые свипы, проверка устойчивости по prompts/секретам и перенос на резервную модель не входят в этот closeout.

Следующий метод Stage 4: **DAIRstega**.
