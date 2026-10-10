# Stage 4 — iteration 12: DAIRstega smoke и core conformance

## Real-Llama smoke

Normalized DAIRstega был запущен на основной модели `meta-llama/Llama-3.2-3B` revision `13afe5124825b4f3751f836b40dafda64c1ed062` при:

- `alpha=8`;
- `beta=1.0`;
- method-local `top_k=100`;
- common generation `top_k=None`, `top_p=1.0`;
- carrier length 16 токенов.

Получено:

- payload 27 бит;
- BPT `1.6875`;
- secret bits read 35, что согласуется с семантикой look-ahead: первые 8 бит заполняют alpha-окно, затем дочитывается ровно 27 новых бит по мере подтверждения payload;
- exact sender/receiver token-ID round-trip;
- exact secret round-trip, BER=0, decoder complete;
- `q_mode=analytic_exact`, `q_source=independent_analytic`;
- strict `D_KL(P_ref || Q)` бесконечен на всех 16 шагах;
- max reverse KL `1.7358786` bit/token;
- max TVD `0.6983451`.

Бесконечный strict KL ожидаем: method-local `top_k=100` после канонического `P_ref` создаёт support loss. Кроме того, exact conditional Q зависит от posterior-состояния перекрывающегося secret window. Поэтому finite reverse KL и TVD нужны как дополнительные сравнительные показатели. Этот smoke подтверждает корректность полного цикла, но не используется как самостоятельная оценка публикационного capacity–concealment trade-off.

Стеготекст smoke:

` more than 200 years ago, when machines tried to mimic the human mind.`

## Synthetic paper-vs-repository conformance

Добавлен независимый diagnostic layer `scripts/stage4_dairstega_conformance.py` и runner `scripts/check_stage4_dairstega_core_conformance.py`.

На 134 fixtures (6 hand-crafted + 128 random):

- normalized vs paper/ADR allocation failures: `0`;
- literal repository allocation diverged from paper/normalized rule на `29/134` fixtures (`21.6%`);
- для `beta=1` divergences: `0`;
- для `beta=0.5` divergences: `29`;
- diagnostic point selections differed 16 раз;
- max boundary displacement: 5273 integer codes;
- repository-prefix mismatch: 1 diagnostic singleton case;
- status: `PASS`.

Интерпретация ограничена conformance fixtures. `21.6%` нельзя трактовать как частоту расхождения на Llama. Причина allocation divergence в проверяемом наборе — дополнительное округление `sqrt(p)` до четырёх знаков в pinned PyTorch-коде; normalized scientific semantics следует непрерывной формуле `p^beta / sum(p^beta)`. Для `beta=1` такого дополнительного округления нет и allocation совпал на всех проверенных fixtures.

Singleton-prefix divergence соответствует заранее зафиксированному ADR-0025: статья требует максимальный общий prefix и даёт `alpha` бит для интервала ширины 1, а публичный `num_same_from_beg()` возвращает `alpha-1`.

## Direct author gate

Подготовлен `scripts/run_stage4_dairstega_author_conformance.py`. Он не модифицирует author checkout и требует `external/DAIRstega` ровно на pinned commit.

Поскольку interval allocation находится inline внутри nested encoder, direct gate:

1. исполняет оригинальные author helpers `bits2int`, `int2bits`, `num_same_from_beg`, извлечённые AST напрямую из pinned `DAIRstega_t.py`;
2. проверяет singleton-prefix поведение;
3. проверяет наличие точных source fragments, определяющих probability floor, top-k, sqrt-rounding, integer allocation, residual handling и secret-point selection;
4. использует отдельный synthetic checker для исполнения literal transcription этих allocation statements на множестве распределений.

Это разделяет прямую проверку оригинального файла и независимое алгоритмическое сравнение, не выдавая транскрипцию за вызов несуществующего публичного core API.

## Следующий gate

После PASS direct-author conformance выполнить publication-characteristic probe по фиксированным `alpha/beta` точкам и проверить направление capacity–concealment trade-off. Финальный параметрический sweep остаётся Stage 5.
