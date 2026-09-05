# Материал для презентации Этапа 2

Цель презентации — показать не научное сравнение методов, а то, что создана воспроизводимая базовая экспериментальная инфраструктура и все три baseline method проходят один и тот же end-to-end protocol.

## Слайд 1. Этап 2: цель и результат

Коротко:

- адаптировать Bins, Huffman, Arithmetic Coding;
- унифицировать inputs / P_reference / encode-decode / metrics;
- получить единый storage/reporting format;
- результат: technical milestone benchmark v0.2.

Главный тезис: различия измеряются после общей точки `P_reference`, а внешние условия фиксируются одинаково.

## Слайд 2. Общая экспериментальная цепочка

Схема:

`prompt -> LM/KV -> raw logits -> canonical P_reference -> stego method -> carrier token -> text channel -> retokenize -> decoder`

Отдельно визуально подчеркнуть, что decoder получает только обычный текст, а не sender token IDs.

## Слайд 3. Единый adapter layer для трех методов

Показать общий lifecycle `StegoMethod -> EncoderSession / DecoderSession` и три method-specific блока:

- Bins: secret bits выбирают bin, затем наиболее вероятный token внутри bin;
- Huffman: top `2^b` -> Huffman tree -> variable-length code;
- Arithmetic Coding: stateful finite-precision interval + look-ahead, причем read bits != confirmed payload.

## Слайд 4. Что унифицировано

Показать одинаковые для методов компоненты:

- model revision;
- prompt id;
- secret id / SHAKE256 stream;
- canonical reference policy;
- termination target;
- text transport;
- metric layer;
- run storage.

Method-internal параметры вынести отдельно, чтобы не смешивать их с common top-k/top-p.

## Слайд 5. Базовые метрики

Сгруппировать по смыслу:

- capacity: BPT;
- efficiency: entropy utilization;
- distribution distortion: KL/TVD;
- realized LM quality: raw-LM NLL/PPL;
- reliability: BER / exact roundtrip;
- computation: ms/token и bits/s.

Коротко отметить: `KL(P_ref || Q)` может быть `inf` при support loss; TVD остается градуированной конечной мерой.

## Слайд 6. Воспроизводимый run и storage

Схема:

`canonical config -> run_id -> results/runs/<run_id>/... -> summary.parquet -> Markdown report`

Показать, что повтор той же canonical config дает тот же `run_id`, а `summary.parquet` делает upsert, а не logical duplicate.

## Слайд 7. Stage-2 smoke results

Таблица:

| Method | BPT | Entropy util. | TVD | PPL | BER | Encode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arithmetic Coding | 2.4375 | 65.84% | 0.0534 | 6.0516 | 0 | 62.551 |
| Bins | 2.0000 | 64.60% | 0.5962 | 17.8125 | 0 | 18.734 |
| Huffman | 2.1875 | 52.61% | 0.4222 | 9.0745 | 0 | 17.292 |

Под таблицей обязательно: `1 model x 1 prompt x 1 secret x 1 operating point x 16 carrier tokens; pipeline validation only`.

## Слайд 8. Что видно уже сейчас, а что утверждать нельзя

Можно сказать:

- pipeline различает методы по capacity/distortion/quality/computation;
- AC в smoke-run лучше по BPT/TVD/PPL, но дороже;
- Huffman имеет малую вычислительную стоимость stego core;
- все три метода прошли exact text-only roundtrip.

Нельзя говорить:

- что AC в целом лучший метод;
- что численные различия статистически значимы;
- что эти operating points оптимальны;
- что timings являются устойчивыми средними.

## Слайд 9. Два направления KL

Показать рядом:

`D_KL(P_reference || Q_stego)` — benchmark-native / Cachin-oriented direction, чувствительно к support loss;

`D_KL(Q_stego || P_reference)` — author-compatible direction для Harvard/Ziegler reproducibility.

На Stage-2 smoke первая KL = `inf` у всех трех методов. Это не ошибка. В Stage 3 считаются оба направления для корректного сопоставления с авторами.

## Слайд 10. Статус ROADMAP и следующий этап

Закрыто технически:

- три baseline adapters;
- unified runner;
- полный baseline metric vector;
- persistent storage/reporting;
- automated checks/readiness gate;
- metric documentation.

Следующий этап: проверить воспроизводимость Bins/Huffman/Arithmetic Coding относительно upstream/publication settings, не смешивая `normalized` и `author-compatible` режимы.
