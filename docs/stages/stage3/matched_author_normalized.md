# Step 3.12 — matched author-compatible vs normalized comparison

## Цель

Проверить, что именно меняется при переносе Bins, Huffman и Arithmetic Coding из author-compatible контура в нормализованный benchmark, не смешивая это с уже обнаруженной неопределённостью исторического Figure-3 batch driver.

Step 3.12 является **парным дифференциальным экспериментом**, а не повторной попыткой воспроизвести Figure 3. Author-side берётся из уже зафиксированного полного sweep Step 3.10; normalized-side запускается заново на тех же входах.

## Зафиксированная выборка

Используются первые 8 frozen CNN/DailyMail contexts и replicate 0. Для каждого контекста сравниваются четыре representative points:

- `bins_b3`;
- `huffman_e3`;
- `arithmetic_t0.9_k300`, `precision=26`;
- `arithmetic_t1.0_k50256`, `precision=26`.

Итого: `8 × 4 = 32` matched pairs.

Для каждой пары используется тот же `paper_sentence_sha256_bitstream_v1` (`seed=1234`, `replicate=0`, 16384 доступных бит). SHA-256 потока обязан совпасть с SHA, записанным в соответствующем author Figure-3 record.

## Выравнивание carrier length

Author-side уже является законченным предложением. Его число `sender_token_ids` фиксирует длину пары `N`.

Normalized-side генерирует **ровно N токенов**, без sentence stopping. Поэтому сравнение не зависит от того, когда normalized генерация поставила бы точку, и не требует копировать недокументированный historical termination driver.

Такой дизайн сравнивает методы при одинаковой длине носителя, но не утверждает, что тексты должны совпадать токен-в-токен.

## Что намеренно не копируется из author-compatible режима

Normalized pipeline сохраняет решения Этапа 2:

- canonical `P_reference` строится общим builder'ом;
- GPT-2 EOT исключается как special token;
- author-only mask token id `628` не добавляется;
- Bins partition строится над normalized `V_allowed` и отдельным `MethodRandomSource`, а не через глобальный NumPy RNG автора;
- Huffman и Arithmetic используют normalized method contracts;
- Arithmetic finite precision и method top-k остаются параметрами normalized adapter, без historical text/BPE repair quirks.

Это и есть различия, влияние которых требуется измерить, а не скрыть.

## Метрики и ADR-0012

До matched run metric/storage layer расширяется двумя явно именованными направлениями KL:

- benchmark-native: `kl_ref_to_stego_* = D_KL(P_reference || Q_stego)`;
- author-direction: `kl_stego_to_ref_* = D_KL(Q_stego || P_reference)`.

Старые generic `kl_*` поля остаются compatibility aliases только для `P_reference -> Q_stego`; новые runs всегда сохраняют direction-qualified поля. Для старых summary rows reverse direction остаётся `null`, а не восстанавливается задним числом.

Для pair comparison используются author `kl_q_stego_to_p_lm_bits_author` и normalized `kl_stego_to_ref_mean_bits`. Это совпадающее **направление**, но не обязательно идентичный reference distribution. Например, normalized `P_reference` при `tau=0.9` содержит canonical temperature transform, тогда как author Arithmetic paper metric считает KL относительно untempered LM distribution. Step 3.12 поэтому не вводит численный pass/fail threshold; эту разницу разбирает Step 3.13.

Дополнительно сохраняются capacity/BPT, TVD, raw-LM NLL, payload, token agreement и exact token-ID decode recovery.

## Execution gates

Run считается пригодным для Step 3.13 только если выполнены все условия:

```text
32 / 32 matched pairs
same frozen GPT-2 Medium revision
same frozen context hashes
same secret-stream hashes
author record status = ok
normalized carrier length = author carrier length
normalized token-ID decode = exact
both KL directions explicitly accounted
```

Token sequence parity не является gate. Scientific closeness также не оценивается в Step 3.12.

## Запуск

```bash
python scripts/check_stage3_matched_author_normalized.py --preflight
python scripts/run_stage3_matched_author_normalized.py
python scripts/check_stage3_matched_author_normalized.py
```

Результаты создаются в:

```text
results/stage3/matched_author_normalized/
├── records.jsonl
├── paired_comparison.csv
└── summary.json
```

После успешного checker следующий этап — **Step 3.13 conformance/discrepancy analysis**.
