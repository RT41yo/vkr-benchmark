# Stage 4 — отчет по совместимости методов с языковыми моделями и токенизаторами

## Итог

Проверены все 14 комбинаций `7 методов × 2 модельно-токенизаторные пары` на одной фиксированной smoke-конфигурации. Все комбинации успешно прошли direct token-ID round-trip, обычный текстовый канал `tokens → text → tokens` и восстановление секрета после повторной токенизации.

Для Llama-3.2-3B все семь методов имеют статус `PASS`. Для Qwen3-4B-Base все семь методов имеют статус `PASS_WITH_ADAPTATION`: перед построением канонического `P_reference` общая инфраструктура исключает 267 output ID модели, отсутствующих в словаре токенизатора, и перенормирует распределение. Эта адаптация относится к модельно-токенизаторной паре и не является специальным правилом отдельного стегометода.

## Условия запуска

- prompt: `The history of artificial intelligence began`;
- `secret_id = 000001`;
- fixed-carrier: 16 токенов для Bins, Huffman, Arithmetic Coding, Discop, ADG и DAIRstega;
- RRC: payload 64 бита, safety cap 128 carrier tokens;
- common generation: `top_k=None`, `top_p=1.0`;
- metric instrumentation отключена: explicit `Q_stego`, KL/TVD, entropy и raw-LM NLL не рассчитывались.

Проверка является технической smoke-проверкой совместимости, а не сравнительным экспериментом по качеству, емкости или скорости.

## Модельно-токенизаторные пары

### Llama-3.2-3B

Revision: `13afe5124825b4f3751f836b40dafda64c1ed062`.

- model output vocab: 128256;
- tokenizer size: 128256;
- output-only ID: 0;
- после исключения двух специальных ID: `|V_allowed| = 128254`.

Все семь методов прошли без модельно-специфической адаптации.

### Qwen3-4B-Base

Revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`.

- model output vocab: 151936;
- tokenizer size: 151669;
- output-only ID: 267;
- после исключения 267 output-only ID и 14 специальных ID: `|V_allowed| = 151655`.

Все семь методов прошли полный цикл после уже зафиксированной общей операции `exclude_and_renormalize_267_output_only_ids`.

## Матрица

| Метод | Llama-3.2-3B | Qwen3-4B-Base |
|---|---|---|
| Bins | PASS | PASS_WITH_ADAPTATION |
| Huffman | PASS | PASS_WITH_ADAPTATION |
| Arithmetic Coding | PASS | PASS_WITH_ADAPTATION |
| Discop | PASS | PASS_WITH_ADAPTATION |
| RRC | PASS | PASS_WITH_ADAPTATION |
| ADG | PASS | PASS_WITH_ADAPTATION |
| DAIRstega | PASS | PASS_WITH_ADAPTATION |

## Round-trip diagnostics

Во всех 14 ячейках одновременно выполнены три условия:

- `direct_secret_roundtrip_exact = true`;
- `text_token_roundtrip_exact = true`;
- `text_secret_roundtrip_exact = true`.

Это означает, что на данной фиксированной smoke-точке не обнаружено ни алгоритмической несовместимости со способом построения `P_reference`, ни сбоя текстового транспорта для выбранных последовательностей.

RRC использовал 12 carrier tokens на Llama и 26 на Qwen для одного и того же payload 64 бита. Это наблюдение не используется как межмодельное сравнение эффективности: матрица не является статистическим экспериментом, а единичная траектория зависит от конкретных `P_reference` вдоль генерации.

Аналогично различия payload у остальных методов между Llama и Qwen не интерпретируются как сравнительный результат Stage 4. Количественное сравнение на второй модели относится к последующим экспериментальным этапам.

## Вывод

Compatibility gate Stage 4 закрыт. Все семь методов технически совместимы с основной моделью Llama-3.2-3B и резервной Qwen3-4B-Base в нормализованной инфраструктуре benchmark v0.3. Для Qwen требуется только общая, заранее зафиксированная нормализация допустимого пространства токенов; специальных изменений логики отдельных стегометодов не требуется.

Машинный результат: `results/stage4/compatibility_matrix.json`.

Компактная таблица: `results/stage4/compatibility_matrix.md`.
