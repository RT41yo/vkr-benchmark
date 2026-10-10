# Stage 4 — iteration 15: compatibility matrix runner

Подготовлен единый технический smoke-runner для 14 комбинаций `7 методов × 2 модели`.

Главное методологическое решение: compatibility run использует обычные encoder/decoder sessions и канонический `P_reference`, но намеренно не вызывает metric instrumentation (`distribution_info`, explicit Q, KL/TVD, entropy/NLL). Это не меняет логику выбора carrier tokens и восстановления секрета, зато не смешивает техническую совместимость с дорогой диагностикой, особенно заметной у ADG.

Runner последовательно загружает Llama-3.2-3B и Qwen3-4B-Base, на каждой модели запускает Bins, Huffman, Arithmetic Coding, Discop, RRC, ADG и DAIRstega, а затем выгружает модель перед загрузкой следующей. Для каждой комбинации сохраняются direct secret round-trip, ordinary-text token round-trip, ordinary-text secret round-trip, размеры model/tokenizer spaces и число исключенных output-only ID.

Финальная матрица будет зафиксирована после реального запуска на локальных pinned checkpoints.
