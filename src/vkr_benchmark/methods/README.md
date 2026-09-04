# Реализации стегометодов

Все методы реализуются через общий lifecycle из `base.py`. Метод получает только:

- `MethodEnvironment` с нормализованным допустимым token space;
- канонический `StepContext` / `P_reference`;
- собственный секрет, ключ/RNG и внутреннее состояние.

Метод не должен загружать LM, токенизировать текст, применять общую temperature / top-k /
top-p policy или считать основные benchmark-метрики самостоятельно.

## Bins

Первым concrete adapter реализован `BinsMethod` (`bins.py`). Алгоритмический reference:
`harvardnlp/NeuralSteganography/block_baseline.py`, commit
`14e982564aeaf9a33f7b4de440deda2184d17f12`.

Normalized-версия сохраняет фиксированное разбиение допустимого словаря на `2^b` bins,
отображение `b` secret bits -> bin и выбор наиболее вероятного токена выбранного bin.
Разбиение использует изолированный `MethodRandomSource`, а не глобальный NumPy RNG.
Метод возвращает точное explicit `Q_stego`.

Подробности и отличия от reference-кода зафиксированы в
`docs/decisions/0004-bins-normalized-adaptation.md`.

## Huffman

`HuffmanMethod` (`huffman.py`) на каждом шаге выбирает top `2**bits_per_word`
кандидатов из канонического `P_reference`, строит детерминированное дерево Хаффмана
и читает секретные биты до достижения листа. Поэтому фактический `bits_consumed`
переменный и равен длине кода выбранного токена.

Decoder заново строит дерево на том же `P_reference` и восстанавливает Huffman-код
наблюдаемого токена. Для каждого шага adapter возвращает точное explicit
`Q_stego(token)=2^(-code_length)` на candidate set.

Алгоритмический reference: `huffman_baseline.py` + `huffman.py` того же Harvard
репозитория и commit. Нормализующее правило tie-break и остальные отличия описаны в
`docs/decisions/0006-huffman-normalized-adaptation.md`.

Huffman использует тот же общий `runner/streaming.py` и `TextChannel`, что и Bins;
отдельной Huffman-specific оркестрации нет. End-to-end smoke script находится в
`scripts/check_huffman_e2e.py`.

## Arithmetic Coding

`ArithmeticMethod` (`arithmetic.py`) сохраняет finite-precision integer-range ядро
Harvard `arithmetic.py`: текущий интервал `[L, R)` делится на подинтервалы
кандидатов, секретное `precision`-битное look-ahead окно задает точку выбора,
а общий двоичный префикс границ выбранного подинтервала считается реально
встроенным payload и используется для перенормировки состояния.

В normalized mode метод получает уже готовый `P_reference`; author-side `temp` не
переносится внутрь adapter. `top_k` остается method-internal candidate cap после
`P_reference`. Finite-precision integer widths дают exact explicit `Q_stego`.

В отличие от Bins/Huffman Arithmetic Coding должен читать вперед `precision` битов.
Поэтому `secret_bits_read` больше `payload_bits`: авторитетный payload хранится в
`EncoderFinalization.payload_bits`. Это учтено в архитектуре метода и будет учтено
в общем runner на следующем AC E2E-шаге.

Подробности: `docs/decisions/0007-arithmetic-normalized-adaptation.md`.

Следующий шаг этапа 2: подключить Arithmetic Coding к общему streaming runner и
`TextChannel`, затем провести Llama/Qwen E2E smoke tests.
