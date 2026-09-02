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

Следующие методы этапа 2: Huffman и Arithmetic Coding.
