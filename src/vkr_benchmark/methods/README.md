# Method implementations

Все методы реализуются через общий lifecycle из `base.py`. Метод получает только
канонический `StepContext`, собственный секрет/ключ/RNG и внутреннее состояние.
Он не должен загружать LM, токенизировать текст, применять общую temperature / top-k /
top-p policy или считать основные benchmark-метрики самостоятельно.

Первым concrete adapter на этапе 2 будет Bins, затем Huffman и Arithmetic Coding.
