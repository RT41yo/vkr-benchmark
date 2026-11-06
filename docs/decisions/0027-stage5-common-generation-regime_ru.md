# ADR-0027: Общий full-support generation regime для Stage 5/6

**Статус:** принято для Stage 5; основной baseline для Stage 6  
**Ветка:** `stage5-pilot-sweeps`  
**Основание:** Stage-5 top-p calibration, commit-base `a31d2b5`

## Контекст

ADR-0026 отделил общие параметры генерации от method-local operating points. Перед Pilot A требовалось выбрать один общий `top_p`, одинаковый для всех семи методов, чтобы последующие trade-off и Парето-сравнения выполнялись в одном внешнем условии.

Первый Stage-5 calibration был намеренно коротким и проверял `top_p={0.95,0.98,1.0}` при representative point каждого метода. Критерием выбора была прежде всего техническая совместимость общей политики `P_reference` со всеми normalized methods, а уже затем стоимость и поведение метрик.

## Наблюдение

Для Bins `block_size=2` на prompt `s5p0001` при `top_p=0.98` получен корректный `UnsupportedConfigurationError`:

```text
current P_reference has no positive-probability token in Bins bin(s) [0, 1, 2]
```

После nucleus truncation три из четырех фиксированных Bins-корзин не содержали ни одного токена с положительной вероятностью в текущем `P_reference`. Поэтому Bins не мог гарантированно представить произвольный очередной 2-битный блок секрета.

`top_p=0.95` отдельно не запускался после этого failure: при одинаковых logits уменьшение `top_p` не увеличивает support и не может вернуть токены, уже отсутствующие при `top_p=0.98`.

При `top_p=1.0` выполнены 14 коротких runs: 7 methods × 2 prompts × 1 paired replicate. Все 14 завершились со `status=ok`, exact text/secret round-trip и `BER=0`.

## Решение

Основной общий generation regime для Pilot A, Pilot B и baseline Stage 6 фиксируется как:

```text
temperature = 1.0
common top_k = null
common top_p = 1.0
exclude_special_tokens = true
kv_cache = true
```

То есть основной `P_reference` использует полный допустимый support после общих benchmark-исключений, без common top-k/top-p truncation.

Для Qwen сохраняется отдельное уже принятое правило: output-only token IDs исключаются до формирования `P_reference`, после чего вероятность перенормируется. Это не считается common top-p truncation.

## Почему не «чинить» Bins под top_p < 1

Динамическое перераспределение surviving tokens между корзинами после nucleus truncation изменило бы семантику фиксированного Bins-разбиения. Такое изменение создало бы другой алгоритм и нарушило бы нормализованное сравнение. Поэтому incompatibility общей truncation policy считается свойством экспериментального режима, а не ошибкой Bins adapter.

## Следствия

1. Pilot A/B не свипают common `top_p`; меняются только method-local параметры и отдельно обозначенные calibration dimensions.
2. Все методы в основном Парето-анализе Llama работают при одном и том же full-support external condition.
3. Если позже понадобится sensitivity analysis другого `top_p`, это будет отдельное внешнее условие и отдельный Парето-фронт; его точки не смешиваются с baseline.
4. Full support делает exact-Q instrumentation особенно дорогим для ADG/Discop, поэтому их стоимость учитывается отдельно от stego algorithm timing согласно ADR-0020.
