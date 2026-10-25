# Stage 5 — пилотные свипы и окончательная фиксация экспериментального дизайна

## 1. Цель

Stage 5 не является уменьшенной копией Stage 6. Его задача — дешево и воспроизводимо принять решения, необходимые для основного эксперимента:

1. выбрать общий generation regime;
2. определить содержательные диапазоны method-local параметров;
3. исключить технически некорректные и явно неинтересные точки;
4. откалибровать RRC fixed-payload regime;
5. оценить prompt/replicate variance;
6. измерить wall-clock стоимость, отдельно от времени самого стегометода;
7. зафиксировать Stage-6 scale и benchmark specification v1.0.

Методологическое разделение external conditions / method-local points / calibration закреплено ADR-0026.

## 2. Выполненная подготовка

### 2.1. Iteration 1 — sweep foundation

Выполнено:

- переиспользуются восемь frozen CNN/DailyMail pilot contexts Stage 3;
- Stage-5 prompt registry получает стабильные `prompt_id`, exact text хранится локально и воспроизводимо регенерируется;
- введен machine-readable sweep manifest;
- dry-run planner заранее вычисляет полный набор configs и canonical `run_id` без загрузки LM;
- batch runner загружает одну LM один раз на sweep, поддерживает filters и `--resume`;
- зафиксирован paired replicate/randomness protocol;
- выполнен первый common-`top_p` calibration.

### 2.2. Common generation calibration — завершено

Технически проверялись значения:

```text
top_p = 0.95, 0.98, 1.00
```

`top_p=0.98` оказался несовместим с Bins `block_size=2`: после common nucleus truncation в текущем `P_reference` не осталось positive-probability tokens в трех из четырех фиксированных bins. Уменьшение `top_p` не может расширить support, поэтому `0.95` не является feasible common regime.

При `top_p=1.0` выполнены 14 short calibration runs: 7 methods × 2 prompts × 1 paired replicate. Все 14 завершились успешно, с exact round-trip и `BER=0`.

Основной common regime зафиксирован ADR-0027:

```text
temperature = 1.0
common top_k = null
common top_p = 1.0
exclude_special_tokens = true
kv_cache = true
```

Полный отчет: `docs/stages/stage5/top_p_calibration_report.md`.

## 3. Pilot A — screening — завершено

Pilot A использовал:

- Llama-3.2-3B;
- common regime из ADR-0027;
- prompts: `s5p0001..s5p0004`;
- 2 paired replicates;
- **128 carrier tokens** для streaming methods;
- RRC калибруется отдельно в fixed-payload lifecycle.

Streaming executable manifest:

`configs/pilots/stage5/pilot_a_streaming_llama.json`

Сетка:

| Метод | Pilot-A grid |
|---|---|
| Bins | `block_size = 1..8` |
| Huffman | `bits_per_word = 1..8` |
| Arithmetic | `precision={16,32,48}` × `top_k={128,512,2048,8192,50000}` |
| ADG | one normalized point |
| Discop | one normalized point |
| DAIRstega | `alpha={8,16,32,48}` × `beta={1.0,0.75,0.5}`, initially `top_k=100` |

Это 45 method points × 4 prompts × 2 replicates = **360 runs**. Sweep завершен полностью: 347 process-successful runs, 13 hard failures; exact text-channel recovery — 318/360. Все 42 non-exact runs прошли post-hoc reliability diagnostics и связаны с изменением token sequence после обычного text transport.

Итоговые отчеты:

- `docs/stages/stage5/pilot_a_analysis.md`;
- `docs/stages/stage5/pilot_a_reliability_analysis.md`.

Screening после Pilot A завершен. Из 45 method-local points в Pilot B перенесен 21 representative point; 24 точки остаются документированными screening results и повторно не запускаются. Полный retained/rejected table: `docs/stages/stage5/pilot_b_operating_point_selection.md`; machine-readable решение: `results/stage5/pilot_a_operating_point_selection.json`.

DAIRstega candidate-pool sensitivity, если потребуется для Stage 6, выполняется отдельным targeted probe и не расширяет Pilot B неявно.

## 4. RRC calibration — precision-margin sweep завершен, numerical issue открыт

RRC не принуждается к fixed-carrier lifecycle.

Первый executable calibration:

`configs/pilots/stage5/rrc_precision_margin_calibration_llama.json`

Параметры:

```text
precision_margin = {20,40,60}
verify_termination = true
fixed payload = 1024 bits
max_carrier_tokens = 2048
4 prompts × 2 paired replicates
```

1024 bits — самый большой предварительный payload candidate и потому stress-case для numerical precision. Sweep `{20,40,60}` завершен: 24/24 exact, `BER=0`, содержательные метрики и carrier length совпадают между margins. Для Decimal range-state arithmetic наблюдается plateau; `precision_margin=40` остается консервативным кандидатом.

Одновременно обнаружен отдельный numerical issue: в части runs forward KL равен `+inf` при reverse KL около `1e-16` и TVD около `1e-12`. Текущая рабочая диагностика связывает это с FP64 cumulative boundaries до Decimal arithmetic; отдельный high-precision boundary probe еще не выполнен. Поэтому окончательный RRC regime для Stage 6 пока не заморожен.

Отчет: `docs/stages/stage5/rrc_precision_calibration_report.md`.

После решения boundary-precision вопроса выполняется отдельный payload calibration по `{128,256,512,1024}` для выбора normalized RRC regime с сопоставимой carrier length.

RRC calibration отвечает на вопрос «какие технические настройки фиксировать дальше?», а не создает искусственную trade-off curve RRC.

## 5. Pilot B — confirmation — manifest подготовлен

Согласованный streaming Pilot B содержит 21 retained point:

| Метод | Retained points |
|---|---:|
| Bins | 4 |
| Huffman | 4 |
| Arithmetic Coding | 6 |
| ADG | 1 |
| Discop | 1 |
| DAIRstega | 5 |
| **Итого** | **21** |

Режим:

- executable manifest: `configs/pilots/stage5/pilot_b_streaming_llama.json`;
- 8 frozen pilot prompts `s5p0001..s5p0008`;
- 3 paired replicates;
- 256 carrier tokens;
- тот же common generation regime из ADR-0027;
- **504 planned streaming runs**.

Pilot B оценивает prompt/secret variance, стабильность trade-off/rank conclusions и фактическую стоимость запуска перед окончательной фиксацией Stage 6. RRC в эти 504 runs не входит: его fixed-payload regime определяется отдельно после boundary-precision investigation.

## 6. Критерии отбора

Каждая точка сохраняет полный benchmark vector; единой агрегированной оценки «лучший метод» нет.

Обязательные оси:

- reliability: decoder completion, `roundtrip_exact`, BER, token-sequence round-trip;
- capacity: BPT, entropy utilization;
- distribution distortion: оба направления KL по ADR-0012 + TVD;
- quality: raw-LM NLL/PPL;
- performance: encode/decode ms/token, payload bit/s, stego algorithm time;
- instrumentation: `metric_instrumentation_total_ms` отдельно от stego time;
- variability: разброс по prompt × replicate.

Точка может быть исключена как технически некорректная или стабильно доминируемая соседней точкой того же метода. При этом для параметризованного метода желательно сохранить security/quality anchor, промежуточную область и capacity-oriented endpoint, если они содержательно различаются.

## 7. Вычислительный бюджет Stage 6

Для каждого retained operating point Pilot B дает как минимум median и p90 wall-clock/run. Оцениваются два разных бюджета:

```text
algorithmic cost = LM + P_ref processing + stego algorithm
wall-clock cost  = algorithmic cost + metric instrumentation + storage/I/O
```

Stage-6 estimate вычисляется как сумма фактической стоимости по всем method points × prompts × replicates с дополнительным резервом на warm-up, I/O и reruns.

Qwen используется для ключевых сравнений, а не обязательно для полного Llama grid. Для Qwen сохраняется уже зафиксированная общая политика: 267 output-only IDs исключаются до `P_reference`, затем распределение перенормируется.

## 8. Выходные артефакты Stage 5

Минимальный closeout:

- dataset pilot runs;
- analytic pilot report;
- Stage-6 budget report;
- retained/rejected operating-point table с причинами;
- benchmark specification v1.0;
- Stage-5 Typst presentation;
- machine-readable sweep manifests и execution logs;
- `stage5_closeout.md`.
