# Stage 5 — отчет по common `top_p` calibration

## 1. Назначение

Этот calibration не является сравнением качества методов. Он проверяет, существует ли единый common `top_p`, при котором все семь normalized methods технически совместимы с одной и той же политикой формирования `P_reference`, и дает первый порядок wall-clock/metric-instrumentation стоимости.

Исходные артефакты:

- `results/stage5/sweeps/stage5-top-p-calibration-llama-v1/support_probe_execution.jsonl`;
- `results/stage5/sweeps/stage5-top-p-calibration-llama-v1/execution.jsonl`;
- `results/stage5/top_p_calibration_summary.json`.

## 2. Support probe

При `Bins(block_size=2)`, `s5p0001`, `r01`:

- `top_p=0.98` — `UnsupportedConfigurationError` через 0.428 s;
- причина: в текущем `P_reference` отсутствовали positive-probability tokens в bins `[0, 1, 2]`;
- `top_p=1.0` — успешный run, exact round-trip, `BER=0`.

`top_p=0.95` после failure `0.98` не запускался: меньший nucleus threshold не может расширить support.

## 3. Full-support calibration

При `top_p=1.0` выполнено 14 runs: 7 methods × 2 prompts × 1 paired replicate, по 8 carrier tokens для streaming methods и 32 payload bits для RRC.

Результат:

- successful: 14/14;
- failed: 0;
- exact round-trip: 14/14;
- max BER: 0;
- сумма фактического wall time по runs: 89.40 s.

Средние значения по двум prompts приведены только как техническая диагностика; выборка слишком мала для научного ранжирования.

| Method | BPT | Entropy util. | KL `P_ref→Q` | KL `Q→P_ref` | TVD | Wall/run | Metric instr. | Stego algorithm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bins | 2.000 | 0.404 | `+inf` | 2.172 | 0.628 | 0.549 s | 0.016 s | 0.048 s |
| Huffman | 1.938 | 0.374 | `+inf` | 1.360 | 0.522 | 0.364 s | 0.011 s | 0.001 s |
| Arithmetic | 3.750 | 0.872 | `+inf` | 0.024 | 0.018 | 1.404 s | 0.356 s | 0.689 s |
| ADG | 2.750 | 0.588 | 0.0101 | 0.0103 | 0.0329 | 29.560 s | 27.461 s | 1.190 s |
| Discop | 3.063 | 0.975 | ~0 | ~0 | ~0 | 10.796 s | 5.928 s | 4.368 s |
| RRC | 3.516 | 0.964 | ~0 | ~0 | ~0 | 1.132 s | 0.037 s | 0.655 s |
| DAIRstega | 1.000 | 0.250 | `+inf` | 0.875 | 0.388 | 0.896 s | 0.013 s | 0.520 s |

Для RRC BPT здесь относится к короткому fixed-payload режиму 32 bits и не интерпретируется как финальная operating point.

## 4. Timing observation

Calibration подтверждает необходимость ADR-0020: metric instrumentation нельзя смешивать с временем самого стегометода.

Особенно заметно:

- ADG: около 1.19 s stego algorithm против 27.46 s metric instrumentation;
- Discop: около 4.37 s stego algorithm против 5.93 s metric instrumentation.

Поэтому Stage-5 Pilot A остается широким только для методов с реальными method-local knobs. ADG и Discop имеют по одной normalized point, а не искусственные параметрические сетки.

## 5. Решение

Common baseline фиксируется:

```text
temperature = 1.0
common top_k = null
common top_p = 1.0
```

Это решение формализовано ADR-0027.

## 6. Ограничения calibration

- 2 prompts;
- 1 paired replicate;
- 8 carrier tokens для streaming methods;
- calibration предназначен для feasibility/timing, а не для статистического вывода;
- значения `entropy_utilization` на столь короткой траектории могут заметно флуктуировать, поэтому Pilot A увеличивает длину до 128 carrier tokens.
