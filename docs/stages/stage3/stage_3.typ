// ============================================================
// Этап 3 ВКР: проверка воспроизводимости базовых методов
// ============================================================

#set page(paper: "presentation-16-9", margin: (x: 34pt, y: 28pt))
#set text(font: "New Computer Modern", size: 12pt, lang: "ru")
#set heading(numbering: none)

#let accent = rgb("#1f5fbf")
#let soft = rgb("#eef4ff")
#let dark = rgb("#1f2937")
#let muted = rgb("#5b6472")
#let good = rgb("#0f7b55")
#let warn = rgb("#b45309")
#let bad = rgb("#b42318")

#let slidecounter = counter("slide")

#let slide-title(body) = {
  slidecounter.step()
  text(size: 17pt, weight: "bold", fill: accent)[
    #context slidecounter.display(). #body
  ]
  v(5pt)
  line(length: 100%, stroke: 0.6pt + accent)
  v(9pt)
}

#let note(body) = block(
  width: 100%, fill: soft,
  stroke: (paint: accent, thickness: 0.5pt),
  radius: 4pt, inset: 8pt,
)[#body]

#let card(title, body) = block(
  fill: luma(248),
  stroke: (paint: luma(215), thickness: 0.4pt),
  radius: 5pt, inset: 8pt, width: 100%,
)[
  #text(size: 12pt, weight: "bold", fill: accent)[#title]
  #v(3pt)
  #text(size: 10.6pt)[#body]
]

#let goodbox(body) = block(
  fill: rgb("#e8f8f1"), stroke: (paint: good, thickness: 0.4pt),
  radius: 4pt, inset: 8pt, width: 100%,
)[#body]

#let warnbox(body) = block(
  fill: rgb("#fff6e8"), stroke: (paint: warn, thickness: 0.4pt),
  radius: 4pt, inset: 8pt, width: 100%,
)[#body]

#let small(body) = text(size: 10pt)[#body]
#let tiny(body) = text(size: 8.8pt, fill: muted)[#body]

// ===================================
// Титульный слайд
// ===================================
#v(1fr)
#align(center)[
  #text(size: 23pt, weight: "bold", fill: accent)[Этап 3 ВКР] \
  #v(5pt)
  #text(size: 16pt, weight: "bold")[Проверка воспроизводимости Bins, Huffman и Arithmetic Coding] \
  #v(12pt)
  #text(size: 12pt)[Author-compatible reproduction → matched comparison → conformance analysis] \
  #v(22pt)
  #text(size: 10.5pt, fill: muted)[Р.В. Тетеревлев · #datetime.today().display("[day].[month].[year]")]
]
#v(1fr)
#pagebreak()

// ===================================
// 1
// ===================================
#slide-title[Как проверялась воспроизводимость]

#grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  card("1. Author-compatible")[
    Зафиксирован Harvard `NeuralSteganography` commit. \
    Воспроизведены Bins, Huffman и Arithmetic Coding на GPT-2 Medium.
  ],
  card("2. Figure 3")[
    23 operating points, 80 CNN/DailyMail contexts, 3 repeats. \
    Проверялись форма кривых, порядок методов и special Arithmetic point.
  ],
  card("3. Matched comparison")[
    Один контекст, один secret stream и одинаковая carrier length для author и normalized. \
    Цель — отделить эффект нормализации от paper-level неопределённостей.
  ],
)

#v(12pt)
#note[
  #small[
  Главный критерий: normalized adapter должен сохранять *принцип метода*, а не обязательно генерировать тот же самый token sequence. Нормализация намеренно меняет общую policy, RNG ownership, masking и numerical boundaries.
  ]
]

#v(12pt)
#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Зафиксированная author-база")[
    Harvard commit: `14e9825…` \
    GPT-2 Medium revision: `6dcaa7a…`
  ],
  card("Контекст")[
    CNN/DailyMail, первые три предложения статьи. \
    Manifest hash зафиксирован и одинаков для всех Stage-3 runs.
  ],
)
#pagebreak()

// ===================================
// 2
// ===================================
#slide-title[Полный Figure-3 sweep: 5520 запусков]

#grid(columns: (0.45fr, 0.55fr), gutter: 12pt,
  [
    #card("Матрица")[
      Bins: `b=1..5` \
      Huffman: `2^1..2^8` кандидатов \
      Arithmetic: `tau=0.4..1.2`, `k=300` \
      Special: `tau=1`, `k=50256`
    ]
    #v(8pt)
    #goodbox[
      #text(weight: "bold", fill: good)[Execution result] \
      #small[5520/5520 outcomes сохранены. \
      5513 достигли first sentence boundary. \
      7 сохранены как `sentence_termination_failure`.]
    ]
    #v(8pt)
    #tiny[Termination failures не отбрасывались скрыто и не заменялись искусственной границей предложения.]
  ],
  [
    #align(center)[
      #image("../../../results/stage3/paper_reproduction/figure3_full/figure3_reproduction.svg", width: 95%)
    ]
  ],
)
#pagebreak()

// ===================================
// 3
// ===================================
#slide-title[Что воспроизвелось из Figure 3]

#text(size: 9.4pt)[
#table(
  columns: (0.38fr, 0.30fr, 0.32fr),
  stroke: 0.3pt, inset: 4pt,
  table.header([*Проверяемое утверждение*], [*Наш результат*], [*Статус*]),
  [Bins остаётся в high-KL области], [`KL 2.223 → 3.293 bits`], [`trend_reproduction`],
  [Huffman уменьшает KL при росте capacity], [`1.925 → 0.525 bits`], [`trend_reproduction`],
  [Arithmetic minimum около 4 bits/word при `tau=1`], [`3.752 bpw`, `KL=0.100844`], [`trend_reproduction`],
  [Arithmetic ниже Huffman/Bins], [ниже на всей общей области], [`trend_reproduction`],
  [Special Arithmetic почти не искажает LM], [`KL=0.0006655 bits`], [`partial_reproduction`],
  [Exact paper anchor `4e-8 nats`], [`0.0004613 nats` при p=26], [`not_reproducible`],
)
]

#v(9pt)
#goodbox[
  #text(weight: "bold", fill: good)[Основной научный вывод публикации воспроизведён:] \
  #small[Arithmetic показывает лучший KL-capacity trade-off, минимум находится около 4 bits/word при `tau=1.0`.]
]

#v(6pt)
#tiny[Общая paper-level оценка Stage 3: `partial_reproduction`, потому что exact special-point anchor и exact historical MC orchestration подтвердить нельзя.]
#pagebreak()

// ===================================
// 4
// ===================================
#slide-title[Почему не совпало значение `4e-8 nats`]

#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Public executable")[
    Arithmetic Coding использует finite precision. \
    В public `run_single.py` зафиксировано `precision=26`. \
    Full sentence special point: \
    *`KL = 4.61e-4 nats`*.
  ],
  card("Precision probe без zero-padding")[
    p26 → `7.38e-4` nats \
    p32 → `2.88e-5` nats \
    p40 → `2.92e-8` nats \
    p48 → `4.01e-10` nats
  ],
)

#v(10pt)
#warnbox[
  #text(weight: "bold", fill: warn)[Интерпретация] \
  #small[`precision=40` попадает в тот же порядок, что paper anchor, но это не доказывает, что авторы использовали p=40. Исторический Figure-3 batch driver не опубликован, поэтому мы не подгоняем реализацию post-hoc.]
]

#v(10pt)
#note[
  #small[*Результат:* discrepancy локализована к finite precision / undocumented historical orchestration, а не к принципу Arithmetic Coding.]
]
#pagebreak()

// ===================================
// 5
// ===================================
#slide-title[Author vs normalized: сохранился ли принцип методов?]

#text(size: 8.7pt)[
#table(
  columns: (0.20fr, 0.15fr, 0.15fr, 0.16fr, 0.18fr, 0.16fr),
  align: center + horizon,
  stroke: 0.3pt, inset: 3.2pt,
  table.header([*Точка*], [*Author BPT*], [*Norm. BPT*], [*Δ BPT*], [*Δ reverse KL*], [*Exact seq.*]),
  [Bins b=3], [3.000], [3.000], [0.000], [+0.203], [0/8],
  [Huffman e=3], [2.473], [2.514], [+0.041], [+0.040], [*6/8*],
  [Arithmetic `.9/300`], [2.934], [3.058], [+0.124], [−0.055], [0/8],
  [Arithmetic `1/50256`], [4.677], [4.594], [−0.084], [−0.000064], [0/8],
)
]

#v(9pt)
#grid(columns: (1fr, 1fr), gutter: 10pt,
  goodbox[
    #text(weight: "bold", fill: good)[32/32 normalized decodes exact] \
    #small[Одинаковые secret streams и carrier lengths. Core principle сохранён на 4/4 representative points.]
  ],
  card("Почему token sequences могут отличаться")[
    Bins: другое partition identity. \
    Huffman: candidate/tree policy. \
    Arithmetic: высокая чувствительность finite-precision interval к малым boundary differences.
  ],
)
#pagebreak()

// ===================================
// 6
// ===================================
#slide-title[Два направления KL — практический результат Этапа 3]

#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Benchmark-native")[
    #text(size: 12.5pt, weight: "bold")[$D_"KL"(P_"reference" || Q_"stego")$] \
    #v(4pt)
    Во всех *32/32* matched runs результат `+inf`. \
    Причина: sparse `Q_stego` теряет часть support `P_reference`.
  ],
  card("Author-comparable")[
    #text(size: 12.5pt, weight: "bold")[$D_"KL"(Q_"stego" || P_"reference")$] \
    #v(4pt)
    Остаётся конечной и позволяет сравнивать distortion с author-compatible направлением.
  ],
)

#v(10pt)
#note[
  #small[
  Рекомендация для specification v1.0: \
  `KL(P_ref || Q)` оставить строгим support diagnostic *без smoothing*; \
  `KL(Q || P_ref)` хранить как finite comparative metric; \
  TVD хранить как дополнительную конечную metric.
  ]
]

#v(8pt)
#tiny[Specification v0.1 не переписывалась задним числом. Изменение относится только к будущей v1.0.]
#pagebreak()

// ===================================
// 7
// ===================================
#slide-title[Итог Этапа 3]

#goodbox[
  #text(size: 14pt, weight: "bold", fill: good)[Normalized adapters сохраняют принципиальное поведение Bins, Huffman и Arithmetic Coding] \
  #v(4pt)
  #small[Это подтверждено author-compatible reproduction, Figure-3 trend reproduction и matched author-vs-normalized analysis.]
]

#v(10pt)
#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Подтверждено")[
    1. Figure-3 trends и method ordering. \
    2. Arithmetic minimum при `tau=1`. \
    3. Core method principle 4/4 matched points. \
    4. Dual-KL interpretation для benchmark.
  ],
  card("Ограничения")[
    1. Exact `4e-8 nats` anchor не воспроизведён при p=26. \
    2. Historical MC driver/sample count неизвестны. \
    3. Matched grid — 4 точки × 8 contexts, а не полный normalized sweep.
  ],
)

#v(12pt)
#note[
  #text(size: 12pt, weight: "bold", fill: accent)[Следующий этап: подключение современных методов — ADG, Discop, RRC и других методов ROADMAP.]
]
