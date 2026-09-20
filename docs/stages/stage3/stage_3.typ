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
  #text(size: 16pt, weight: "bold")[Проверка воспроизводимости Bins, Huffman и Arithmetic Coding]
  #v(22pt)
  #text(size: 10.5pt, fill: muted)[Р.В. Тетеревлев · #datetime.today().display("[day].[month].[year]")]
]
#v(1fr)
#pagebreak()

// ===================================
// 1
// ===================================
#slide-title[Пайплайн экспериментов этапа 3]

#let arrow = align(center + horizon)[#text(size: 22pt, weight: "bold")[→]]
#let down = align(center + horizon)[#text(size: 20pt, weight: "bold")[↓]]
#let left = align(center + horizon)[#text(size: 22pt, weight: "bold")[←]]

#grid(columns: (1fr, 22pt, 1fr, 22pt, 1fr), gutter: 8pt,
  card("1. Авторский код")[
    Harvard `NeuralSteganography` \
    commit `14e982…`
  ],
  arrow,
  card("2. Smoke-тесты")[
    GPT-2 Small \
    Bins / Huffman / Arithmetic \
    Проверка исполняемости кода
  ],
  arrow,
  card("3. Условия статьи")[
    GPT-2 Medium \
    CNN/DailyMail \
    80 контекстов × 3 повтора
  ],

  [], [], [], [], down,

  card("6. Matched comparison")[
    4 representative points × 8 контекстов \
    Author-compatible vs normalized \
    32 matched-пары
  ],
  left,
  card("5. Интерпретация")[
    Тренды Figure 3 воспроизведены \
    `4e-8 nats` при `precision=26` - нет \
    Итог: `partial_reproduction`
  ],
  left,
  card("4. Figure 3 reproduction")[
    23 экспериментальные точки \
    Bins: 5, Huffman: 8, AC: 9 + 1 \
    `23 × 80 × 3 = 5520`
  ],
)

#v(12pt)
#note[
  #small[
  Сначала воспроизводим author-compatible эксперимент Figure 3 (Ziegler et al.),
  затем отдельно проверяем, что normalized-адаптеры в benchmark сохраняют принцип
  исходных методов.
  ]
]
#pagebreak()

// ===================================
// 2
// ===================================
#slide-title[Как формировались запуски на этапе 3]

#grid(columns: (1fr, 1fr), gutter: 14pt,
  card("(А) Воспроизведение Figure 3 (Ziegler et al.)")[
    #text(weight: "bold")[Цель:] воспроизвести author-compatible режим (график Figure 3).

    #v(6pt)
    23 экспериментальные точки: \
    Bins - 5 точек: `b = 1, 2, 3, 4, 5` \
    Huffman - 8 точек: `e = 1..8` (`2^e = 2, 4, 8, ..., 256` кандидатов) \
    Arithmetic - 9 точек: `tau = 0.4, 0.5, ..., 1.2`; `k = 300`; `precision = 26` \
    Special Arithmetic - 1 точка: `tau = 1.0`; `k = 50256`; `precision = 26`

    #v(8pt)
    Для каждой точки: \
    `80 контекстов (CNN/DailyMail) × 3 Monte-Carlo повтора = 240 запусков`

    #v(8pt)
    #goodbox[
      #text(weight: "bold", fill: good)[Итого:] \
      `23 × 80 × 3 = 5520 запусков`
    ]

    #v(6pt)
    #small[
      5513 запусков успешно завершились формированием первого предложения; \ 7 - не достигли условия завершения (не встретились токены `.` или `!` или `?`, которые удовлетворяют правилу остановки генерации предложения).
    ]
  ],

  card("(Б) Сравнение author-compatible и normalized режимов")[
    #text(weight: "bold")[Цель:] проверить normalized-адаптацию базовых стегометодов.

    #v(6pt)
    4 экспериментальные точки: \
    Bins `b = 3` \
    Huffman `e = 3` \
    Arithmetic `tau = 0.9, k = 300` \
    Arithmetic `tau = 1.0, k = 50256`

    #v(8pt)
    Для каждой точки: \
    `8 контекстов = 8 matched-пар`

    #v(8pt)
    #goodbox[
      #text(weight: "bold", fill: good)[Итого:] \
      `4 × 8 = 32 matched-пары`
    ]

    #v(6pt)
    #small[
      В каждой паре фиксировались одна модель, один контекст,
      один поток секретных битов и одинаковая длина стеготекста в токенах.
    ]
  ],
)

#v(10pt)
#note[
  #small[
  Эксперимент A отвечает на вопрос «воспроизводится ли Figure 3?». \
  Эксперимент Б отвечает на вопрос «сохраняют ли normalized-реализации принцип исходных методов?».
  ]
]
#pagebreak()

// ===================================
// 3
// ===================================
#slide-title[Результаты воспроизведения авторской реализации]

#grid(columns: (1fr, 1fr), gutter: 12pt,
  [
    #align(center)[
      #image("Figure_3_author.png", width: 75%)
    ]
    #align(center)[#small[Авторская реализация Figure 3 (Ziegler et al.)]]
  ],
  [
    #align(center)[
      #image("Figure_3_reprod.png", width: 70%)
    ]
    #align(center)[#small[Воспроизведение авторской реализации]]
  ],
)

#text(size: 8.2pt)[
#table(
  columns: (0.42fr, 0.31fr, 0.27fr),
  stroke: 0.3pt, inset: 3pt,
  table.header([*Утверждение*], [*Наш результат*], [*Статус*]),
  [Bins остается в high-KL области], [`KL 2.223 → 3.293`], [`trend`],
  [Huffman уменьшает KL], [`1.925 → 0.525`], [`trend`],
  [AC minimum около 4 bits/word], [`3.752 bpw`, `KL=0.100844`], [`trend`],
  [AC ниже Huffman/Bins], [ниже на общей области], [`trend`],
  [Special AC почти не искажает LM], [`KL=0.0006655`], [`partial`],
  [`4e-8 nats` из статьи], [`0.0004613 nats` при `p=26`], [`not reproduced`],
)
]

#goodbox[
  #small[Основной вывод воспроизведен: *Arithmetic показывает лучший KL-capacity trade-off; минимум находится около 4 bits/word при `tau=1.0`*]
]

#tiny[
  Общая оценка этапа 3: частичное воспроизведение, так как точное значение `4e-8 nats` и исходную схему Monte-Carlo эксперимента авторов подтвердить нельзя. \ 
  
]
#pagebreak()

// ===================================
// 4
// ===================================
#slide-title[Почему не совпало значение `4e-8 nats`]

#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Опубликованный код")[
    Arithmetic Coding использует конечную точность вычислений. \
    В публичном `run_single.py` указано `precision=26`. \
    Для специальной точки полного эксперимента получено: \
    *`KL = 4.61e-4 nats`* (а не `4e-8 nats`, как в статье).
  ],
  card("Проверка влияния precision")[
    `p = 26` → `7.38e-4` nats \
    `p = 32` → `2.88e-5` nats \
    `p = 40` → `2.92e-8` nats \
    `p = 48` → `4.01e-10` nats
  ],
)

#v(10pt)
#warnbox[
  #text(weight: "bold", fill: warn)[Интерпретация] \
  #small[
    При `precision = 40` значение KL получается того же порядка, что и `4e-8 nats`. \
    Но это не доказывает, что авторы использовали `precision = 40` (полный скрипт запуска для получения данных для Figure 3 не опубликован).
  ]
]

#v(10pt)
#warnbox[
  #text(weight: "bold", fill: warn)[Результат] \
  #small[
    Расхождение связано с конечной точностью Arithmetic Coding
    и неполной документированностью исходного эксперимента, а не с нарушением принципа метода.
  ]
]
#pagebreak()

// ===================================
// 5
// ===================================
#slide-title[Сравнение author-compatible и normalized режимов]

#align(center)[
  #image("Author-Norm_comparison.png", width: 78%)
]
#align(center)[
  #small[Сопоставленное сравнение по 4 контрольным конфигурациям: одинаковые модель, контекст, secret stream и длина стеготекста в токенах.]
]

#v(7pt)

#grid(columns: (0.62fr, 0.38fr), gutter: 10pt,
  [
    #text(size: 7.4pt)[
    #table(
      columns: (0.30fr, 0.18fr, 0.18fr, 0.17fr, 0.17fr),
      stroke: 0.3pt,
      inset: 2.6pt,
      table.header(
        [*Точка*],
        [*BPT author*],
        [*BPT norm*],
        [*KL author*],
        [*KL norm*],
      ),
      [Bins `b=3`], [`3.000`], [`3.000`], [`2.698`], [`2.901`],
      [Huffman `e=3`], [`2.473`], [`2.514`], [`1.064`], [`1.104`],
      [AC `tau=0.9, k=300`], [`2.934`], [`3.058`], [`0.103`], [`0.048`],
      [AC `tau=1.0, k=50256`], [`4.677`], [`4.594`], [`0.000605`], [`0.000542`],
    )
    ]

    #tiny[
      KL указана в направлении `D_KL(Q_stego || P_reference)`, чтобы сопоставлять normalized-результаты с author-compatible метрикой.
    ]
  ],

  [
    #goodbox[
      #small[
        Bins сохраняет точную пропускную способность. \
        Huffman дает самое близкое совпадение: 6/8 одинаковых последовательностей токенов. \
        Arithmetic Coding сохраняет интервальную механику, но чувствителен к малым численным отличиям.
      ]
    ]
  ],
)
#pagebreak()

// ===================================
// 6
// ===================================
#slide-title[Почему нужны два направления KL]

#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("1. Строгое направление benchmark")[
    #text(size: 12.5pt, weight: "bold")[$D_"KL"(P_"reference" || Q_"stego")$]

    #v(5pt)
    #small[
      Вопрос метрики: \
      *Не потерял ли стегометод часть токенов, которым P_reference назначает ненулевую вероятность?*

      #v(5pt)
      Именно это направление ближе к информационно-теоретической модели Кашена.

      #v(5pt)
      На практике для Bins, Huffman и части AC часто получается `+inf`, если:
    ]

    #v(3pt)
    #align(center)[
      #text(size: 11pt)[$P_"reference"(x) > 0$, но $Q_"stego"(x) = 0$]
    ]

    #v(3pt)
    #small[
      То есть LM считает токен возможным, а стегометод никогда его не выбирает.
    ]
  ],

  card("2. Обратное направление для сравнения")[
    #text(size: 12.5pt, weight: "bold")[$D_"KL"(Q_"stego" || P_"reference")$]

    #v(5pt)
    #small[
      Вопрос метрики: \
      *насколько события, которые реально производит стегометод, выглядят вероятными для LM?*

      #v(5pt)
      Это направление обычно остается конечным, потому что токены с
      $Q_"stego"(x)=0$ не дают вклада в сумму.

      #v(5pt)
      Именно это направление используется для сопоставления с author-compatible результатами Figure 3.
    ]
  ],
)

#note[
  #small[
    Результат этапа 3: при сравнении строгое направление дало `+inf` во всех `32/32` normalized-запусках.
    Поэтому одну KL нельзя использовать как единственную численную меру отличия.
  ]
]

#goodbox[
  #text(weight: "bold", fill: good)[Итог для будущей спецификации] \
  #small[
    Оставить $D_"KL"(P_"reference" || Q_"stego")$ как строгую диагностику потери support; \
    хранить $D_"KL"(Q_"stego" || P_"reference")$ как конечную сравнительную метрику; \
    дополнительно использовать TVD как конечную меру различия распределений.
  ]
]
#pagebreak()

// ===================================
// 7
// ===================================
#slide-title[Итог этапа 3]

#goodbox[
  #text(size: 14pt, weight: "bold", fill: good)[
    Нормализованные реализации сохраняют принцип работы Bins, Huffman и Arithmetic Coding
  ] \
  #v(4pt)
  #small[
    Это подтверждено воспроизведением авторского эксперимента, анализом трендов Figure 3
    и сопоставленным сравнением author-compatible и normalized режимов.
  ]
]

#v(10pt)
#grid(columns: (1fr, 1fr), gutter: 12pt,
  card("Что подтверждено")[
    1. Основные тренды Figure 3 и порядок методов. \
    2. Минимум Arithmetic Coding при `tau=1.0`. \
    3. Сохранение принципа методов в 4/4 контрольных точках. \
    4. Необходимость двух направлений KL для бенчмарка.
  ],
  card("Ограничения")[
    1. Точное значение `4e-8 nats` не воспроизведено при `precision=26`. \
    2. Исходная схема Monte-Carlo эксперимента авторов неизвестна. \
    3. Сопоставленное сравнение выполнено для 4 точек × 8 контекстов, а не для полного normalized sweep.
  ],
)

#v(12pt)
#note[
  #text(size: 12pt, weight: "bold", fill: accent)[
    Следующий этап: подключение современных методов - ADG, Discop, RRC и других методов из roadmap.
  ]
]
