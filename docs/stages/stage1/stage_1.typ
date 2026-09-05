// ============================================================
// Этап 1 ВКР: результаты и спецификация бенчмарка v0.1
// 6 слайдов по образцу presentation_v.0.4.typ
// ============================================================

#set page(paper: "presentation-16-9", margin: (x: 32pt, y: 26pt))
#set text(font: "New Computer Modern", size: 12pt, lang: "ru")
#set heading(numbering: none)

#let accent = blue
#let slidecounter = counter("slide")

#let slide-title(body) = {
  slidecounter.step()
  text(size: 15pt, weight: "bold", fill: accent)[
    #context slidecounter.display(). #body
  ]
  v(5pt)
  line(length: 100%, stroke: 0.5pt + accent)
  v(7pt)
}

#let keybox(body) = block(
  stroke: (paint: accent, thickness: 0.4pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let lightbox(body) = block(
  fill: luma(246), stroke: 0.25pt, inset: 8pt,
  width: 100%, radius: 3pt,
)[#body]

#let goodbox(body) = block(
  fill: rgb("#ecf8f0"), stroke: (paint: rgb("#1a7f37"), thickness: 0.35pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let warnbox(body) = block(
  fill: rgb("#fff6df"), stroke: (paint: rgb("#b7791f"), thickness: 0.35pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let small(body) = text(size: 10.2pt)[#body]
#let tiny(body) = text(size: 9pt)[#body]
#let cap(body) = text(size: 9pt, fill: luma(90))[#body]

#let zc(body) = table.cell(fill: accent)[#text(size: 10pt, fill: white, weight: "bold")[#body]]


// ============================================================
// Титульный слайд
// ============================================================
#v(1fr)
#align(center)[
  #text(size: 22pt, weight: "bold", fill: accent)[Этап 1 ВКР] \
  #v(5pt)
  #text(size: 17pt, weight: "bold")[Фиксация методологии, анализ программных реализаций \ и выбор языковых моделей] \
  #v(10pt)
  #text(size: 12pt)[Спецификация бенчмарка v0.1.] \
  #v(18pt)
]
#v(1fr)
#pagebreak()


// ============================================================
// 1. Репозитории и методы
// ============================================================
#slide-title[Результаты анализа доступных репозиториев и методов]

#small[
На этапе 1 изучены открытые реализации методов генеративной лингвистической стеганографии и сопоставлены с исходными публикациями. \ Главный вывод: код полезен как источник алгоритмов, но не как готовый единый бенчмарк.
]

#v(7pt)
#table(
  columns: (0.21fr, 0.24fr, 0.26fr, 0.29fr),
  align: (left + horizon, left + horizon, left + horizon, left + horizon),
  stroke: 0.3pt, inset: 4.5pt,
  table.header([*Группа*], [*Методы*], [*Источник кода*], [*Решение*]),
  [Исторические baseline], [1) Bins; \ 2) Huffman; \ 3) Arithmetic Coding], [`NeuralSteganography`], [1) Включить в бенчмарк.\ 2) Перенести ядро в общий интерфейс],
  [Адаптивная группировка], [ADG], [`ADG-steganography`], [1) Включить в бенчмарк. \ 2) Нужен собственный декодер],
  [Сохранение распределения], [1) Discop; \ 2) RRC], [`Discop`; \ `RRC_steganography`], [1) Включить в бенчмарк. \ 2) Считать KL/TVD внешним модулем],
  [Управляемый компромисс], [DAIRstega], [`DAIRstega`], [1) Включить; в бенчмарк. \ 2) Нужен собственный декодер],
)

#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 12pt)[
  #goodbox[
    #text(fill: rgb("#1a7f37"), weight: "bold")[Итоговый набор методов] \
    #v(2pt)
    #text(size: 12.5pt, weight: "bold")[Bins, Huffman, Arithmetic Coding, ADG, Discop, RRC, DAIRstega]
  ]
][
  #warnbox[
    #text(fill: rgb("#8a5a00"), weight: "bold")[Главный риск] \
    #text(size: 12.5pt, weight: "bold")[Исходные репозитории используют разные модели, токенизаторы, метрики и правила генерации. Поэтому сравнивать их «как есть» некорректно.]
  ]
]

#v(5pt)
#keybox[
  #small[*Вывод:* на этапе 2 нужно реализовать методы в общей инфраструктуре, а исходные репозитории использовать как reference implementations для сверки.]
]
#pagebreak()


// ============================================================
// 2. Пробные запуски LLM
// ============================================================
#slide-title[Результаты пробных запусков LLM]

#small[
Проверялась пригодность модели как вычислительной основы стенда: VRAM, скорость, полный вектор logits, KV-cache, токенизатор и др.
]

#v(7pt)
#table(
  columns: (0.38fr, 0.31fr, 0.31fr),
  align: (left + horizon, left + horizon, left + horizon),
  stroke: 0.3pt, inset: 4.3pt,
  table.header([*Показатель*], [*Qwen3-4B-Base*], [*Llama-3.2-3B*]),
  [VRAM после загрузки], [7.545 ГБ], [*5.985 ГБ*],
  [Пик VRAM, 2048 + 256], [7.987 ГБ], [*6.354 ГБ*],
  [Decode, контекст 128], [70.35 токен/с], [*94.82 токен/с*],
  [Decode, контекст 512], [66.59 токен/с], [*90.45 токен/с*],
  [Decode, контекст 2048], [55.47 токен/с], [*77.12 токен/с*],
  [`tokens -> text -> tokens`], [30/30], [30/30],
  [Output-only ID], [267], [*0*],
)

#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 12pt)[
  #goodbox[
    #text(fill: rgb("#1a7f37"), weight: "bold")[Обе модели прошли] \
    #small[BF16, полный logits-вектор, KV-cache, контекст 2048 + 256 токенов, повторяемость при одном вычислительном пути.]
  ]
][
  #warnbox[
    #text(fill: rgb("#8a5a00"), weight: "bold")[Важный нюанс] \
    #small[KV-cache и полный пересчет prefix дают близкие, но не идентичные logits. Encoder и decoder должны использовать один и тот же путь.]
  ]
]

#v(5pt)
#cap[Пробные запуски выполнены на NVIDIA GeForce RTX 5070 Ti, 16GB VRAM, BF16, batch size = 1.]
#pagebreak()


// ============================================================
// 3. Перечень выбранных LLM
// ============================================================
#slide-title[Перечень выбранных языковых моделей]

#grid(columns: (1fr, 1fr), gutter: 14pt)[
  #block(fill: rgb("#ecf8f0"), stroke: (paint: rgb("#1a7f37"), thickness: 0.45pt), inset: 10pt, radius: 4pt)[
    #text(size: 14pt, weight: "bold", fill: rgb("#1a7f37"))[Основная модель] \
    #v(5pt)
    #text(size: 18pt, weight: "bold")[Llama-3.2-3B] \
    #v(7pt)
    #tiny[Ревизия: \
    `13afe5124825b4f3751f836b40dafda64c1ed062`]
    #v(8pt)
    #small[
    Причины выбора: меньше VRAM, выше скорость, `model vocab = tokenizer vocab`, нет output-only ID.
    ]
  ]
][
  #block(fill: rgb("#edf4ff"), stroke: (paint: accent, thickness: 0.45pt), inset: 10pt, radius: 4pt)[
    #text(size: 14pt, weight: "bold", fill: accent)[Резервная модель] \
    #v(5pt)
    #text(size: 18pt, weight: "bold")[Qwen3-4B-Base] \
    #v(7pt)
    #tiny[Ревизия: \
    `906bfd4b4dc7f14ee4320094d8b41684abff8539`]
    #v(8pt)
    #small[
    Пригодна для стенда, но требует исключения 267 output-only ID и использует больше памяти.
    ]
  ]
]

#v(12pt)
#keybox[
  #text(fill: accent, weight: "bold")[Единые условия дальнейших экспериментов] \
  #small[BF16 для LM; \ FP32 для обработки вероятностей; \ batch size 1; \ без квантования; \ `eval()`; \ без градиентов; \ фиксированные ревизии моделей; \ единый путь с KV-cache.]
]

#v(6pt)
#small[*Роль Qwen:* проверка переносимости выводов на другую современную LM и резерв на случай ограничений доступа к Llama.]
#pagebreak()


// ============================================================
// Слайды 4–7: спецификация бенчмарка v0.1
// Для вставки после первых трех слайдов презентации этапа 1
// ============================================================

#set page(paper: "presentation-16-9", margin: (x: 32pt, y: 26pt))
#set text(font: "New Computer Modern", size: 12pt, lang: "ru")
#set heading(numbering: none)

#let accent = blue
#let slidecounter = counter("slide")
#slidecounter.update(3)

#let slide-title(body) = {
  slidecounter.step()
  text(size: 15pt, weight: "bold", fill: accent)[
    #context slidecounter.display(). #body
  ]
  v(5pt)
  line(length: 100%, stroke: 0.5pt + accent)
  v(7pt)
}

#let keybox(body) = block(
  stroke: (paint: accent, thickness: 0.4pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let lightbox(body) = block(
  fill: luma(247), stroke: 0.3pt, inset: 8pt,
  width: 100%, radius: 3pt,
)[#body]

#let goodbox(body) = block(
  fill: rgb("#ecf8f0"), stroke: (paint: rgb("#1a7f37"), thickness: 0.35pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let warnbox(body) = block(
  fill: rgb("#fff6df"), stroke: (paint: rgb("#b7791f"), thickness: 0.35pt),
  inset: 8pt, width: 100%, radius: 3pt,
)[#body]

#let small(body) = text(size: 10.2pt)[#body]
#let tiny(body) = text(size: 9pt)[#body]
#let cap(body) = text(size: 9pt, fill: luma(90))[#body]


// ============================================================
// 4. Зачем нужна спецификация
// ============================================================
#slide-title[Cпецификация бенчмарка v0.1]

#small[
Цель бенчмарка - обеспечить воспроизводимое сравнение методов генеративной лингвистической стеганографии в одинаковых контролируемых условиях. \
Цель Спецификации - зафиксировать единые правила сравнения методов.
]

#v(8pt)
#align(center)[
  #block(fill: luma(247), stroke: 0.35pt, radius: 4pt, inset: 10pt, width: 92%)[
    #text(size: 14pt, weight: "bold")[
      внешние условия + стегометод и его параметры #text(fill: accent)[ → ] вектор метрик
    ]
  ]
]

#v(10pt)
#grid(columns: (1fr, 1fr), gutter: 14pt)[
  #lightbox[
    #text(fill: accent, weight: "bold")[Внешние условия эксперимента:] \
    #small[
    Контролируемые факторы, внешние по отношению к конкретному методу: языковая модель, токенизатор, промпт, правило формирования $P_"reference"$, секретный поток, длина генерации, окружение.
    ]
  ]
][
  #lightbox[
    #text(fill: accent, weight: "bold")[Управляющие параметры стегометода:] \
    #small[
    То, что относится к самому алгоритму: число корзин Bins, размер кандидатного множества Huffman, параметры ADG, RRC, DAIRstega и других методов.
    ]
  ]
]

#v(9pt)
#keybox[
  #small[*Идея сравнения:* сначала фиксируем внешние условия и сравниваем методы; затем контролируемо меняем отдельные условия, например LM, чтобы проверить устойчивость выводов.]
]

#v(5pt)
#cap[v0.1 — методологический контракт этапа 1; окончательные масштабы экспериментов фиксируются в v1.0 после пилотных переборов параметров.]

#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 12pt)[
  #goodbox[
    #text(fill: rgb("#1a7f37"), weight: "bold")[Два режима исследовательской работы:] \
    #small[(1) *Нормализованный сравнительный эксперимент* (метод запускается по правилам настоящей спецификации). \ (2) *Проверка соответствия авторской реализации* (метод запускается в исходной или максимально близкой к публикации).]
  ]
][
  #warnbox[
    #text(fill: rgb("#8a5a00"), weight: "bold")[Программно-аппаратное окружение] \
    #small[NVIDIA GeForce RTX 5070 Ti, 16 ГБ VRAM; Python: 3.12.14; PyTorch: 2.7.1+cu128; версия CUDA runtime, используемая PyTorch: 12.8; Transformers: 4.52.4; tokenizers: 0.21.1; safetensors: 0.5.3; huggingface-hub: 0.32.5; NumPy: 2.0.2; BF16; квантование: отключено; пошаговая генерация с KV-кэшем...]
  ]
]
#pagebreak()


// ============================================================
// 5. Архитектура одного запуска
// ============================================================
#slide-title[Архитектура одного запуска]

#small[
Один запуск строится вокруг одного префикса: модель дает логиты следующего токена, общий модуль формирует эталонное распределение, затем либо выполняется контрольная генерация, либо подключается стегометод.
]

#v(7pt)
#align(center)[
#table(
  columns: (1fr,),
  align: center,
  stroke: none,
  inset: 2pt,
  [#text(size: 12pt, weight: "bold")[промпт / предыдущий токен]],
  [#text(fill: accent, size: 13pt)[↓]],
  [#block(fill: rgb("#edf4ff"), stroke: (paint: accent, thickness: 0.4pt), radius: 4pt, inset: 7pt, width: 72%)[LM-адаптер (Llama / Qwen → логиты + KV-кэш)]],
  [#text(fill: accent, size: 13pt)[↓]],
  [#block(fill: rgb("#edf4ff"), stroke: (paint: accent, thickness: 0.4pt), radius: 4pt, inset: 7pt, width: 72%)[общий модуль формирования $P_"reference"$]],
  [#text(fill: accent, size: 13pt)[↓]],
  [#text(size: 13pt, weight: "bold")[$P_"reference"$]],
)]

#v(6pt)
#grid(columns: (1fr, 1fr), gutter: 14pt)[
  #goodbox[
    #text(fill: rgb("#1a7f37"), weight: "bold")[Контрольная генерация] \
    #small[Без секрета: токены выбираются из $P_"reference"$. Это baseline для сравнения качества, скорости и обнаружимости.]
  ]
][
  #warnbox[
    #text(fill: rgb("#8a5a00"), weight: "bold")[Стегогенерация] \
    #small[Стегометод получает $P_"reference"$ и секретные биты, выбирает токен и задает $Q_"stego"$.]
  ]
]

#v(5pt)
#keybox[
  #small[*Важно:* KL/TVD измеряют отличие $Q_"stego"$ от $P_"reference"$, то есть дополнительное изменение распределения, создаваемое стегометодом.]
]

#v(5pt)
#keybox[
  #small[*$P_"reference"$ - распределение следующего токена после всех общих преобразований, но до вмешательства стегометода*.]
    #v(5pt)
    #tiny[`prefix -> LM + KV-cache -> BF16 logits -> FP32 -> mask invalid/special IDs -> temperature -> softmax -> top-k/top-p -> renormalization -> P_reference`]
]
#pagebreak()


// ============================================================
// 6. Что измеряет бенчмарк
// ============================================================
#slide-title[Метрики]

#small[
Для каждого стегометода измеряются как минимум следующие группы характеристик:
]

#v(7pt)
#table(
  columns: (0.25fr, 0.2fr, 0.45fr),
  align: (left + horizon, left + horizon, left + horizon),
  stroke: 0.3pt, inset: 4.8pt,
  table.header([*Группа*], [*Метрики*], [*Что показывает*]),
  [Емкость стегоканала], [BPT (бит/токен)], [сколько секретных бит передано на один сгенерированный токен],
  [Использование энтропии], [Entropy utilization (%)], [какая доля неопределенности LM превращена в полезную емкость],
  [Искажение распределения], [KL (бит/токен), TVD], [насколько стегометод изменил распределение следующего токена],
  [Качество стеготекста], [NLL (нат/токен), PPL], [насколько вероятен текст с точки зрения оценочной LM],
  [Надежность извлечения секрета], [round-trip, BER (%)], [восстанавливается ли секрет после передачи текста],
  [Вычислительная эффективность], [мс, мс/токен, бит/с, VRAM], [сколько ресурсов требует кодирование и декодирование],
  [Обнаружимость (стегоанализ)], [ROC-AUC], [насколько стегоанализатор отличает стеготекст от контрольной генерации],
)

#v(8pt)
#keybox[
  #small[*Главный смысл:* \ бенчмарк должен показать компромиссы «емкость - искажение», «емкость - качество», «емкость - надежность», «емкость - обнаружимость», а не выбрать победителя по одной метрике.]
]
#pagebreak()


// ============================================================
// 7. Что зафиксировано сейчас и что уточняется
// ============================================================
#slide-title[Что фиксирует спецификация v0.1 и что уточняется позже]

#grid(columns: (1fr, 1fr), gutter: 14pt)[
  #goodbox[
    #text(fill: rgb("#1a7f37"), weight: "bold")[Зафиксировано в v0.1] \
    #v(3pt)
    #small[
    - основная LM: Llama-3.2-3B \
    - резервная LM: Qwen3-4B-Base \
    - единый путь с KV-кэшем \
    - правило формирования $P_"reference"$ \
    - независимые секретные потоки \
    - допустимые токены и исключения \
    - набор метрик \
    - полный round-trip \
    - структура конфигурации и результатов
    ]
  ]
][
  #warnbox[
    #text(fill: rgb("#8a5a00"), weight: "bold")[Уточняется в v1.0] \
    #v(3pt)
    #small[
    - число промптов \
    - число секретов и повторов \
    - диапазоны параметров методов \
    - окончательная длина генерации \
    - способ получения $Q_"stego"$ для каждого метода \
    - финальный стегоанализатор \
    - разбиение данных для ROC-AUC \
    - вычислительный бюджет основных экспериментов
    ]
  ]
]

#v(10pt)
#keybox[
  #text(fill: accent, weight: "bold")[Следующий шаг] \
  #small[Реализовать данный контракт в репозитории бенчмарка этапа 2 и подключить первые методы: Bins, Huffman и Arithmetic Coding.]
]

#v(6pt)
#cap[Спецификация v0.1 не закрывает все будущие решения, а задает проверяемую основу для реализации и пилотных экспериментов.]
