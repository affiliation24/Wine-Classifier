# Классификатор типа вина по описанию

Небольшой проект по deep learning: модель читает текстовое описание вина (как в карточке товара или заметке дегустатора) и определяет тип вина:

- красное
- белое
- розовое

Проект можно запускать локально как простое веб-приложение (интерфейс на Gradio).

**Демо:** ссылка на HuggingFace Spaces — `https://pyrowoon-wine-guide.hf.space`

---

## Что умеет

Вы вводите описание обычным текстом, а модель возвращает:

- предсказанный тип вина
- «уверенность» по каждому классу в процентах

Пример:

```
Ввод:  "Насыщенный аромат темной вишни и черной смородины, заметен дуб.
        Танины плотные, послевкусие долгое и пряное."

Вывод: Красное
       Красное: 94.3%
       Белое:    3.1%
       Розовое:  2.6%
```

---

## Как это работает (простыми словами)

Модель объединяет два источника информации:

- **Текст**: берётся предобученная языковая модель BERT, которая «понимает» смысл фраз и превращает текст в набор чисел (вектор признаков).
- **Слова‑подсказки из предметной области**: дополнительно считаются простые признаки по ключевым словам, которые часто встречаются в описаниях (например, «дуб», «цитрус», «шампанское» и т. п.).

Дальше эти признаки склеиваются и подаются в небольшой классификатор (несколько полносвязных слоёв), который выдаёт 3 вероятности: красное / белое / розовое.

Если коротко: **BERT даёт смысл текста**, а **доменные признаки помогают на конкретных винных терминах**.

---

## Данные

- **Источник**: датасет с Kaggle — [Wine Dataset](https://www.kaggle.com/datasets/elvinrustam/wine-dataset)
- **Размер**: 1274 описания после очистки
- **Классы**: красное (566), белое (584), розовое (124)
- **Балансировка**: редкий класс «розовое» увеличивался простым оверсемплингом через `sklearn.utils.resample`

---

## Обучение (кратко)

Параметры обучения:

| Параметр | Значение |
|---|---|
| База | `bert-base-uncased` |
| Оптимизатор | AdamW |
| Скорость обучения | 2e-5 |
| Batch size | 16 |
| Эпохи | 3 |
| Максимальная длина текста | 128 |
| Разделение train/test | 80/20 (стратифицированно) |

Результаты после 3 эпох:

| Эпоха | Loss | Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 1 | 0.8673 | 67.06% | 0.7043 |
| 2 | 0.4405 | 83.92% | 0.8463 |
| 3 | 0.3278 | 89.02% | 0.8927 |

---

## Структура проекта

```
wine-classifier/
├── README.md
├── app.py                                  # веб-интерфейс
├── data
│   └── WineDataset.csv                     # исходный датасет
├── mlflow.db
├── mlruns
│   └── 2
│       └── 13a8dc1a376040a0b22151574baba286
│           └── artifacts
│               ├── confusion_matrix.png
│               └── training_curves.png
├── plots                                   # папка с графиками 
│   ├── confusion_matrix.png
│   ├── eda_overview.png
│   └── training_curves.png
├── requirements.txt                        # зависимости Python
├── train.ipynb                             # обучающий ноутбук
└── wine_model
    ├── model.pt                            # веса обученной модели
    └── tokenizer
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── vocab.txt
```

---

## Как запустить локально

1) Склонируйте репозиторий (или Space) и перейдите в папку проекта:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

2) Установите зависимости:

```bash
pip install -r requirements.txt
```

3) Запустите приложение:

```bash
python app.py
```

После этого откройте ссылку, которую покажет консоль (обычно `http://127.0.0.1:7860`).

---

## Зависимости (главные)

- `torch`
- `transformers`
- `gradio`

Остальные перечислены в `requirements.txt`.

---

## Примеры для проверки

Красное:

> "Enticing aromas of ripe cherries, blackberries, and light vanilla notes. The taste is rich, with velvety tannins and a warm, spicy finish. Perfect with steak or aged cheeses."

Белое:

> "Fresh aromas of green apple, ripe pear, and white blossoms. Crisp and vibrant on the palate with balanced acidity, leading to a clean, mineral finish. Perfect as an aperitif or with light seafood."

Розовое:

> "Delicate aromas of fresh strawberries, raspberries, and a hint of citrus zest. Light and refreshing on the palate with bright acidity and a subtle floral finish. Ideal for warm afternoons, picnics, or pairing with light salads."

---

## Технологии

- **PyTorch** — обучение и инференс модели
- **HuggingFace Transformers** — BERT и токенизатор
- **Gradio** — веб-интерфейс для ввода текста
- **scikit-learn** — предобработка и балансировка классов
- **Pandas** — работа с данными
- **MLflow** - хранение гиперпараметров
---

## Зачем проект

Это учебный pet‑проект: практиковался PyTorch, дообучение BERT и развёртывание модели в виде простого приложения.
