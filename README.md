# PlayingCardClassifier
Классификация игральных карт на PyTorch с использованием transfer learning 
(EfficientNet-B0) и подбором гиперпараметров через Optuna. REST API на FastAPI, сборка в Docker.

## Описание

Модель классифицирует изображения игральных карт на 53 класса (52 карты + джокер). 
В основе — EfficientNet-B0, предобученный на ImageNet, у которого отрезан исходный 
классификатор и приделан свой на 53 выхода. Модель обучается на датасете игральных карт 
(взят с Kaggle), гиперпараметры подбираются через Optuna, а для инференса используется FastAPI-эндпоинт.

## Результаты

В репозитории лежит модель, которая обучалась на лучших  гиперпараметрах, 
найденных Optuna за 10 trials по 5 эпох: 
learning rate 0.0062, batch size 16, оптимизатор SGD, weight decay 0.00032. 
Выдала 95.85% accuracy на валидации. Финальное обучение проводилось в 10 эпох, 
лучшая valid accuracy — 0.9811.

## Структура проекта

В корне лежат основные модули. 
`api.py` — FastAPI-приложение с эндпоинтом `/predict`. 
`config.py` хранит пути к артефактам и настройки Optuna. 
`dataset.py` отвечает за датасет, трансформации и загрузку данных через kagglehub. 
`model.py` содержит архитектуру классификатора и функцию загрузки обученного чекпоинта. 
`train.py` обучает модель — либо с дефолтными параметрами, либо с лучшими из файла. 
`tune.py` запускает поиск гиперпараметров через Optuna. 
`test.py` считает accuracy на тестовой выборке. 
`inference.py` показывает предсказания на случайных картинках. 
`visualize.py` строит график лоссов. 
Плюс `Dockerfile`, `.dockerignore` и `requirements.txt`.

## Датасет

Используется датасет Playing Cards Image Dataset от gpiosenka с Kaggle 
(https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).
В нём 53 класса, они сбалансированы, изображения приводятся к 128×128.
Датасет скачивается автоматически через kagglehub при первом запуске — ничего вручную качать не нужно.

## Быстрый старт через Docker

Сначала клонируем репозиторий (через HTTPS как ниже или другим удобным для вас способом):

    git clone https://github.com/dary56/PlayingCardClassifier.git
    cd PlayingCardClassifier

Собираем образ. Первая сборка займёт несколько минут — Docker скачает
базовый образ и установит CPU-версию PyTorch со всеми зависимостями 
(для инференса CPU достаточно, а образ получается в разы меньше, чем с CUDA-сборкой).

    docker build -t card-classifier .

Запускаем контейнер:

    docker run -p 8000:8000 card-classifier

Открываем в браузере http://localhost:8000/docs — это Swagger UI, через
который можно сразу протестировать API. Разворачиваем `POST /predict`,
нажимаем «Try it out», загружаем картинку карты (не обязательно из датасета, любую) 
и жмём «Execute». В ответ придёт JSON вида:

    {
    "class": "queen of clubs",
    "probability": 0.8749886155128479
    }

## Использование без Docker

Если хочется обучить модель самостоятельно или подобрать другие гиперпараметры,
можно работать локально. Клонируем репозиторий и создаём виртуальное окружение (ниже команды для Windows):

    python -m venv .venv
    .venv\Scripts\activate

Дальше ставим PyTorch (https://pytorch.org/get-started/locally/). 
Тут есть выбор — с GPU или только CPU:

    # Если есть NVIDIA GPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    # Если GPU нет или нужен только CPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Если CUDA доступна, вычисления пойдут на GPU, иначе на CPU. 
Учтите, что на CPU запуск tune и train даже на несколько эпох займёт значительно больше времени.
После этого ставим остальные зависимости:

    pip install -r requirements.txt

## Пошаговое использование

Сначала имеет смысл запустить подбор гиперпараметров:

    python tune.py

По умолчанию Optuna делает 30 trials по 5 эпох каждый. Результаты складываются в SQLite 
(`data/optuna.db`), а лучшие параметры — в `data/best_params.json`.

Дальше обучаем финальную модель:

    python train.py

Если `data/best_params.json` существует, обучение пойдёт с лучшими параметрами из Optuna. 
Если файла нет — с дефолтными (даёт 93.58%% accuracy на тесте). На выходе получаем 
`data/best_model.pth` с весами, `data/losses.json` с лоссами по эпохам и `data/classes.json` с именами классов.

Проверяем качество на тесте:

    python test.py

Опционально можно построить график лоссов:

    python visualize.py

Он сохранит картинку в `data/loss_plot.png`. 

Можно также посмотреть предсказания на случайных картинках из тестовой выборки:

    python inference.py

## API

Эндпоинта `POST /predict` принимает изображение карты через `multipart/form-data` 
и возвращает JSON с полем `class` (имя класса) и `probability` (вероятность).
Когда модель обучена, можно запускать API:

    uvicorn api:app --reload

После успешного запуска появится строка по типу:

    INFO:     Uvicorn running on http://127.0.0.1:8000

Нужно открыть ссылку в браузере, в конце добавив `/docs`. Разворачиваем `POST /predict`,
нажимаем «Try it out», загружаем картинку карты (не обязательно из датасета, любую) 
и жмём «Execute». Получаем в ответ JSON с классом и вероятностью.
