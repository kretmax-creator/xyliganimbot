# xyliganimbot

Telegram-бот для поиска ответов в базе знаний (документ Google Docs).

## Описание

Бот позволяет искать информацию в базе знаний через Telegram. База знаний хранится в Google Docs и автоматически импортируется в бот.

## Требования

- Python 3.x
- Telegram Bot Token
- Доступ к Google Docs (публичный документ или через API)

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\Scripts\activate  # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Скопируйте `.env.example` в `.env` и заполните необходимые переменные
5. Скопируйте `config.yaml.example` в `config.yaml` и настройте конфигурацию

## Запуск

```bash
python -m src
```

## Запуск в Docker (Ubuntu)

Сборка и запуск на виртуалке с Ubuntu (терминал).

### Требования

- Docker установлен: `docker --version`
- Проект на сервере (например, примаплен в `/home/test/shared/xyliganimbot`). Используются те же папки, что и при разработке: `data/`, `logs/`, `models/` — создавать их не нужно.

### 1. Подготовка на хосте

```bash
cd /home/test/shared/xyliganimbot

# Конфиг (если ещё нет)
cp config.yaml.example config.yaml
# Отредактируйте config.yaml: chats.allowed, admins, bot_username

# Секреты (если ещё нет): скопируйте .env.example в .env и укажите TELEGRAM_BOT_TOKEN
cp .env.example .env
# Отредактируйте .env
```

Импорт базы знаний и загрузка модели — отдельно (скрипты вне бота или `/admin load_model`, `/admin vectorize` после первого запуска). Без `data/knowledge.md`, `data/knowledge_cache.json` и модели в `models/` поиск работать не будет.

### 2. Сборка образа

```bash
docker build -t xyliganimbot .
```

В образе используется PyTorch только для CPU (без CUDA/cuDNN): меньше размер, нет зависимости от видеокарты и тяжёлых загрузок.

### 3. Запуск контейнера

Бот использует те же каталоги, что и в разработке. Пути — корень проекта на сервере (например `/home/test/shared/xyliganimbot`). Токен и секреты берутся из `.env` в папке с ботом:

```bash
PROJECT=/home/test/shared/xyliganimbot

docker run -d \
  --name xyliganimbot \
  --env-file "$PROJECT/.env" \
  -v "$PROJECT/data:/app/data" \
  -v "$PROJECT/logs:/app/logs" \
  -v "$PROJECT/models:/app/models" \
  -v "$PROJECT/config.yaml:/app/config.yaml" \
  xyliganimbot
```

В `.env` должны быть минимум `TELEGRAM_BOT_TOKEN`; при необходимости — `LOG_LEVEL`, `GOOGLE_SERVICE_ACCOUNT_KEY` и т.п.

### 4. Просмотр логов

```bash
docker logs -f xyliganimbot
```

Успешный старт: в логах есть `Bot started, waiting for messages...`. При пустых `data/` и `models/` появятся предупреждения — поиск заработает после импорта контента, загрузки модели и `/admin vectorize`.

Файлы логов также в `logs/` на хосте (`logs/app.log`, `logs/audit.log`).

### 5. Остановка и удаление контейнера

```bash
docker stop xyliganimbot
docker rm xyliganimbot
```

### 6. Перезапуск после изменений

```bash
PROJECT=/home/test/shared/xyliganimbot

docker stop xyliganimbot
docker rm xyliganimbot
docker build -t xyliganimbot .
docker run -d --name xyliganimbot \
  --env-file "$PROJECT/.env" \
  -v "$PROJECT/data:/app/data" \
  -v "$PROJECT/logs:/app/logs" \
  -v "$PROJECT/models:/app/models" \
  -v "$PROJECT/config.yaml:/app/config.yaml" \
  xyliganimbot
```

Модель в `models/` загружается администратором вручную (или через `/admin load_model` при наличии сети). При сборке образа модели не используются.

## Конфигурация

Основная конфигурация находится в `config.yaml`. Секретные данные (токены, ключи) настраиваются через переменные окружения в файле `.env`.

## Команды бота

- `/help` — список доступных команд и описание работы бота
- `/search запрос` — поиск в базе знаний (в личке); в группах — при упоминании бота
- Админские (только для пользователей из `admins` в конфиге):
  - `/admin load_model` — загрузка embedding-модели в `models/`
  - `/admin vectorize` — векторизация контента и сохранение в кэш

Импорт контента из Google Docs выполняется отдельным скриптом (`python testing/test_import_content.py`), не командой бота.

## Структура проекта

```
xyliganimbot/
├── src/              # Исходный код бота
│   ├── bot.py        # Основной модуль бота
│   ├── config.py     # Работа с конфигурацией
│   ├── google_docs.py # Работа с Google Docs
│   ├── search.py     # Логика поиска
│   ├── logging.py    # Логирование
│   ├── audit.py      # Аудит
│   └── handlers/     # Обработчики команд и сообщений
├── data/             # Кэш документа (knowledge.md, knowledge_cache.json, images/)
├── models/           # Embedding-модели (загружаются вручную)
├── logs/             # Логи приложения
├── docs/             # Документация
├── Dockerfile        # Образ для Docker
└── k8s/              # Kubernetes манифесты
```

## Документация

Подробная документация находится в папке `docs/`:
- `docs/idea.md` — описание идеи проекта
- `docs/vision.md` — техническое видение
- `docs/tasklist_v0.1.md` — план разработки
- `docs/BACKLOG.md` — бэклог задач
- `docs/TESTING.md` — план тестирования

## Лицензия

[Указать лицензию при необходимости]
