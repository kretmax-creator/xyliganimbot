# xyliganimbot — Telegram-бот для поиска в базе знаний
# Python 3.11, один процесс. Модели и данные — через volume mount.

FROM python:3.11-slim

WORKDIR /app

# Зависимости: сначала CPU-версия torch (лёгкая), потом остальное — без тяжёлого torch+CUDA
COPY requirements.txt .
# Сначала ставим CPU-версию torch (она лёгкая)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Потом остальные зависимости (уже без тяжёлого torch+cuda)
RUN pip install --no-cache-dir -r requirements.txt

# Код и конфиг по умолчанию
COPY src/ src/
COPY config.yaml.example config.yaml

# Пустые каталоги: данные, логи и модели монтируются при запуске
RUN mkdir -p data data/images logs models /app/models/.cache/huggingface /app/models/.cache/torchinductor

# Создаём системного пользователя с UID/GID 10001 для совместимости с k8s securityContext
RUN addgroup --gid 10001 app && adduser --uid 10001 --gid 10001 --home /app/models --disabled-password --gecos "" app

# Явно направляем кэш HF/torch в rw-volume /app/models
ENV USER=app \
    LOGNAME=app \
    HOME=/app/models \
    XDG_CACHE_HOME=/app/models/.cache \
    HF_HOME=/app/models/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/app/models/.cache/torchinductor

# Секреты — только через -e / --env-file. Модели не в образе.
CMD ["python", "-m", "src"]
