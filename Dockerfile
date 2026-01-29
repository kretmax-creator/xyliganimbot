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
RUN mkdir -p data data/images logs models

# Секреты — только через -e / --env-file. Модели не в образе.
CMD ["python", "-m", "src"]
