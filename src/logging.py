"""
Модуль логирования для xyliganimbot.

Обеспечивает настройку и использование логирования в приложении.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/app.log",
    audit_file: str = "logs/audit.log",
    log_user_messages: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Настраивает логирование для приложения.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов приложения
        audit_file: Путь к файлу аудита
        log_user_messages: Логировать ли тексты сообщений пользователей
        max_bytes: Максимальный размер файла лога перед ротацией
        backup_count: Количество резервных копий логов
    """
    # Создаем директорию для логов, если её нет
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = Path(audit_file).parent
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Преобразуем строковый уровень в константу logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Настройка форматтера
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Настройка ротирующего файлового обработчика для основного лога
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Настройка обработчика для консоли (для разработки)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Настройка отдельного логгера для аудита
    audit_logger = logging.getLogger("audit")
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    audit_handler.setFormatter(formatter)
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    # Предотвращаем дублирование в корневом логгере
    audit_logger.propagate = False

    # Сохраняем настройку логирования сообщений пользователей
    # (будет использоваться в обработчиках)
    root_logger.log_user_messages = log_user_messages


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер с указанным именем.

    Args:
        name: Имя логгера (обычно __name__ модуля)

    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


def get_audit_logger() -> logging.Logger:
    """
    Получить логгер для аудита.

    Returns:
        Логгер для записи аудита
    """
    return logging.getLogger("audit")
