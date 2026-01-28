"""
Модуль работы с конфигурацией для xyliganimbot.

Обеспечивает загрузку и валидацию конфигурации из config.yaml
и переменных окружения.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from dotenv import load_dotenv

from src.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Загружает конфигурацию из config.yaml.

    Args:
        config_path: Путь к файлу конфигурации. Если None, используется
                     config.yaml в корне проекта.

    Returns:
        Словарь с конфигурацией

    Raises:
        FileNotFoundError: Если файл конфигурации не найден
        yaml.YAMLError: Если файл имеет некорректный формат YAML
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: {e}") from e


def load_secrets() -> Dict[str, Optional[str]]:
    """
    Загружает секреты из переменных окружения.

    Загружает переменные из .env файла и возвращает словарь
    с секретными данными.

    Returns:
        Словарь с секретами (ключи: TELEGRAM_BOT_TOKEN, GOOGLE_SERVICE_ACCOUNT_KEY и т.д.)
    """
    # Загрузка переменных окружения из .env файла
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    secrets = {
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "GOOGLE_SERVICE_ACCOUNT_KEY": os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL"),
    }

    return secrets


def validate_config(config: Dict[str, Any], secrets: Dict[str, Optional[str]]) -> None:
    """
    Валидирует обязательные параметры конфигурации.

    Args:
        config: Словарь с конфигурацией из config.yaml
        secrets: Словарь с секретами из переменных окружения

    Raises:
        ValueError: Если отсутствуют обязательные параметры
    """
    errors = []

    # Проверка обязательных секретов
    if not secrets.get("TELEGRAM_BOT_TOKEN"):
        errors.append("TELEGRAM_BOT_TOKEN not set in environment variables")

    # Проверка белого списка чатов (доступ только по чатам)
    chats = config.get("chats", {}).get("allowed", [])

    if not chats:
        errors.append("At least one chat must be in whitelist (chats.allowed)")

    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_message)


def get_whitelists(config: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Извлекает белые списки из конфигурации.

    Доступ проверяется только по чату: если чат в списке, команды всех участников
    обрабатываются (кроме админских — те только для пользователей из admins).

    Returns:
        Словарь с ключами:
        - "chats": список chat_id чатов
        - "admins": список telegram_id администраторов
    """
    chats = config.get("chats", {}).get("allowed", [])
    admins = config.get("admins", [])

    chats_list = [int(cid) for cid in chats] if chats else []
    admins_list = [int(aid) for aid in admins] if admins else []

    return {
        "chats": chats_list,
        "admins": admins_list,
    }


def get_bot_username(config: Dict[str, Any]) -> str:
    """
    Извлекает username бота из конфигурации (без @).

    Используется для фильтрации упоминаний в групповых чатах и удаления
    упоминания из текста запроса.

    Returns:
        Username бота, по умолчанию "xyliganim_bot"
    """
    return (config.get("bot_username") or "xyliganim_bot").strip().lstrip("@")


def get_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает настройки логирования из конфигурации.

    Args:
        config: Словарь с конфигурацией

    Returns:
        Словарь с настройками логирования
    """
    logging_config = config.get("logging", {})
    return {
        "level": logging_config.get("level", "INFO"),
        "file": logging_config.get("file", "logs/app.log"),
        "audit_file": logging_config.get("audit_file", "logs/audit.log"),
        "log_user_messages": logging_config.get("log_user_messages", False),
        "max_bytes": logging_config.get("max_bytes", 10485760),  # 10MB
        "backup_count": logging_config.get("backup_count", 5),
    }


def get_google_docs_config(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Извлекает настройки Google Docs из конфигурации.

    Args:
        config: Словарь с конфигурацией

    Returns:
        Словарь с настройками Google Docs (url, document_id)
    """
    google_docs = config.get("google_docs", {})
    return {
        "url": google_docs.get("url"),
        "document_id": google_docs.get("document_id"),
    }


def get_cache_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает настройки кэша из конфигурации.

    Args:
        config: Словарь с конфигурацией

    Returns:
        Словарь с настройками кэша
    """
    cache_config = config.get("cache", {})
    return {
        "update_interval_days": cache_config.get("update_interval_days", 7),
        "auto_update": cache_config.get("auto_update", True),
    }
