"""
Основной модуль Telegram-бота для xyliganimbot.

Обеспечивает подключение к Telegram Bot API, получение обновлений
через long polling и базовую обработку сообщений.
"""

import sys
from typing import Dict, List

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from src.config import (
    load_config,
    load_secrets,
    validate_config,
    get_whitelists,
    get_logging_config,
)
from src.logging import setup_logging, get_logger, get_audit_logger

# Логгеры будут инициализированы после setup_logging()
logger = None
audit_logger = None

# Белые списки будут загружены при старте
whitelists: Dict[str, List[int]] = {}


def is_user_allowed(user_id: int, chat_id: int) -> bool:
    """
    Проверяет, разрешен ли доступ пользователю и чату.

    Args:
        user_id: Telegram ID пользователя
        chat_id: Telegram ID чата

    Returns:
        True если доступ разрешен, False иначе
    """
    users = whitelists.get("users", [])
    chats = whitelists.get("chats", [])

    # Проверяем, есть ли пользователь в белом списке
    user_allowed = not users or user_id in users

    # Проверяем, есть ли чат в белом списке
    chat_allowed = not chats or chat_id in chats

    return user_allowed and chat_allowed


def is_admin(user_id: int) -> bool:
    """
    Проверяет, является ли пользователь администратором.

    Args:
        user_id: Telegram ID пользователя

    Returns:
        True если пользователь администратор, False иначе
    """
    admins = whitelists.get("admins", [])
    return user_id in admins


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Базовый обработчик входящих сообщений.

    На текущей итерации только логирует входящие сообщения.
    В последующих итерациях будет добавлена обработка команд и поиск.

    Args:
        update: Обновление от Telegram
        context: Контекст обработчика
    """
    logger.debug(f"handle_message called for update_id={update.update_id}")
    
    user = update.effective_user
    chat = update.effective_chat
    message = update.message

    if not user or not chat or not message:
        logger.warning(
            f"Received update without user, chat or message: update_id={update.update_id}, "
            f"update_type={update.update_type}"
        )
        return

    # Проверка белого списка
    if not is_user_allowed(user.id, chat.id):
        logger.warning(
            f"Access denied: user_id={user.id}, username={user.username}, "
            f"chat_id={chat.id}, chat_type={chat.type}"
        )
        return

    # Логирование входящего сообщения
    import logging
    root_logger = logging.getLogger()
    log_user_messages = getattr(root_logger, "log_user_messages", False)
    message_text = message.text if message.text else "[non-text message]"

    if log_user_messages:
        logger.info(
            f"Received message: user_id={user.id}, username={user.username}, "
            f"chat_id={chat.id}, text='{message_text}'"
        )
    else:
        logger.info(
            f"Received message: user_id={user.id}, username={user.username}, "
            f"chat_id={chat.id}"
        )

    # Запись в аудит
    audit_logger.info(
        f"Message from user_id={user.id}, username={user.username}, "
        f"chat_id={chat.id}"
    )


def create_application(token: str) -> Application:
    """
    Создает и настраивает приложение Telegram-бота.

    Args:
        token: Токен Telegram-бота

    Returns:
        Настроенное приложение
    """
    application = Application.builder().token(token).build()

    # Регистрация обработчика всех сообщений
    # В группах бот получает сообщения, если его упоминают или он администратор
    # Обрабатываем все сообщения для отладки
    application.add_handler(
        MessageHandler(filters.ALL, handle_message)
    )

    return application


def main() -> None:
    """
    Основная функция запуска бота.

    Загружает конфигурацию, настраивает логирование,
    создает приложение и запускает long polling.
    """
    try:
        # Загрузка конфигурации
        config = load_config()
        secrets = load_secrets()

        # Валидация конфигурации
        validate_config(config, secrets)

        # Загрузка белых списков
        global whitelists
        whitelists = get_whitelists(config)

        # Настройка логирования
        logging_config = get_logging_config(config)
        log_level = secrets.get("LOG_LEVEL") or logging_config["level"]

        setup_logging(
            level=log_level,
            log_file=logging_config["file"],
            audit_file=logging_config["audit_file"],
            log_user_messages=logging_config["log_user_messages"],
            max_bytes=logging_config["max_bytes"],
            backup_count=logging_config["backup_count"],
        )

        # Инициализация логгеров после настройки
        global logger, audit_logger
        logger = get_logger(__name__)
        audit_logger = get_audit_logger()

        # Получение токена
        token = secrets.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment variables")

        logger.info("Starting xyliganimbot...")
        logger.info(
            f"Whitelists loaded: {len(whitelists.get('users', []))} users, "
            f"{len(whitelists.get('chats', []))} chats, "
            f"{len(whitelists.get('admins', []))} admins"
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    try:
        # Создание приложения
        application = create_application(token)

        # Запуск long polling
        logger.info("Bot started, waiting for messages...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
