"""
Основной модуль Telegram-бота для xyliganimbot.

Обеспечивает подключение к Telegram Bot API, получение обновлений
через long polling и базовую обработку сообщений.
"""

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from src.logging import setup_logging, get_logger, get_audit_logger

# Логгеры будут инициализированы после setup_logging()
logger = None
audit_logger = None


def is_user_allowed(user_id: int, chat_id: int) -> bool:
    """
    Проверяет, разрешен ли доступ пользователю и чату.

    Временная упрощенная реализация. В итерации 3 будет заменена
    на загрузку белых списков из config.yaml.

    Args:
        user_id: Telegram ID пользователя
        chat_id: Telegram ID чата

    Returns:
        True если доступ разрешен, False иначе
    """
    # Временная реализация - пока разрешаем всем
    # В итерации 3 будет загрузка из config.yaml
    return True


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

    Загружает токен из переменных окружения, настраивает логирование,
    создает приложение и запускает long polling.
    """
    # Загрузка переменных окружения из .env файла
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # Загрузка токена из переменных окружения
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not set in environment variables")
        sys.exit(1)

    # Настройка логирования
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Временная загрузка log_user_messages из config.yaml
    # В итерации 3 будет полная загрузка конфигурации
    log_user_messages = False
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config and "logging" in config:
                    log_user_messages = config["logging"].get("log_user_messages", False)
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
    
    setup_logging(level=log_level, log_user_messages=log_user_messages)
    
    # Инициализация логгеров после настройки
    global logger, audit_logger
    logger = get_logger(__name__)
    audit_logger = get_audit_logger()

    logger.info(f"Starting xyliganimbot... (log_user_messages={log_user_messages})")

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
