"""
Основной модуль Telegram-бота для xyliganimbot.

Обеспечивает подключение к Telegram Bot API, получение обновлений
через long polling и обработку команд и сообщений.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
)

from src.config import (
    load_config,
    load_secrets,
    validate_config,
    get_whitelists,
    get_logging_config,
)
from src.logging import setup_logging, get_logger, get_audit_logger
from src.search import load_index_from_cache
from src.handlers import handle_help_command, handle_search_query, init_search_context

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


async def check_access(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Проверяет доступ пользователя и чата к боту.

    Args:
        update: Обновление от Telegram
        context: Контекст обработчика

    Returns:
        True если доступ разрешен, False иначе
    """
    user = update.effective_user
    chat = update.effective_chat

    if not user or not chat:
        logger.warning("Received update without user or chat")
        return False

    if not is_user_allowed(user.id, chat.id):
        logger.warning(
            f"Access denied: user_id={user.id}, username={user.username}, "
            f"chat_id={chat.id}, chat_type={chat.type}"
        )
        return False

    return True


def create_application(token: str) -> Application:
    """
    Создает и настраивает приложение Telegram-бота.

    Args:
        token: Токен Telegram-бота

    Returns:
        Настроенное приложение
    """
    application = Application.builder().token(token).build()

    # Регистрация обработчика команды /help
    async def help_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await check_access(update, context):
            await handle_help_command(update, context)

    application.add_handler(CommandHandler("help", help_wrapper))

    # Регистрация обработчика текстовых сообщений (поисковые запросы)
    async def search_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await check_access(update, context):
            await handle_search_query(update, context)

    # Обрабатываем только текстовые сообщения
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, search_wrapper)
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

        # Загрузка поискового индекса
        project_root = Path(__file__).parent.parent
        cache_file = project_root / "data" / "knowledge_cache.json"
        html_file = project_root / "data" / "knowledge.html"
        sections_file = project_root / "data" / "sections.json"
        images_dir = project_root / "data" / "images"

        search_index_data = load_index_from_cache(cache_file)
        if search_index_data:
            logger.info("Search index loaded successfully")
            init_search_context(
                index=search_index_data,
                html_file=html_file,
                sections_file=sections_file,
                images_dir=images_dir,
            )
        else:
            logger.warning(
                "Search index not found in cache. "
                "Bot will work, but search functionality will be limited."
            )
            # Инициализируем с пустым индексом, чтобы бот мог работать
            init_search_context(
                index={"section_index": {}, "content_index": {}},
                html_file=html_file,
                sections_file=sections_file,
                images_dir=images_dir,
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
