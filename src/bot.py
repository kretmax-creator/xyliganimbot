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
    get_bot_username,
)
from src.logging import setup_logging, get_logger, get_audit_logger
from src.search import load_index_from_cache, load_embeddings_from_cache
from src.handlers import handle_help_command, handle_admin_command, handle_search_query, init_search_context

# Логгеры будут инициализированы после setup_logging()
logger = None
audit_logger = None

# Белые списки будут загружены при старте
whitelists: Dict[str, List[int]] = {}


def is_chat_allowed(chat_id: int) -> bool:
    """
    Проверяет, разрешён ли доступ чату.

    Белый список только для чатов: если пользователь в разрешённом чате,
    его команды обрабатываются (кроме админских — те только для admins).

    Args:
        chat_id: Telegram ID чата

    Returns:
        True если чат в белом списке, False иначе
    """
    chats = whitelists.get("chats", [])
    return not chats or chat_id in chats


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

    if not is_chat_allowed(chat.id):
        logger.warning(
            f"Access denied: user_id={user.id}, username={user.username}, "
            f"chat_id={chat.id}, chat_type={chat.type}"
        )
        return False

    return True


def create_application(token: str, bot_username: str) -> Application:
    """
    Создает и настраивает приложение Telegram-бота.

    Args:
        token: Токен Telegram-бота
        bot_username: Username бота (без @) из конфига, для фильтра упоминаний

    Returns:
        Настроенное приложение
    """
    application = Application.builder().token(token).build()
    application.bot_data["bot_username"] = bot_username

    # Регистрация обработчика команды /help
    async def help_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await check_access(update, context):
            await handle_help_command(update, context)

    application.add_handler(CommandHandler("help", help_wrapper))

    # Регистрация обработчика команды /admin (только для администраторов)
    async def admin_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await check_access(update, context):
            return
        user = update.effective_user
        if not user:
            logger.warning("Received /admin command without user")
            return
        if not is_admin(user.id):
            try:
                await update.message.reply_text("Команда доступна только администраторам.")
            except Exception as e:
                logger.error(f"Error sending admin refusal: {e}", exc_info=True)
            logger.warning(f"Admin command denied for user_id={user.id}, username={user.username}")
            return
        await handle_admin_command(update, context)

    application.add_handler(CommandHandler("admin", admin_wrapper))

    # Поиск: в группах — только при упоминании бота; в личке — только по команде /search
    async def search_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await check_access(update, context):
            await handle_search_query(update, context)

    # В групповых чатах — только сообщения с упоминанием бота
    search_filter_groups = (
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS & filters.Mention(bot_username)
    )
    application.add_handler(MessageHandler(search_filter_groups, search_wrapper))

    # Команда /search — для личных чатов и по желанию в группах
    async def search_cmd_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await check_access(update, context):
            return
        query = " ".join(context.args).strip() if context.args else ""
        if not query:
            try:
                await update.message.reply_text(
                    "Используйте: /search ваш запрос\nНапример: /search как настроить VPN"
                )
            except Exception as e:
                logger.error(f"Error sending /search hint: {e}", exc_info=True)
            return
        await handle_search_query(update, context, query=query)

    application.add_handler(CommandHandler("search", search_cmd_wrapper))

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
            f"Whitelists loaded: {len(whitelists.get('chats', []))} chats, "
            f"{len(whitelists.get('admins', []))} admins"
        )

        # Загрузка embeddings или поискового индекса
        project_root = Path(__file__).parent.parent
        cache_file = project_root / "data" / "knowledge_cache.json"
        markdown_file = project_root / "data" / "knowledge.md"
        # Проверяем также старый формат HTML для обратной совместимости
        if not markdown_file.exists():
            markdown_file = project_root / "data" / "knowledge.html"
        images_dir = project_root / "data" / "images"
        models_dir = project_root / "models"

        # Проверки наличия необходимых данных (без автозагрузки)
        logger.info("Checking required data availability...")
        
        # Проверка наличия документа
        if not markdown_file.exists():
            logger.warning(f"Markdown/HTML file not found: {markdown_file}")
        else:
            logger.info(f"Markdown/HTML file found: {markdown_file}")
        
        # Проверка наличия embedding-модели (для семантического поиска)
        model_name = "intfloat/multilingual-e5-small"
        model_path = models_dir / model_name
        if model_path.exists():
            logger.info(f"Embedding model found: {model_path}")
        else:
            logger.warning(
                f"Embedding model not found: {model_path}. "
                f"Semantic search will not be available. Use /admin load_model to download it."
            )

        # Сначала пытаемся загрузить embeddings для семантического поиска
        embeddings_data = load_embeddings_from_cache(cache_file)
        if embeddings_data:
            logger.info("Embeddings loaded successfully for semantic search")
            # Проверяем наличие модели для семантического поиска
            from src.search import load_embedding_model
            model = load_embedding_model()
            if model is None:
                logger.warning(
                    "Embeddings found but model not available. "
                    "Semantic search will not work. Falling back to token-based search."
                )
                embeddings_data = None  # Переключаемся на token-based поиск
        
        if embeddings_data:
            # Используем embeddings для семантического поиска
            init_search_context(
                index=embeddings_data,
                markdown_file=markdown_file,
                images_dir=images_dir,
            )
        else:
            # Если embeddings нет, загружаем обычный token-based индекс
            logger.info("Embeddings not found, trying to load token-based search index...")
            search_index_data = load_index_from_cache(cache_file)
            if search_index_data:
                logger.info("Token-based search index loaded successfully")
                init_search_context(
                    index=search_index_data,
                    markdown_file=markdown_file,
                    images_dir=images_dir,
                )
            else:
                logger.warning(
                    "Neither embeddings nor search index found in cache. "
                    "Bot will work, but search functionality will be limited. "
                    "Use /admin vectorize to create embeddings."
                )
                # Инициализируем с пустым индексом, чтобы бот мог работать
                init_search_context(
                    index={"section_index": {}, "content_index": {}},
                    markdown_file=markdown_file,
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
        bot_username = get_bot_username(config)
        application = create_application(token, bot_username)

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
