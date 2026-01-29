"""
Обработчики команд для xyliganimbot.

Обеспечивает обработку команд бота, таких как /help и /admin.
"""

from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from src.logging import get_logger
from src.audit import log_operation
from src.model_loader import download_model
from src.search import vectorize_content, load_embeddings_from_cache, load_index_from_cache
from src.handlers.messages import init_search_context

logger = get_logger(__name__)

# Имя модели по умолчанию (совпадает с bot.py и model_loader)
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


def _get_project_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Возвращает (project_root, cache_file, markdown_file, images_dir, models_dir)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    cache_file = project_root / "data" / "knowledge_cache.json"
    markdown_file = project_root / "data" / "knowledge.md"
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    images_dir = project_root / "data" / "images"
    models_dir = project_root / "models"
    return project_root, cache_file, markdown_file, images_dir, models_dir


async def handle_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /help.

    Отправляет пользователю информацию о доступных командах
    и краткое описание работы бота.

    Args:
        update: Обновление от Telegram
        context: Контекст обработчика
    """
    user = update.effective_user
    chat = update.effective_chat

    if not user or not chat:
        logger.warning("Received /help command without user or chat")
        return

    logger.info(f"Help command from user_id={user.id}, username={user.username}")

    help_text = (
        "🤖 *Xyliganimbot* — бот для поиска ответов в базе знаний\n\n"
        "📝 *Как использовать:*\n"
        "• В группе: напишите запрос, упомянув бота (@бот), или используйте /search запрос. Можно просто обратиться к боту с вопросом.\n"
        "• В личке: команда /search запрос\n\n"
        "🔍 *Команды:*\n"
        "/help — показать это сообщение\n"
        "/search запрос — поиск по базе знаний\n\n"
        "💡 *Примеры:*\n"
        "• @бот как настроить VPN?\n"
        "• /search где найти инструкцию"
    )

    try:
        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Help message sent to user_id={user.id}")
        include_request = context.bot_data.get("log_user_messages", False) if context else False
        log_operation(
            telegram_id=user.id,
            username=user.username,
            operation="help",
            result="ok",
            request_text="/help",
            include_request_text=include_request,
        )
    except Exception as e:
        logger.error(f"Error sending help message: {e}", exc_info=True)
        if context:
            log_operation(
                telegram_id=user.id,
                username=user.username,
                operation="help",
                result="error",
                request_text="/help",
                include_request_text=context.bot_data.get("log_user_messages", False),
                error=str(e),
            )


async def handle_admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /admin с подкомандами load_model и vectorize.

    Вызывается только для пользователей, прошедших проверку is_admin в bot.py.
    """
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat or not update.message:
        logger.warning("Received /admin command without user, chat or message")
        return

    subcommand = (context.args or [None])[0]
    if subcommand is None:
        await update.message.reply_text(
            "Команда /admin требует подкоманду.\n\n"
            "Доступные команды:\n"
            "/admin load_model — загрузить embedding-модель\n"
            "/admin vectorize — векторизовать контент и обновить кэш"
        )
        return

    subcommand = subcommand.lower().strip()
    if subcommand not in ("load_model", "vectorize"):
        await update.message.reply_text(
            f"Неизвестная подкоманда «{subcommand}».\n\n"
            "Доступные команды:\n"
            "/admin load_model — загрузить embedding-модель\n"
            "/admin vectorize — векторизовать контент и обновить кэш"
        )
        include_req = context.bot_data.get("log_user_messages", False)
        log_operation(
            telegram_id=user.id,
            username=user.username,
            operation="admin_unknown",
            result="ok",
            request_text=f"/admin {subcommand}",
            include_request_text=include_req,
        )
        return

    project_root, cache_file, markdown_file, images_dir, models_dir = _get_project_paths()
    include_req = context.bot_data.get("log_user_messages", False)
    req_text = f"/admin {subcommand}"

    if subcommand == "load_model":
        logger.info(f"Admin load_model from user_id={user.id}, username={user.username}")
        try:
            await update.message.reply_text("Начинаю загрузку модели…")
            ok = download_model(model_name=DEFAULT_MODEL_NAME, models_dir=models_dir)
            if ok:
                await update.message.reply_text("Модель загружена успешно.")
                logger.info(f"Admin load_model completed successfully for user_id={user.id}")
                log_operation(user.id, user.username, "admin_load_model", "ok", req_text, include_req)
            else:
                await update.message.reply_text(
                    "Ошибка при загрузке модели. Проверьте логи и наличие sentence-transformers."
                )
                logger.warning(f"Admin load_model failed for user_id={user.id}")
                log_operation(user.id, user.username, "admin_load_model", "error", req_text, include_req, error="download failed")
        except Exception as e:
            logger.error(f"Admin load_model error: {e}", exc_info=True)
            await update.message.reply_text(f"Ошибка: {e}")
            log_operation(user.id, user.username, "admin_load_model", "error", req_text, include_req, error=str(e))

    elif subcommand == "vectorize":
        logger.info(f"Admin vectorize from user_id={user.id}, username={user.username}")
        try:
            if not markdown_file.exists():
                await update.message.reply_text(
                    f"Файл документа не найден: {markdown_file}. "
                    "Сначала загрузите контент (например, из Google Docs)."
                )
                log_operation(user.id, user.username, "admin_vectorize", "error", req_text, include_req, error="markdown file not found")
                return
            await update.message.reply_text("Векторизация запущена…")
            ok = vectorize_content(
                markdown_file=markdown_file,
                cache_file=cache_file,
                model_name=DEFAULT_MODEL_NAME,
            )
            if not ok:
                await update.message.reply_text(
                    "Ошибка при векторизации. Проверьте наличие модели и логи."
                )
                logger.warning(f"Admin vectorize failed for user_id={user.id}")
                log_operation(user.id, user.username, "admin_vectorize", "error", req_text, include_req, error="vectorize failed")
                return
            # Перезагрузить контекст поиска после успешной векторизации
            embeddings_data = load_embeddings_from_cache(cache_file)
            if embeddings_data:
                init_search_context(
                    index=embeddings_data,
                    markdown_file=markdown_file,
                    images_dir=images_dir,
                )
                await update.message.reply_text("Векторизация завершена. Контекст поиска обновлён.")
            else:
                search_index_data = load_index_from_cache(cache_file)
                if search_index_data:
                    init_search_context(
                        index=search_index_data,
                        markdown_file=markdown_file,
                        images_dir=images_dir,
                    )
                await update.message.reply_text("Векторизация завершена.")
            logger.info(f"Admin vectorize completed successfully for user_id={user.id}")
            log_operation(user.id, user.username, "admin_vectorize", "ok", req_text, include_req)
        except Exception as e:
            logger.error(f"Admin vectorize error: {e}", exc_info=True)
            await update.message.reply_text(f"Ошибка: {e}")
            log_operation(user.id, user.username, "admin_vectorize", "error", req_text, include_req, error=str(e))
