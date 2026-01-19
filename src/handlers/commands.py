"""
Обработчики команд для xyliganimbot.

Обеспечивает обработку команд бота, таких как /help.
"""

from telegram import Update
from telegram.ext import ContextTypes

from src.logging import get_logger

logger = get_logger(__name__)


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
        "Просто отправьте текстовый запрос, и бот найдет релевантные разделы "
        "в базе знаний.\n\n"
        "🔍 *Команды:*\n"
        "/help — показать это сообщение\n\n"
        "💡 *Примеры запросов:*\n"
        "• Как настроить?\n"
        "• Где найти информацию о...\n"
        "• Что делать если..."
    )

    try:
        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Help message sent to user_id={user.id}")
    except Exception as e:
        logger.error(f"Error sending help message: {e}", exc_info=True)
