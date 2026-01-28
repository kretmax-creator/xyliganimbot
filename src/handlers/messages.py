"""
Обработчики текстовых сообщений для xyliganimbot.

Обеспечивает обработку текстовых запросов как поисковых запросов
к базе знаний и отправку результатов пользователю.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from telegram import Update
from telegram.ext import ContextTypes

from src.search import search
from src.logging import get_logger

logger = get_logger(__name__)

# Глобальные переменные для хранения путей и индекса
# Инициализируются при старте бота
search_index: Optional[Dict[str, Any]] = None
markdown_file_path: Optional[Path] = None
images_dir_path: Optional[Path] = None


def escape_html(text: str) -> str:
    """
    Экранирует специальные символы HTML для безопасной отправки в Telegram.

    Args:
        text: Текст для экранирования

    Returns:
        Экранированный текст
    """
    # Экранируем основные HTML-символы
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def init_search_context(
    index: Dict[str, Any],
    markdown_file: Path,
    images_dir: Path,
) -> None:
    """
    Инициализирует контекст поиска.

    Args:
        index: Поисковый индекс
        markdown_file: Путь к Markdown-файлу с документом (или HTML для обратной совместимости)
        images_dir: Путь к директории с изображениями
    """
    global search_index, markdown_file_path, images_dir_path
    
    search_index = index
    markdown_file_path = markdown_file
    images_dir_path = images_dir
    
    logger.info(f"Search context initialized:")
    logger.info(f"  Markdown/HTML file: {markdown_file}")
    logger.info(f"  Images dir: {images_dir}")
    logger.info(f"  Index type: {'embeddings' if 'embeddings' in index else 'token-based'}")


def format_search_results(results: List[Dict[str, Any]], max_text_length: int = 1000) -> str:
    """
    Форматирует результаты поиска в текстовое сообщение с HTML-разметкой.

    Args:
        results: Список результатов поиска
        max_text_length: Максимальная длина текста раздела в ответе

    Returns:
        Отформатированное текстовое сообщение в HTML
    """
    if not results:
        return "❌ По вашему запросу ничего не найдено.\n\nПопробуйте изменить формулировку запроса."

    message_parts = [f"🔍 Найдено разделов: {len(results)}"]

    for i, result in enumerate(results, 1):
        section_title = result.get("section_title", "Без названия")
        text = result.get("text", "")
        # Используем score (для semantic search) или relevance_score (для token-based)
        score = result.get("score", result.get("relevance_score", 0))
        
        # Если score < 0.3, добавляем предупреждение о низкой уверенности
        confidence_warning = ""
        if isinstance(score, float) and score < 0.3:
            confidence_warning = " ⚠️ (низкая уверенность)"

        # Обрезаем текст, если он слишком длинный
        if text and len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        # Экранируем заголовок и текст для безопасного HTML
        escaped_title = escape_html(section_title)
        
        # Форматируем score для отображения
        if isinstance(score, float):
            score_text = f" (релевантность: {score:.1%})"
        else:
            score_text = f" (релевантность: {score})"
        
        message_parts.append(f"\n📌 <b>{escaped_title}</b>{escape_html(score_text)}{confidence_warning}")
        
        if text:
            # Экранируем текст для HTML и убираем лишние пустые строки
            escaped_text = escape_html(text.strip())
            # Заменяем множественные переносы строк на одинарные
            escaped_text = "\n".join(line.strip() for line in escaped_text.split("\n") if line.strip())
            message_parts.append(f"\n{escaped_text}")
        else:
            message_parts.append("\n(Текст раздела недоступен)")

        if i < len(results):
            message_parts.append("\n" + "─" * 30)

    return "\n".join(message_parts)


async def send_search_response(
    update: Update,
    results: List[Dict[str, Any]],
    max_images: int = 3,
) -> None:
    """
    Отправляет результаты поиска пользователю.

    Args:
        update: Обновление от Telegram
        results: Список результатов поиска
        max_images: Максимальное количество изображений для отправки (отключено)
    """
    if not update.message:
        logger.warning("Cannot send search response: no message in update")
        return

    # Форматируем текстовый ответ
    text_response = format_search_results(results)

    # Telegram ограничение: 4096 символов на сообщение
    MAX_MESSAGE_LENGTH = 4096

    try:
        # Если сообщение слишком длинное, разбиваем на части
        if len(text_response) > MAX_MESSAGE_LENGTH:
            # Разбиваем по разделам (по разделителю)
            separator = "\n" + "─" * 30
            sections = text_response.split(separator)
            
            parts = []
            current_part = sections[0] if sections else ""
            
            for i, section in enumerate(sections[1:], 1):
                section_with_separator = separator + section
                
                # Если добавление следующего раздела превысит лимит
                if len(current_part) + len(section_with_separator) > MAX_MESSAGE_LENGTH - 100:
                    # Сохраняем текущую часть
                    if current_part:
                        parts.append(current_part)
                    current_part = section_with_separator.lstrip(separator)
                else:
                    current_part += section_with_separator
            
            if current_part:
                parts.append(current_part)
            
            # Отправляем каждую часть отдельным сообщением
            for i, part in enumerate(parts):
                if i == 0:
                    # Первое сообщение - как ответ
                    await update.message.reply_text(part, parse_mode="HTML")
                else:
                    # Последующие сообщения - обычные
                    await update.message.chat.send_message(part, parse_mode="HTML")
                logger.info(f"Search response part {i+1}/{len(parts)} sent ({len(part)} chars)")
        else:
            # Отправляем одним сообщением
            await update.message.reply_text(text_response, parse_mode="HTML")
            logger.info(f"Search response sent: {len(results)} results ({len(text_response)} chars)")

        # Отправка изображений отключена по запросу пользователя

    except Exception as e:
        logger.error(f"Error sending search response: {e}", exc_info=True)
        try:
            # Fallback: отправляем простое сообщение об ошибке
            await update.message.reply_text(
                "Произошла ошибка при обработке запроса. Попробуйте позже."
            )
        except Exception as e2:
            logger.error(f"Error sending error message: {e2}", exc_info=True)


async def handle_search_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик текстовых сообщений как поисковых запросов.

    Выполняет поиск по базе знаний и отправляет результаты пользователю.

    Args:
        update: Обновление от Telegram
        context: Контекст обработчика
    """
    user = update.effective_user
    chat = update.effective_chat
    message = update.message

    if not user or not chat or not message:
        logger.warning("Received message without user, chat or message")
        return

    query = message.text
    if not query or not query.strip():
        logger.debug("Empty query received")
        try:
            await message.reply_text("Пожалуйста, введите поисковый запрос.")
        except Exception as e:
            logger.error(f"Error sending empty-query message: {e}")
        return

    logger.info(
        f"Search query from user_id={user.id}, username={user.username}, "
        f"query='{query[:50]}...'"
    )

    # Проверяем наличие необходимых данных
    from pathlib import Path
    
    # Проверка наличия документа (Markdown или HTML для обратной совместимости)
    if not markdown_file_path or not markdown_file_path.exists():
        logger.warning("Markdown/HTML file not found")
        try:
            await message.reply_text(
                "Документ базы знаний не найден. "
                "Обратитесь к администратору для загрузки контента."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
        return
    
    # Проверяем наличие поискового индекса и embeddings
    has_embeddings = (
        isinstance(search_index, dict)
        and search_index.get("embeddings") is not None
        and len(search_index.get("embeddings", [])) > 0
    )
    has_token_index = isinstance(search_index, dict) and (
        bool(search_index.get("section_index")) or bool(search_index.get("content_index"))
    )

    if not search_index or (not has_embeddings and not has_token_index):
        logger.warning("Search index missing or embeddings not loaded")
        try:
            await message.reply_text(
                "⚠️ База знаний не индексирована. "
                "Обратитесь к администратору для выполнения команды /admin vectorize."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
        return

    # Для семантического поиска проверяем наличие модели
    if has_embeddings:
        from src.search import load_embedding_model
        model = load_embedding_model()
        if model is None:
            logger.warning("Embedding model not available")
            try:
                await message.reply_text(
                    "⚠️ Модель поиска не загружена. "
                    "Обратитесь к администратору для выполнения команды /admin load_model."
                )
            except Exception as e:
                logger.error(f"Error sending error message: {e}")
            return

    # Выполняем поиск
    try:
        results = search(
            query=query,
            index=search_index,
            markdown_file=markdown_file_path,
            limit=5,  # Ограничиваем до 5 результатов по требованию
        )

        # Логируем результаты с score
        if results:
            scores = [r.get("score", r.get("relevance_score", 0)) for r in results]
            scores_str = ", ".join([f"{s:.3f}" if isinstance(s, float) else str(s) for s in scores[:3]])
            logger.info(
                f"Search completed: {len(results)} results found "
                f"(scores: [{scores_str}])"
            )
        else:
            logger.info("Search completed: no results found")

        # Отправляем результаты
        await send_search_response(update, results)

    except Exception as e:
        logger.error(f"Error processing search query: {e}", exc_info=True)
        try:
            await message.reply_text(
                "Произошла ошибка при поиске. Попробуйте позже."
            )
        except Exception as e2:
            logger.error(f"Error sending error message: {e2}", exc_info=True)
