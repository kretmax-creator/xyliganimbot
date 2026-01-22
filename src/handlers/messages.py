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
html_file_path: Optional[Path] = None
sections_file_path: Optional[Path] = None
images_dir_path: Optional[Path] = None


def escape_markdown(text: str) -> str:
    """
    Экранирует специальные символы Markdown для безопасной отправки в Telegram.

    Args:
        text: Текст для экранирования

    Returns:
        Экранированный текст
    """
    # Символы, которые нужно экранировать в Markdown
    special_chars = ['*', '_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    return text


def init_search_context(
    index: Dict[str, Any],
    html_file: Path,
    sections_file: Path,
    images_dir: Path,
) -> None:
    """
    Инициализирует контекст поиска.

    Args:
        index: Поисковый индекс
        html_file: Путь к HTML-файлу с документом
        sections_file: Путь к файлу с заголовками разделов
        images_dir: Путь к директории с изображениями
    """
    global search_index, html_file_path, sections_file_path, images_dir_path
    search_index = index
    html_file_path = html_file
    sections_file_path = sections_file
    images_dir_path = images_dir
    logger.info("Search context initialized")


def format_search_results(results: List[Dict[str, Any]], max_text_length: int = 1000) -> str:
    """
    Форматирует результаты поиска в текстовое сообщение.

    Args:
        results: Список результатов поиска
        max_text_length: Максимальная длина текста раздела в ответе

    Returns:
        Отформатированное текстовое сообщение
    """
    if not results:
        return "❌ По вашему запросу ничего не найдено.\n\nПопробуйте изменить формулировку запроса."

    message_parts = [f"🔍 Найдено разделов: {len(results)}\n"]

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

        # Экранируем заголовок и текст для безопасного Markdown
        escaped_title = escape_markdown(section_title)
        
        # Форматируем score для отображения
        if isinstance(score, float):
            score_text = f" (релевантность: {score:.1%})"
        else:
            score_text = f" (релевантность: {score})"
        
        message_parts.append(f"\n📌 *{escaped_title}*{score_text}{confidence_warning}")
        
        if text:
            # Экранируем текст, чтобы избежать ошибок парсинга Markdown
            escaped_text = escape_markdown(text)
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
        max_images: Максимальное количество изображений для отправки
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
                    await update.message.reply_text(part, parse_mode="Markdown")
                else:
                    # Последующие сообщения - обычные
                    await update.message.chat.send_message(part, parse_mode="Markdown")
                logger.info(f"Search response part {i+1}/{len(parts)} sent ({len(part)} chars)")
        else:
            # Отправляем одним сообщением
            await update.message.reply_text(text_response, parse_mode="Markdown")
            logger.info(f"Search response sent: {len(results)} results ({len(text_response)} chars)")

        # Отправляем изображения, если они есть
        if images_dir_path and images_dir_path.exists():
            image_files = list(images_dir_path.glob("*.png")) + list(
                images_dir_path.glob("*.jpg")
            ) + list(images_dir_path.glob("*.jpeg"))

            if image_files:
                # Отправляем ограниченное количество изображений
                images_to_send = image_files[:max_images]
                logger.info(f"Sending {len(images_to_send)} images")

                for image_path in images_to_send:
                    try:
                        with open(image_path, "rb") as photo:
                            await update.message.reply_photo(photo=photo)
                    except Exception as e:
                        logger.warning(f"Error sending image {image_path}: {e}")

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
        return

    logger.info(
        f"Search query from user_id={user.id}, username={user.username}, "
        f"query='{query[:50]}...'"
    )

    # Проверяем наличие необходимых данных
    from pathlib import Path
    
    # Проверка наличия документа
    if not html_file_path or not html_file_path.exists():
        logger.warning("HTML file not found")
        try:
            await message.reply_text(
                "Документ базы знаний не найден. "
                "Обратитесь к администратору для загрузки контента."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
        return
    
    # Проверка наличия файла с разделами
    if not sections_file_path or not sections_file_path.exists():
        logger.warning("Sections file not found")
        try:
            await message.reply_text(
                "Файл с разделами не найден. "
                "Обратитесь к администратору."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
        return
    
    # Проверяем, инициализирован ли поисковый индекс
    if not search_index:
        logger.error("Search index not initialized")
        try:
            await message.reply_text(
                "Поисковая система не инициализирована. "
                "Обратитесь к администратору."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
        return
    
    # Проверка наличия embeddings для семантического поиска
    has_embeddings = isinstance(search_index, dict) and "embeddings" in search_index
    if has_embeddings:
        # Проверяем наличие модели для семантического поиска
        from src.search import load_embedding_model
        model = load_embedding_model()
        if model is None:
            logger.warning("Embedding model not available, falling back to token-based search")
            # Можно продолжить с token-based поиском, если есть token-based индекс
            if "section_index" not in search_index and "content_index" not in search_index:
                try:
                    await message.reply_text(
                        "Модель для семантического поиска не найдена. "
                        "Обратитесь к администратору для загрузки модели."
                    )
                except Exception as e:
                    logger.error(f"Error sending error message: {e}")
                return

    # Выполняем поиск
    try:
        results = search(
            query=query,
            index=search_index,
            html_file=html_file_path,
            sections_file=sections_file_path,
            limit=5,
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
