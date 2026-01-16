"""
Модуль поисковой системы для xyliganimbot.

Обеспечивает построение обратного индекса для быстрого поиска
по базе знаний и разбиение документа на разделы.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from html.parser import HTMLParser
from html import unescape

from src.logging import get_logger

logger = get_logger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Парсер HTML для извлечения текста без тегов."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_text = ""

    def handle_data(self, data):
        """Обработка текстовых данных."""
        self.text_parts.append(data)

    def get_text(self) -> str:
        """Возвращает извлеченный текст."""
        return " ".join(self.text_parts)


def load_sections(sections_file: Path) -> List[str]:
    """
    Загружает заголовки разделов из JSON-файла.

    Args:
        sections_file: Путь к файлу с заголовками разделов

    Returns:
        Список заголовков разделов

    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл имеет некорректный формат
    """
    if not sections_file.exists():
        error_msg = f"Sections file not found: {sections_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(sections_file, "r", encoding="utf-8") as f:
            sections = json.load(f)
            if not isinstance(sections, list):
                raise ValueError("Sections file must contain a list of strings")
            logger.info(f"Loaded {len(sections)} section headers from {sections_file}")
            return sections
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in sections file {sections_file}: {e}"
        logger.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e


def normalize_text(text: str) -> str:
    """
    Нормализует текст: приводит к нижнему регистру и удаляет лишние пробелы.

    Args:
        text: Исходный текст

    Returns:
        Нормализованный текст
    """
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем лишние пробелы
    text = " ".join(text.split())
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Разбивает текст на токены (слова).

    Args:
        text: Текст для токенизации

    Returns:
        Список токенов (слов)
    """
    # Нормализуем текст
    normalized = normalize_text(text)
    # Разбиваем на слова (удаляем знаки препинания)
    # Используем регулярное выражение для извлечения слов
    tokens = re.findall(r"\b[а-яёa-z0-9]+\b", normalized, re.IGNORECASE)
    return tokens


def extract_text_from_html(html_content: str) -> str:
    """
    Извлекает текст из HTML, удаляя теги.

    Args:
        html_content: HTML-контент

    Returns:
        Текст без HTML-тегов
    """
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    text = parser.get_text()
    # Декодируем HTML-сущности
    text = unescape(text)
    return text


def find_section_in_html(html_content: str, section_title: str) -> Optional[int]:
    """
    Находит позицию заголовка раздела в HTML.

    Args:
        html_content: HTML-контент
        section_title: Заголовок раздела для поиска

    Returns:
        Позиция начала заголовка в HTML или None, если не найдено
    """
    # Декодируем HTML-сущности для поиска
    decoded_html = unescape(html_content)
    
    # Нормализуем заголовок для поиска
    normalized_title = normalize_text(section_title)
    
    # Пробуем найти точное совпадение (с учетом HTML-тегов)
    escaped_title = re.escape(section_title)
    patterns = [
        rf'<[^>]*>{escaped_title}</[^>]*>',  # В любых тегах
        rf'>{escaped_title}<',  # Между тегами
    ]

    for pattern in patterns:
        match = re.search(pattern, decoded_html, re.IGNORECASE)
        if match:
            return match.start()

    # Если не нашли, ищем по нормализованному тексту
    # Извлекаем текст из HTML и ищем в нем
    html_text = extract_text_from_html(decoded_html)
    normalized_html_text = normalize_text(html_text)
    
    if normalized_title in normalized_html_text:
        # Находим позицию в оригинальном HTML
        # Ищем по словам из заголовка
        title_words = tokenize_text(section_title)
        if not title_words:
            return None
            
        # Ищем первое слово заголовка в HTML
        first_word = title_words[0]
        html_lower = decoded_html.lower()
        
        # Ищем все вхождения первого слова
        pos = 0
        while True:
            pos = html_lower.find(first_word.lower(), pos)
            if pos == -1:
                break
            
            # Проверяем, что это действительно начало заголовка
            # Извлекаем текст вокруг этой позиции
            context_start = max(0, pos - 50)
            context_end = min(len(decoded_html), pos + len(section_title) + 50)
            context = decoded_html[context_start:context_end]
            context_text = extract_text_from_html(context)
            context_normalized = normalize_text(context_text)
            
            # Проверяем, содержит ли контекст заголовок
            if normalized_title in context_normalized:
                return context_start + context.find(section_title[:20]) if section_title[:20] in context else pos
            
            pos += len(first_word)
    
    return None


def parse_html_sections(html_content: str, sections: List[str]) -> Dict[str, str]:
    """
    Разбивает HTML-документ на разделы по заголовкам.

    Args:
        html_content: HTML-контент документа
        sections: Список заголовков разделов

    Returns:
        Словарь: заголовок раздела -> текст раздела
    """
    sections_content = {}
    html_lower = html_content.lower()

    for i, section_title in enumerate(sections):
        # Находим позицию текущего раздела
        current_pos = find_section_in_html(html_content, section_title)

        if current_pos is None:
            logger.warning(f"Section '{section_title}' not found in HTML")
            sections_content[section_title] = ""
            continue

        # Находим позицию следующего раздела
        next_pos = None
        if i + 1 < len(sections):
            next_section_title = sections[i + 1]
            next_pos = find_section_in_html(html_content, next_section_title)

        # Извлекаем текст раздела
        if next_pos is not None and next_pos > current_pos:
            section_html = html_content[current_pos:next_pos]
        else:
            # Последний раздел - берем до конца
            section_html = html_content[current_pos:]

        # Извлекаем текст из HTML раздела
        section_text = extract_text_from_html(section_html)
        sections_content[section_title] = section_text
        logger.debug(f"Extracted section '{section_title}': {len(section_text)} chars")

    return sections_content


def build_search_index(sections_content: Dict[str, str]) -> Dict[str, Any]:
    """
    Строит двухуровневый обратный индекс для поиска.

    Args:
        sections_content: Словарь заголовок раздела -> текст раздела

    Returns:
        Словарь с индексами:
        - section_index: ключевые слова -> список разделов
        - content_index: ключевые слова -> разделы -> позиции в тексте
    """
    section_index: Dict[str, Set[str]] = {}
    content_index: Dict[str, Dict[str, List[int]]] = {}

    for section_title, section_text in sections_content.items():
        # Токенизируем заголовок раздела
        title_tokens = tokenize_text(section_title)
        for token in title_tokens:
            if token not in section_index:
                section_index[token] = set()
            section_index[token].add(section_title)

        # Токенизируем содержимое раздела
        content_tokens = tokenize_text(section_text)
        # Находим позиции токенов в тексте
        normalized_text = normalize_text(section_text)
        words = re.findall(r"\b[а-яёa-z0-9]+\b", normalized_text, re.IGNORECASE)

        for token in content_tokens:
            if token not in content_index:
                content_index[token] = {}

            if section_title not in content_index[token]:
                content_index[token][section_title] = []

            # Находим все позиции токена в тексте
            positions = [i for i, word in enumerate(words) if word == token]
            content_index[token][section_title].extend(positions)

    # Преобразуем set в list для JSON-сериализации
    section_index_serializable = {
        token: list(sections) for token, sections in section_index.items()
    }

    logger.info(
        f"Built search index: {len(section_index_serializable)} tokens in section index, "
        f"{len(content_index)} tokens in content index"
    )

    return {
        "section_index": section_index_serializable,
        "content_index": content_index,
    }


def build_index_from_html(
    html_file: Path, sections_file: Path
) -> Optional[Dict[str, Any]]:
    """
    Строит индекс из HTML-файла и заголовков разделов.

    Args:
        html_file: Путь к HTML-файлу
        sections_file: Путь к файлу с заголовками разделов

    Returns:
        Словарь с индексами или None при ошибке
    """
    try:
        # Загружаем заголовки разделов
        sections = load_sections(sections_file)

        # Загружаем HTML-контент
        if not html_file.exists():
            logger.error(f"HTML file not found: {html_file}")
            return None

        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        logger.info(f"Building search index from {html_file}")

        # Разбиваем на разделы
        sections_content = parse_html_sections(html_content, sections)

        # Строим индекс
        index = build_search_index(sections_content)

        return index

    except FileNotFoundError as e:
        logger.error(f"Error building index: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error building index: {e}", exc_info=True)
        return None


def save_index_to_cache(cache_file: Path, index: Dict[str, Any]) -> None:
    """
    Сохраняет индекс в кэш.

    Args:
        cache_file: Путь к файлу кэша
        index: Словарь с индексами
    """
    try:
        # Загружаем существующий кэш
        cache_data = {}
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

        # Добавляем индекс в кэш
        cache_data["search_index"] = index

        # Сохраняем обновленный кэш
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Search index saved to cache: {cache_file}")

    except Exception as e:
        logger.error(f"Error saving index to cache: {e}", exc_info=True)


def load_index_from_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    """
    Загружает индекс из кэша.

    Args:
        cache_file: Путь к файлу кэша

    Returns:
        Словарь с индексами или None, если не найден
    """
    try:
        if not cache_file.exists():
            logger.debug(f"Cache file not found: {cache_file}")
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        index = cache_data.get("search_index")
        if index:
            logger.info(f"Search index loaded from cache: {cache_file}")
            return index
        else:
            logger.debug("No search index found in cache")
            return None

    except Exception as e:
        logger.warning(f"Error loading index from cache: {e}")
        return None
