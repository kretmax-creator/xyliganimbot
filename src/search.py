"""
Модуль поисковой системы для xyliganimbot.

Обеспечивает построение обратного индекса для быстрого поиска
по базе знаний и разбиение документа на разделы.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from html.parser import HTMLParser
from html import unescape
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    util = None
    np = None


from src.logging import get_logger

logger = get_logger(__name__)

# Глобальная переменная для кэширования embedding-модели
_embedding_model: Optional[SentenceTransformer] = None


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


def extract_images_from_html_section(html_section: str) -> List[str]:
    """
    Извлекает пути к изображениям из HTML-фрагмента раздела.

    Args:
        html_section: HTML-фрагмент раздела

    Returns:
        Список относительных путей к изображениям (например, ["images/file1.png"])
    """
    images = []
    # Ищем все теги <img> с атрибутом src
    # Паттерн для поиска src в тегах img (поддерживает одинарные и двойные кавычки)
    pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    matches = re.findall(pattern, html_section, re.IGNORECASE)
    
    for src in matches:
        # Обрабатываем только локальные пути, которые начинаются с "images/"
        # Это уже обновленные пути из update_image_paths_in_html()
        if src.startswith('images/') or src.startswith('./images/'):
            # Нормализуем путь: убираем ./ если есть
            normalized_path = src.replace('./', '')
            if normalized_path not in images:
                images.append(normalized_path)
                logger.debug(f"Found image in section: {normalized_path}")
    
    return images


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


def parse_html_sections(html_content: str, sections: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Разбивает HTML-документ на разделы по заголовкам и извлекает изображения для каждого раздела.

    Args:
        html_content: HTML-контент документа
        sections: Список заголовков разделов

    Returns:
        Кортеж (sections_content, sections_images):
        - sections_content: словарь заголовок раздела -> текст раздела
        - sections_images: словарь заголовок раздела -> список путей к изображениям
    """
    sections_content = {}
    sections_images = {}
    html_lower = html_content.lower()

    for i, section_title in enumerate(sections):
        # Находим позицию текущего раздела
        current_pos = find_section_in_html(html_content, section_title)

        if current_pos is None:
            logger.warning(f"Section '{section_title}' not found in HTML")
            sections_content[section_title] = ""
            sections_images[section_title] = []
            continue

        # Находим позицию следующего раздела
        next_pos = None
        if i + 1 < len(sections):
            next_section_title = sections[i + 1]
            next_pos = find_section_in_html(html_content, next_section_title)

        # Извлекаем HTML раздела
        if next_pos is not None and next_pos > current_pos:
            section_html = html_content[current_pos:next_pos]
        else:
            # Последний раздел - берем до конца
            section_html = html_content[current_pos:]

        # Извлекаем текст из HTML раздела
        section_text = extract_text_from_html(section_html)
        sections_content[section_title] = section_text
        
        # Извлекаем изображения из HTML раздела
        section_image_paths = extract_images_from_html_section(section_html)
        sections_images[section_title] = section_image_paths
        
        logger.debug(
            f"Extracted section '{section_title}': {len(section_text)} chars, "
            f"{len(section_image_paths)} images"
        )

    return sections_content, sections_images


def load_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Optional[SentenceTransformer]:
    """
    Загружает embedding-модель ТОЛЬКО из локальной папки models/.
    
    Модель должна быть загружена заранее через модуль model_loader или команду /admin load_model.
    Автозагрузка из HuggingFace не выполняется.
    
    Args:
        model_name: Имя модели (по умолчанию paraphrase-multilingual-MiniLM-L12-v2)
    
    Returns:
        Загруженная модель или None, если не найдена локально
    """
    global _embedding_model
    
    if not EMBEDDINGS_AVAILABLE:
        logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
        return None
    
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        # ТОЛЬКО локальная загрузка из папки models/
        models_dir = Path("models") / model_name
        if models_dir.exists():
            logger.info(f"Loading embedding model from local path: {models_dir}")
            _embedding_model = SentenceTransformer(str(models_dir))
            logger.info(f"Embedding model loaded successfully: {model_name}")
            return _embedding_model
        else:
            logger.error(
                f"Model not found locally: {models_dir}. "
                f"Use /admin load_model or src.model_loader.download_model() to download it first."
            )
            return None
    
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}", exc_info=True)
        return None


def vectorize_sections(
    sections_content: Dict[str, str]
) -> Optional[Tuple[Any, List[str]]]:
    """
    Векторизует разделы документа через embedding-модель.
    
    Args:
        sections_content: Словарь заголовок раздела -> текст раздела
    
    Returns:
        Кортеж (embeddings_array, section_titles_list) или None при ошибке
        embeddings_array: numpy array с векторами разделов
        section_titles_list: список заголовков разделов в том же порядке
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.error("Embeddings not available. Cannot vectorize sections.")
        return None
    
    model = load_embedding_model()
    if model is None:
        logger.error("Failed to load embedding model. Cannot vectorize sections.")
        return None
    
    try:
        # Формируем тексты для векторизации: "Заголовок. Текст раздела"
        texts_to_encode = []
        section_titles = []
        
        for section_title, section_text in sections_content.items():
            # Комбинируем заголовок и текст для лучшего понимания контекста
            combined_text = f"{section_title}. {section_text}".strip()
            texts_to_encode.append(combined_text)
            section_titles.append(section_title)
        
        if not texts_to_encode:
            logger.warning("No sections to vectorize")
            return None
        
        logger.info(f"Vectorizing {len(texts_to_encode)} sections...")
        
        # Векторизуем все разделы одним батчем
        embeddings = model.encode(texts_to_encode, show_progress_bar=False)
        
        # Преобразуем в numpy array, если еще не массив
        if np is not None:
            embeddings_array = np.array(embeddings)
        else:
            embeddings_array = embeddings
        
        logger.info(
            f"Vectorization completed: {len(section_titles)} sections, "
            f"embedding dimension: {embeddings_array.shape[1]}"
        )
        
        return embeddings_array, section_titles
    
    except Exception as e:
        logger.error(f"Error vectorizing sections: {e}", exc_info=True)
        return None


def build_embeddings_from_html(
    html_file: Path, sections_file: Path
) -> Optional[Dict[str, Any]]:
    """
    Строит embeddings из HTML-файла и заголовков разделов.
    Заменяет build_index_from_html для семантического поиска.
    
    Args:
        html_file: Путь к HTML-файлу
        sections_file: Путь к файлу с заголовками разделов
    
    Returns:
        Словарь с embeddings данными или None при ошибке:
        - embeddings: список списков (для JSON-сериализации)
        - section_titles: список заголовков разделов
        - sections_content: словарь заголовок -> текст
        - sections_images: словарь заголовок -> список путей к изображениям
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
        
        logger.info(f"Building embeddings from {html_file}")
        
        # Разбиваем на разделы
        sections_content, sections_images = parse_html_sections(html_content, sections)
        
        # Векторизуем разделы
        vectorization_result = vectorize_sections(sections_content)
        
        if vectorization_result is None:
            logger.error("Failed to vectorize sections. Falling back to token-based search.")
            return None
        
        embeddings_array, section_titles = vectorization_result
        
        # Преобразуем numpy array в список списков для JSON-сериализации
        embeddings_list = embeddings_array.tolist()
        
        return {
            "embeddings": embeddings_list,
            "section_titles": section_titles,
            "sections_content": sections_content,
            "sections_images": sections_images,
        }
    
    except FileNotFoundError as e:
        logger.error(f"Error building embeddings: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error building embeddings: {e}", exc_info=True)
        return None


def save_embeddings_to_cache(cache_file: Path, embeddings_data: Dict[str, Any]) -> None:
    """
    Сохраняет embeddings в кэш.
    
    Args:
        cache_file: Путь к файлу кэша
        embeddings_data: Словарь с embeddings данными
    """
    try:
        # Загружаем существующий кэш
        cache_data = {}
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        
        # Добавляем embeddings в кэш
        cache_data["embeddings"] = embeddings_data
        
        # Сохраняем обновленный кэш
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embeddings saved to cache: {cache_file}")
    
    except Exception as e:
        logger.error(f"Error saving embeddings to cache: {e}", exc_info=True)


def vectorize_content(
    html_file: Path,
    sections_file: Path,
    cache_file: Path,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> bool:
    """
    Векторизует загруженный контент и сохраняет embeddings в кэш.
    
    Это независимый процесс, который работает с локальными файлами.
    Требует наличия модели в папке models/ (загруженной через model_loader).
    
    Args:
        html_file: Путь к HTML-файлу (data/knowledge.html)
        sections_file: Путь к файлу с заголовками разделов (data/sections.json)
        cache_file: Путь к файлу кэша (data/knowledge_cache.json)
        model_name: Имя модели для векторизации
    
    Returns:
        True если успешно, False при ошибке
    """
    try:
        # Проверяем наличие необходимых файлов
        if not html_file.exists():
            logger.error(f"HTML file not found: {html_file}")
            return False
        
        if not sections_file.exists():
            logger.error(f"Sections file not found: {sections_file}")
            return False
        
        logger.info(f"Starting vectorization of content from {html_file}")
        
        # Строим embeddings из HTML-файла
        embeddings_data = build_embeddings_from_html(html_file, sections_file)
        
        if embeddings_data is None:
            logger.error("Failed to build embeddings. Check if model is loaded and files are valid.")
            return False
        
        # Сохраняем embeddings в кэш
        save_embeddings_to_cache(cache_file, embeddings_data)
        
        # Обновляем section_images в основном кэше, если нужно
        try:
            from src.google_docs import load_cache, save_cache
            
            cache_data = load_cache(cache_file)
            if cache_data is None:
                cache_data = {}
            
            # Добавляем section_images из embeddings_data
            sections_images = embeddings_data.get('sections_images', {})
            if sections_images:
                section_images_serializable = {
                    section_title: image_paths
                    for section_title, image_paths in sections_images.items()
                }
                cache_data['section_images'] = section_images_serializable
                logger.info(
                    f"Updated section_images: {len(sections_images)} sections, "
                    f"total {sum(len(imgs) for imgs in sections_images.values())} images"
                )
            
            # Сохраняем обновленный кэш
            save_cache(cache_file, cache_data)
        except Exception as e:
            logger.warning(f"Could not update section_images in cache: {e}")
        
        logger.info("Content vectorization completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error vectorizing content: {e}", exc_info=True)
        return False


def load_embeddings_from_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    """
    Загружает embeddings из кэша.
    
    Args:
        cache_file: Путь к файлу кэша
    
    Returns:
        Словарь с embeddings данными или None, если не найден
        embeddings_data содержит:
        - embeddings: список списков (конвертируется обратно в numpy array)
        - section_titles: список заголовков разделов
        - sections_content: словарь заголовок -> текст
        - sections_images: словарь заголовок -> список путей к изображениям
    """
    try:
        if not cache_file.exists():
            logger.debug(f"Cache file not found: {cache_file}")
            return None
        
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        embeddings_data = cache_data.get("embeddings")
        if embeddings_data:
            # Конвертируем список списков обратно в numpy array
            if np is not None and "embeddings" in embeddings_data:
                embeddings_list = embeddings_data["embeddings"]
                embeddings_data["embeddings"] = np.array(embeddings_list)
            
            logger.info(f"Embeddings loaded from cache: {cache_file}")
            return embeddings_data
        else:
            logger.debug("No embeddings found in cache")
            return None
    
    except Exception as e:
        logger.warning(f"Error loading embeddings from cache: {e}")
        return None


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
        sections_content, sections_images = parse_html_sections(html_content, sections)

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


def get_section_text(
    section_title: str, html_file: Path, sections_file: Path
) -> Optional[str]:
    """
    Извлекает текст раздела из HTML-файла.

    Args:
        section_title: Заголовок раздела
        html_file: Путь к HTML-файлу
        sections_file: Путь к файлу с заголовками разделов

    Returns:
        Текст раздела или None, если не найден
    """
    try:
        if not html_file.exists():
            logger.warning(f"HTML file not found: {html_file}")
            return None

        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        sections = load_sections(sections_file)
        sections_content, _ = parse_html_sections(html_content, sections)

        return sections_content.get(section_title)

    except Exception as e:
        logger.warning(f"Error extracting section text: {e}")
        return None


def get_text_snippet(text: str, max_length: int = 300) -> str:
    """
    Возвращает snippet текста или полный текст для небольших разделов.

    Args:
        text: Текст раздела
        max_length: Максимальная длина snippet (по умолчанию 300 символов)

    Returns:
        Snippet или полный текст, если раздел небольшой (< 500 символов)
    """
    if not text:
        return ""
    
    # Если раздел небольшой (< 500 символов), возвращаем полный текст
    if len(text) < 500:
        return text
    
    # Для больших разделов возвращаем snippet
    if len(text) > max_length:
        # Пытаемся обрезать по границе слова
        snippet = text[:max_length]
        last_space = snippet.rfind(' ')
        if last_space > max_length * 0.8:  # Если пробел не слишком далеко от конца
            snippet = snippet[:last_space]
        return snippet + "..."
    
    return text


def semantic_search(
    query: str,
    embeddings_data: Dict[str, Any],
    model: Optional[SentenceTransformer] = None,
    limit: int = 5,
    min_score: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Выполняет семантический поиск через косинусное сходство векторов.

    Args:
        query: Поисковый запрос
        embeddings_data: Словарь с embeddings данными:
            - embeddings: список списков (векторы разделов)
            - section_titles: список заголовков разделов
            - sections_content: словарь заголовок -> текст
        model: Embedding-модель (если None, загружается)
        limit: Максимальное количество результатов
        min_score: Минимальный score для включения результата (0-1)

    Returns:
        Список словарей с результатами:
        - section_title: название раздела
        - score: косинусное сходство (0-1)
        - text: текст раздела (snippet или полный)
    """
    if not EMBEDDINGS_AVAILABLE or util is None:
        logger.error("sentence-transformers not available for semantic search")
        return []
    
    if not query or not query.strip():
        logger.debug("Empty query provided for semantic search")
        return []
    
    try:
        # Загружаем модель, если не предоставлена
        if model is None:
            model = load_embedding_model()
            if model is None:
                logger.error("Failed to load embedding model for semantic search")
                return []
        
        # Получаем данные из embeddings_data
        embeddings_list = embeddings_data.get("embeddings", [])
        section_titles = embeddings_data.get("section_titles", [])
        sections_content = embeddings_data.get("sections_content", {})
        
        # Проверяем наличие данных
        # embeddings_list может быть списком или numpy array
        if embeddings_list is None:
            logger.warning("Embeddings is None")
            return []
        
        try:
            embeddings_len = len(embeddings_list)
            if embeddings_len == 0:
                logger.warning("Empty embeddings")
                return []
        except (TypeError, ValueError):
            logger.warning("Cannot determine embeddings length")
            return []
        
        if not section_titles or len(section_titles) == 0:
            logger.warning("Empty section titles")
            return []
        
        # Преобразуем список списков в numpy array (если еще не массив)
        if np is not None:
            if isinstance(embeddings_list, np.ndarray):
                embeddings_array = embeddings_list
            else:
                embeddings_array = np.array(embeddings_list)
            # Приводим к float32 для совместимости с PyTorch
            if embeddings_array.dtype != np.float32:
                embeddings_array = embeddings_array.astype(np.float32)
        else:
            embeddings_array = embeddings_list
        
        # Векторизуем запрос
        logger.info(f"Vectorizing query: '{query[:50]}...'")
        query_embedding = model.encode(query, show_progress_bar=False)
        
        # Приводим query_embedding к numpy array и float32, если нужно
        if np is not None:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
        
        # Выполняем семантический поиск через util.semantic_search()
        # Возвращает список списков: [[{corpus_id, score}, ...], ...]
        # Для одного запроса возвращается [[{corpus_id, score}, ...]]
        hits = util.semantic_search(
            query_embedding,
            embeddings_array,
            top_k=limit * 2  # Берем больше, чтобы отфильтровать по min_score
        )
        
        if not hits or len(hits) == 0 or not hits[0]:
            logger.info("No results found in semantic search")
            return []
        
        # Формируем результаты
        # hits[0] содержит результаты для первого (и единственного) запроса
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            score = float(hit['score'])
            
            # Фильтруем по минимальному score
            if score < min_score:
                continue
            
            # Получаем заголовок раздела
            if idx >= len(section_titles):
                logger.warning(f"Index {idx} out of range for section_titles")
                continue
            
            section_title = section_titles[idx]
            
            # Получаем текст раздела
            section_text = sections_content.get(section_title, "")
            
            # Формируем snippet или полный текст
            text = get_text_snippet(section_text)
            
            results.append({
                "section_title": section_title,
                "score": score,
                "text": text,
            })
            
            # Ограничиваем количество результатов
            if len(results) >= limit:
                break
        
        scores_str = ", ".join([f"{r['score']:.3f}" for r in results[:3]])
        logger.info(
            f"Semantic search completed: {len(results)} results "
            f"(scores: [{scores_str}])"
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        return []


def search(
    query: str,
    index: Dict[str, Any],
    html_file: Optional[Path] = None,
    sections_file: Optional[Path] = None,
    sections_content: Optional[Dict[str, str]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Выполняет поиск по индексу и возвращает ранжированные результаты.
    
    Поддерживает два типа поиска:
    1. Семантический поиск (если index содержит embeddings)
    2. Token-based поиск (если index содержит section_index/content_index)

    Args:
        query: Поисковый запрос
        index: Словарь с индексами или embeddings данными
        html_file: Опционально - путь к HTML-файлу для извлечения текста разделов
        sections_file: Опционально - путь к файлу с заголовками разделов
        sections_content: Опционально - уже распарсенные разделы (словарь заголовок -> текст)
        limit: Максимальное количество результатов (по умолчанию 5)

    Returns:
        Список словарей с результатами:
        - section_title: название раздела
        - score: оценка релевантности (0-1 для semantic, relevance_score для token-based)
        - text: текст раздела (snippet или полный)
    """
    if not query or not query.strip():
        logger.debug("Empty query provided")
        return []

    if not index:
        logger.warning("Index is empty or None")
        return []

    # Определяем тип индекса: embeddings или token-based
    has_embeddings = "embeddings" in index and "section_titles" in index
    has_token_index = "section_index" in index or "content_index" in index

    # Если есть embeddings, используем семантический поиск
    if has_embeddings:
        logger.info("Using semantic search (embeddings found)")
        results = semantic_search(
            query=query,
            embeddings_data=index,
            limit=limit,
            min_score=0.3
        )
        return results

    # Иначе используем token-based поиск
    if not has_token_index:
        logger.warning("Index type not recognized (neither embeddings nor token-based)")
        return []

    logger.info("Using token-based search")
    section_index = index.get("section_index", {})
    content_index = index.get("content_index", {})

    if not section_index and not content_index:
        logger.warning("Both section_index and content_index are empty")
        return []

    # Токенизируем запрос
    query_tokens = tokenize_text(query)
    if not query_tokens:
        logger.debug(f"No tokens found in query: {query}")
        return []

    logger.info(f"Searching for query: '{query}' ({len(query_tokens)} tokens)")

    # Словарь для подсчета релевантности разделов
    # Ключ: название раздела, значение: (section_matches, content_matches)
    section_scores: Dict[str, tuple[int, int]] = {}

    # Поиск по индексу разделов (вес: 2)
    for token in query_tokens:
        if token in section_index:
            sections = section_index[token]
            for section_title in sections:
                if section_title not in section_scores:
                    section_scores[section_title] = (0, 0)
                section_matches, content_matches = section_scores[section_title]
                section_scores[section_title] = (section_matches + 1, content_matches)

    # Поиск по индексу содержимого (вес: 1)
    for token in query_tokens:
        if token in content_index:
            token_sections = content_index[token]
            for section_title in token_sections:
                if section_title not in section_scores:
                    section_scores[section_title] = (0, 0)
                section_matches, content_matches = section_scores[section_title]
                # Подсчитываем количество вхождений токена в разделе
                positions = token_sections[section_title]
                section_scores[section_title] = (
                    section_matches,
                    content_matches + len(positions),
                )

    # Вычисляем общую релевантность и формируем результаты
    results = []
    for section_title, (section_matches, content_matches) in section_scores.items():
        # Релевантность: совпадения в заголовке * 5 + совпадения в содержимом
        # Заголовки имеют больший вес, так как они более релевантны для поиска
        relevance_score = section_matches * 5 + content_matches
        total_matches = section_matches + content_matches

        # Извлекаем текст раздела
        section_text = None
        if sections_content and section_title in sections_content:
            # Используем уже распарсенные разделы (быстрее)
            section_text = sections_content[section_title]
        elif html_file and sections_file:
            # Парсим HTML только если не предоставлены готовые разделы
            section_text = get_section_text(section_title, html_file, sections_file)

        # Формируем snippet или полный текст
        text = get_text_snippet(section_text) if section_text else ""

        # Нормализуем score для совместимости с semantic search (0-1)
        # Используем простую нормализацию: делим на максимально возможный score
        # Максимальный score = количество токенов в запросе * 5 (все в заголовках)
        max_possible_score = len(query_tokens) * 5 if query_tokens else 1
        normalized_score = min(relevance_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

        result = {
            "section_title": section_title,
            "score": normalized_score,  # Используем score для совместимости
            "relevance_score": relevance_score,  # Сохраняем для обратной совместимости
            "text": text,
        }

        results.append(result)

    # Сортируем по убыванию релевантности
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Ограничиваем количество результатов
    limited_results = results[:limit]

    scores_str = ", ".join([f"{r['score']:.3f}" for r in limited_results[:3]])
    logger.info(
        f"Token-based search completed: found {len(results)} sections, "
        f"returning top {len(limited_results)} "
        f"(scores: [{scores_str}])"
    )

    return limited_results
