"""
Модуль работы с Google Docs для xyliganimbot.

Обеспечивает импорт документа из Google Docs в формате ZIP-архива
с HTML и изображениями, извлечение и сохранение локально.
"""

import re
import json
import zipfile
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO

from src.logging import get_logger

logger = get_logger(__name__)


def load_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    """
    Загружает кэш метаданных документа из JSON-файла.

    Args:
        cache_file: Путь к файлу кэша

    Returns:
        Словарь с метаданными или None, если файл не существует
    """
    if not cache_file.exists():
        logger.debug(f"Cache file not found: {cache_file}")
        return None

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            logger.info(f"Cache loaded from {cache_file}")
            return cache
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in cache file {cache_file}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading cache from {cache_file}: {e}")
        return None


def save_cache(cache_file: Path, cache_data: Dict[str, Any]) -> None:
    """
    Сохраняет кэш метаданных документа в JSON-файл.

    Args:
        cache_file: Путь к файлу кэша
        cache_data: Словарь с метаданными для сохранения
    """
    try:
        # Создаем директорию, если её нет
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cache saved to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache to {cache_file}: {e}")


def check_cache_needs_update(
    cache: Optional[Dict[str, Any]],
    document_id: str,
    update_interval_days: int = 7
) -> bool:
    """
    Проверяет, нужно ли обновлять кэш документа.

    Args:
        cache: Словарь с метаданными кэша или None
        document_id: ID документа для проверки
        update_interval_days: Интервал обновления в днях

    Returns:
        True, если кэш нужно обновить, False иначе
    """
    if cache is None:
        logger.info("No cache found, update needed")
        return True

    # Проверяем document_id
    cached_document_id = cache.get('document_id')
    if cached_document_id != document_id:
        logger.info(
            f"Document ID changed: {cached_document_id} -> {document_id}, update needed"
        )
        return True

    # Проверяем дату последнего обновления
    last_updated_str = cache.get('last_updated')
    if not last_updated_str:
        logger.info("No last_updated in cache, update needed")
        return True

    try:
        last_updated = datetime.fromisoformat(last_updated_str)
        now = datetime.now()
        days_since_update = (now - last_updated).days

        if days_since_update >= update_interval_days:
            logger.info(
                f"Cache is {days_since_update} days old (threshold: {update_interval_days}), update needed"
            )
            return True
        else:
            logger.info(
                f"Cache is fresh ({days_since_update} days old), no update needed"
            )
            return False
    except ValueError as e:
        logger.warning(f"Invalid date format in cache: {last_updated_str}, {e}")
        return True


def get_image_paths(images_dir: Path) -> List[str]:
    """
    Получает список путей к изображениям в директории.

    Args:
        images_dir: Директория с изображениями

    Returns:
        Список относительных путей к изображениям
    """
    if not images_dir.exists():
        return []

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    # Используем set для устранения дубликатов (Windows не различает регистр)
    images_set = set()

    # Собираем все файлы с нужными расширениями (case-insensitive)
    for file_path in images_dir.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in image_extensions:
                images_set.add(file_path)

    # Преобразуем в список и сортируем для предсказуемости
    images = sorted(images_set)

    # Возвращаем относительные пути от data/
    relative_paths = []
    for img in images:
        # Получаем путь относительно data/
        # Если images_dir = data/images, то img.relative_to(data/) = images/filename.png
        try:
            # Пытаемся получить путь относительно родителя images_dir
            rel_path = img.relative_to(images_dir.parent)
            relative_paths.append(str(rel_path).replace('\\', '/'))
        except ValueError:
            # Если не получается, используем полный путь
            relative_paths.append(str(img).replace('\\', '/'))

    return relative_paths


def extract_document_id(url: str) -> Optional[str]:
    """
    Извлекает document_id из URL Google Docs.

    Args:
        url: URL документа Google Docs

    Returns:
        document_id или None, если не удалось извлечь
    """
    # Паттерн для извлечения ID из URL вида:
    # https://docs.google.com/document/d/DOCUMENT_ID/edit
    pattern = r'/document/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def fetch_document_zip(document_id: str) -> bytes:
    """
    Получает ZIP-архив с документом и изображениями из Google Docs.

    Args:
        document_id: ID документа Google Docs

    Returns:
        Байты ZIP-архива

    Raises:
        urllib.error.URLError: При ошибках сети или доступа
        ValueError: Если документ недоступен
    """
    export_url = f"https://docs.google.com/document/d/{document_id}/export?format=zip"

    try:
        logger.info(f"Fetching ZIP archive for document {document_id} from Google Docs...")
        with urllib.request.urlopen(export_url, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch document: HTTP {response.status}")
            zip_data = response.read()
            logger.info(f"Successfully fetched ZIP archive ({len(zip_data)} bytes)")
            return zip_data
    except urllib.error.URLError as e:
        logger.error(f"Network error while fetching document {document_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while fetching document {document_id}: {e}")
        raise ValueError(f"Failed to fetch document: {e}") from e


def extract_files_from_zip(zip_data: bytes, output_dir: Path) -> Dict[str, Any]:
    """
    Распаковывает ZIP-архив и извлекает HTML и изображения.

    Args:
        zip_data: Байты ZIP-архива
        output_dir: Директория для сохранения файлов

    Returns:
        Словарь с информацией об извлеченных файлах:
        - html_file: Path к HTML-файлу (или None)
        - images: список путей к изображениям
        - all_files: список всех извлеченных файлов
    """
    html_file = None
    images = []
    all_files = []

    # Создаем директорию для изображений
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(zip_data), 'r') as zip_ref:
            # Получаем список всех файлов в архиве
            file_list = zip_ref.namelist()
            logger.info(f"Found {len(file_list)} files in ZIP archive")

            for file_name in file_list:
                # Пропускаем директории
                if file_name.endswith('/'):
                    continue

                # Извлекаем файл
                file_data = zip_ref.read(file_name)
                file_path = output_dir / file_name

                # Создаем директорию для файла, если нужно
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Сохраняем файл
                with open(file_path, 'wb') as f:
                    f.write(file_data)

                all_files.append(file_path)

                # Определяем тип файла
                file_lower = file_name.lower()

                # HTML файл (обычно называется index.html или похоже)
                if file_lower.endswith('.html') and (html_file is None or 'index' in file_lower):
                    html_file = file_path
                    logger.info(f"Found HTML file: {file_name}")

                # Изображения
                elif any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    # Перемещаем изображение в папку images/
                    image_name = Path(file_name).name
                    image_path = images_dir / image_name
                    # Если файл уже существует, добавляем суффикс
                    counter = 1
                    while image_path.exists():
                        stem = image_path.stem
                        suffix = image_path.suffix
                        image_path = images_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    # Копируем файл
                    with open(image_path, 'wb') as f:
                        f.write(file_data)
                    # Удаляем оригинальный файл из корня
                    if file_path != image_path:
                        file_path.unlink()
                    images.append(image_path)
                    logger.info(f"Extracted image: {image_name} -> {image_path}")

        logger.info(f"Extracted {len(images)} images and {1 if html_file else 0} HTML file(s)")
        return {
            'html_file': html_file,
            'images': images,
            'all_files': all_files
        }

    except zipfile.BadZipFile as e:
        error_msg = f"Invalid ZIP archive: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error extracting ZIP archive: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e


def update_image_paths_in_html(html_file: Path, images_dir: Path) -> None:
    """
    Обновляет ссылки на изображения в HTML-файле на локальные пути.

    Args:
        html_file: Путь к HTML-файлу
        images_dir: Директория с изображениями
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Получаем список всех изображений в папке
        image_files = list(images_dir.glob('*'))
        image_names = {img.name.lower(): img.name for img in image_files}

        # Обновляем ссылки на изображения
        # Ищем все теги img с src
        def replace_image_src(match):
            src = match.group(1)
            # Если это уже локальный путь, пропускаем
            if src.startswith('images/') or src.startswith('./images/'):
                return match.group(0)

            # Пытаемся найти изображение по имени файла
            src_lower = Path(src).name.lower()
            if src_lower in image_names:
                return f'src="images/{image_names[src_lower]}"'
            return match.group(0)

        # Паттерн для поиска src в тегах img
        pattern = r'src="([^"]+)"'
        html_content = re.sub(pattern, replace_image_src, html_content, flags=re.IGNORECASE)

        # Сохраняем обновленный HTML
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Updated image paths in HTML file: {html_file}")

    except Exception as e:
        logger.warning(f"Could not update image paths in HTML: {e}")


def import_document(url: str, output_dir: Path) -> Dict[str, Any]:
    """
    Импортирует документ из Google Docs и сохраняет локально.

    Args:
        url: URL документа Google Docs
        output_dir: Директория для сохранения файлов (обычно data/)

    Returns:
        Словарь с результатами импорта:
        - success: bool - успешность импорта
        - document_id: str - ID документа
        - html_file: Optional[Path] - путь к HTML-файлу
        - images: List[Path] - список путей к изображениям
        - error: Optional[str] - сообщение об ошибке (если есть)

    Raises:
        ValueError: Если URL некорректный или документ недоступен
    """
    logger.info(f"Starting document import from URL: {url}")

    # Извлекаем document_id
    document_id = extract_document_id(url)
    if not document_id:
        error_msg = f"Failed to extract document_id from URL: {url}"
        logger.error(error_msg)
        return {
            'success': False,
            'document_id': None,
            'html_file': None,
            'images': [],
            'error': error_msg
        }

    logger.info(f"Extracted document_id: {document_id}")

    try:
        # Создаем директорию, если её нет
        output_dir.mkdir(parents=True, exist_ok=True)

        # Получаем ZIP-архив
        zip_data = fetch_document_zip(document_id)

        # Распаковываем архив
        extracted = extract_files_from_zip(zip_data, output_dir)

        html_file = extracted['html_file']
        images = extracted['images']

        # Если HTML файл найден, обновляем ссылки на изображения
        if html_file:
            images_dir = output_dir / "images"
            update_image_paths_in_html(html_file, images_dir)

            # Переименовываем HTML файл в knowledge.html, если нужно
            target_html = output_dir / "knowledge.html"
            if html_file != target_html:
                # Удаляем старый файл, если существует
                if target_html.exists():
                    target_html.unlink()
                html_file.rename(target_html)
                html_file = target_html

        logger.info(f"Document successfully imported:")
        logger.info(f"  HTML file: {html_file}")
        logger.info(f"  Images: {len(images)} files")

        # Сохраняем метаданные в кэш
        cache_file = output_dir / "knowledge_cache.json"
        images_dir = output_dir / "images"
        image_paths = get_image_paths(images_dir)

        # Формируем путь к HTML-файлу относительно data/
        html_path = None
        if html_file:
            try:
                html_path = str(html_file.relative_to(output_dir.parent)).replace('\\', '/')
            except ValueError:
                html_path = str(html_file).replace('\\', '/')

        cache_data = {
            'content': html_path or str(html_file) if html_file else None,
            'last_updated': datetime.now().isoformat(),
            'document_id': document_id,
            'images': image_paths
        }

        # Строим поисковый индекс и извлекаем связь изображений с разделами
        if html_file:
            try:
                from src.search import (
                    load_sections,
                    parse_html_sections,
                    build_search_index,
                )

                sections_file = output_dir / "sections.json"
                if sections_file.exists():
                    logger.info("Building search index and extracting section images...")
                    
                    # Загружаем заголовки разделов
                    sections = load_sections(sections_file)
                    
                    # Загружаем HTML-контент
                    with open(html_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    # Разбиваем на разделы и извлекаем изображения
                    sections_content, sections_images = parse_html_sections(html_content, sections)
                    
                    # Сохраняем связь изображений с разделами в кэш
                    # Преобразуем Path объекты в строки для JSON-сериализации
                    section_images_serializable = {
                        section_title: image_paths
                        for section_title, image_paths in sections_images.items()
                    }
                    cache_data['section_images'] = section_images_serializable
                    logger.info(
                        f"Extracted images for {len(sections_images)} sections, "
                        f"total {sum(len(imgs) for imgs in sections_images.values())} images"
                    )
                    
                    # Строим поисковый индекс
                    index = build_search_index(sections_content)
                    if index:
                        cache_data['search_index'] = index
                        logger.info("Search index built successfully")
                    else:
                        logger.warning("Failed to build search index")
                else:
                    logger.warning(f"Sections file not found: {sections_file}, skipping index build")
            except Exception as e:
                logger.error(f"Error building search index: {e}", exc_info=True)

        save_cache(cache_file, cache_data)

        return {
            'success': True,
            'document_id': document_id,
            'html_file': html_file,
            'images': images,
            'error': None
        }

    except urllib.error.URLError as e:
        error_msg = f"Network error while importing document: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'document_id': document_id,
            'html_file': None,
            'images': [],
            'error': error_msg
        }
    except ValueError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return {
            'success': False,
            'document_id': document_id,
            'html_file': None,
            'images': [],
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error while importing document: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'document_id': document_id,
            'html_file': None,
            'images': [],
            'error': error_msg
        }
