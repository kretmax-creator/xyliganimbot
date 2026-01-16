"""
Модуль работы с Google Docs для xyliganimbot.

Обеспечивает импорт документа из Google Docs в формате ZIP-архива
с HTML и изображениями, извлечение и сохранение локально.
"""

import re
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO

from src.logging import get_logger

logger = get_logger(__name__)


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
