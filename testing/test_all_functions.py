"""
Тестовый скрипт для проверки всех функций xyliganimbot.

Проверяет:
1. Загрузку контента из Google Docs
2. Загрузку embedding-модели
3. Векторизацию контента
"""

import sys
import os
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')  # UTF-8
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Добавляем корневую директорию в путь
# Скрипт может запускаться из корня: python testing/test_all_functions.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_google_docs_config
from src.logging import setup_logging, get_logger
from src.google_docs import import_document
from src.model_loader import download_model
from src.search import vectorize_content

# Настройка логирования
setup_logging(
    level="INFO",
    log_file="logs/app.log",
    audit_file="logs/audit.log",
    log_user_messages=False,
    max_bytes=10485760,
    backup_count=5,
)

logger = get_logger(__name__)


def test_import_content():
    """Тест загрузки контента из Google Docs."""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Загрузка контента из Google Docs")
    print("=" * 60)
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        google_docs_config = get_google_docs_config(config)
        url = google_docs_config.get("url")
        
        if not url:
            print("[ERROR] ОШИБКА: URL Google Docs не найден в config.yaml")
            return False
        
        print(f"URL документа: {url}")
        
        # Импортируем документ
        output_dir = project_root / "data"
        result = import_document(url, output_dir)
        
        if result.get("success"):
            print("[OK] Контент успешно загружен!")
            print(f"   - HTML файл: {result.get('html_file')}")
            print(f"   - Изображений: {len(result.get('images', []))}")
            print(f"   - Document ID: {result.get('document_id')}")
            return True
        else:
            print(f"[ERROR] ОШИБКА при загрузке контента: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] ИСКЛЮЧЕНИЕ при загрузке контента: {e}")
        logger.error(f"Error in test_import_content: {e}", exc_info=True)
        return False


def test_download_model():
    """Тест загрузки embedding-модели."""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Загрузка embedding-модели")
    print("=" * 60)
    
    try:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        models_dir = project_root / "models"
        
        print(f"Модель: {model_name}")
        print(f"Директория: {models_dir}")
        
        # Загружаем модель
        success = download_model(model_name=model_name, models_dir=models_dir)
        
        if success:
            print("[OK] Модель успешно загружена!")
            model_path = models_dir / model_name
            if model_path.exists():
                print(f"   - Путь: {model_path}")
                # Проверяем размер директории
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                print(f"   - Размер: {total_size / (1024*1024):.2f} MB")
            return True
        else:
            print("[ERROR] ОШИБКА при загрузке модели")
            return False
            
    except Exception as e:
        print(f"[ERROR] ИСКЛЮЧЕНИЕ при загрузке модели: {e}")
        logger.error(f"Error in test_download_model: {e}", exc_info=True)
        return False


def test_vectorize_content():
    """Тест векторизации контента."""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Векторизация контента")
    print("=" * 60)
    
    try:
        markdown_file = project_root / "data" / "knowledge.md"
        # Проверяем также старый формат HTML для обратной совместимости
        if not markdown_file.exists():
            markdown_file = project_root / "data" / "knowledge.html"
        cache_file = project_root / "data" / "knowledge_cache.json"
        
        # Проверяем наличие файлов
        if not markdown_file.exists():
            print(f"[ERROR] ОШИБКА: Markdown/HTML файл не найден: {markdown_file}")
            print("   Сначала выполните тест загрузки контента (ТЕСТ 1)")
            return False
        
        print(f"Markdown/HTML файл: {markdown_file}")
        print(f"Кэш: {cache_file}")
        
        # Векторизуем контент
        success = vectorize_content(
            markdown_file=markdown_file,
            cache_file=cache_file
        )
        
        if success:
            print("[OK] Контент успешно векторизован!")
            
            # Проверяем кэш
            if cache_file.exists():
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                embeddings_data = cache_data.get('embeddings')
                if embeddings_data:
                    embeddings = embeddings_data.get('embeddings', [])
                    section_titles = embeddings_data.get('section_titles', [])
                    print(f"   - Разделов векторизовано: {len(section_titles)}")
                    print(f"   - Размерность векторов: {len(embeddings[0]) if embeddings else 0}")
                    print(f"   - Всего векторов: {len(embeddings)}")
            
            return True
        else:
            print("[ERROR] ОШИБКА при векторизации контента")
            return False
            
    except Exception as e:
        print(f"[ERROR] ИСКЛЮЧЕНИЕ при векторизации: {e}")
        logger.error(f"Error in test_vectorize_content: {e}", exc_info=True)
        return False


def main():
    """Основная функция тестирования."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ФУНКЦИЙ XYLIGANIMBOT")
    print("=" * 60)
    
    results = []
    
    # Тест 1: Загрузка контента
    results.append(("Загрузка контента", test_import_content()))
    
    # Тест 2: Загрузка модели
    results.append(("Загрузка модели", test_download_model()))
    
    # Тест 3: Векторизация (только если контент загружен)
    markdown_file = project_root / "data" / "knowledge.md"
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    if markdown_file.exists():
        results.append(("Векторизация контента", test_vectorize_content()))
    else:
        print("\n[SKIP] Пропуск теста векторизации: контент не загружен")
        results.append(("Векторизация контента", None))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    for test_name, result in results:
        if result is None:
            status = "[SKIP] ПРОПУЩЕН"
        elif result:
            status = "[OK] УСПЕШНО"
        else:
            status = "[ERROR] ОШИБКА"
        print(f"{status} - {test_name}")
    
    # Общий результат
    successful = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    print(f"\nВсего тестов: {total}, Успешно: {successful}, Ошибок: {total - successful}")
    
    if successful == total and total > 0:
        print("\n[SUCCESS] Все тесты пройдены успешно!")
        return 0
    else:
        print("\n[WARNING] Некоторые тесты завершились с ошибками")
        return 1


if __name__ == "__main__":
    sys.exit(main())
