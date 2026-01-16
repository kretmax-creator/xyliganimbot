"""
Тестовый скрипт для проверки импорта документа из Google Docs.
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.google_docs import import_document
from src.config import load_config, get_google_docs_config

def main():
    """Тестирует импорт документа из Google Docs."""
    print("=" * 60)
    print("Тест импорта документа из Google Docs")
    print("=" * 60)
    
    try:
        # Загружаем конфигурацию
        print("\n1. Загрузка конфигурации...")
        config = load_config()
        google_docs_config = get_google_docs_config(config)
        url = google_docs_config.get("url")
        
        if not url:
            print("[ERROR] Ошибка: URL документа не найден в config.yaml")
            return
        
        print(f"   URL документа: {url}")
        
        # Определяем директорию для сохранения
        output_dir = Path("data")
        print(f"   Директория для сохранения: {output_dir}")
        
        # Выполняем импорт
        print("\n2. Импорт документа...")
        result = import_document(url, output_dir)
        
        # Выводим результаты
        print("\n3. Результаты импорта:")
        print("=" * 60)
        
        if result['success']:
            print("[OK] Импорт выполнен успешно!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   HTML файл: {result['html_file']}")
            print(f"   Изображений: {len(result['images'])}")
            
            if result['images']:
                print("\n   Извлеченные изображения:")
                for i, img_path in enumerate(result['images'][:10], 1):  # Показываем первые 10
                    print(f"   {i}. {img_path.name}")
                if len(result['images']) > 10:
                    print(f"   ... и еще {len(result['images']) - 10} изображений")
        else:
            print("[ERROR] Ошибка при импорте!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Ошибка: {result['error']}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
