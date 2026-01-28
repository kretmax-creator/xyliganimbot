"""
Тест загрузки контента из Google Docs.
"""

import sys
from pathlib import Path

# Скрипт может запускаться из корня: python testing/test_import_content.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_google_docs_config
from src.logging import setup_logging, get_logger
from src.google_docs import import_document

setup_logging(level="INFO", log_file="logs/app.log", audit_file="logs/audit.log", 
              log_user_messages=False, max_bytes=10485760, backup_count=5)

logger = get_logger(__name__)

def main():
    print("=" * 60)
    print("ТЕСТ: Загрузка контента из Google Docs")
    print("=" * 60)
    
    config = load_config()
    google_docs_config = get_google_docs_config(config)
    url = google_docs_config.get("url")
    
    if not url:
        print("[ERROR] URL Google Docs не найден в config.yaml")
        return 1
    
    print(f"URL документа: {url}")
    
    output_dir = project_root / "data"
    result = import_document(url, output_dir)
    
    if result.get("success"):
        print("[OK] Контент успешно загружен!")
        print(f"   - HTML файл: {result.get('html_file')}")
        print(f"   - Изображений: {len(result.get('images', []))}")
        print(f"   - Document ID: {result.get('document_id')}")
        return 0
    else:
        print(f"[ERROR] Ошибка: {result.get('error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
