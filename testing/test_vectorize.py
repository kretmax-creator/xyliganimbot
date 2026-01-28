"""
Тест векторизации контента.
"""

import sys
import json
from pathlib import Path

# Скрипт может запускаться из корня: python testing/test_vectorize.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.logging import setup_logging, get_logger
from src.search import vectorize_content

setup_logging(level="INFO", log_file="logs/app.log", audit_file="logs/audit.log", 
              log_user_messages=False, max_bytes=10485760, backup_count=5)

logger = get_logger(__name__)

def main():
    print("=" * 60)
    print("ТЕСТ: Векторизация контента")
    print("=" * 60)
    
    markdown_file = project_root / "data" / "knowledge.md"
    # Проверяем также старый формат HTML для обратной совместимости
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    cache_file = project_root / "data" / "knowledge_cache.json"
    
    if not markdown_file.exists():
        print(f"[ERROR] Markdown/HTML файл не найден: {markdown_file}")
        print("   Сначала выполните: python test_import_content.py")
        return 1
    
    print(f"Markdown/HTML файл: {markdown_file}")
    print(f"Кэш: {cache_file}")
    print("\n[INFO] Векторизация может занять некоторое время...")
    
    success = vectorize_content(
        markdown_file=markdown_file,
        cache_file=cache_file
    )
    
    if success:
        print("[OK] Контент успешно векторизован!")
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            embeddings_data = cache_data.get('embeddings')
            if embeddings_data:
                embeddings = embeddings_data.get('embeddings', [])
                section_titles = embeddings_data.get('section_titles', [])
                print(f"   - Разделов векторизовано: {len(section_titles)}")
                print(f"   - Размерность векторов: {len(embeddings[0]) if embeddings else 0}")
                print(f"   - Всего векторов: {len(embeddings)}")
        
        return 0
    else:
        print("[ERROR] Ошибка при векторизации контента")
        return 1

if __name__ == "__main__":
    sys.exit(main())
