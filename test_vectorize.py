"""
Тест векторизации контента.
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
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
    
    html_file = project_root / "data" / "knowledge.html"
    sections_file = project_root / "data" / "sections.json"
    cache_file = project_root / "data" / "knowledge_cache.json"
    
    if not html_file.exists():
        print(f"[ERROR] HTML файл не найден: {html_file}")
        print("   Сначала выполните: python test_import_content.py")
        return 1
    
    if not sections_file.exists():
        print(f"[ERROR] Файл с разделами не найден: {sections_file}")
        return 1
    
    print(f"HTML файл: {html_file}")
    print(f"Файл разделов: {sections_file}")
    print(f"Кэш: {cache_file}")
    print("\n[INFO] Векторизация может занять некоторое время...")
    
    success = vectorize_content(
        html_file=html_file,
        sections_file=sections_file,
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
