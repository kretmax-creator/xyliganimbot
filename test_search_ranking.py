"""
Быстрый тест для проверки ранжирования после изменения веса заголовков.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.search import (
    load_index_from_cache,
    search,
    parse_html_sections,
    load_sections,
)
from src.logging import setup_logging

setup_logging(level="INFO", log_file="logs/app.log")

# Загружаем индекс
cache_file = Path("data/knowledge_cache.json")
index = load_index_from_cache(cache_file)

if not index:
    print("[ERROR] Индекс не найден")
    sys.exit(1)

# Парсим разделы
html_file = Path("data/knowledge.html")
sections_file = Path("data/sections.json")

sections_content = None
if html_file.exists() and sections_file.exists():
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    sections = load_sections(sections_file)
    sections_content = parse_html_sections(html_content, sections)

print("=" * 70)
print("ТЕСТ РАНЖИРОВАНИЯ С УВЕЛИЧЕННЫМ ВЕСОМ ЗАГОЛОВКОВ")
print("=" * 70)
print("\nФормула: relevance_score = section_matches * 5 + content_matches")
print("(вес заголовка увеличен с 2 до 5)\n")

# Тестовые запросы
test_queries = [
    "токен",
    "пароль",
    "VPN",
    "токен пароль",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Запрос: '{query}'")
    print(f"{'='*70}")
    
    results = search(query, index, sections_content=sections_content, limit=5)
    
    if not results:
        print("Результаты не найдены")
        continue
    
    print(f"\nНайдено результатов: {len(results)}")
    print(f"\nРанжирование результатов:")
    print(f"{'№':<4} {'Раздел':<50} {'Релевантность':<15} {'Заголовок':<10} {'Содержимое':<12}")
    print("-" * 95)
    
    for i, result in enumerate(results, 1):
        section_title = result['section_title']
        if len(section_title) > 47:
            section_title = section_title[:44] + "..."
        
        print(
            f"{i:<4} {section_title:<50} "
            f"{result['relevance_score']:<15} "
            f"{result['section_matches']:<10} "
            f"{result['content_matches']:<12}"
        )
    
    # Анализ ранжирования
    print(f"\nАнализ:")
    for i, result in enumerate(results):
        if i == 0:
            continue
        prev_score = results[i-1]['relevance_score']
        curr_score = result['relevance_score']
        if prev_score < curr_score:
            print(f"  [WARN] Результат {i+1} имеет большую релевантность, чем результат {i}!")
        elif prev_score == curr_score:
            print(f"  [INFO] Результаты {i} и {i+1} имеют одинаковую релевантность")
    
    # Проверка приоритета заголовков
    has_title_match = any(r['section_matches'] > 0 for r in results)
    if has_title_match:
        title_results = [r for r in results if r['section_matches'] > 0]
        content_only = [r for r in results if r['section_matches'] == 0]
        if content_only:
            print(f"\n  [OK] Разделы с совпадениями в заголовке ({len(title_results)}) выше разделов только с содержимым ({len(content_only)})")
        else:
            print(f"\n  [OK] Все результаты имеют совпадения в заголовке")

print("\n" + "=" * 70)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 70)
