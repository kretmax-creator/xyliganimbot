"""
Тестовый скрипт для проверки функции поиска.
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.search import (
    load_index_from_cache,
    search,
    build_index_from_html,
)
from src.logging import setup_logging


def test_search():
    """Тестирует функцию поиска."""
    print("=" * 60)
    print("Тест: Поиск по индексу")
    print("=" * 60)

    # Настройка логирования
    setup_logging(level="INFO", log_file="logs/app.log")

    # Загружаем индекс из кэша
    cache_file = Path("data/knowledge_cache.json")
    index = load_index_from_cache(cache_file)

    if not index:
        print("[ERROR] Индекс не найден в кэше")
        print("Запустите импорт документа для создания индекса")
        return

    print(f"[OK] Индекс загружен из кэша")
    print(f"Токенов в индексе разделов: {len(index.get('section_index', {}))}")
    print(f"Токенов в индексе содержимого: {len(index.get('content_index', {}))}")

    # Тестовые запросы
    test_queries = [
        "токен",
        "пароль",
        "VPN",
        "настройка",
        "токен пароль",
        "VPN контур",
        "несуществующий запрос xyz123",
        "",  # Пустой запрос
    ]

    html_file = Path("data/knowledge.html")
    sections_file = Path("data/sections.json")

    # Парсим разделы один раз для всех запросов (оптимизация)
    sections_content = None
    if html_file.exists() and sections_file.exists():
        try:
            from src.search import parse_html_sections, load_sections
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            sections = load_sections(sections_file)
            sections_content = parse_html_sections(html_content, sections)
            print(f"[OK] Разделы распарсены: {len(sections_content)} разделов")
        except Exception as e:
            print(f"[WARN] Не удалось распарсить разделы: {e}")

    print("\n" + "=" * 60)
    print("Тестирование поисковых запросов:")
    print("=" * 60)

    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        print("-" * 60)

        results = search(
            query,
            index,
            sections_content=sections_content,
            limit=5,
        )

        if not results:
            print("  [INFO] Результаты не найдены")
        else:
            print(f"  [OK] Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"\n  Результат {i}:")
                print(f"    Раздел: {result['section_title']}")
                print(f"    Релевантность: {result['relevance_score']}")
                print(f"    Совпадений в заголовке: {result['section_matches']}")
                print(f"    Совпадений в содержимом: {result['content_matches']}")
                print(f"    Всего совпадений: {result['matches']}")
                if "text" in result:
                    try:
                        text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                        print(f"    Текст (превью): {text_preview}")
                    except UnicodeEncodeError:
                        # Пропускаем вывод текста при проблемах с кодировкой
                        print(f"    Текст: {len(result['text'])} символов")

    print("\n" + "=" * 60)
    print("[SUCCESS] Тест поиска завершен!")
    print("=" * 60)


def main():
    """Запускает тесты."""
    try:
        test_search()
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
