"""
Интерактивный скрипт для ручной проверки поиска.
Позволяет вводить запросы и анализировать результаты поиска.
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.search import (
    load_index_from_cache,
    search,
    parse_html_sections,
    load_sections,
)
from src.logging import setup_logging


def format_result(result: dict, index: int) -> str:
    """Форматирует результат поиска для вывода."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Результат #{index + 1}")
    lines.append(f"{'='*70}")
    lines.append(f"Раздел: {result['section_title']}")
    lines.append(f"Релевантность: {result['relevance_score']}")
    lines.append(f"  - Совпадений в заголовке: {result['section_matches']}")
    lines.append(f"  - Совпадений в содержимом: {result['content_matches']}")
    lines.append(f"  - Всего совпадений: {result['matches']}")
    
    if "text" in result:
        text = result["text"]
        # Показываем первые 200 символов текста
        preview = text[:200] + "..." if len(text) > 200 else text
        lines.append(f"\nТекст раздела (превью):")
        lines.append(f"  {preview}")
        lines.append(f"\nПолный размер текста: {len(text)} символов")
    
    return "\n".join(lines)


def print_search_results(query: str, results: list):
    """Выводит результаты поиска."""
    print(f"\n{'='*70}")
    print(f"Запрос: '{query}'")
    print(f"{'='*70}")
    
    if not results:
        print("\n[INFO] Результаты не найдены")
        print("Возможные причины:")
        print("  - Запрос не содержит слов из документа")
        print("  - Все слова в запросе игнорируются (знаки препинания)")
        return
    
    print(f"\nНайдено результатов: {len(results)}")
    print(f"Показано: {len(results)} (лимит: 5)")
    
    # Показываем статистику по релевантности
    if results:
        scores = [r["relevance_score"] for r in results]
        print(f"\nСтатистика релевантности:")
        print(f"  Максимальная: {max(scores)}")
        print(f"  Минимальная: {min(scores)}")
        print(f"  Средняя: {sum(scores) / len(scores):.2f}")
    
    # Выводим каждый результат
    for i, result in enumerate(results):
        try:
            print(format_result(result, i))
        except UnicodeEncodeError:
            # Если проблема с кодировкой, выводим без текста
            print(f"\n{'='*70}")
            print(f"Результат #{i + 1}")
            print(f"{'='*70}")
            print(f"Раздел: {result['section_title']}")
            print(f"Релевантность: {result['relevance_score']}")
            print(f"Совпадений: {result['matches']}")
            if "text" in result:
                print(f"Размер текста: {len(result['text'])} символов")


def analyze_query_tokens(query: str, index: dict):
    """Анализирует, какие токены из запроса найдены в индексе."""
    from src.search import tokenize_text
    
    query_tokens = tokenize_text(query)
    if not query_tokens:
        print("\n[WARN] Запрос не содержит токенов после обработки")
        return
    
    print(f"\nАнализ токенов запроса:")
    print(f"  Токенов в запросе: {len(query_tokens)}")
    print(f"  Токены: {query_tokens}")
    
    section_index = index.get("section_index", {})
    content_index = index.get("content_index", {})
    
    found_tokens = []
    missing_tokens = []
    
    for token in query_tokens:
        in_section = token in section_index
        in_content = token in content_index
        
        if in_section or in_content:
            found_tokens.append(token)
            sections_count = len(section_index.get(token, []))
            content_sections = len(content_index.get(token, {}))
            print(f"\n  Токен '{token}':")
            print(f"    - В индексе разделов: {'да' if in_section else 'нет'} ({sections_count} разделов)")
            print(f"    - В индексе содержимого: {'да' if in_content else 'нет'} ({content_sections} разделов)")
        else:
            missing_tokens.append(token)
            print(f"\n  Токен '{token}': НЕ НАЙДЕН в индексе")
    
    if missing_tokens:
        print(f"\n[WARN] Токены не найдены в индексе: {missing_tokens}")
    if found_tokens:
        print(f"\n[OK] Найдено токенов: {len(found_tokens)} из {len(query_tokens)}")


def main():
    """Главная функция интерактивного скрипта."""
    print("=" * 70)
    print("ИНТЕРАКТИВНАЯ ПРОВЕРКА ПОИСКА")
    print("=" * 70)
    
    # Настройка логирования
    setup_logging(level="INFO", log_file="logs/app.log")
    
    # Загружаем индекс из кэша
    cache_file = Path("data/knowledge_cache.json")
    index = load_index_from_cache(cache_file)
    
    if not index:
        print("\n[ERROR] Индекс не найден в кэше")
        print("Запустите импорт документа для создания индекса:")
        print("  python test_import.py")
        return
    
    print(f"\n[OK] Индекс загружен из кэша")
    print(f"  Токенов в индексе разделов: {len(index.get('section_index', {}))}")
    print(f"  Токенов в индексе содержимого: {len(index.get('content_index', {}))}")
    
    # Парсим разделы один раз для всех запросов (оптимизация)
    html_file = Path("data/knowledge.html")
    sections_file = Path("data/sections.json")
    sections_content = None
    
    if html_file.exists() and sections_file.exists():
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            sections = load_sections(sections_file)
            sections_content = parse_html_sections(html_content, sections)
            print(f"  Разделов распарсено: {len(sections_content)}")
        except Exception as e:
            print(f"  [WARN] Не удалось распарсить разделы: {e}")
            print("  Поиск будет работать без извлечения текста разделов")
    
    print("\n" + "=" * 70)
    print("ИНСТРУКЦИЯ:")
    print("  - Введите поисковый запрос и нажмите Enter")
    print("  - Для выхода введите: exit, quit или q")
    print("  - Для анализа токенов введите: analyze <запрос>")
    print("  - Для показа справки введите: help")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n> Введите запрос: ").strip()
            
            if not query:
                continue
            
            # Команды выхода
            if query.lower() in ["exit", "quit", "q"]:
                print("\nВыход из программы. До свидания!")
                break
            
            # Команда помощи
            if query.lower() == "help":
                print("\nДоступные команды:")
                print("  exit, quit, q - выход из программы")
                print("  analyze <запрос> - анализ токенов запроса")
                print("  help - показать эту справку")
                print("\nПримеры запросов:")
                print("  токен")
                print("  пароль")
                print("  VPN контур")
                print("  настройка токена")
                continue
            
            # Команда анализа токенов
            if query.lower().startswith("analyze "):
                analyze_query = query[8:].strip()
                if analyze_query:
                    analyze_query_tokens(analyze_query, index)
                else:
                    print("[ERROR] Укажите запрос для анализа: analyze <запрос>")
                continue
            
            # Выполняем поиск
            results = search(
                query,
                index,
                sections_content=sections_content,
                limit=5,
            )
            
            # Выводим результаты
            print_search_results(query, results)
            
            # Предлагаем анализ токенов
            if not results:
                print("\nХотите проанализировать токены запроса? (y/n): ", end="")
                answer = input().strip().lower()
                if answer == "y":
                    analyze_query_tokens(query, index)
        
        except KeyboardInterrupt:
            print("\n\nВыход из программы. До свидания!")
            break
        except Exception as e:
            print(f"\n[ERROR] Ошибка при выполнении поиска: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
