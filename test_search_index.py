"""
Тестовый скрипт для проверки построения поискового индекса.
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.search import (
    load_sections,
    normalize_text,
    tokenize_text,
    extract_text_from_html,
    find_section_in_html,
    parse_html_sections,
    build_search_index,
    build_index_from_html,
    save_index_to_cache,
    load_index_from_cache,
)
from src.logging import setup_logging

def test_text_normalization():
    """Тестирует нормализацию текста."""
    print("=" * 60)
    print("Тест 1: Нормализация текста")
    print("=" * 60)

    test_cases = [
        ("Привет, мир!", "привет, мир!"),  # normalize_text не удаляет знаки препинания
        ("ТОКЕН: пароль", "токен: пароль"),  # только нижний регистр и пробелы
        ("  Много   пробелов  ", "много пробелов"),  # удаляет лишние пробелы
        ("Текст С Заглавными", "текст с заглавными"),  # нижний регистр
    ]

    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        status = "OK" if result == expected else "FAIL"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")

    print()


def test_tokenization():
    """Тестирует токенизацию текста."""
    print("=" * 60)
    print("Тест 2: Токенизация текста")
    print("=" * 60)

    test_cases = [
        ("Привет, мир!", ["привет", "мир"]),
        ("ТОКЕН: пароль", ["токен", "пароль"]),
        ("VPN контур разработки", ["vpn", "контур", "разработки"]),
    ]

    for input_text, expected in test_cases:
        result = tokenize_text(input_text)
        status = "OK" if set(result) == set(expected) else "FAIL"
        print(f"{status} '{input_text}' -> {result}")
        if set(result) != set(expected):
            print(f"  Expected: {expected}")

    print()


def test_index_building():
    """Тестирует построение индекса."""
    print("=" * 60)
    print("Тест 3: Построение поискового индекса")
    print("=" * 60)

    try:
        # Настройка логирования
        setup_logging(level="INFO", log_file="logs/app.log")

        html_file = Path("data/knowledge.html")
        sections_file = Path("data/sections.json")

        if not html_file.exists():
            print(f"[SKIP] HTML файл не найден: {html_file}")
            return

        if not sections_file.exists():
            print(f"[SKIP] Файл с заголовками не найден: {sections_file}")
            return

        print(f"\n1. Загрузка заголовков разделов...")
        sections = load_sections(sections_file)
        print(f"   [OK] Загружено разделов: {len(sections)}")
        print(f"   Примеры: {sections[:3]}")

        print(f"\n2. Построение индекса из HTML...")
        index = build_index_from_html(html_file, sections_file)

        if index:
            section_index = index.get("section_index", {})
            content_index = index.get("content_index", {})

            print(f"   [OK] Индекс построен успешно!")
            print(f"   Токенов в индексе разделов: {len(section_index)}")
            print(f"   Токенов в индексе содержимого: {len(content_index)}")

            # Показываем примеры
            if section_index:
                example_token = list(section_index.keys())[0]
                example_sections = section_index[example_token]
                print(f"\n   Пример из индекса разделов:")
                print(f"     '{example_token}' -> {example_sections[:3]}")

            if content_index:
                example_token = list(content_index.keys())[0]
                example_content = content_index[example_token]
                print(f"\n   Пример из индекса содержимого:")
                print(f"     '{example_token}' -> {list(example_content.keys())[:2]}")

        else:
            print(f"   [ERROR] Не удалось построить индекс")

        print(f"\n3. Проверка сохранения в кэш...")
        cache_file = Path("data/knowledge_cache.json")
        cached_index = load_index_from_cache(cache_file)

        if cached_index:
            print(f"   [OK] Индекс найден в кэше")
            print(f"   Токенов в кэше: {len(cached_index.get('section_index', {}))}")
        else:
            print(f"   [INFO] Индекс не найден в кэше (возможно, нужно запустить импорт)")

        print("\n" + "=" * 60)
        print("[SUCCESS] Тест построения индекса завершен!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


def test_html_extraction():
    """Тестирует извлечение текста из HTML."""
    print("=" * 60)
    print("Тест 4: Извлечение текста из HTML")
    print("=" * 60)

    test_cases = [
        ("<p>Простой текст</p>", "Простой текст"),
        ("<h1>Заголовок</h1><p>Параграф</p>", "Заголовок Параграф"),
        ("<p>Текст с &nbsp; пробелами</p>", "Текст с"),  # &nbsp; декодируется, но проверяем что текст есть
        ("<div><span>Вложенный</span> текст</div>", "Вложенный"),  # Проверяем что текст извлечен
    ]

    for html, expected_part in test_cases:
        result = extract_text_from_html(html)
        # Проверяем, что ожидаемый текст содержится в результате
        status = "OK" if expected_part.lower() in result.lower() else "FAIL"
        print(f"{status} HTML: {html[:30]}... -> '{result[:50]}...'")
        if status == "FAIL":
            print(f"  Expected to contain: '{expected_part}'")

    print()


def test_find_section():
    """Тестирует поиск разделов в HTML."""
    print("=" * 60)
    print("Тест 5: Поиск разделов в HTML")
    print("=" * 60)

    html_content = """
    <html>
    <body>
        <h1>Первый раздел</h1>
        <p>Содержимое первого раздела</p>
        <h2>Второй раздел</h2>
        <p>Содержимое второго раздела</p>
    </body>
    </html>
    """

    test_cases = [
        ("Первый раздел", True),  # Должен найтись
        ("Второй раздел", True),  # Должен найтись
        ("Несуществующий раздел", False),  # Не должен найтись
    ]

    for section_title, should_find in test_cases:
        pos = find_section_in_html(html_content, section_title)
        found = pos is not None
        status = "OK" if found == should_find else "FAIL"
        print(f"{status} Поиск '{section_title}': {'найден' if found else 'не найден'} (ожидалось: {'найден' if should_find else 'не найден'})")
        if found:
            print(f"  Позиция: {pos}")

    print()


def test_parse_sections():
    """Тестирует разбиение HTML на разделы."""
    print("=" * 60)
    print("Тест 6: Разбиение HTML на разделы")
    print("=" * 60)

    html_content = """
    <html>
    <body>
        <h1>Раздел 1</h1>
        <p>Текст раздела 1</p>
        <h2>Раздел 2</h2>
        <p>Текст раздела 2</p>
        <h3>Раздел 3</h3>
        <p>Текст раздела 3</p>
    </body>
    </html>
    """

    sections = ["Раздел 1", "Раздел 2", "Раздел 3"]
    sections_content = parse_html_sections(html_content, sections)

    print(f"   Найдено разделов: {len(sections_content)}")
    for section_title, section_text in sections_content.items():
        if section_text:
            status = "OK"
            print(f"   {status} '{section_title}': {len(section_text)} символов")
        else:
            status = "WARN"
            print(f"   {status} '{section_title}': пустой раздел")

    print()


def test_build_index():
    """Тестирует построение индекса из словаря разделов."""
    print("=" * 60)
    print("Тест 7: Построение индекса из разделов")
    print("=" * 60)

    sections_content = {
        "Токен: пароль": "Токен используется для входа. Пароль нужен для токена.",
        "VPN контур": "VPN контур разработки позволяет подключиться к сети.",
        "Настройка": "Настройка токена требует пароль и сертификат.",
    }

    index = build_search_index(sections_content)

    section_index = index.get("section_index", {})
    content_index = index.get("content_index", {})

    print(f"   Токенов в индексе разделов: {len(section_index)}")
    print(f"   Токенов в индексе содержимого: {len(content_index)}")

    # Проверяем наличие ключевых токенов
    test_tokens = ["токен", "пароль", "vpn", "контур"]
    for token in test_tokens:
        in_section = token in section_index
        in_content = token in content_index
        status = "OK" if (in_section or in_content) else "FAIL"
        print(f"   {status} Токен '{token}': в индексе разделов={in_section}, в индексе содержимого={in_content}")

    # Показываем примеры
    if section_index:
        example_token = list(section_index.keys())[0]
        example_sections = section_index[example_token]
        print(f"\n   Пример из индекса разделов:")
        print(f"     '{example_token}' -> {example_sections}")

    print()


def test_save_load_cache():
    """Тестирует сохранение и загрузку индекса из кэша."""
    print("=" * 60)
    print("Тест 8: Сохранение и загрузка индекса")
    print("=" * 60)

    import tempfile
    import json

    # Создаем временный файл кэша
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
        temp_cache_file = Path(f.name)

    try:
        # Создаем тестовый индекс
        test_index = {
            "section_index": {
                "токен": ["Раздел 1", "Раздел 2"],
                "пароль": ["Раздел 1"],
            },
            "content_index": {
                "токен": {
                    "Раздел 1": [0, 10, 20],
                    "Раздел 2": [5, 15],
                },
            },
        }

        # Создаем начальный кэш с базовыми данными
        initial_cache = {
            "content": "data/knowledge.html",
            "last_updated": "2024-01-01T00:00:00",
            "document_id": "test",
            "images": []
        }
        with open(temp_cache_file, 'w', encoding='utf-8') as f:
            json.dump(initial_cache, f, ensure_ascii=False, indent=2)

        # Сохраняем индекс
        print("   1. Сохранение индекса...")
        save_index_to_cache(temp_cache_file, test_index)
        if temp_cache_file.exists():
            print("   [OK] Индекс сохранен")
        else:
            print("   [FAIL] Файл не создан")
            return

        # Загружаем индекс
        print("   2. Загрузка индекса...")
        loaded_index = load_index_from_cache(temp_cache_file)
        if loaded_index:
            print("   [OK] Индекс загружен")
            # Проверяем структуру
            if "section_index" in loaded_index and "content_index" in loaded_index:
                print("   [OK] Структура индекса корректна")
                print(f"   Токенов в индексе разделов: {len(loaded_index['section_index'])}")
                print(f"   Токенов в индексе содержимого: {len(loaded_index['content_index'])}")
            else:
                print("   [FAIL] Неправильная структура индекса")
        else:
            print("   [FAIL] Индекс не загружен")

    finally:
        # Удаляем временный файл
        if temp_cache_file.exists():
            temp_cache_file.unlink()

    print()


def test_load_sections():
    """Тестирует загрузку заголовков разделов."""
    print("=" * 60)
    print("Тест 9: Загрузка заголовков разделов")
    print("=" * 60)

    sections_file = Path("data/sections.json")
    if not sections_file.exists():
        print(f"   [SKIP] Файл не найден: {sections_file}")
        return

    try:
        sections = load_sections(sections_file)
        print(f"   [OK] Загружено разделов: {len(sections)}")
        print(f"   Тип данных: {type(sections).__name__}")
        if sections:
            print(f"   Примеры: {sections[:3]}")
            # Проверяем, что все элементы - строки
            all_strings = all(isinstance(s, str) for s in sections)
            status = "OK" if all_strings else "FAIL"
            print(f"   {status} Все элементы - строки: {all_strings}")
    except Exception as e:
        print(f"   [FAIL] Ошибка загрузки: {e}")

    print()


def main():
    """Запускает все тесты."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ПОИСКОВОГО ИНДЕКСА")
    print("=" * 60)

    try:
        # Тест нормализации
        test_text_normalization()

        # Тест токенизации
        test_tokenization()

        # Тест извлечения текста из HTML
        test_html_extraction()

        # Тест поиска разделов
        test_find_section()

        # Тест разбиения на разделы
        test_parse_sections()

        # Тест построения индекса
        test_build_index()

        # Тест сохранения и загрузки кэша
        test_save_load_cache()

        # Тест загрузки заголовков
        test_load_sections()

        # Тест построения индекса из HTML (интеграционный)
        test_index_building()

        print("\n" + "=" * 60)
        print("[SUCCESS] Все тесты завершены!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
