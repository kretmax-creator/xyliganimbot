"""
Тест семантического поиска.
"""

import sys
import os
import json
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')  # UTF-8
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Скрипт может запускаться из корня: python testing/test_semantic_search.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.logging import setup_logging, get_logger
from src.search import (
    load_embeddings_from_cache,
    load_embedding_model,
    semantic_search,
    search,
)

setup_logging(level="INFO", log_file="logs/app.log", audit_file="logs/audit.log", 
              log_user_messages=False, max_bytes=10485760, backup_count=5)

logger = get_logger(__name__)


def main():
    print("=" * 60)
    print("ТЕСТ: Семантический поиск")
    print("=" * 60)
    
    cache_file = project_root / "data" / "knowledge_cache.json"
    markdown_file = project_root / "data" / "knowledge.md"
    # Проверяем также старый формат HTML для обратной совместимости
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    
    # Проверяем наличие файлов
    if not cache_file.exists():
        print(f"[ERROR] Кэш не найден: {cache_file}")
        print("   Сначала выполните: python test_vectorize.py")
        return 1
    
    if not markdown_file.exists():
        print(f"[ERROR] Markdown/HTML файл не найден: {markdown_file}")
        return 1
    
    print(f"Кэш: {cache_file}")
    print(f"Markdown/HTML файл: {markdown_file}")
    
    # Загружаем embeddings
    print("\n[INFO] Загрузка embeddings из кэша...")
    embeddings_data = load_embeddings_from_cache(cache_file)
    
    if not embeddings_data:
        print("[ERROR] Embeddings не найдены в кэше")
        print("   Сначала выполните: python test_vectorize.py")
        return 1
    
    print(f"[OK] Embeddings загружены:")
    print(f"   - Разделов: {len(embeddings_data.get('section_titles', []))}")
    print(f"   - Векторов: {len(embeddings_data.get('embeddings', []))}")
    
    # Загружаем модель
    print("\n[INFO] Загрузка embedding-модели...")
    model = load_embedding_model()
    
    if not model:
        print("[ERROR] Модель не найдена")
        print("   Сначала выполните: python test_download_model.py")
        return 1
    
    print("[OK] Модель загружена")
    
    # Тестовые запросы из плана тестирования (docs/TEST_PLAN.md)
    test_queries = [
        # 1.1 Базовые запросы
        "7.1 Outlook после смены пароля ВРМ",            # TC-1.1.1 (Точное совпадение)
        "Как настроить Outlook?",                        # TC-1.1.2 (Синоним)
        "сменить заводской PIN",                         # TC-1.1.3 (По содержимому)
        
        # 1.2 Сложные запросы
        "Как настроить токен для входа в систему?",      # TC-1.2.1
        "настройка VPN, но не Иннотех",                  # TC-1.2.2
        "How to configure VPN?",                         # TC-1.2.3
        
        # 1.3 Граничные случаи
        "VPN",                                           # TC-1.3.1
        "Мне нужно настроить VPN подключение для доступа к внутренним ресурсам банка, но я не знаю какой сертификат выбрать и какой адрес сервера использовать", # TC-1.3.2
        "ВПН",                                           # TC-1.3.3 (опечатка)
        "Как настроить VPN?",                            # TC-1.3.4 (спецсимволы)
        
        # 1.6 Синонимы и сленг
        "Флешка заблочилась, что делать?",               # TC-1.6.1
        "Не работает видео в Дионе",                     # TC-1.6.2
        "Как попасть на удаленку?",                      # TC-1.6.3
        "Учетка заблочилась",                            # TC-1.6.4 (вместо "Комп висит")
        
        # 1.7 Причинно-следственные связи
        "Почему меня заблокировали в девкорпе?",         # TC-1.7.1
        "Зачем нужна виртуалка?",                        # TC-1.7.2
        "Что будет, если я поставлю DLP на свой ноутбук?", # TC-1.7.3
        "Сакура ругается на связь с сервером",           # TC-1.7.4
        
        # 1.8 Точный поиск сущностей
        "Какой адрес у шлюза для разработки?",           # TC-1.8.1
        "Номер телефона поддержки Иннотеха",             # TC-1.8.2
        "Какой заводской пароль на токене?",             # TC-1.8.3
        "Кому писать по проблемам с DLP?",               # TC-1.8.4
        
        # 1.9 Процедурные знания
        "Как настроить аутлук?",                         # TC-1.9.1
        "Продление учетки Иннотех",                      # TC-1.9.2
        "Настройки VPN",                                 # TC-1.9.3
        "Можно отправить письмо на яндекс почту?",       # TC-1.9.4
    ]
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ПОИСКА")
    print("=" * 60)
    
    all_passed = True
    
    for query in test_queries:
        print(f"\n--- Запрос: '{query}' ---")
        
        # Тест 1: Прямой вызов semantic_search()
        print("[TEST] Прямой вызов semantic_search()...")
        results = semantic_search(
            query=query,
            embeddings_data=embeddings_data,
            model=model,
            limit=3,
            min_score=0.25  # Обновлен порог для модели E5
        )
        
        if results:
            print(f"[OK] Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                title = result.get('section_title', 'Без названия')
                text = result.get('text', '')
                text_preview = text[:100] + '...' if len(text) > 100 else text
                print(f"   {i}. {title} (score: {score:.3f})")
                if text_preview:
                    try:
                        print(f"      {text_preview}")
                    except UnicodeEncodeError:
                        print(f"      [текст раздела]")
        else:
            print("[WARNING] Результаты не найдены")
            all_passed = False
        
        # Тест 2: Вызов через search() (универсальная функция)
        print("\n[TEST] Вызов через search() (универсальная функция)...")
        results_unified = search(
            query=query,
            index=embeddings_data,
            markdown_file=markdown_file,
            limit=3,
        )
        
        if results_unified:
            print(f"[OK] Найдено результатов: {len(results_unified)}")
            for i, result in enumerate(results_unified, 1):
                score = result.get('score', result.get('relevance_score', 0))
                title = result.get('section_title', 'Без названия')
                print(f"   {i}. {title} (score: {score:.3f})")
        else:
            print("[WARNING] Результаты не найдены")
            all_passed = False
    
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    if all_passed:
        print("[SUCCESS] Все тесты пройдены успешно!")
        return 0
    else:
        print("[WARNING] Некоторые тесты завершились с предупреждениями")
        return 1


if __name__ == "__main__":
    sys.exit(main())
