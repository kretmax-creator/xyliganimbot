"""
Комплексный скрипт для настройки модели E5:
1. Загрузка модели intfloat/multilingual-e5-small
2. Пересоздание embeddings кэша с новой моделью
3. Автоматическая проверка работы поиска
"""

import sys
import os
import json
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')  # UTF-8
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Скрипт может запускаться из корня: python testing/test_e5_setup.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.logging import setup_logging, get_logger
from src.model_loader import download_model
from src.search import vectorize_content, load_embeddings_from_cache, load_embedding_model, semantic_search

setup_logging(
    level="INFO",
    log_file="logs/app.log",
    audit_file="logs/audit.log",
    log_user_messages=False,
    max_bytes=10485760,
    backup_count=5,
)

logger = get_logger(__name__)


def step1_download_model():
    """Шаг 1: Загрузка модели E5."""
    print("\n" + "=" * 60)
    print("ШАГ 1: Загрузка модели intfloat/multilingual-e5-small")
    print("=" * 60)
    
    model_name = "intfloat/multilingual-e5-small"
    models_dir = project_root / "models"
    
    print(f"Модель: {model_name}")
    print(f"Директория: {models_dir}")
    print("\n[INFO] Загрузка может занять несколько минут...")
    
    success = download_model(model_name=model_name, models_dir=models_dir)
    
    if success:
        print("[OK] Модель успешно загружена!")
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"   - Путь: {model_path}")
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            print(f"   - Размер: {total_size / (1024*1024):.2f} MB")
        return True
    else:
        print("[ERROR] Ошибка при загрузке модели")
        return False


def step2_recreate_embeddings():
    """Шаг 2: Пересоздание embeddings кэша с новой моделью."""
    print("\n" + "=" * 60)
    print("ШАГ 2: Пересоздание embeddings кэша с моделью E5")
    print("=" * 60)
    
    markdown_file = project_root / "data" / "knowledge.md"
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    
    cache_file = project_root / "data" / "knowledge_cache.json"
    
    if not markdown_file.exists():
        print(f"[ERROR] Markdown/HTML файл не найден: {markdown_file}")
        print("   Сначала выполните: python test_import_content.py")
        return False
    
    print(f"Markdown/HTML файл: {markdown_file}")
    print(f"Кэш: {cache_file}")
    
    # Делаем резервную копию старого кэша, если он существует
    if cache_file.exists():
        backup_file = cache_file.with_suffix('.json.backup')
        print(f"[INFO] Создание резервной копии старого кэша: {backup_file}")
        import shutil
        shutil.copy2(cache_file, backup_file)
    
    print("\n[INFO] Векторизация может занять некоторое время...")
    
    success = vectorize_content(
        markdown_file=markdown_file,
        cache_file=cache_file,
        model_name="intfloat/multilingual-e5-small"
    )
    
    if success:
        print("[OK] Embeddings успешно пересозданы!")
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            embeddings_data = cache_data.get('embeddings')
            if embeddings_data:
                embeddings = embeddings_data.get('embeddings', [])
                section_titles = embeddings_data.get('section_titles', [])
                print(f"   - Разделов векторизовано: {len(section_titles)}")
                if embeddings:
                    print(f"   - Размерность векторов: {len(embeddings[0])}")
                    print(f"   - Всего векторов: {len(embeddings)}")
        
        return True
    else:
        print("[ERROR] Ошибка при пересоздании embeddings")
        return False


def step3_test_search():
    """Шаг 3: Автоматическая проверка работы поиска."""
    print("\n" + "=" * 60)
    print("ШАГ 3: Автоматическая проверка работы поиска")
    print("=" * 60)
    
    cache_file = project_root / "data" / "knowledge_cache.json"
    markdown_file = project_root / "data" / "knowledge.md"
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"
    
    if not cache_file.exists():
        print(f"[ERROR] Кэш не найден: {cache_file}")
        return False
    
    if not markdown_file.exists():
        print(f"[ERROR] Markdown/HTML файл не найден: {markdown_file}")
        return False
    
    # Загружаем embeddings
    print("[INFO] Загрузка embeddings из кэша...")
    embeddings_data = load_embeddings_from_cache(cache_file)
    
    if not embeddings_data:
        print("[ERROR] Embeddings не найдены в кэше")
        return False
    
    print(f"[OK] Embeddings загружены:")
    print(f"   - Разделов: {len(embeddings_data.get('section_titles', []))}")
    
    # Загружаем модель
    print("\n[INFO] Загрузка embedding-модели...")
    model = load_embedding_model()
    
    if not model:
        print("[ERROR] Модель не найдена")
        return False
    
    print("[OK] Модель загружена")
    
    # Тестовые запросы для проверки
    test_queries = [
        ("VPN", "Должен найти раздел про VPN"),
        ("Как настроить Outlook?", "Должен найти раздел про Outlook"),
        ("токен", "Должен найти раздел про токен"),
        ("!!!", "Должен быть отфильтрован (только символы)"),
        ("7.1 Outlook после смены пароля ВРМ", "Точное совпадение"),
    ]
    
    print("\n[INFO] Выполнение тестовых запросов...")
    all_passed = True
    
    for query, description in test_queries:
        print(f"\n--- Запрос: '{query}' ({description}) ---")
        
        results = semantic_search(
            query=query,
            embeddings_data=embeddings_data,
            model=model,
            limit=3,
            min_score=0.25
        )
        
        if results:
            print(f"[OK] Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                title = result.get('section_title', 'Без названия')
                print(f"   {i}. {title} (score: {score:.3f})")
        else:
            if query == "!!!":
                print("[OK] Запрос корректно отфильтрован (только символы)")
            else:
                print("[WARNING] Результаты не найдены")
                all_passed = False
    
    return all_passed


def main():
    """Основная функция."""
    print("\n" + "=" * 60)
    print("НАСТРОЙКА МОДЕЛИ E5 И ПЕРЕСОЗДАНИЕ EMBEDDINGS КЭША")
    print("=" * 60)
    
    results = []
    
    # Шаг 1: Загрузка модели
    results.append(("Загрузка модели E5", step1_download_model()))
    
    # Шаг 2: Пересоздание embeddings (только если модель загружена)
    if results[0][1]:
        results.append(("Пересоздание embeddings", step2_recreate_embeddings()))
    else:
        print("\n[SKIP] Пропуск пересоздания embeddings: модель не загружена")
        results.append(("Пересоздание embeddings", None))
    
    # Шаг 3: Тестирование поиска (только если embeddings пересозданы)
    if len(results) > 1 and results[1][1]:
        results.append(("Тестирование поиска", step3_test_search()))
    else:
        print("\n[SKIP] Пропуск тестирования: embeddings не пересозданы")
        results.append(("Тестирование поиска", None))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    
    for step_name, result in results:
        if result is None:
            status = "[SKIP] ПРОПУЩЕН"
        elif result:
            status = "[OK] УСПЕШНО"
        else:
            status = "[ERROR] ОШИБКА"
        print(f"{status} - {step_name}")
    
    # Общий результат
    successful = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    print(f"\nВсего шагов: {total}, Успешно: {successful}, Ошибок: {total - successful}")
    
    if successful == total and total > 0:
        print("\n[SUCCESS] Все шаги выполнены успешно!")
        print("\n[INFO] Теперь можно протестировать поиск вручную через:")
        print("   python test_semantic_search.py")
        return 0
    else:
        print("\n[WARNING] Некоторые шаги завершились с ошибками")
        return 1


if __name__ == "__main__":
    sys.exit(main())
