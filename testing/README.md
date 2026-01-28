# Тестирование xyliganimbot

В этой папке собраны все материалы по тестированию: планы, тесткейсы, отчёты и скрипты автотестов.

## Запуск тестов

Из **корня проекта**:

```bash
python testing/test_comprehensive_suite.py   # полный набор по COMPREHENSIVE_TEST_SUITE (40 кейсов)
python testing/test_phase3_search.py         # фаза 3 итерации 8.2 (проблемные кейсы)
python testing/test_semantic_search.py       # семантический поиск
python testing/test_vectorize.py             # векторизация базы знаний
python testing/test_all_functions.py         # общий прогон функций
python testing/test_e5_setup.py              # проверка окружения E5
python testing/test_download_model.py        # загрузка модели
python testing/test_import_content.py        # импорт контента
```

Отчёты `COMPREHENSIVE_TEST_REPORT.txt` и `PHASE3_TEST_RESULTS.txt` создаются в этой папке.

## Содержимое папки

### Планы и тесткейсы

| Файл | Описание |
|------|----------|
| **TEST_PLAN.md** | План тестирования семантического поиска и ответов бота |
| **COMPREHENSIVE_TEST_SUITE.md** | Полный набор тесткейсов (Auto/Manual), источник для test_comprehensive_suite.py |
| **testcases_gemini.md** | Тесткейсы (в т.ч. от Gemini) |

### Отчёты

| Файл | Описание |
|------|----------|
| **COMPREHENSIVE_TEST_REPORT.md** / **.txt** | Отчёт автотестов по COMPREHENSIVE_TEST_SUITE |
| **PHASE3_TEST_RESULTS.txt** | Результаты фазы 3 (итерация 8.2) |
| **TEST_REPORT.md** | Отчёт о тестировании семантического поиска |
| **MANUAL_TEST_RESULTS.md** | Результаты ручного тестирования |
| **MANUAL_TEST_RESULTS_E5.md** | Результаты ручного тестирования с моделью E5 |
| **QA_ANALYSIS_E5.md** | Анализ качества (E5) |

### Решения и комментарии

| Файл | Описание |
|------|----------|
| **SOLUTIONS_FOR_FAILING_TESTS.md** | Варианты решений для проваленных тестов и что реализовано |

### Скрипты

| Скрипт | Назначение |
|--------|------------|
| **test_comprehensive_suite.py** | Автотесты по COMPREHENSIVE_TEST_SUITE (40 кейсов) |
| **test_phase3_search.py** | Тесты фазы 3 итерации 8.2 |
| **test_semantic_search.py** | Проверка семантического поиска |
| **test_vectorize.py** | Векторизация и кэш |
| **test_all_functions.py** | Общий прогон |
| **test_e5_setup.py** | Окружение и модель E5 |
| **test_download_model.py** | Загрузка модели |
| **test_import_content.py** | Импорт контента |

Требования: `data/knowledge_cache.json`, `data/knowledge.md` (или `.html`), модель E5 в `models/` (для семантических тестов).
