# -*- coding: utf-8 -*-
"""
Автоматизированные тесты Фазы 3 итерации 8.2.

Проверяет проблемные тесткейсы из Фазы 1 и Фазы 2:
- Фаза 1: адаптивный порог, абсолютный порог, нерелевантные запросы -> пустой список
- Фаза 2: отрицания, boost точных совпадений, запросы вне контекста

Запуск: python test_phase3_search.py
Требует: data/knowledge_cache.json, data/knowledge.md, models/ с E5
"""

import os
import sys
from pathlib import Path

# UTF-8 на Windows для вывода
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Скрипт может запускаться из корня: python testing/test_phase3_search.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.search import search, load_embeddings_from_cache
from src.logging import setup_logging, get_logger

setup_logging(
    level="WARNING",
    log_file="logs/app.log",
    audit_file="logs/audit.log",
    log_user_messages=False,
)
logger = get_logger(__name__)


def load_index_and_markdown():
    """Загружает индекс и путь к markdown."""
    cache_file = project_root / "data" / "knowledge_cache.json"
    markdown_file = project_root / "data" / "knowledge.md"
    if not markdown_file.exists():
        markdown_file = project_root / "data" / "knowledge.html"

    if not cache_file.exists():
        raise FileNotFoundError(f"Кэш не найден: {cache_file}")
    if not markdown_file.exists():
        raise FileNotFoundError(f"Markdown не найден: {markdown_file}")

    index = load_embeddings_from_cache(cache_file)
    if not index:
        raise ValueError("Кэш пуст или не загружен")
    emb = index.get("embeddings")
    if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
        raise ValueError("Embeddings не найдены в кэше")

    return index, markdown_file


# Проблемные тесткейсы Фазы 1 и Фазы 2 (критерии из tasklist и QA_ANALYSIS_E5)
PHASE3_TEST_CASES = [
    # (id, query, check_fn, description)
    (
        "TC-1.4.2/TC-2.2.2",
        "абсолютно нерелевантный запрос xyz123",
        lambda r: len(r) == 0,
        "Nerelevantnyj zapros -> pustoj spisok",
    ),
    (
        "TC-1.5.2",
        "как приготовить пиццу",
        lambda r: len(r) == 0,
        "Zapros vne konteksta -> pustoj spisok",
    ),
    (
        "TC-2.2.2",
        "вфапфвапфпавапф",
        lambda r: len(r) == 0,
        "Bessmyslennyj zapros -> pustoj spisok",
    ),
    (
        "TC-1.2.2",
        "настройка VPN, но не Иннотех",
        lambda r: not any("иннотех" in (x.get("section_title") or "").lower() for x in r),
        "Otricanie: net razдела Innoteh v rezultatakh",
    ),
    (
        "TC-1.8.4",
        "Кому писать по проблемам с DLP?",
        lambda r: len(r) >= 1
        and any(
            "поддержка" in (x.get("section_title") or "").lower()
            for x in r[:2]
        ),
        "Razdel 9. Podderzhka v top-2",
    ),
    (
        "TC-1.9.4",
        "Можно отправить письмо на яндекс почту?",
        lambda r: len(r) == 0 or (r and r[0].get("score", 0) < 0.75),
        "Vne konteksta -> pustoj ili nizkij score",
    ),
    # Позитивные: не сломали базовый поиск
    (
        "TC-1.1.1",
        "7.1 Outlook после смены пароля ВРМ",
        lambda r: len(r) >= 1 and "outlook" in (r[0].get("section_title") or "").lower(),
        "Tochnoe sovpadenie Outlook v pervom rezultate",
    ),
    (
        "TC-1.1.3",
        "сменить заводской PIN",
        lambda r: len(r) >= 1
        and ("3.1" in (r[0].get("section_title") or "") or "токен" in (r[0].get("section_title") or "").lower()),
        "Zapros po soderzhimomu -> razdel 3.1 ili token v top",
    ),
    (
        "TC-1.3.1",
        "VPN",
        lambda r: len(r) >= 1 and r[0].get("score", 0) >= 0.6,
        "Korotkij zapros VPN -> est rezultaty, score >= 0.6",
    ),
]


def run_phase3_tests():
    """Запускает все тесты Фазы 3 и возвращает отчёт."""
    lines = []
    def out(s=""):
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode("ascii"))

    out("=" * 60)
    out("Фаза 3 итерации 8.2: автоматизированное тестирование поиска")
    out("=" * 60)

    try:
        index, markdown_file = load_index_and_markdown()
    except (FileNotFoundError, ValueError) as e:
        out(f"[SKIP] Не удалось загрузить данные: {e}")
        out("   Выполните: python test_vectorize.py (и при необходимости test_download_model.py)")
        return {"passed": 0, "failed": 0, "skipped": len(PHASE3_TEST_CASES), "details": [], "output": "\n".join(lines)}

    out(f"Индекс: {len(index.get('section_titles', []))} разделов")
    out(f"Markdown: {markdown_file}")
    out()

    passed = 0
    failed = 0
    details = []

    for tc_id, query, check_fn, description in PHASE3_TEST_CASES:
        try:
            results = search(
                query=query,
                index=index,
                markdown_file=markdown_file,
                limit=5,
            )
            ok = check_fn(results)
            if ok:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
            first_score = results[0].get("score", None) if results else None
            first_title = (results[0].get("section_title", "") or "")[:50] if results else ""
            details.append(
                {
                    "id": tc_id,
                    "query": query[:50],
                    "status": status,
                    "n_results": len(results),
                    "first_score": first_score,
                    "first_title": first_title,
                }
            )
            out(f"  [{status}] {tc_id}: {description}")
            if not ok and results:
                out(f"       n={len(results)}, first_score={first_score}, first_title={first_title!r}...")
        except Exception as e:
            failed += 1
            details.append({"id": tc_id, "query": query[:50], "status": "ERROR", "error": str(e)})
            out(f"  [ERROR] {tc_id}: {repr(e)}")

    total = passed + failed
    pct = (100 * passed / total) if total else 0
    out()
    out("Итог:")
    out(f"  Пройдено: {passed}/{total} ({pct:.0f}%)")
    out(f"  Провалено: {failed}")
    if pct >= 90:
        out("  Целевая метрика >90%: достигнута")
    else:
        out("  Целевая метрика >90%: не достигнута")

    # Сохраняем отчёт в файл (UTF-8) для документации
    report_path = Path(__file__).resolve().parent / "PHASE3_TEST_RESULTS.txt"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return {"passed": passed, "failed": failed, "total": total, "pct": pct, "details": details, "output": "\n".join(lines)}


if __name__ == "__main__":
    report = run_phase3_tests()
    sys.exit(0 if report.get("failed", 0) == 0 and report.get("passed", 0) > 0 else 1)
