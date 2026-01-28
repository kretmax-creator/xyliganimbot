# -*- coding: utf-8 -*-
"""
Автотесты по testing/COMPREHENSIVE_TEST_SUITE.md.

Запускает search() для всех тесткейсов с типом Auto и проверяет ожидаемый результат.
Запуск: python testing/test_comprehensive_suite.py (из корня проекта)
Требует: data/knowledge_cache.json, data/knowledge.md, models/ с E5
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Скрипт может запускаться из корня: python testing/test_comprehensive_suite.py
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.search import search, load_embeddings_from_cache
from src.logging import setup_logging

setup_logging(level="WARNING", log_file="logs/app.log", audit_file="logs/audit.log", log_user_messages=False)


def load_index_and_markdown():
    cache_file = project_root / "data" / "knowledge_cache.json"
    md = project_root / "data" / "knowledge.md"
    if not md.exists():
        md = project_root / "data" / "knowledge.html"
    if not cache_file.exists():
        raise FileNotFoundError(f"Кэш не найден: {cache_file}")
    if not md.exists():
        raise FileNotFoundError(f"Markdown не найден: {md}")
    index = load_embeddings_from_cache(cache_file)
    if not index:
        raise ValueError("Кэш пуст или не загружен")
    emb = index.get("embeddings")
    if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
        raise ValueError("Embeddings не найдены в кэше")
    return index, md


def _combined(r: List[Dict], idx: int = 0) -> str:
    if not r or idx >= len(r):
        return ""
    t = r[idx]
    return ((t.get("section_title") or "") + " " + (t.get("text") or "")).lower()


def _any_contains(results: List[Dict], phrases: List[str], top_n: int = 5) -> bool:
    phrases_l = [p.lower() for p in phrases]
    for i in range(min(top_n, len(results))):
        s = _combined(results, i)
        if any(p in s for p in phrases_l):
            return True
    return False


def _first_contains(results: List[Dict], phrases: List[str]) -> bool:
    return len(results) >= 1 and _any_contains(results[:1], phrases, 1)


# Тесткейсы из COMPREHENSIVE_TEST_SUITE.md (тип Auto)
# (id, query, check_fn, priority)
SUITE = [
    # 1.1 Базовая релевантность
    ("TC-1.1.1", "7.1 Outlook после смены пароля ВРМ", lambda r: len(r) >= 1 and _first_contains(r, ["outlook", "7.1"]) and (r[0].get("score") or 0) > 0.8, "Critical"),
    ("TC-1.1.2", "Как настроить Outlook?", lambda r: _any_contains(r, ["outlook", "7.1"], 3), "Critical"),
    ("TC-1.1.3", "сменить заводской PIN", lambda r: _first_contains(r, ["3.1", "токен", "1234567890", "pin"]), "High"),
    ("TC-1.1.4", "обновление сертификатов", lambda r: _any_contains(r, ["сертификат", "3.4", "vpn"], 3), "High"),
    ("TC-1.1.5", "неверный пароль", lambda r: _any_contains(r, ["troubleshooting", "7.", "6.1", "пароль", "смена"]), "High"),
    ("TC-1.1.6", "смена пароля от УЗ", lambda r: _any_contains(r, ["6.", "учетн", "пароль", "смена"]), "High"),
    # 1.2 Синонимы и сленг
    ("TC-1.2.1", "Флешка заблочилась, что делать?", lambda r: _any_contains(r, ["3.1", "токен", "troubleshooting"]), "High"),
    ("TC-1.2.2", "Не работает видео в Дионе", lambda r: _any_contains(r, ["dion", "дион", "8.1", "сетев"]), "Medium"),
    ("TC-1.2.3", "Как попасть на удаленку?", lambda r: _any_contains(r, ["vpn", "врп", "контур", "3.4", "1.1"]), "High"),
    ("TC-1.2.4", "Учетка заблочилась", lambda r: _any_contains(r, ["2.1", "обязательн", "troubleshooting", "блокир"]), "High"),
    ("TC-1.2.5", "Протух пароль", lambda r: _any_contains(r, ["6.1", "пароль", "смена"]), "Medium"),
    ("TC-1.2.6", "пароль от компа", lambda r: _any_contains(r, ["учетн", "пароль", "windows", "врп"]), "Medium"),
    # 1.3 Сложные запросы
    ("TC-1.3.1", "Как настроить токен для входа в систему?", lambda r: _any_contains(r, ["3.1", "3.2", "3.3", "токен"]), "Medium"),
    ("TC-1.3.2", "Мне нужно настроить VPN подключение для доступа к внутренним ресурсам банка, но я не знаю какой сертификат выбрать и какой адрес сервера использовать", lambda r: _any_contains(r, ["3.4", "3.5", "vpn"]) and len(r) >= 1, "Medium"),
    ("TC-1.3.3", "Почему меня заблокировали в девкорпе?", lambda r: _any_contains(r, ["блокир", "nac", "сакура", "2.1"]), "Medium"),
    ("TC-1.3.4", "Сакура ругается на связь с сервером", lambda r: _any_contains(r, ["2.1", "обязательн", "troubleshooting", "сакура"]), "Medium"),
    # 1.4 Отрицания
    ("TC-1.4.1", "настройка VPN, но не Иннотех", lambda r: not _any_contains(r, ["иннотех"], 5) and _any_contains(r, ["vpn"]), "Low"),
    ("TC-1.4.2", "антивирус без установки на ноутбук", lambda r: _any_contains(r, ["установк", "по", "правил", "обязательн", "7.", "troubleshooting"]), "Low"),
    # 1.5 Точный поиск фактов
    ("TC-1.5.1", "Какой адрес у шлюза для разработки?", lambda r: _any_contains(r, ["ext.vpn.vtb.ru", "ext.vpn"]), "High"),
    ("TC-1.5.2", "Номер телефона поддержки Иннотеха", lambda r: _any_contains(r, ["9.", "поддержка", "8-800", "8 800"]), "High"),
    ("TC-1.5.3", "Какой заводской пароль на токене?", lambda r: _any_contains(r, ["3.1", "1234567890", "pin", "токен"]), "High"),
    ("TC-1.5.4", "Кому писать по проблемам с DLP?", lambda r: _any_contains(r, ["security@phoenixit", "поддержка", "dlp", "9."]), "Medium"),
    # 2.1 Опечатки и транслитерация
    ("TC-2.1.1", "ВПН", lambda r: _any_contains(r, ["vpn"]), "High"),
    ("TC-2.1.2", "Аутлук", lambda r: _any_contains(r, ["outlook"]), "High"),
    ("TC-2.1.3", "рутокен", lambda r: _any_contains(r, ["rutoken", "токен", "3.1"]), "High"),
    ("TC-2.1.4", "иннатех", lambda r: _any_contains(r, ["иннотех", "4.3", "6.2"]), "Medium"),
    # 2.2 Фильтрация нерелевантного
    ("TC-2.2.1", "как приготовить пиццу", lambda r: len(r) == 0, "High"),
    ("TC-2.2.2", "абсолютно нерелевантный запрос xyz123", lambda r: len(r) == 0, "High"),
    ("TC-2.2.3", "!!!", lambda r: len(r) == 0, "Medium"),
    ("TC-2.2.4", "Можно отправить письмо на яндекс почту?", lambda r: len(r) == 0 or (r and r[0].get("score", 0) < 0.75), "Medium"),
    # 2.3 Технические ограничения
    ("TC-2.3.1", "VPN", lambda r: len(r) >= 1 and _any_contains(r, ["vpn"]), "High"),
    # 3.1 Форматирование — проверяем только что search возвращает данные для snippet/full
    ("TC-3.1.2", "3.1 Первоначальная настройка токена", lambda r: len(r) >= 1 and (r[0].get("text") or "").count("...") >= 0, "Medium"),
    ("TC-3.1.3", "9. Поддержка", lambda r: len(r) >= 1 and len((r[0].get("text") or "")) > 0, "Medium"),
    # Раздел 5: User stories
    ("TC-5.1", "токен перестал работать после обновления", lambda r: _any_contains(r, ["токен", "обновлен", "troubleshooting"]), "—"),
    ("TC-5.2", "где можно поменять пароль от домена test.vtb.ru?", lambda r: _any_contains(r, ["пароль", "6.1", "6.2", "смен"]), "—"),
    ("TC-5.3", "нужно все сертификаты автоматически обновить", lambda r: _any_contains(r, ["сертификат", "3.4"]), "—"),
    ("TC-5.4", "учетка заблочена", lambda r: _any_contains(r, ["блокир", "2.1", "6.2"]), "—"),
    ("TC-5.5", "не могу в джире авторизоваться", lambda r: _any_contains(r, ["jira", "confluence", "vpn", "доступ"]), "—"),
    ("TC-5.6", "Сакура ругается", lambda r: _any_contains(r, ["сакура", "2.1", "troubleshooting"]), "—"),
    ("TC-5.7", "забыл пин код токена", lambda r: _any_contains(r, ["3.1", "pin", "токен", "поддержка", "9."]), "—"),
]


def run_suite() -> Dict[str, Any]:
    lines = []
    def out(s=""):
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode("ascii"))

    out("=" * 70)
    out("Автотесты по testing/COMPREHENSIVE_TEST_SUITE.md")
    out("=" * 70)

    try:
        index, markdown_file = load_index_and_markdown()
    except (FileNotFoundError, ValueError) as e:
        out(f"[SKIP] {e}")
        return {"passed": 0, "failed": 0, "total": 0, "details": [], "output": "\n".join(lines)}

    out(f"Индекс: {len(index.get('section_titles', []))} разделов")
    out("")

    passed = failed = 0
    details = []
    by_priority = {"Critical": [], "High": [], "Medium": [], "Low": [], "—": []}

    for tc_id, query, check_fn, prio in SUITE:
        try:
            results = search(query=query, index=index, markdown_file=markdown_file, limit=5)
            ok = check_fn(results)
            if ok:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
            first_score = results[0].get("score") if results else None
            first_title = (results[0].get("section_title") or "")[:45] if results else ""
            details.append({"id": tc_id, "query": query[:40], "status": status, "score": first_score, "title": first_title})
            by_priority.setdefault(prio, []).append((tc_id, status))
            out(f"  [{status}] {tc_id}: {query[:45]}")
            if not ok and results:
                out(f"       first_score={first_score}, first_title={first_title!r}")
        except Exception as e:
            failed += 1
            details.append({"id": tc_id, "query": query[:40], "status": "ERROR", "error": str(e)})
            by_priority.setdefault(prio, []).append((tc_id, "ERROR"))
            out(f"  [ERROR] {tc_id}: {repr(e)}")

    total = passed + failed
    pct = (100 * passed / total) if total else 0
    out("")
    out("Итог:")
    out(f"  Пройдено: {passed}/{total} ({pct:.1f}%)")
    out(f"  Провалено: {failed}")
    out("")
    for prio in ["Critical", "High", "Medium", "Low", "—"]:
        items = by_priority.get(prio, [])
        if not items:
            continue
        p_ok = sum(1 for _, s in items if s == "PASS")
        out(f"  По приоритету {prio}: {p_ok}/{len(items)} PASS")

    report_path = Path(__file__).resolve().parent / "COMPREHENSIVE_TEST_REPORT.txt"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return {"passed": passed, "failed": failed, "total": total, "pct": pct, "details": details, "output": "\n".join(lines)}


if __name__ == "__main__":
    report = run_suite()
    sys.exit(0 if report.get("failed", 0) == 0 else 1)
