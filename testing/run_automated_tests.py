# -*- coding: utf-8 -*-
"""
Запуск автоматических тестов xyliganimbot.

Выполняет:
1. Юнит-тесты (audit, messages helpers) — всегда
2. Комплексный набор по поиску (test_comprehensive_suite) — если есть data/knowledge_cache.json и data/knowledge.md

Запуск из корня проекта: python testing/run_automated_tests.py
Выход: 0 — все тесты пройдены, 1 — есть провалы или ошибки.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Настраиваем логирование минимально для тестов
from src.logging import setup_logging
setup_logging(level="WARNING", log_file="logs/app.log", audit_file="logs/audit.log", log_user_messages=False)


def run_unittest_suites():
    """Запуск unittest: test_audit, test_messages_helpers."""
    import unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("testing.test_audit"))
    suite.addTests(loader.loadTestsFromName("testing.test_messages_helpers"))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result.wasSuccessful(), result.failures, result.errors


def run_comprehensive_suite():
    """Запуск test_comprehensive_suite (поиск). Возвращает (success, failed_count)."""
    cache_file = project_root / "data" / "knowledge_cache.json"
    md_file = project_root / "data" / "knowledge.md"
    if not md_file.exists():
        md_file = project_root / "data" / "knowledge.html"
    if not cache_file.exists() or not md_file.exists():
        print("\n[SKIP] test_comprehensive_suite: нет data/knowledge_cache.json или knowledge.md")
        return True, 0
    try:
        # Импорт из папки testing (скрипт не пакет)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_comprehensive_suite",
            project_root / "testing" / "test_comprehensive_suite.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        report = mod.run_suite()
        failed = report.get("failed", 0)
        passed = report.get("passed", 0)
        total = report.get("total", 0)
        print(f"\nКомплексный набор: {passed}/{total} пройдено, {failed} провалено")
        return failed == 0, failed
    except Exception as e:
        print(f"\n[ERROR] test_comprehensive_suite: {e}")
        return False, 1


def main():
    print("=" * 60)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ xyliganimbot")
    print("=" * 60)

    ok_unit, failures, errors = run_unittest_suites()
    unit_success = ok_unit and not failures and not errors

    ok_comp, comp_failed = run_comprehensive_suite()

    print("")
    print("=" * 60)
    if unit_success and ok_comp:
        print("ИТОГ: Все тесты пройдены.")
        return 0
    print("ИТОГ: Есть провалы или ошибки.")
    if not unit_success:
        print(f"  Юнит-тесты: провалов {len(failures)}, ошибок {len(errors)}")
    if not ok_comp:
        print(f"  Комплексный набор: провалено {comp_failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
