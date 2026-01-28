# -*- coding: utf-8 -*-
"""
Автотесты для вспомогательных функций обработчиков (strip_bot_mention и т.д.).

Запуск: python -m unittest testing.test_messages_helpers (из корня)
"""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.handlers.messages import strip_bot_mention


class TestStripBotMention(unittest.TestCase):
    """Тесты strip_bot_mention."""

    def test_removes_mention_at_start(self):
        self.assertEqual(
            strip_bot_mention("@xyliganim_bot как настроить vpn", "xyliganim_bot"),
            "как настроить vpn",
        )

    def test_removes_mention_in_middle(self):
        # После удаления @username может остаться двойной пробел — поиск это допускает
        result = strip_bot_mention("подскажи @xyliganim_bot настройка outlook", "xyliganim_bot")
        self.assertIn("подскажи", result)
        self.assertIn("настройка outlook", result)
        self.assertNotIn("@xyliganim_bot", result)

    def test_removes_mention_case_insensitive(self):
        self.assertEqual(
            strip_bot_mention("@Xyliganim_Bot запрос", "xyliganim_bot"),
            "запрос",
        )

    def test_no_mention_returns_stripped(self):
        self.assertEqual(
            strip_bot_mention("  просто запрос  ", "xyliganim_bot"),
            "просто запрос",
        )

    def test_empty_bot_username_returns_stripped_text(self):
        self.assertEqual(
            strip_bot_mention("@xyliganim_bot текст", None),
            "@xyliganim_bot текст",
        )
        self.assertEqual(strip_bot_mention("  текст  ", ""), "текст")

    def test_empty_text_returns_empty(self):
        self.assertEqual(strip_bot_mention("", "xyliganim_bot"), "")
        self.assertEqual(strip_bot_mention("   ", "xyliganim_bot"), "")


if __name__ == "__main__":
    unittest.main()
