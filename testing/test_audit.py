# -*- coding: utf-8 -*-
"""
Автотесты для модуля аудита (итерации 10–11).

Проверяет src.audit.log_operation: формат записи, учёт include_request_text.
Запуск: python -m unittest testing.test_audit (из корня) или python testing/test_audit.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.audit import log_operation, MAX_REQUEST_LENGTH


class TestAuditLogOperation(unittest.TestCase):
    """Тесты log_operation."""

    @patch("src.audit._get_audit_logger")
    def test_log_operation_hides_request_when_include_false(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_operation(
            telegram_id=123,
            username="testuser",
            operation="search",
            result="found_3",
            request_text="секретный запрос",
            include_request_text=False,
        )
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("request=[hidden]", call_msg)
        self.assertIn("telegram_id=123", call_msg)
        self.assertIn("username=testuser", call_msg)
        self.assertIn("operation=search", call_msg)
        self.assertIn("result=found_3", call_msg)

    @patch("src.audit._get_audit_logger")
    def test_log_operation_includes_request_when_include_true(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_operation(
            telegram_id=456,
            username=None,
            operation="search",
            result="no_results",
            request_text="как настроить vpn",
            include_request_text=True,
        )
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("request=как настроить vpn", call_msg)
        self.assertIn("username=-", call_msg)

    @patch("src.audit._get_audit_logger")
    def test_log_operation_truncates_long_request(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        long_text = "x" * (MAX_REQUEST_LENGTH + 100)
        log_operation(
            telegram_id=789,
            username="u",
            operation="search",
            result="ok",
            request_text=long_text,
            include_request_text=True,
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("...", call_msg)
        self.assertLessEqual(len(call_msg), 1000)  # разумная длина строки

    @patch("src.audit._get_audit_logger")
    def test_log_operation_with_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_operation(
            telegram_id=111,
            username=None,
            operation="admin_load_model",
            result="error",
            request_text="/admin load_model",
            include_request_text=False,
            error="Network timeout",
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("error=Network timeout", call_msg)
        self.assertIn("result=error", call_msg)

    @patch("src.audit._get_audit_logger")
    def test_log_operation_empty_request(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_operation(
            telegram_id=222,
            username="bot",
            operation="help",
            result="ok",
            request_text=None,
            include_request_text=True,
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("request=-", call_msg)


if __name__ == "__main__":
    unittest.main()
