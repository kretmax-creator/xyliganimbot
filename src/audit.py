"""
Модуль аудита для xyliganimbot.

Запись в logs/audit.log для каждой операции: команда/запрос, результат, ошибки.
Учёт настройки log_user_messages при логировании текста запроса.
"""

import logging
from typing import Optional

# Максимальная длина запроса в аудите (при включённом log_user_messages)
MAX_REQUEST_LENGTH = 500


def _get_audit_logger() -> logging.Logger:
    return logging.getLogger("audit")


def log_operation(
    telegram_id: int,
    username: Optional[str],
    operation: str,
    result: str,
    request_text: Optional[str] = None,
    include_request_text: bool = False,
    error: Optional[str] = None,
) -> None:
    """
    Записывает операцию в audit.log.

    Args:
        telegram_id: Telegram ID пользователя
        username: Username пользователя (может быть None)
        operation: Тип операции (help, search, admin_load_model, admin_vectorize и т.д.)
        result: Результат (ok, found N, error, denied)
        request_text: Текст запроса/команды (логируется только если include_request_text)
        include_request_text: Логировать ли текст запроса (из конфига log_user_messages)
        error: Текст ошибки при неуспехе
    """
    logger = _get_audit_logger()
    req = ""
    if include_request_text and request_text:
        req = request_text.strip()[:MAX_REQUEST_LENGTH]
        if len(request_text.strip()) > MAX_REQUEST_LENGTH:
            req += "..."
    elif request_text and not include_request_text:
        req = "[hidden]"
    else:
        req = "-"

    parts = [
        f"telegram_id={telegram_id}",
        f"username={username or '-'}",
        f"operation={operation}",
        f"request={req}",
        f"result={result}",
    ]
    if error:
        parts.append(f"error={error[:200]}")

    logger.info(" | ".join(parts))
