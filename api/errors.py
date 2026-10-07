"""One error shape for the whole API:  {"error": {"status": 404, "code": "unknown_symbol", "message": "...", "details": [...]}}"""

from __future__ import annotations


class ApiError(Exception):
    """An expected, user-explainable failure. The message is shown to the caller, so it must be safe to show."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(status: int, code: str, message: str, details: list | None = None) -> dict:
    body = {"status": status, "code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}
