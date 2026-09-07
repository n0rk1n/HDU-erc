from __future__ import annotations


class ConfigError(ValueError):
    """Raised when startup configuration or a local schema is invalid."""


class DomainError(Exception):
    """A stable, API-mappable domain failure."""

    def __init__(self, code: str, message: str, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class InvalidIdentifier(DomainError):
    def __init__(self, message: str = "invalid identifier") -> None:
        super().__init__("invalid_identifier", message, 422)


class InvalidMessageState(DomainError):
    def __init__(self, message: str = "invalid message state") -> None:
        super().__init__("invalid_message", message, 409)


class UserNotFound(DomainError):
    def __init__(self, message: str = "user not found") -> None:
        super().__init__("user_not_found", message, 404)


class AlreadyRated(DomainError):
    def __init__(self) -> None:
        super().__init__("already_rated", "message already rated", 409)


class MessageNotFound(DomainError):
    def __init__(self) -> None:
        super().__init__("message_not_found", "message not found", 404)


class TurnInProgress(DomainError):
    def __init__(self, message: str = "turn already in progress") -> None:
        super().__init__("turn_in_progress", message, 409)


class ModelError(DomainError):
    def __init__(self, message: str = "model error") -> None:
        super().__init__("model_error", message, 502)


class DatabaseError(DomainError):
    def __init__(self, message: str = "database error") -> None:
        super().__init__("database_error", message, 500)


class ProcessInterrupted(DomainError):
    def __init__(self, message: str = "process interrupted") -> None:
        super().__init__("process_interrupted", message, 503)


SAFE_PUBLIC_ERROR_MESSAGES: dict[str, str] = {
    "reply_format_error": "model reply does not match the required messages object",
    "model_error": "model generation failed",
    "database_error": "database error",
    "process_interrupted": "generation interrupted",
}
