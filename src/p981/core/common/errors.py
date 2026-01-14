"""스테이지 오류 정의."""


class StageError(Exception):
    """스테이지 오케스트레이션에서 발생하는 기본 오류."""


class StageValidationError(StageError):
    """스테이지 입력 검증 실패 시 발생."""
