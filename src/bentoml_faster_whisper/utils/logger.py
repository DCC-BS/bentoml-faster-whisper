"""Structured logging for the service, following the DCC-BS backend logging pattern.

A single pipeline renders everything: structlog events *and* stdlib records (BentoML,
uvicorn, our own ``logging.getLogger`` modules, and loguru forwarded in) all flow through
the root handler — JSON lines in production (``IS_PROD=true``), a Rich console renderer in
development.

Environment variables:
- ``IS_PROD``  — ``true`` selects JSON output; anything else (default) uses the dev console.
- ``LOG_LEVEL`` — root level for app diagnostics (default ``INFO``).
"""

import functools
import inspect
import logging
import os
import sys
from collections.abc import Callable
from typing import TypeVar, cast

import ctranslate2
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, WrappedLogger

_F = TypeVar("_F", bound=Callable)

# Libraries whose INFO chatter (per-connection open/close, etc.) pollutes prod logs.
_QUIET_LIBRARIES = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock")

# Loggers that BentoML / uvicorn attach their own handlers to. We clear those handlers so
# records flow through the single root pipeline instead of being double-formatted.
_PIPELINE_LOGGERS = ("bentoml", "uvicorn", "uvicorn.error", "uvicorn.asgi", "circus")


def _exception_info(exc_info: object) -> tuple[type[BaseException] | None, BaseException | None]:
    if exc_info is True:
        exc_info = sys.exc_info()
    if isinstance(exc_info, BaseException):
        return type(exc_info), exc_info
    if isinstance(exc_info, tuple) and len(exc_info) >= 2:
        exc_type = exc_info[0]
        exc_value = exc_info[1]
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            val = exc_value if isinstance(exc_value, BaseException) else None
            return exc_type, val
    return None, None


def _is_client_error(exc_type: type[BaseException], exc_value: BaseException | None = None) -> bool:
    """True for exceptions BentoML turns into a 4xx (bad input), not a 5xx server fault."""
    from bentoml.exceptions import BentoMLException
    from pydantic import ValidationError
    from starlette.exceptions import HTTPException

    if issubclass(exc_type, ValidationError):
        return True
    if issubclass(exc_type, HTTPException):
        if exc_value is not None:
            status_code = getattr(exc_value, "status_code", 400)
            return status_code < 500
        return True
    if issubclass(exc_type, BentoMLException):
        source = exc_value if exc_value is not None else exc_type
        try:
            return getattr(source, "error_code").value < 500
        except Exception:
            return False
    return False


def _drop_client_error_traceback(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Strip the traceback for client (4xx) errors so only genuine 5xx faults carry a stack.

    Runs before ``format_exc_info`` in the pipeline; a bad request stays a single concise
    line while real server errors keep their full traceback.
    """
    exc_info = event_dict.get("exc_info")
    if exc_info:
        exc_type, exc_value = _exception_info(exc_info)
        if exc_type is not None and _is_client_error(exc_type, exc_value):
            event_dict.pop("exc_info", None)
    return event_dict


class ClientErrorFilter(logging.Filter):
    """Demote client 4xx/validation errors from ERROR to WARNING.

    Simplifies the message to show the status/fields instead of dumping a stack trace.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc_type, exc_value = _exception_info(record.exc_info)
            if exc_type is not None and _is_client_error(exc_type, exc_value):
                record.levelno = logging.WARNING
                record.levelname = "WARNING"

                error_msg = ""
                try:
                    from bentoml.exceptions import BentoMLException
                    from pydantic import ValidationError
                    from starlette.exceptions import HTTPException as StarletteHTTPException

                    if isinstance(exc_value, ValidationError):
                        errors = exc_value.errors()
                        error_details = []
                        for err in errors:
                            loc = ".".join(str(part) for part in err.get("loc", []))
                            msg = err.get("msg", "invalid value")
                            error_details.append(f"{loc}: {msg}")
                        error_msg = f"Validation Error (400): {'; '.join(error_details)}"
                    elif isinstance(exc_value, StarletteHTTPException):
                        status_code = getattr(exc_value, "status_code", 400)
                        detail = getattr(exc_value, "detail", "Invalid request")
                        error_msg = f"HTTP Error ({status_code}): {detail}"
                    elif isinstance(exc_value, BentoMLException):
                        status_code = 400
                        try:
                            status_code = exc_value.error_code.value
                        except Exception:
                            pass
                        error_msg = f"BentoML Error ({status_code}): {str(exc_value)}"
                    else:
                        error_msg = f"Client Error: {str(exc_value)}"
                except Exception as e:
                    error_msg = f"Client Error parsing failed: {e}"

                if error_msg:
                    record.msg = f"{record.getMessage()} - {error_msg}"
                    record.args = ()
                record.exc_info = None

        return True


def _drop_color_message_key(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Uvicorn duplicates its message with ANSI codes under "color_message" — drop it."""
    event_dict.pop("color_message", None)
    return event_dict


def _configure_library_loggers(level: int) -> None:
    for name in _PIPELINE_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
        lib_logger.setLevel(level)

    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


from dcc_backend_common.logger import get_logger as dcc_get_logger
from dcc_backend_common.logger import init_logger as dcc_init_logger


def _configure_library_loggers(level: int) -> None:
    for name in _PIPELINE_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
        lib_logger.setLevel(level)

    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


def configure_logging() -> None:
    ctranslate2.set_log_level(logging.WARN)
    # dcc_init_logger() reads IS_PROD via get_env_or_throw and aborts if it is unset.
    # IS_PROD is documented above as optional (default: dev console), so default it here
    # rather than let a missing flag crash every import of the service module (e.g. CI
    # collecting tests with an empty .env, or `bentoml build` introspecting the service).
    os.environ.setdefault("IS_PROD", "false")
    dcc_init_logger()

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    _configure_library_loggers(level)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(level)
        handler.addFilter(ClientErrorFilter())


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a bound structlog logger (optionally named, typically ``__name__``)."""
    return dcc_get_logger(name)


def log_exceptions(func: _F) -> _F:
    """Log any exception escaping ``func`` (with traceback) at the point of failure, then
    re-raise it — the structlog replacement for loguru's ``@logger.catch(reraise=True)``.

    Handles generator functions too: their body runs during iteration, so the wrapper must
    drive the generator to see exceptions (a plain try/except around the call would miss them).
    """
    log = get_logger(getattr(func, "__module__", __name__))
    qualname = getattr(func, "__qualname__", repr(func))

    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def gen_wrapper(*args: object, **kwargs: object):
            try:
                yield from func(*args, **kwargs)
            except Exception:
                log.exception("Unhandled exception", function=qualname)
                raise

        return cast(_F, gen_wrapper)

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object):
        try:
            return func(*args, **kwargs)
        except Exception:
            log.exception("Unhandled exception", function=qualname)
            raise

    return cast(_F, wrapper)
