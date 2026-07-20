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
import structlog
from structlog.stdlib import BoundLogger, ProcessorFormatter
from structlog.types import EventDict, Processor, WrappedLogger

_F = TypeVar("_F", bound=Callable)

# Libraries whose INFO chatter (per-connection open/close, etc.) pollutes prod logs.
_QUIET_LIBRARIES = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock")

# Loggers that BentoML / uvicorn attach their own handlers to. We clear those handlers so
# records flow through the single root pipeline instead of being double-formatted.
_PIPELINE_LOGGERS = ("bentoml", "uvicorn", "uvicorn.error", "uvicorn.asgi", "circus")

# BentoML logs every request-handling exception here at ERROR with a full traceback,
# including client input errors that already return a 4xx.
_HTTP_APP_LOGGER = "bentoml._internal.server.http_app"


def _exception_type(exc_info: object) -> type[BaseException] | None:
    if exc_info is True:
        exc_info = sys.exc_info()
    if isinstance(exc_info, BaseException):
        return type(exc_info)
    if isinstance(exc_info, tuple) and exc_info:
        exc_type = exc_info[0]
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            return exc_type
    return None


def _is_client_error(exc_type: type[BaseException]) -> bool:
    """True for exceptions BentoML turns into a 4xx (bad input), not a 5xx server fault."""
    from bentoml.exceptions import BentoMLException
    from pydantic import ValidationError

    if issubclass(exc_type, ValidationError):
        return True
    if issubclass(exc_type, BentoMLException):
        try:
            return exc_type.error_code.value < 500
        except Exception:
            return False
    return False


def _drop_client_error_traceback(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Strip the traceback for client (4xx) errors so only genuine 5xx faults carry a stack.

    Runs before ``format_exc_info`` in the pipeline; a bad request stays a single concise
    line while real server errors keep their full traceback.
    """
    exc_type = _exception_type(event_dict.get("exc_info"))
    if exc_type is not None and _is_client_error(exc_type):
        event_dict.pop("exc_info", None)
    return event_dict


def _drop_color_message_key(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Uvicorn duplicates its message with ANSI codes under "color_message" — drop it."""
    event_dict.pop("color_message", None)
    return event_dict


def _configure_library_loggers() -> None:
    for name in _PIPELINE_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True

    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_logging() -> None:
    ctranslate2.set_log_level(logging.WARN)

    is_prod = os.getenv("IS_PROD", "false").lower() == "true"
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Applied to records that did NOT originate from structlog (BentoML, uvicorn, our
    # stdlib/loguru modules) so they end up with the same shape.
    foreign_pre_chain: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _drop_color_message_key,
        timestamper,
    ]

    if is_prod:
        renderer_processors: list[Processor] = [
            ProcessorFormatter.remove_processors_meta,
            _drop_client_error_traceback,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        renderer_processors = [
            ProcessorFormatter.remove_processors_meta,
            _drop_client_error_traceback,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ProcessorFormatter(processors=renderer_processors, foreign_pre_chain=foreign_pre_chain))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Route Python warnings through the pipeline instead of raw stderr (JSON in prod too).
    logging.captureWarnings(True)

    _configure_library_loggers()


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a bound structlog logger (optionally named, typically ``__name__``)."""
    return structlog.get_logger(name) if name else structlog.get_logger()


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
