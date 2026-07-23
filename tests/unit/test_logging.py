import logging
import os
import unittest
from unittest.mock import patch

from bentoml_faster_whisper.utils.logger import configure_logging


class TestLoggingConfiguration(unittest.TestCase):
    def setUp(self):
        # Store original levels and handlers to restore them after the test
        self.original_handlers = logging.getLogger().handlers[:]
        self.original_root_level = logging.getLogger().level

        self.original_levels = {}
        for name in ("bentoml", "uvicorn", "circus", "httpx"):
            self.original_levels[name] = logging.getLogger(name).level

    def tearDown(self):
        # Restore logging state
        logging.getLogger().handlers = self.original_handlers
        logging.getLogger().setLevel(self.original_root_level)

        for name, level in self.original_levels.items():
            logging.getLogger(name).setLevel(level)

    def test_configure_logging_respects_log_level(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "warn", "IS_PROD": "false"}):
            configure_logging()

            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.WARNING)

            # Ensure the root stream handler has the correct level
            self.assertTrue(len(root_logger.handlers) >= 1)
            handler = root_logger.handlers[0]
            self.assertEqual(handler.level, logging.WARNING)

            # Ensure core pipeline loggers are set to WARNING
            self.assertEqual(logging.getLogger("bentoml").level, logging.WARNING)
            self.assertEqual(logging.getLogger("uvicorn").level, logging.WARNING)

            # Ensure quiet libraries are at least WARNING
            self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)

    def test_configure_logging_respects_log_level_debug(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "debug", "IS_PROD": "false"}):
            configure_logging()

            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.DEBUG)

            # Ensure the root stream handler has the correct level
            self.assertTrue(len(root_logger.handlers) >= 1)
            handler = root_logger.handlers[0]
            self.assertEqual(handler.level, logging.DEBUG)

            # Ensure core pipeline loggers are set to DEBUG
            self.assertEqual(logging.getLogger("bentoml").level, logging.DEBUG)
            self.assertEqual(logging.getLogger("uvicorn").level, logging.DEBUG)

            # Ensure quiet libraries inherit DEBUG level when global LOG_LEVEL is DEBUG
            self.assertEqual(logging.getLogger("httpx").level, logging.DEBUG)
            self.assertEqual(logging.getLogger("faster_whisper").level, logging.DEBUG)

    def test_client_error_filter(self):
        import sys
        from pydantic import BaseModel, ValidationError
        from starlette.exceptions import HTTPException

        # Initialize logging to set up the handler and filter
        configure_logging()
        handler = logging.getLogger().handlers[0]

        # Find the ClientErrorFilter
        from bentoml_faster_whisper.utils.logger import ClientErrorFilter

        client_filter = None
        for f in handler.filters:
            if isinstance(f, ClientErrorFilter):
                client_filter = f
                break

        self.assertIsNotNone(client_filter)
        assert client_filter is not None

        # 1. Test standard exception: should NOT be demoted
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError:
            exc_info = sys.exc_info()

        record = logging.LogRecord("name", logging.ERROR, "pathname", 12, "msg", (), exc_info)
        self.assertTrue(client_filter.filter(record))
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertIsNotNone(record.exc_info)

        # 2. Test ValidationError: should be demoted and traceback removed
        class DummyModel(BaseModel):
            value: int

        try:
            DummyModel(value="not-an-int")
        except ValidationError:
            exc_info = sys.exc_info()

        record = logging.LogRecord("name", logging.ERROR, "pathname", 12, "msg", (), exc_info)
        self.assertTrue(client_filter.filter(record))
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertIsNone(record.exc_info)
        self.assertIn("Validation Error", record.msg)

        # 3. Test Starlette HTTPException (status 400): should be demoted
        try:
            raise HTTPException(status_code=400, detail="bad input")
        except HTTPException:
            exc_info = sys.exc_info()

        record = logging.LogRecord("name", logging.ERROR, "pathname", 12, "msg", (), exc_info)
        self.assertTrue(client_filter.filter(record))
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertIsNone(record.exc_info)
        self.assertIn("HTTP Error (400)", record.msg)

        # 4. Test Starlette HTTPException (status 500): should NOT be demoted
        try:
            raise HTTPException(status_code=500, detail="server fault")
        except HTTPException:
            exc_info = sys.exc_info()

        record = logging.LogRecord("name", logging.ERROR, "pathname", 12, "msg", (), exc_info)
        self.assertTrue(client_filter.filter(record))
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertIsNotNone(record.exc_info)
