import logging
import os
import unittest
from unittest.mock import patch

from helpers.logger import configure_logging


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

            # Ensure quiet libraries stay at least WARNING even if global is DEBUG
            self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
