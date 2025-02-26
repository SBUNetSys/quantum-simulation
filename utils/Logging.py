import logging
import sys
from logging.handlers import RotatingFileHandler

"""
Logging class to log messages to stdout for debugging purposes.
"""


class Logger:
    def __init__(self, name, level=logging.DEBUG, logging_enabled=True,
                 save_to_file=False, file_name="log.txt",
                 max_file_size=5 * 1024 * 1024, backup_count=5):
        """
        Initialize a logger with optional file rotation capability.

        Args:
            name: Name of the logger
            level: Logging level (default: DEBUG)
            logging_enabled: Whether to output logs to stdout (default: True)
            save_to_file: Whether to save logs to a file (default: False)
            file_name: Name of the log file (default: "log.txt")
            max_file_size: Maximum size of each log file in bytes (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
        """
        self.logger = logging.getLogger(name)
        if self.logger.handlers:
            self.logger.handlers.clear()
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        if logging_enabled:
            ch = logging.StreamHandler(sys.stdout)  # Send logs to stdout
        else:
            ch = logging.NullHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler with rotation
        if save_to_file:
            fh = RotatingFileHandler(
                filename=file_name,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, message, color=None):
        if color:
            message = self.gen_color_msg(message, color)
        self.logger.info(message)

    def error(self, message, color=None):
        if color:
            message = self.gen_color_msg(message, color)
        self.logger.error(message)

    def debug(self, message, color=None):
        if color:
            message = self.gen_color_msg(message, color)
        self.logger.debug(message)

    def warning(self, message, color=None):
        if color:
            message = self.gen_color_msg(message, color)
        self.logger.warning(message)

    def critical(self, message, color=None):
        if color:
            message = self.gen_color_msg(message, color)
        self.logger.critical(message)

    @staticmethod
    def gen_color_msg(msg, color):
        if color == "red":
            return f"\033[91m{msg}\033[0m"
        elif color == "blue":
            return f"\033[94m{msg}\033[0m"
        elif color == "green":
            return f"\033[92m{msg}\033[0m"
        elif color == "yellow":
            return f"\033[93m{msg}\033[0m"
        elif color == "purple":
            return f"\033[95m{msg}\033[0m"
        elif color == "cyan":
            return f"\033[96m{msg}\033[0m"
        elif color == "orange":
            return f"\033[33m{msg}\033[0m"
        elif color == "light_blue":
            return f"\033[34m{msg}\033[0m"
        elif color == "light_green":
            return f"\033[32m{msg}\033[0m"
        elif color == "light_red":
            return f"\033[31m{msg}\033[0m"
        else:
            return msg
