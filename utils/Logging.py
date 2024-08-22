import logging
import sys

"""
Logging class to log messages to stdout for debugging purposes.
"""


class Logger:
    def __init__(self, name, level=logging.DEBUG, logging_enabled=True, save_to_file=False, file_name="log.txt"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if logging_enabled:
            ch = logging.StreamHandler(sys.stdout)  # Send logs to stdout
        else:
            ch = logging.NullHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if save_to_file:
            fh = logging.FileHandler(file_name)
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
