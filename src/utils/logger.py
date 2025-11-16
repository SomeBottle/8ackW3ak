"""
日志记录模块
"""

import traceback
import os
import logging
from getpass import getuser
from time import time
from logging.handlers import TimedRotatingFileHandler


class __CustomLogger(logging.Logger):
    # 针对 logger.error 方法加入traceback输出
    def error(self, msg, *args, **kwargs):
        # 调用父类的 error 方法
        super().error(f"{msg} \n {traceback.format_exc()}", *args, **kwargs)


logging.setLoggerClass(__CustomLogger)


class Logger:
    """
    日志记录模块
    """

    def __init__(self, name: str, log_save_dir: str):
        """
        初始化日志记录器

        :param name: 日志记录器名称
        :param log_save_dir: 日志保存目录，实际目录是 log_save_dir/logs_{name}
        """
        self._logger = logging.getLogger(f"logger_{name}")
        self._logger.setLevel(logging.DEBUG)
        self._log_dir = os.path.join(log_save_dir, f"logs_{name}")
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        # 文件输出
        self._file_handler = TimedRotatingFileHandler(
            os.path.join(self._log_dir, f"exp.log"),
            when="midnight",
            delay=True,
            interval=1,
            backupCount=10,
        )
        # STDOUT
        self._stdout_handler = logging.StreamHandler()
        self._stdout_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_handler.setFormatter(formatter)
        self._stdout_handler.setFormatter(formatter)
        self._logger.addHandler(self._stdout_handler)
        self._logger.addHandler(self._file_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger
