import logging, coloredlogs
from enum import Enum

logger = logging.getLogger(__name__)
logger_levels = Enum('level', ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"])
logging.getLogger('matplotlib.font_manager').disabled = True


def config_logger(level: str):
    coloredlogs.install(level=level)