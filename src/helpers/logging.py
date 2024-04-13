import logging, coloredlogs
from enum import Enum

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)
logger_levels = Enum('level', ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"])
logging.getLogger('matplotlib.font_manager').disabled = True
