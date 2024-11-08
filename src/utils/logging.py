"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import logging, coloredlogs
import warnings
from enum import Enum

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)
logger_levels = Enum('level', ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"])
logging.getLogger('matplotlib.font_manager').disabled = True

sklearnex_logger = logging.getLogger('sklearnex')
sklearnex_logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore", module="daal4py")
warnings.filterwarnings("ignore", module="sklearn", category=FutureWarning)
