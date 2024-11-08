"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

from utils import logger, get_seed, extract_filename
from xgboost import XGBClassifier

# --- GPU Acceleration ---

gpu_enabled = False
gpu_device_id = None
gpu_allocated_memory = 0
GPU_LIMIT = 10000


def enable_gpu_acceleration():
    global gpu_enabled
    gpu_enabled = True


def disable_gpu_acceleration():
    global gpu_enabled
    gpu_enabled = False


def is_gpu_enabled():
    return gpu_enabled


def get_gpu_device():
    return gpu_device_id


def set_gpu_device(device_id: int = 0):
    try:
        import rmm
    except ImportError:
        logger.error("[GPU] RMM is not installed. Please install RMM to use GPU acceleration.")
        raise ImportError

    logger.info(f'[{extract_filename(__file__)}] Using GPU acceleration.')

    global gpu_device_id
    global gpu_allocated_memory
    gpu_device_id = device_id
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=gpu_allocated_memory * 1024 ** 3,
        devices=[device_id]
    )


def get_gpu_allocated_memory():
    return gpu_allocated_memory


def set_gpu_allocated_memory(memory: int):
    global gpu_allocated_memory
    gpu_allocated_memory = memory


# ---- CPU/GPU classifiers ----

surrogate_classifiers = []


def get_surrogate_classifiers() -> list[object]:
    return surrogate_classifiers


def set_surrogate_classifiers(classifiers: list[object]):
    global surrogate_classifiers
    surrogate_classifiers = classifiers


def get_test_classifier() -> object:
    if is_gpu_enabled():
        return XGBClassifier(random_state=get_seed(), device=f'cuda:{get_gpu_device()}')
    return XGBClassifier(random_state=get_seed())
