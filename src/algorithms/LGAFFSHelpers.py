"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

from utils import get_seed, is_gpu_enabled, GPU_LIMIT


def get_random_forest(data_size: int):
    if is_gpu_enabled() and data_size >= GPU_LIMIT:
        return get_gpu_random_forest()
    else:
        return get_cpu_random_forest()


def get_cpu_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=get_seed())


def get_gpu_random_forest():
    from cuml import RandomForestClassifier
    return RandomForestClassifier(random_state=get_seed(), n_streams=1)
