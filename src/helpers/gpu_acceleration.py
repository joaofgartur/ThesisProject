import pynvml
import rmm
from cuml import RandomForestClassifier as RandomForestClassifier_GPU

from helpers import get_seed

gpu_device_id = None
GPU_LIMIT = 10000


def set_gpu_device(device_id: int = 0):
    global gpu_device_id
    gpu_device_id = device_id
    gpu_memory_limits = get_gpu_memory_limits(device_id)
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=gpu_memory_limits,
        devices=[device_id]
    )


def get_gpu_device():
    global gpu_device_id
    return gpu_device_id


def get_gpu_memory_limits(device_id: int = 0):
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_memory_limit = meminfo.total

    return gpu_memory_limit


def get_gpu_random_forest():
    return RandomForestClassifier_GPU(random_state=get_seed(), n_streams=1)