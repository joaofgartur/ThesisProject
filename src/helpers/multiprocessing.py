num_processes = None
num_threads = None


def set_num_processes(num_workers: int = 1):
    global num_processes
    num_processes = num_workers


def get_num_processes():
    global num_processes
    return num_processes


def set_num_threads(num_workers: int = 1):
    global num_threads
    num_threads = num_workers


def get_num_threads():
    global num_threads
    return num_threads
