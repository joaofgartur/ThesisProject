import ctypes
import time
from threading import Thread, Event


class ResourceManager(object):

    def __init__(self, period: int = 300):
        self.period = period
        self.cleanup_thread = None
        self.stop_event = Event()

    def __cleanup__(self):
        libc = ctypes.CDLL("libc.so.6")

        while not self.stop_event.is_set():
            time.sleep(self.period)
            try:
                libc.malloc_trim(0)  # memory cleanup
            except Exception as e:
                print(f"Error during memory cleanup: {e}")

    def start(self):
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.stop_event.clear()
            self.cleanup_thread = Thread(target=self.__cleanup__, daemon=True)
            self.cleanup_thread.start()

    def stop(self):
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.stop_event.set()
            self.cleanup_thread.join()
