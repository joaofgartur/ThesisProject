import abc

from datasets import Dataset


class Algorithm(metaclass=abc.ABCMeta):

    def __init__(self, learning_settings: dict):
        self.learning_settings = learning_settings
        """"""

    @abc.abstractmethod
    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        """"""
