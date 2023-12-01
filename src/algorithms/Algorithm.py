import abc

from datasets import Dataset


class Algorithm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        """"""

    @abc.abstractmethod
    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        """"""
