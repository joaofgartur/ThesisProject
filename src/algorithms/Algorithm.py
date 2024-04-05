import abc

from datasets import Dataset


class Algorithm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        """"""

    @abc.abstractmethod
    def fit(self, data: Dataset, sensitive_attribute: str):
        """"""

    @abc.abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """"""

    def set_validation_data(self, validation_data: Dataset):
        """"""
        pass
