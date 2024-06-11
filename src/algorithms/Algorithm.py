import abc

from datasets import Dataset


class Algorithm(metaclass=abc.ABCMeta):

    def __init__(self):
        """"""
        self.is_binary = True
        self.sensitive_attribute = None
        self.needs_auxiliary_data = False
        self.auxiliary_data = None

    @abc.abstractmethod
    def fit(self, data: Dataset, sensitive_attribute: str):
        """"""

    @abc.abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """"""

    def set_validation_data(self, validation_data: Dataset):
        """"""
        pass
