from algorithms.Algorithm import Algorithm
from datasets import Dataset


class DoNothing(Algorithm):

    def __init__(self):
        """

        """
        pass

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        return dataset
