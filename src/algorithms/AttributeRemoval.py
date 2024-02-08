from algorithms.Algorithm import Algorithm
from datasets import Dataset


class AttributeRemoval(Algorithm):

    def __init__(self):
        """

        """
        pass

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        dataset.features = dataset.features.drop(columns=[sensitive_attribute])
        return dataset
