from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import convert_to_standard_dataset


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float):
        super().__init__()
        self.repair_level = repair_level
        self.transformer = None
        self.sensitive_attribute = None

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute
        self.transformer = DIR(repair_level=self.repair_level)

    def transform(self, data: Dataset) -> Dataset:
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.fit_transform(standard_data)

        transformed_dataset = update_dataset(dataset=data,
                                             features=transformed_data.features,
                                             targets=transformed_data.labels)

        return transformed_dataset
