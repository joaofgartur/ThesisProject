from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import convert_to_standard_dataset


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float):
        super().__init__()
        self.repair_level = repair_level

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # transform dataset
        transformer = DIR(repair_level=self.repair_level)
        transformed_dataset = transformer.fit_transform(standard_dataset)

        # convert into regular dataset
        new_dataset = update_dataset(dataset=dataset,
                                     features=transformed_dataset.features,
                                     targets=transformed_dataset.labels)

        return new_dataset
