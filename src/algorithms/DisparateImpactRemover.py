from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import convert_to_standard_dataset, set_dataset_features_and_targets, logger


class DisparateImpactRemover(Algorithm):

    def __init__(self, repair_level: float):
        super().__init__()
        self.repair_level = repair_level

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        logger.info(f"Repairing dataset {dataset.name} via {self.__class__.__name__}...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # transform dataset
        transformer = DIR(repair_level=self.repair_level)
        transformed_dataset = transformer.fit_transform(standard_dataset)

        # convert into regular dataset
        new_dataset = set_dataset_features_and_targets(dataset=dataset,
                                                       features=transformed_dataset.features,
                                                       targets=transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
