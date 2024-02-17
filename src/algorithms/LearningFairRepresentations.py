from aif360.algorithms.preprocessing import LFR
from sklearn.preprocessing import StandardScaler

from algorithms.Algorithm import Algorithm
from algorithms.algorithms import scale_dataset
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset
from helpers import (convert_to_standard_dataset, set_dataset_features_and_labels, logger)


class LearningFairRepresentations(Algorithm):

    def __init__(self):
        super().__init__()

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        logger.info(f"Repairing dataset {dataset.name} via Massaging...")

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # define privileged and unprivileged group
        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        # transform dataset
        transformer = LFR(unprivileged_groups=unprivileged_groups,
                          privileged_groups=privileged_groups,
                          k=10, Ax=0.1, Ay=1.0, Az=2.0,
                          verbose=1
                          )
        transformer = transformer.fit(standard_dataset, maxiter=5000, maxfun=5000)
        transformed_dataset = transformer.transform(standard_dataset)

        print(transformed_dataset.labels)

        # convert into regular dataset
        new_dataset = set_dataset_features_and_labels(dataset, transformed_dataset.features, transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
