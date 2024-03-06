from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset, update_dataset
from helpers import (convert_to_standard_dataset, logger)


class LearningFairRepresentations(Algorithm):

    def __init__(self):
        super().__init__()

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # define privileged and unprivileged group
        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        # transform dataset
        transformer = LFR(unprivileged_groups=unprivileged_groups,
                          privileged_groups=privileged_groups,
                          k=10, Ax=0.1, Ay=1.0, Az=2.0,
                          verbose=1)
        transformed_dataset = transformer.fit_transform(standard_dataset)

        # convert into regular dataset
        new_dataset = update_dataset(dataset=dataset,
                                     features=transformed_dataset.features,
                                     targets=transformed_dataset.labels)

        return new_dataset
