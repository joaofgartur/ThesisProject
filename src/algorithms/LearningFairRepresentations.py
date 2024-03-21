from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset, update_dataset
from helpers import (convert_to_standard_dataset)


class LearningFairRepresentations(Algorithm):

    def __init__(self):
        super().__init__()
        self.transformer = None
        self.sensitive_attribute = None

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute

        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        privileged_groups = [{self.sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{self.sensitive_attribute: NEGATIVE_OUTCOME}]

        self.transformer = LFR(unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               k=10, Ax=0.1, Ay=1.0, Az=2.0,
                               verbose=1)
        self.transformer.fit(standard_data)

    def transform(self, data: Dataset, ) -> Dataset:
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.transform(standard_data)

        transformed_dataset = update_dataset(dataset=data,
                                             features=transformed_data.features,
                                             targets=transformed_data.labels)

        return transformed_dataset
