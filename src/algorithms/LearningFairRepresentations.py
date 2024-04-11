from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from datasets import Dataset, update_dataset
from helpers import (convert_to_standard_dataset)


class LearningFairRepresentations(Algorithm):

    def __init__(self, seed: int,
                 k: int = 5,
                 ax: float = 0.01,
                 ay: float = 1.0,
                 az: float = 50.0,
                 print_interval: int = 1000,
                 verbose: int = 0):
        super().__init__()
        self.transformer = None
        self.seed = seed
        self.k = k
        self.Ax = ax
        self.Ay = ay
        self.Az = az
        self.print_interval = print_interval
        self.verbose = verbose

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute

        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        privileged_groups = [{self.sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{self.sensitive_attribute: NEGATIVE_OUTCOME}]

        self.transformer = LFR(unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               k=self.k,
                               Ax=self.Ax,
                               Ay=self.Ay,
                               Az=self.Az,
                               print_interval=self.print_interval,
                               verbose=self.verbose,
                               seed=self.seed)
        self.transformer.fit(standard_data)

    def transform(self, data: Dataset, ) -> Dataset:
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.transform(standard_data)

        transformed_dataset = update_dataset(dataset=data,
                                             features=transformed_data.features,
                                             targets=transformed_data.labels)

        return transformed_dataset
