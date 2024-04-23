from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME, PRIVILEGED, UNPRIVILEGED
from datasets import Dataset, update_dataset
from helpers import (convert_to_standard_dataset)

import numpy as np


class AIF360LearningFairRepresentations(Algorithm):

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

        privileged_groups = [{self.sensitive_attribute: PRIVILEGED}]
        unprivileged_groups = [{self.sensitive_attribute: UNPRIVILEGED}]

        self.transformer = LFR(unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               k=self.k,
                               Ax=self.Ax,
                               Ay=self.Ay,
                               Az=self.Az,
                               print_interval=self.print_interval,
                               verbose=1,
                               seed=self.seed)
        self.transformer.fit(standard_data, 150000, 150000)

    def transform(self, data: Dataset, ) -> Dataset:
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.transform(standard_data)

        transformed_dataset = update_dataset(dataset=data,
                                             features=transformed_data.features,
                                             targets=transformed_data.labels)
    
        unique_values = np.unique(transformed_dataset.targets)
        if len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 1):
            print('All targets are either 0 or 1. This is not expected. Check the optimization results.')
            return data

        return transformed_dataset
