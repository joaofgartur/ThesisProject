import copy

from aif360.algorithms.preprocessing import LFR

from algorithms.Algorithm import Algorithm
from constants import PRIVILEGED, UNPRIVILEGED
from datasets import Dataset, update_dataset
from helpers import (convert_to_standard_dataset, get_seed, )

import numpy as np


class LearnedFairRepresentations(Algorithm):

    def __init__(self,
                 k: int = 5,
                 ax: float = 0.01,
                 ay: float = 1.0,
                 az: float = 50.0,
                 print_interval: int = 1000,
                 verbose: int = 0):
        super().__init__()
        self.transformer = None
        self.k = k
        self.Ax = ax
        self.Ay = ay
        self.Az = az
        self.print_interval = print_interval
        self.verbose = verbose

    def __check_error(self, transformed_data) -> bool:
        target_classes = np.unique(transformed_data.targets)

        print(f"\t[LFR] Target classes: {target_classes}", flush=True)

        return len(target_classes) == 1 and (target_classes[0] == 0 or target_classes[0] == 1)

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
                               verbose=self.verbose,
                               seed=get_seed())

        self.transformer.fit(standard_data, 150000, 150000)

    def transform(self, data: Dataset, ) -> Dataset:
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_dataset = copy.deepcopy(data)
        try:
            transformed_data = self.transformer.transform(standard_data)
            transformed_dataset = update_dataset(dataset=data,
                                                 features=transformed_data.features,
                                                 targets=transformed_data.labels)
        except ValueError:
            transformed_dataset.error_flag = True

        transformed_dataset.error_flag = self.__check_error(transformed_dataset)

        return transformed_dataset
