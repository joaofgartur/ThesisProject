"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
from aif360.algorithms.preprocessing import Reweighing as Aif360Reweighing

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import convert_to_standard_dataset
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME, PRIVILEGED, UNPRIVILEGED
from errors import error_check_dataset, error_check_sensitive_attribute


class Reweighing(Algorithm):

    def __init__(self):
        super().__init__()
        self.transformer = None
        self.sensitive_attribute = None

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute

        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        privileged_groups = [{self.sensitive_attribute: PRIVILEGED}]
        unprivileged_groups = [{self.sensitive_attribute: UNPRIVILEGED}]

        self.transformer = Aif360Reweighing(unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
        self.transformer.fit(standard_data)

    def transform(self, data: Dataset) -> Dataset:
        """
        Apply reweighing technique to modify dataset features based on computed weights.

        Parameters
        ----------
        data :
            Original dataset object containing features and targets.

        Returns
        -------
        new_dataset :
            Modified dataset with reweighing applied.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If the sensitive attribute is not present in the dataset.
        """
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.transform(standard_data)

        transformed_dataset = update_dataset(dataset=data,
                                             features=transformed_data.features,
                                             targets=transformed_data.labels)

        transformed_dataset.instance_weights = transformed_data.instance_weights

        return transformed_dataset
