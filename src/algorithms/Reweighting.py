"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
from aif360.algorithms.preprocessing import Reweighing as Aif360Reweighing

from algorithms.Algorithm import Algorithm
from datasets import Dataset, update_dataset
from helpers import convert_to_standard_dataset
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from errors import error_check_dataset, error_check_sensitive_attribute

SENSITIVE_ATTRIBUTE = "sensitive_attribute"
OUTCOME = "outcome"
WEIGHT = "weight"


class Reweighing(Algorithm):

    def __init__(self):
        super().__init__()

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        """
        Apply reweighing technique to modify dataset features based on computed weights.

        Parameters
        ----------
        dataset :
            Original dataset object containing features and targets.
        sensitive_attribute :
            Name of the data column representing the relevant attribute.

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

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # define privileged and unprivileged group
        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        # transform dataset
        transformer = Aif360Reweighing(unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)
        transformer.fit(standard_dataset)
        transformed_dataset = transformer.transform(standard_dataset)

        # convert into regular dataset
        new_dataset = update_dataset(dataset=dataset,
                                     features=transformed_dataset.features,
                                     targets=transformed_dataset.labels)

        new_dataset.instance_weights = transformed_dataset.instance_weights

        return new_dataset
