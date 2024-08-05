"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing as Aif360Reweighing

from algorithms.Algorithm import Algorithm
from datasets import Dataset, convert_to_standard_dataset
from utils import get_generator
from constants import PRIVILEGED, UNPRIVILEGED


class Reweighing(Algorithm):

    def __init__(self):
        super().__init__()
        self.transformer = None
        self.sensitive_attribute = None

    def __resample(self, data: pd.DataFrame, weights: np.array) -> (np.ndarray, np.ndarray):

        privileged_group = data[data[self.sensitive_attribute] == PRIVILEGED]
        unprivileged_group = data[data[self.sensitive_attribute] == UNPRIVILEGED]
        n_samples = np.abs(len(privileged_group) - len(unprivileged_group))

        sample_indexes = get_generator().choice(data.index, size=n_samples, replace=True, p=weights)
        sampled_df = data.loc[sample_indexes]
        sampled_data = pd.concat([data, sampled_df], ignore_index=True)

        return sampled_data.to_numpy(), sample_indexes

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
        normalized_weights = np.divide(transformed_data.instance_weights,
                                       transformed_data.instance_weights.sum())

        np_array = np.concatenate((transformed_data.features, transformed_data.labels), axis=1)
        resampling_data = pd.DataFrame(np_array,
                                       columns=standard_data.feature_names + standard_data.label_names)

        resampled_data, indexes = self.__resample(resampling_data, normalized_weights)
        resampled_features, resampled_targets = resampled_data[:, :-1], resampled_data[:, -1]

        data.update(features=resampled_features, targets=resampled_targets)

        sampled_values = data.protected_features.loc[indexes]
        new_sensitive_values = pd.concat([data.get_protected_features(), sampled_values]).reset_index(drop=True)
        data.protected_features = new_sensitive_values

        return data
