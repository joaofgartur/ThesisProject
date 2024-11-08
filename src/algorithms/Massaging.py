"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from constants import PRIVILEGED, UNPRIVILEGED, NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from errors import error_check_dataset, error_check_sensitive_attribute


class Massaging(Algorithm):

    def __init__(self):
        super().__init__()
        self.m = None
        self.demotion_candidates = None
        self.promotion_candidates = None
        self.sensitive_attribute = None

    def __compute_m(self, dataset: Dataset, sensitive_attribute: str) -> int:
        """
        Function that computes the number of necessary modifications to the target values.

        Parameters
        ----------
        dataset :
            Dataset object that contains the data features and targets.
        sensitive_attribute :
            Name of the data column representing the relevant attribute.

        Returns
        -------
        m :
            Number of necessary modifications.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If the sensitive attribute is not present in the dataset.
            - If there are no data instances that match the context in the dataset.
        """

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        data = pd.concat([dataset.features, dataset.targets], axis='columns').reset_index(drop=True)
        target = dataset.targets.columns[0]

        count_unprivileged = data[data[sensitive_attribute] == UNPRIVILEGED].shape[0]
        count_unprivileged_positive = data[(data[sensitive_attribute] == UNPRIVILEGED) &
                                           (data[target] == POSITIVE_OUTCOME)].shape[0]

        count_privileged = data[data[sensitive_attribute] == PRIVILEGED].shape[0]
        count_privileged_positive = data[(data[sensitive_attribute] == PRIVILEGED)
                                         & (data[target] == POSITIVE_OUTCOME)].shape[0]

        if count_privileged + count_unprivileged == 0:
            raise ValueError("There are no data instances that match the context in the dataset.")

        return int(np.divide((count_unprivileged * count_privileged_positive)
                             - (count_privileged * count_unprivileged_positive),
                             count_privileged + count_unprivileged))

    def __compute_class_probabilities__(self, dataset: Dataset) -> np.ndarray:
        """
        Compute class probabilities for the given dataset using a Gaussian Naive Bayes model.

        Parameters
        ----------
        dataset :
            Dataset object containing features and targets.

        Returns
        -------
        class_probabilities :
            Array containing predicted class probabilities for each instance in the dataset.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If both 'train_size' and 'test_size' are not provided in the learning settings.

        Notes
        -----
        - This method uses a Gaussian Naive Bayes model for learning.
        - In case of errors during class probability computation, a log entry is created, and a `ValueError` is raised.
        """
        _TRAIN_SIZE_KEY = "test_size"
        _TEST_SIZE_KEY = "train_size"

        error_check_dataset(dataset)

        try:
            from sklearn.naive_bayes import GaussianNB

            model = GaussianNB()
            features = dataset.features.to_numpy()
            targets = dataset.targets.to_numpy().ravel()
            model.fit(features, targets)
            class_probabilities = model.predict_proba(features)

            return class_probabilities

        except Exception as e:
            logging.error(f"An error occurred during class probability computation: {e}")
            raise

    def __rank_candidates__(self, dataset: Dataset, sensitive_attribute: str) -> Tuple[np.array, np.array]:
        """
        Rank candidates for promotion and demotion based on class probabilities.

        Parameters
        ----------
        dataset :
            Dataset object containing features and targets.
        sensitive_attribute :
            Name of the data column representing the relevant attribute.

        Returns
        -------
        Tuple of DataFrames :
            - promotion_candidates: DataFrame containing candidates for promotion.
            - demotion_candidates: DataFrame containing candidates for demotion.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If the sensitive attribute is not present in the dataset.
        """

        _INDEX = "index"
        _CLASS_PROBABILITY = "class_probability"

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        # learn class probabilities
        class_probabilities = self.__compute_class_probabilities__(dataset)
        positive_class_probabilities = class_probabilities[:, 0]

        data = pd.concat([dataset.features, dataset.targets], axis='columns').reset_index(drop=True)
        target = dataset.targets.columns[0]

        # select candidates for promotion
        pr_candidates_indexes = data.index[
            (data[sensitive_attribute] == UNPRIVILEGED) & (data[target] == NEGATIVE_OUTCOME)].tolist()

        promotion_candidates = pd.DataFrame({_CLASS_PROBABILITY: positive_class_probabilities})
        promotion_candidates = promotion_candidates.iloc[pr_candidates_indexes].sort_values(by=_CLASS_PROBABILITY,
                                                                                            ascending=False)

        # select candidates for demotion
        dem_candidates_indexes = data.index[
            (data[sensitive_attribute] == PRIVILEGED) & (data[target] == POSITIVE_OUTCOME)].tolist()
        demotion_candidates = pd.DataFrame({_CLASS_PROBABILITY: positive_class_probabilities})
        demotion_candidates = demotion_candidates.iloc[dem_candidates_indexes].sort_values(by=_CLASS_PROBABILITY)

        return promotion_candidates, demotion_candidates

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute

        self.promotion_candidates, self.demotion_candidates = self.__rank_candidates__(data, self.sensitive_attribute)
        self.m = self.__compute_m(data, self.sensitive_attribute)

    def transform(self, data: Dataset) -> Dataset:
        """
        Apply massaging technique to modify dataset targets.

        Parameters
        ----------
        data :
            Original dataset object containing features and targets.

        Returns
        -------
        new_dataset :
            Modified dataset with massaging applied.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If the sensitive attribute is not present in the dataset.
        """

        for _ in range(self.m):

            pr_index = self.promotion_candidates.index[0]
            data.targets.iloc[pr_index] = POSITIVE_OUTCOME

            dem_index = self.demotion_candidates.index[0]
            data.targets.iloc[dem_index] = NEGATIVE_OUTCOME

            self.promotion_candidates = self.promotion_candidates.drop(index=pr_index)
            self.demotion_candidates = self.demotion_candidates.drop(index=dem_index)

        self.promotion_candidates = None
        self.demotion_candidates = None
        self.m = None

        return data
