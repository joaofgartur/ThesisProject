"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import copy
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from constants import PRIVILEGED, UNPRIVILEGED, NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from errors import error_check_dataset, error_check_sensitive_attribute
from helpers import logger


class Massaging(Algorithm):

    def __init__(self, learning_settings: dict):
        self.learning_settings = learning_settings

    def __compute_m__(self, dataset: Dataset, sensitive_attribute: str) -> int:
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

        data, outcome_label = dataset.merge_features_and_targets()

        count_unprivileged = data[data[sensitive_attribute] == UNPRIVILEGED].shape[0]
        count_unprivileged_positive = data[(data[sensitive_attribute] == UNPRIVILEGED) &
                                           (data[outcome_label] == POSITIVE_OUTCOME)].shape[0]

        count_privileged = data[data[sensitive_attribute] == PRIVILEGED].shape[0]
        count_privileged_positive = data[(data[sensitive_attribute] == PRIVILEGED)
                                         & (data[outcome_label] == POSITIVE_OUTCOME)].shape[0]

        if count_privileged + count_unprivileged == 0:
            raise ValueError("There are no data instances that match the context in the dataset.")

        m = (((count_unprivileged * count_privileged_positive) - (count_privileged * count_unprivileged_positive))
             // (count_privileged + count_unprivileged))

        return m

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

        if _TRAIN_SIZE_KEY not in self.learning_settings or _TEST_SIZE_KEY not in self.learning_settings:
            raise ValueError("Both train size and test size should be provided in the learning settings.")

        try:
            x_train, __, y_train, __ = train_test_split(dataset.features, dataset.targets,
                                                        test_size=self.learning_settings[_TEST_SIZE_KEY],
                                                        train_size=self.learning_settings[_TRAIN_SIZE_KEY])

            model = GaussianNB()
            model.fit(x_train, y_train.values.ravel())

            class_probabilities = model.predict_proba(dataset.features)

            return class_probabilities

        except Exception as e:
            logging.error(f"An error occurred during class probability computation: {e}")
            raise

    def __ranking__(self, dataset: Dataset, sensitive_attribute: str) -> Tuple[np.array, np.array]:
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

        data, outcome_label = dataset.merge_features_and_targets()

        # select candidates for promotion
        pr_candidates_indexes = data.index[
            (data[sensitive_attribute] == UNPRIVILEGED) & (data[outcome_label] == NEGATIVE_OUTCOME)].tolist()
        promotion_candidates = pd.DataFrame({_INDEX: data.index, _CLASS_PROBABILITY: positive_class_probabilities})

        promotion_candidates = promotion_candidates.iloc[pr_candidates_indexes].sort_values(by=_CLASS_PROBABILITY,
                                                                                            ascending=False)

        # select candidates for demotion
        dem_candidates_indexes = data.index[
            (data[sensitive_attribute] == PRIVILEGED) & (data[outcome_label] == POSITIVE_OUTCOME)].tolist()
        demotion_candidates = pd.DataFrame({_INDEX: data.index, _CLASS_PROBABILITY: positive_class_probabilities})
        demotion_candidates = demotion_candidates.iloc[dem_candidates_indexes].sort_values(by=_CLASS_PROBABILITY)

        return promotion_candidates, demotion_candidates

    def repair(self, dataset: Dataset, sensitive_attribute: str) -> Dataset:
        """
            Apply massaging technique to modify dataset targets.

            Parameters
            ----------
            dataset :
                Original dataset object containing features and targets.
            sensitive_attribute :
                Name of the data column representing the relevant attribute.

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
        logger.info(f"Repairing dataset {dataset.name} via Massaging...")

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        new_dataset = copy.deepcopy(dataset)
        promotion_candidates, demotion_candidates = self.__ranking__(dataset, sensitive_attribute)
        m = self.__compute_m__(dataset, sensitive_attribute)

        for __ in range(m):
            top_promotion = promotion_candidates.iloc[0]
            pr_index = top_promotion["index"].astype('int32')
            new_dataset.targets.iloc[pr_index] = POSITIVE_OUTCOME

            top_demotion = demotion_candidates.iloc[0]
            dem_index = top_demotion["index"].astype('int32')
            new_dataset.targets.iloc[dem_index] = NEGATIVE_OUTCOME

            promotion_candidates = promotion_candidates.drop(index=pr_index)
            demotion_candidates = demotion_candidates.drop(index=dem_index)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
