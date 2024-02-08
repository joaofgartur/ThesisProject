"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import copy
import itertools

import pandas as pd
import numpy as np

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import logger
from metrics import simple_probability, joint_probability
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from errors import error_check_dataset, error_check_sensitive_attribute

SENSITIVE_ATTRIBUTE = "sensitive_attribute"
OUTCOME = "outcome"
WEIGHT = "weight"


class Reweighing(Algorithm):

    def __init__(self):
        """

        """
        pass

    def __compute_weights(self, dataset: Dataset, sensitive_attribute: str) -> pd.DataFrame:
        """
        Compute weights based on probabilities of sensitive attribute-outcome combinations.

        Parameters
        ----------
        dataset :
            Dataset object containing features and targets.
        sensitive_attribute :
            Name of the data column representing the relevant attribute.

        Returns
        -------
        weights_df :
            DataFrame containing computed weights for sensitive attribute-outcome combinations.

        Raises
        ------
        ValueError
            - If an invalid dataset is provided.
            - If the dataset does not contain both features and targets.
            - If the sensitive attribute is not present in the dataset.
        """

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        data, outcome_column = dataset.merge_features_and_targets()
        weights_data = []

        list_values = data[sensitive_attribute].unique()
        list_outcomes = [POSITIVE_OUTCOME, NEGATIVE_OUTCOME]
        permutations = [permutation for permutation in itertools.product(list_values, list_outcomes)]

        for permutation in permutations:
            value = permutation[0]
            outcome = permutation[1]
            prob_expected = simple_probability(data, sensitive_attribute, value) * simple_probability(data,
                                                                                                      outcome_column,
                                                                                                      outcome)
            prob_actual = joint_probability(data, sensitive_attribute, value, outcome_column, outcome)

            weight = prob_expected / prob_actual
            weights_data.append({SENSITIVE_ATTRIBUTE: value, OUTCOME: outcome, WEIGHT: weight})

        return pd.DataFrame(weights_data)

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
        logger.info(f"Repairing dataset {dataset.name} via Reweighing...")

        error_check_dataset(dataset)
        error_check_sensitive_attribute(dataset, sensitive_attribute)

        new_dataset = copy.deepcopy(dataset)
        weights = self.__compute_weights(dataset, sensitive_attribute)

        data, outcome_column = dataset.merge_features_and_targets()

        weights_dataset = []
        for __, instance in data.iterrows():
            instance_weight = weights[(weights[SENSITIVE_ATTRIBUTE] == instance[sensitive_attribute]) &
                                      (weights[OUTCOME] == instance[outcome_column])][WEIGHT].values
            weights_dataset.append(instance_weight)

        weights_array = np.concatenate(weights_dataset)
        n_rows = dataset.features.shape[0]
        new_dataset.features = dataset.features.sample(n=n_rows, weights=weights_array, replace=True).reset_index()

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
