"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import itertools

import pandas as pd
from aif360.algorithms.preprocessing import Reweighing as Aif360Reweighing
from sklearn.preprocessing import StandardScaler

from algorithms.Algorithm import Algorithm
from algorithms.algorithms import scale_dataset
from datasets import Dataset
from helpers import logger, convert_to_standard_dataset, modify_dataset
from metrics import simple_probability, joint_probability
from constants import POSITIVE_OUTCOME, NEGATIVE_OUTCOME
from errors import error_check_dataset, error_check_sensitive_attribute

SENSITIVE_ATTRIBUTE = "sensitive_attribute"
OUTCOME = "outcome"
WEIGHT = "weight"


class Reweighing(Algorithm):

    def __init__(self, learning_settings: dict):
        super().__init__(learning_settings)

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

        # convert dataset into aif360 dataset
        standard_dataset = convert_to_standard_dataset(dataset, sensitive_attribute)

        # standardize features
        scaler = StandardScaler()
        standard_dataset = scale_dataset(scaler, standard_dataset)

        # define privileged and unprivileged group
        privileged_groups = [{sensitive_attribute: POSITIVE_OUTCOME}]
        unprivileged_groups = [{sensitive_attribute: NEGATIVE_OUTCOME}]

        # transform dataset
        transformer = Aif360Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        transformer.fit(standard_dataset)
        transformed_dataset = transformer.transform(standard_dataset)

        # convert into regular dataset
        new_dataset = modify_dataset(dataset, transformed_dataset.features, transformed_dataset.labels)

        logger.info(f"Dataset {dataset.name} repaired.")

        return new_dataset
