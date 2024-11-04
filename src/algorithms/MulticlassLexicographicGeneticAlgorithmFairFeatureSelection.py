"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import numpy as np
import pandas as pd

from constants import NUM_DECIMALS
from datasets import Dataset

from algorithms.LexicographicGeneticAlgorithmFairFeatureSelection import LexicographicGeneticAlgorithmFairFeatureSelection
from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from evaluation import FairnessEvaluator
from utils import dict_to_dataframe


class MulticlassLexicographicGeneticAlgorithmFairFeatureSelection(Algorithm):
    """
    Class representing a multiclass lexicographic genetic algorithm for fair feature selection.

    Attributes
    ----------
    epsilon : float
        The epsilon value for lexicographic selection.
    is_binary : bool
        Flag indicating if the algorithm is binary.
    algorithm : LexicographicGeneticAlgorithmFairFeatureSelection
        The underlying lexicographic genetic algorithm.
    sensitive_attribute : str
        The sensitive attribute used by the algorithm.
    population : list
        The current population of individuals.

    Methods
    -------
    __init__(genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0, max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01, verbose: bool = False):
        Initializes the MulticlassLexicographicGeneticAlgorithmFairFeatureSelection object with the specified parameters.
    __fitness_metrics(data: Dataset, predictions: pd.DataFrame, sensitive_attribute):
        Computes the fitness metrics for the predictions.
    __fairness_fitness(data: Dataset, predictions: pd.DataFrame):
        Computes the fairness fitness of the predictions.
    fit(data: Dataset, sensitive_attribute: str):
        Fits the algorithm to the data.
    transform(dataset: Dataset) -> Dataset:
        Transforms the dataset using the fitted algorithm.
    """

    def __init__(self, genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01,
                 verbose: bool = False):
        """
        Initializes the MulticlassLexicographicGeneticAlgorithmFairFeatureSelection object with the specified parameters.

        Parameters
        ----------
        genetic_parameters : GeneticBasicParameters
            The basic parameters for the genetic algorithm.
        min_feature_prob : float, optional
            The minimum feature probability (default is 0.0).
        max_feature_prob : float, optional
            The maximum feature probability (default is 1.0).
        n_splits : int, optional
            The number of splits for cross-validation (default is 5).
        epsilon : float, optional
            The epsilon value for lexicographic selection (default is 0.01).
        verbose : bool, optional
            Flag to enable verbose logging (default is False).
        """

        super().__init__()

        # genetic parameters
        self.epsilon = epsilon
        self.is_binary = False
        self.algorithm = LexicographicGeneticAlgorithmFairFeatureSelection(
            genetic_parameters=genetic_parameters,
            n_splits=n_splits,
            min_feature_prob=min_feature_prob,
            max_feature_prob=max_feature_prob,
            verbose=verbose
        )

        self.algorithm.__fairness_fitness = self.__fairness_fitness

        self.sensitive_attribute = None
        self.population = None

    def __fitness_metrics(self, data: Dataset, predictions: pd.DataFrame, sensitive_attribute):
        """
        Computes the fitness metrics for the predictions.

        Parameters
        ----------
        data : Dataset
            The dataset to compute the fitness metrics for.
        predictions : pd.DataFrame
            The predictions to compute the fitness metrics for.
        sensitive_attribute : str
            The sensitive attribute used by the algorithm.

        Returns
        -------
        dict
            The fitness metrics.
        """

        fairness_evaluator = FairnessEvaluator(data.features, data.targets, predictions, sensitive_attribute)

        ds = fairness_evaluator.discrimination_score()
        di = fairness_evaluator.disparate_impact()
        consistency = fairness_evaluator.consistency(k=3)
        false_positive_error_rate_balance_score = fairness_evaluator.false_positive_error_rate_balance_score()
        false_negative_error_rate_balance_score = fairness_evaluator.false_negative_error_rate_balance_score()

        return {
            'di': di,
            'ds': ds,
            'consistency': consistency,
            'false_positive_error_rate_balance_score': false_positive_error_rate_balance_score,
            'false_negative_error_rate_balance_score': false_negative_error_rate_balance_score
        }

    def __fairness_fitness(self, data: Dataset, predictions: pd.DataFrame):
        """
        Computes the fairness fitness of the predictions.

        Parameters
        ----------
        data : Dataset
            The dataset to compute the fairness fitness for.
        predictions : pd.DataFrame
            The predictions to compute the fairness fitness for.

        Returns
        -------
        dict
            The fairness fitness.
        """

        original_attribute_values = data.protected_features[self.sensitive_attribute]

        dummy_values = data.get_dummy_protected_feature(self.sensitive_attribute)

        metrics = pd.DataFrame()
        for value in dummy_values:
            data.protected_features[self.sensitive_attribute] = dummy_values[value]
            df = pd.DataFrame(dummy_values[value], columns=[value])
            value_df = pd.concat([dict_to_dataframe({'value': value}),
                                  self.__fitness_metrics(data, predictions, df)], axis=1)
            metrics = pd.concat([metrics, value_df])

        data.protected_features[self.sensitive_attribute] = original_attribute_values

        result = {}
        for metric in metrics.columns:
            if metrics[metric].dtype == 'object':
                continue
            sum_of_squares = np.round(np.sum((metrics[metric] - 1.0)**2), decimals=NUM_DECIMALS)
            result.update({metric: sum_of_squares})

        return result

    def fit(self, data: Dataset, sensitive_attribute: str):
        """
        Fits the algorithm to the data.

        Parameters
        ----------
        data : Dataset
            The dataset to fit the algorithm to.
        sensitive_attribute : str
            The sensitive attribute used by the algorithm.
        """
        self.sensitive_attribute = sensitive_attribute
        self.algorithm.fit(data, sensitive_attribute)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset using the fitted algorithm.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return self.algorithm.transform(dataset)
