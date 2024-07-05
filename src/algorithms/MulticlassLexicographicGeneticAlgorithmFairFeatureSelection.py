import numpy as np
import pandas as pd

from constants import NUM_DECIMALS
from datasets import Dataset

from algorithms import LexicographicGeneticAlgorithmFairFeatureSelection
from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from evaluation import FairnessEvaluator
from helpers import dict_to_dataframe


class MulticlassLexicographicGeneticAlgorithmFairFeatureSelection(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01,
                 verbose: bool = False):
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

    def __fitness_metrics(self, data: Dataset, predictions: Dataset, sensitive_attribute):
        fairness_evaluator = FairnessEvaluator(data, predictions, sensitive_attribute)

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

    def __fairness_fitness(self, data: Dataset, predictions: Dataset):

        original_attribute_values = data.protected_attributes[self.sensitive_attribute]

        dummy_values = data.get_dummy_protected_feature(self.sensitive_attribute)

        metrics = pd.DataFrame()
        for value in dummy_values:
            data.protected_attributes[self.sensitive_attribute] = dummy_values[value]
            value_df = pd.concat([dict_to_dataframe({'value': value}),
                                  self.__fitness_metrics(data, predictions, self.sensitive_attribute)], axis=1)
            metrics = pd.concat([metrics, value_df])

        data.protected_attributes[self.sensitive_attribute] = original_attribute_values

        result = {}
        for metric in metrics.columns:
            if metrics[metric].dtype == 'object':
                continue
            sum_of_squares = np.round(np.sum((metrics[metric] - 1.0)**2), decimals=NUM_DECIMALS)
            result.update({metric: sum_of_squares})

        return result

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.sensitive_attribute = sensitive_attribute
        self.algorithm.fit(data, sensitive_attribute)

    def transform(self, dataset: Dataset) -> Dataset:
        return self.algorithm.transform(dataset)
