"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import itertools
import math
from random import sample

import numpy as np
import pandas as pd

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from algorithms.LGAFFSHelpers import get_random_forest
from constants import PRED_OUTCOME
from datasets import Dataset
from evaluation import FairnessEvaluator
from evaluation.ModelEvaluator import ModelEvaluator
from utils import get_generator, get_seed, logger


class LexicographicGeneticAlgorithmFairFeatureSelection(Algorithm):
    """
    Class representing a lexicographic genetic algorithm for fair feature selection.

    Attributes
    ----------
    genetic_parameters : GeneticBasicParameters
        The basic parameters for the genetic algorithm.
    min_feature_prob : float
        The minimum feature probability.
    max_feature_prob : float
        The maximum feature probability.
    n_splits : int
        The number of splits for cross-validation.
    epsilon : float
        The epsilon value for lexicographic selection.
    verbose : bool
        Flag to enable verbose logging.
    sensitive_attribute : str
        The sensitive attribute used by the algorithm.
    population : list
        The current population of individuals.
    cache_path : str
        The path to cache evaluated individuals.
    evaluated_individuals : dict
        The dictionary of evaluated individuals.

    Methods
    -------
    __init__(genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0, max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01, verbose: bool = False):
        Initializes the LexicographicGeneticAlgorithmFairFeatureSelection object with the specified parameters.
    __gen_individual(feature_prob: float):
        Generates an individual with the given feature probability.
    __gen_ramped_population():
        Generates a ramped population.
    __crossover(parent1, parent2):
        Performs crossover between two parents.
    __mutation(individual: list):
        Mutates an individual.
    __tournament(population):
        Performs tournament selection on the population.
    __select_best(population):
        Selects the best individuals from the population.
    __performance_fitness(targets: pd.DataFrame, predictions: pd.DataFrame):
        Computes the performance fitness of the predictions.
    __fairness_fitness(data: Dataset, predictions: pd.DataFrame):
        Computes the fairness fitness of the predictions.
    __fitness(data: Dataset, individual, folds):
        Computes the fitness of an individual.
    __evaluate_population(data: Dataset, population, folds):
        Evaluates the population.
    __phenotype(data: Dataset, individual: list) -> pd.DataFrame:
        Applies the phenotype to the data.
    fit(data: Dataset, sensitive_attribute: str):
        Fits the algorithm to the data.
    __clean_cache__():
        Cleans the cache.
    transform(data: Dataset) -> Dataset:
        Transforms the dataset using the fitted algorithm.
    """

    def __init__(self, genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01,
                 verbose: bool = False):
        super().__init__()

        self.algorithm_name = 'LGAFFS'

        self.genetic_parameters = genetic_parameters

        self.min_feature_prob = min_feature_prob
        self.max_feature_prob = max_feature_prob
        self.n_splits = n_splits
        self.epsilon = epsilon

        self.verbose = verbose

        self.sensitive_attribute = None
        self.population = None
        self.cache_path = None

        self.evaluated_individuals = {}

    def __gen_individual(self, feature_prob: float):
        """
        Generates an individual with the given feature probability.

        Parameters
        ----------
        feature_prob : float
            The feature probability to generate the individual.

        Returns
        -------
        list
            The generated individual.
        """
        features_prob = get_generator().uniform(low=self.min_feature_prob, high=self.max_feature_prob,
                                                size=self.genetic_parameters.individual_size)
        return [np.where(features_prob > feature_prob, 1, 0), [], []]

    def __gen_ramped_population(self):
        """
        Generates a ramped population.

        Returns
        -------
        list
            The generated ramped population.
        """
        n = self.genetic_parameters.population_size
        pop_feature_probs = get_generator().uniform(low=self.min_feature_prob, high=self.max_feature_prob, size=n)
        return [self.__gen_individual(float(prob)) for prob in pop_feature_probs]

    def __crossover(self, parent1, parent2):
        """
        Performs crossover between two parents.

        Parameters
        ----------
        parent1 : list
            The first parent.
        parent2 : list
            The second parent.

        Returns
        -------
        tuple
            The offspring generated from the crossover.
        """
        if get_generator().random() < self.genetic_parameters.probability_crossover:
            mask = get_generator().random(parent1[0].shape[0]) < 0.5
            offspring1 = np.where(mask, parent1[0], parent2[0])
            offspring2 = np.where(mask, parent2[0], parent1[0])
            return [offspring1, {}, {}], [offspring2, {}, {}]
        return parent1, parent2

    def __mutation(self, individual: list):
        """
        Mutates an individual.

        Parameters
        ----------
        individual : list
            The individual to mutate.

        Returns
        -------
        list
            The mutated individual.
        """
        mutation_mask = get_generator().random(individual[0].shape[0]) < self.genetic_parameters.probability_mutation
        mutated_genome = np.where(mutation_mask, 1 - individual[0], individual[0])
        return [mutated_genome, {}, {}]

    def __tournament(self, population):
        """
        Performs tournament selection on the population.

        Parameters
        ----------
        population : list
            The population to perform tournament selection on.

        Returns
        -------
        tuple
            The selected individuals.
        """
        pool1 = sample(population, self.genetic_parameters.tournament_size)
        pool2 = sample(population, self.genetic_parameters.tournament_size)
        return self.__select_best(pool1), self.__select_best(pool2)

    def __select_best(self, population):
        """
        Selects the best individuals from the population.

        Parameters
        ----------
        population : list
            The population to select the best individuals from.

        Returns
        -------
        list
            The best individuals.
        """

        def sort_population(_population: list, objective: str, index: int):
            _population.sort(key=lambda x: x[index][objective], reverse=True)
            return _population

        def select_top_individuals(_population, _metric, epsilon: float = 0.0, fairness_metric=False):
            index = 2 if fairness_metric else 1
            _population = sort_population(_population, _metric, index)
            best_value = _population[0][index][_metric] - epsilon
            last_index = np.argmax([individual[index][_metric] < best_value - epsilon for individual in _population])
            return _population[:last_index + 1]

        def lexicographic_selection(_population, metrics):
            for _metric in metrics:
                if len(_population) == 1:
                    break
                _population = select_top_individuals(_population, _metric, True)
            return _population[0]

        performance_metrics = list(population[0][1].keys())
        population = select_top_individuals(population, performance_metrics[0])

        lexicographic_permutations = np.array(list(itertools.permutations(performance_metrics)))
        winners = {}
        winners_count = {}
        for permutation in lexicographic_permutations:
            best_individual = lexicographic_selection(population, permutation)
            key = tuple(best_individual[0])
            winners[key] = best_individual
            winners_count[key] = 1 + winners_count.get(key, 0)

        # select the individual with the most wins
        best_individual_key = max(winners_count, key=winners_count.get)
        return winners[best_individual_key]

    def __performance_fitness(self, targets: pd.DataFrame, predictions: pd.DataFrame):
        """
        Computes the performance fitness of the predictions.

        Parameters
        ----------
        targets : pd.DataFrame
            The target values.
        predictions : pd.DataFrame
            The predicted values.

        Returns
        -------
        dict
            The performance fitness.
        """
        performance_evaluator = ModelEvaluator(targets, predictions)
        specificity = performance_evaluator.specificity()
        sensitivity = performance_evaluator.sensitivity()
        performance_metric = math.sqrt(specificity * sensitivity)
        return {'performance': performance_metric}

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
        df = pd.DataFrame(data.protected_features[self.sensitive_attribute], columns=[self.sensitive_attribute])
        evaluator = FairnessEvaluator(data.features, data.targets, predictions, df)

        ds = evaluator.discrimination_score()
        di = evaluator.disparate_impact()
        consistency = evaluator.consistency(k=3)
        false_positive_error_rate_balance_score = evaluator.false_positive_error_rate_balance_score()
        false_negative_error_rate_balance_score = evaluator.false_negative_error_rate_balance_score()

        return {
            'di': di,
            'ds': ds,
            'consistency': consistency,
            'false_positive_error_rate_balance_score': false_positive_error_rate_balance_score,
            'false_negative_error_rate_balance_score': false_negative_error_rate_balance_score
        }

    def __fitness(self, data: Dataset, individual, folds):
        """
        Computes the fitness of an individual.

        Parameters
        ----------
        data : Dataset
            The dataset to compute the fitness for.
        individual : list
            The individual to compute the fitness for.
        folds : int
            The number of folds for cross-validation.

        Returns
        -------
        list
            The fitness of the individual.
        """
        from sklearn.model_selection import cross_val_predict

        genotype = tuple(individual[0])
        if genotype in self.evaluated_individuals:
            return self.evaluated_individuals[genotype]

        if np.sum(individual[0]) == 0:
            return [individual[0], {'performance': -1}, {}]

        features = self.__phenotype(data, individual)

        model = get_random_forest(features.shape[0])

        x = features.to_numpy().astype(np.float32)
        y = data.targets.to_numpy().astype(np.float32).ravel()

        predictions = pd.DataFrame(cross_val_predict(model, x, y, cv=folds), columns=[PRED_OUTCOME])

        individual = [individual[0], self.__performance_fitness(data.targets, predictions),
                      self.__fairness_fitness(data, predictions)]

        self.evaluated_individuals[genotype] = individual

        return individual

    def __evaluate_population(self, data: Dataset, population, folds):
        """
        Evaluates the population.

        Parameters
        ----------
        data : Dataset
            The dataset to evaluate the population on.
        population : list
            The population to evaluate.
        folds : int
            The number of folds for cross-validation.

        Returns
        -------
        list
            The evaluated population.
        """

        for i, individual in enumerate(population):
            population[i] = self.__fitness(data, individual, folds)

        return population

    def __phenotype(self, data: Dataset, individual: list) -> pd.DataFrame:
        """
        Applies the phenotype to the data.

        Parameters
        ----------
        data : Dataset
            The dataset to apply the phenotype to.
        individual : list
            The individual to apply the phenotype.

        Returns
        -------
        pd.DataFrame
            The transformed dataset.
        """
        features = data.features.copy()
        features_to_drop = np.argwhere(individual[0] == 0).ravel()

        return features.drop(data.features.columns[features_to_drop], axis=1)

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
        self.genetic_parameters.individual_size = data.features.shape[1]
        self.population = self.__gen_ramped_population()
        self.sensitive_attribute = sensitive_attribute
        self.evaluated_individuals = {}

        self.cache_path = f'{self.algorithm_name}_{data.name}_{get_seed()}_{self.sensitive_attribute}'

    def __clean_cache__(self):
        """
        Cleans the cache.
        """
        self.evaluated_individuals = None
        self.population = None

    def transform(self, data: Dataset) -> Dataset:
        """
        Transforms the dataset using the fitted algorithm.

        Parameters
        ----------
        data : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        from sklearn.model_selection import KFold

        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=get_seed())
        best_individual = []

        for i in range(self.genetic_parameters.num_generations):
            self.population = self.__evaluate_population(data, self.population, folds)

            best_individual = self.__select_best(self.population)
            new_population = [best_individual]

            while len(new_population) < len(self.population):
                individual_1, individual_2 = self.__tournament(self.population)
                new_individual_1, new_individual_2 = self.__crossover(individual_1, individual_2)
                new_population.append(self.__mutation(new_individual_1))
                new_population.append(self.__mutation(new_individual_2))

            self.population = new_population

            if self.verbose and i % 5 == 0:
                logger.info(f'\t[LGAFFS] Generation {i + 1}/{self.genetic_parameters.num_generations} -'
                            f' Best individual: {best_individual[0]}')

        if self.verbose:
            logger.info(f'\t[LGAFFS] Generation {self.genetic_parameters.num_generations}/'
                        f'{self.genetic_parameters.num_generations} - Best individual: {best_individual[0]}')

        if not best_individual:
            data.error_flag = True
            self.__clean_cache__()
            return data

        data.features = self.__phenotype(data, best_individual)

        self.__clean_cache__()

        return data
