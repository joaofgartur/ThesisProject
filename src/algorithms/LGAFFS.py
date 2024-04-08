import copy
import itertools
import math
from random import sample

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters, uniform_crossover, bit_flip_mutation, select_best
from constants import NUM_DECIMALS
from datasets import Dataset, update_dataset
from evaluation import FairnessEvaluator
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import logger


class LGAFFS(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01):
        super().__init__()

        # genetic parameters
        self.genetic_parameters = genetic_parameters

        self.min_feature_prob = min_feature_prob
        self.max_feature_prob = max_feature_prob
        self.n_splits = n_splits
        self.epsilon = epsilon

        self.sensitive_attribute = None
        self.population = None

    def __gen_individual(self, feature_prob: float = 1.0):
        if feature_prob:
            features_prob = np.random.uniform(low=self.min_feature_prob, high=self.max_feature_prob,
                                              size=self.genetic_parameters.individual_size)
            # genome, performance, fairness
            return [np.array(
                [1 if features_prob[i] > feature_prob else 0 for i in range(self.genetic_parameters.individual_size)]),
                    [], []]
        return [np.random.randint(2, size=self.genetic_parameters.individual_size), {}, {}]

    def __gen_ramped_population(self):
        n = self.genetic_parameters.population_size
        pop_feature_probs = np.random.uniform(low=self.min_feature_prob, high=self.max_feature_prob, size=n)
        return [self.__gen_individual(float(pop_feature_probs[i])) for i in range(n)]

    def __uniform_crossover(self, parent1, parent2):
        return uniform_crossover(parent1, parent2, self.genetic_parameters.probability_crossover)

    def __mutation(self, individual: list):
        return bit_flip_mutation(individual, self.genetic_parameters.probability_mutation)

    def __tournament(self, population):

        def one_tour(local_population):
            pool = sample(local_population, self.genetic_parameters.tournament_size)
            return self.__select_best(pool)

        mate_pool = []
        for _ in range(2):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool[0], mate_pool[1]

    def __select_best(self, population):
        return select_best(population)

    def __performance_fitness(self, data: Dataset, predictions: Dataset):
        performance_evaluator = ModelEvaluator(data, predictions)

        performance_metric = np.round(
            math.sqrt(performance_evaluator.specificity() * performance_evaluator.sensitivity()), decimals=NUM_DECIMALS)

        return {'performance': performance_metric}

    def __fairness_fitness(self, data: Dataset, predictions: Dataset):
        fairness_evaluator = FairnessEvaluator(data, predictions, self.sensitive_attribute)

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

    def __fitness(self, data: Dataset, individual, folds: KFold):

        # invalid individual
        if sum(individual[0]) == 0:
            return [individual[0], {'performance': -1}, {}]

        data = self.__phenotype(data, individual)

        model = RandomForestClassifier()
        predictions = cross_val_predict(model, data.features.to_numpy(), data.targets.to_numpy().ravel(), cv=folds)
        predicted_data = update_dataset(data, targets=predictions)

        return [individual[0], self.__performance_fitness(data, predicted_data),
                self.__fairness_fitness(data, predicted_data)]

    def __phenotype(self, data: Dataset, individual: list) -> Dataset:
        transformed_data = copy.deepcopy(data)

        features_to_drop = np.argwhere(individual[0] == 0).ravel()
        selected_features = data.features.drop(data.features.columns[features_to_drop], axis=1)
        transformed_data.features = selected_features

        return transformed_data

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.genetic_parameters.individual_size = data.features.shape[1]
        self.population = self.__gen_ramped_population()
        self.sensitive_attribute = sensitive_attribute

    def transform(self, data: Dataset) -> Dataset:
        folds = KFold(n_splits=self.n_splits)
        best_individual = []

        for i in range(self.genetic_parameters.num_generations):
            logger.info(f'[LGAFFS] Generation {i + 1}/{self.genetic_parameters.num_generations}')

            for j in range(len(self.population)):
                self.population[j] = self.__fitness(data, self.population[j], folds)

            best_individual = self.__select_best(self.population)
            new_population = [best_individual]

            while len(new_population) < len(self.population):
                individual_1, individual_2 = self.__tournament(self.population)

                # crossover
                new_individual_1, new_individual_2 = self.__uniform_crossover(individual_1, individual_2)

                # mutation
                new_population.append(self.__mutation(new_individual_1))
                new_population.append(self.__mutation(new_individual_2))

            self.population = new_population

        if not best_individual:
            return data

        print(f'[LGAFFS] Best individual: {best_individual[0]}')

        return self.__phenotype(data, best_individual)
