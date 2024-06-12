import copy
import itertools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from random import sample

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from constants import NUM_DECIMALS
from datasets import Dataset, update_dataset
from evaluation import FairnessEvaluator
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import get_generator, get_seed


class LexicographicGeneticAlgorithmFairFeatureSelection(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01, verbose: bool = False):
        super().__init__()

        # genetic parameters
        self.genetic_parameters = genetic_parameters

        self.min_feature_prob = min_feature_prob
        self.max_feature_prob = max_feature_prob
        self.n_splits = n_splits
        self.epsilon = epsilon

        self.verbose = verbose

        self.sensitive_attribute = None
        self.population = None

    def __gen_individual(self, feature_prob: float = 1.0):
        if feature_prob:
            features_prob = get_generator().uniform(low=self.min_feature_prob, high=self.max_feature_prob,
                                                    size=self.genetic_parameters.individual_size)
            # genome, performance, fairness
            return [np.array(
                [1 if features_prob[i] > feature_prob else 0 for i in range(self.genetic_parameters.individual_size)]),
                [], []]
        return [get_generator().randint(2, size=self.genetic_parameters.individual_size), {}, {}]

    def __gen_ramped_population(self):
        n = self.genetic_parameters.population_size
        pop_feature_probs = get_generator().uniform(low=self.min_feature_prob, high=self.max_feature_prob, size=n)
        return [self.__gen_individual(float(pop_feature_probs[i])) for i in range(n)]

    def __crossover(self, parent1, parent2):

        def uniform_crossover(_parent1: list, _parent2: list, probability_crossover: float):
            value = get_generator().random()
            if value < probability_crossover:

                offspring1 = [np.zeros(_parent1[0].shape, dtype=int), {}, {}]
                offspring2 = [np.zeros(_parent1[0].shape, dtype=int), {}, {}]

                indexes_choice = get_generator().choice(_parent1[0].shape[0])
                index_mask = indexes_choice < 0.5

                offspring1[0][index_mask] = _parent1[0][index_mask]
                offspring2[0][index_mask] = _parent2[0][index_mask]

                offspring1[0][~index_mask] = _parent2[0][~index_mask]
                offspring2[0][~index_mask] = _parent1[0][~index_mask]

                return offspring1, offspring2
            else:
                return _parent1, _parent2

        return uniform_crossover(parent1, parent2, self.genetic_parameters.probability_crossover)

    def __mutation(self, individual: list):

        def bit_flip_mutation(_individual: list, probability_mutation: float):
            new_individual = copy.deepcopy(_individual)
            random_probs = get_generator().random(_individual[0].shape[0])
            for i in range(random_probs.shape[0]):
                if random_probs[i] < probability_mutation:
                    new_individual[0][i] = 1 - new_individual[0][i]
            return new_individual

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

        def sort_population(_population: list, objective: str, index: int):
            _population.sort(key=lambda x: x[index][objective], reverse=True)
            return _population

        def select_top_individuals(_population, _metric, epsilon: float = 0.0, fairness_metric=False):
            index = 1
            if fairness_metric:
                index = 2

            _population = sort_population(_population, _metric, index)
            best_value = _population[0][index][_metric] - epsilon
            last_index = np.argmax([individual[index][_metric] < best_value - epsilon for individual in _population])
            return _population[:last_index + 1]

        def lexicographic_selection(_population, metrics):
            for _metric in metrics:
                if len(_population) == 1:
                    break
                _population = select_top_individuals(_population, _metric)

            return _population[0]

            # sort by performance metrics
        for metric in list(population[0][1].keys()):
            population = select_top_individuals(population, metric)

        # lexicographic fairness selection
        lexicographic_permutations = list(itertools.permutations(list(population[0][1].keys())))
        winners = {}
        winners_count = {}
        for permutation in lexicographic_permutations:
            best_individual = lexicographic_selection(population, permutation)

            individual_key = str(best_individual[0])
            if individual_key not in winners.keys():
                winners.update({individual_key: best_individual})
                winners_count.update({individual_key: 1})
            else:
                winners_count.update({individual_key: winners_count[individual_key] + 1})

        # select the individual with the most wins
        best_individual_key = max(winners_count, key=lambda x: winners_count[x])

        return winners[best_individual_key]

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

        if sum(individual[0]) == 0:
            return [individual[0], {'performance': -1}, {}]

        data = self.__phenotype(data, individual)

        model = RandomForestClassifier(random_state=get_seed())
        predictions = cross_val_predict(model, data.features.to_numpy(), data.targets.to_numpy().ravel(), cv=folds)
        predicted_data = update_dataset(data, targets=predictions)

        return [individual[0], self.__performance_fitness(data, predicted_data),
                self.__fairness_fitness(data, predicted_data)]

    def __multithread_fitness(self, data: Dataset, population, folds: KFold):
        num_threads = min(os.cpu_count(), 15)

        def evaluate_fitness(individual):
            return self.__fitness(data, individual, folds)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            fitness_results = list(executor.map(evaluate_fitness, population))

        return fitness_results

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
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=get_seed())
        best_individual = []

        for i in range(self.genetic_parameters.num_generations):

            for j, individual in enumerate(self.__multithread_fitness(data, self.population, folds)):
                self.population[j] = individual

            best_individual = self.__select_best(self.population)
            new_population = [best_individual]

            while len(new_population) < len(self.population):
                individual_1, individual_2 = self.__tournament(self.population)

                # crossover
                new_individual_1, new_individual_2 = self.__crossover(individual_1, individual_2)

                # mutation
                new_population.append(self.__mutation(new_individual_1))
                new_population.append(self.__mutation(new_individual_2))

            self.population = new_population

            if self.verbose:
                print(f'Generation {i + 1}/{self.genetic_parameters.num_generations} -'
                      f' Best individual: {best_individual[0]}')

        if not best_individual:
            return data

        return self.__phenotype(data, best_individual)
