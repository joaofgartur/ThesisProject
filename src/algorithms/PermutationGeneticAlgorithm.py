import copy
import itertools
from random import sample

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from helpers import abs_diff, logger
from protocol.assessment import surrogate_models, get_model_evaluators


class PermutationGeneticAlgorithm(Algorithm):

    def __init__(self, base_algorithm: Algorithm, num_gen: int = 1, pop_size: int = 1, indiv_size: int = 1,
                 tour_size: int = 2, elite_num: int = 2,
                 prob_mut: float = 0.05, prob_cross: float = 0.8, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0):

        self.base_algorithm = base_algorithm
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.individual_size = indiv_size
        self.tour_size = tour_size
        self.elite_num = elite_num
        self.prob_mut = prob_mut
        self.prob_cross = prob_cross
        self.min_feature_prob = min_feature_prob
        self.max_feature_prob = max_feature_prob
        self.validation_data = None
        self.sensitive_attribute = ''
        self.population = []
        self.decoder = {}

    def __generate_individual(self) -> np.ndarray:
        return np.random.permutation(np.arange(0, self.individual_size))[:self.individual_size]

    def __delete_individual(self, population, individual):
        for i in range(len(population)):
            if all(population[i][0]) == all(individual[0]) and all(population[i][1]) == all(individual[1]):
                population.pop(i)
                break
        return population

    def __generate_population(self):
        return [[self.__generate_individual(), []] for _ in range(self.pop_size)]

    def __uniform_crossover(self, parent1, parent2):
        value = np.random.random()
        if value < self.prob_cross:

            offspring1 = [np.zeros(parent1[0].shape, dtype=int), []]
            offspring2 = [np.zeros(parent1[0].shape, dtype=int), []]

            indexes_choice = np.random.rand(parent1[0].shape[0])
            index_mask = indexes_choice < 0.5

            offspring1[0][index_mask] = parent1[0][index_mask]
            offspring2[0][index_mask] = parent2[0][index_mask]

            offspring1[0][~index_mask] = parent2[0][~index_mask]
            offspring2[0][~index_mask] = parent1[0][~index_mask]

            return offspring1, offspring2
        else:
            return parent1, parent2

    def __scramble_mutation(self, individual):
        mutated_individual = copy.deepcopy(individual)

        if np.random.random() < self.prob_mut:
            index_1, index_2 = np.random.choice(self.individual_size, 2, replace=False)
            segment = mutated_individual[0][index_1:index_2]
            np.random.shuffle(segment)
            mutated_individual[0][index_1:index_2] = segment

        return mutated_individual

    def __best(self, population):

        def __sort_population(local_population, objective, reverse):
            local_population.sort(key=lambda x: x[1][objective], reverse=reverse)
            return local_population

        def __select_top(local_population, objective, reverse=False):
            local_population = __sort_population(local_population, objective, reverse)
            best_value = local_population[0][1][objective]
            last_index = np.argmax(
                [individual[1][objective] < best_value for individual in local_population])
            return local_population[:last_index + 1]

        # select best performing individuals
        metric = 'performance'
        population = __select_top(population, metric, reverse=True)

        # drop the performance metric
        metrics = list(population[0][1].keys())
        metrics.remove(metric)

        # for each permutation select the winner, a.k.a the best individual
        permutations = list(itertools.permutations(metrics))
        winners = {}
        winners_count = {}
        for permutation in permutations:

            for metric in permutation:
                if len(population) > 1:
                    break
                population = __select_top(population, metric)

            # choose the best individual
            best_individual = population[0]

            individual_key = str(best_individual[0])  # use genome as dict key
            if individual_key not in winners.keys():
                winners.update({individual_key: best_individual})
                winners_count.update({individual_key: 1})
            else:
                winners_count.update({individual_key: winners_count[individual_key] + 1})

        # select the individual with the most wins
        best_individual_key = max(winners_count, key=lambda x: winners_count[x])

        return winners[best_individual_key]

    def __new_population(self, elite_pop, offsprings):
        offset = len(offsprings) - len(elite_pop)

        best_offsprings = []
        for _ in range(len(offsprings)):
            best_offspring = self.__best(offsprings)
            best_offsprings.append(best_offspring)
            offsprings = self.__delete_individual(offsprings, best_offspring)

        new_population = elite_pop + best_offsprings[:offset]
        return new_population

    def __tournament(self, population):

        def one_tour(local_population):
            pool = sample(local_population, self.tour_size)
            return self.__best(pool)

        mate_pool = []
        for _ in range(len(population)):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool

    def __fitness(self, data: Dataset, individual):

        def _metric_score(series: pd.Series):
            return abs_diff(series.min(), series.max())

        def _is_invalid(_individual):
            unique_values = []
            for value in _individual[0]:
                if value in unique_values:
                    return True
                unique_values.append(value)
            return False

        if _is_invalid(individual):
            return {'performance': -1}

        transformed_data = self.__phenotype(data, individual)

        model = RandomForestClassifier()

        individual_fairness, individual_performance = get_model_evaluators(model,
                                                                           transformed_data,
                                                                           self.validation_data,
                                                                           self.sensitive_attribute)

        fairness_metrics = individual_fairness.evaluate(stats=False)

        result = {'performance': individual_performance.accuracy()}

        for metric in fairness_metrics:
            if metric == 'Label':
                continue

            result.update({metric: _metric_score(fairness_metrics[metric])})

        return result

    def __phenotype(self, data: Dataset, individual):
        dummy_values = data.get_dummy_protected_feature(self.sensitive_attribute)
        values_permutation = [self.decoder[i] for i in individual[0]]

        transformed_data = copy.deepcopy(data)

        for value in values_permutation:
            transformed_data.set_feature(self.sensitive_attribute, dummy_values[value])
            self.base_algorithm.fit(transformed_data, self.sensitive_attribute)
            transformed_data = self.base_algorithm.transform(transformed_data)

        return transformed_data

    def set_validation_data(self, validation_data: Dataset):
        self.validation_data = validation_data

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.decoder = data.features_mapping[sensitive_attribute]
        self.individual_size = len(data.features_mapping[sensitive_attribute])
        self.population = self.__generate_population()
        self.sensitive_attribute = sensitive_attribute

    def transform(self, dataset: Dataset) -> Dataset:
        population = self.population

        population = [[individual[0], self.__fitness(dataset, individual)] for individual in population]
        best_individual = self.__best(population)
        logger.info(f'[PGA] Generation {0}/{self.num_gen} Best Individual: {[self.decoder[i] for i in best_individual[0]]}')

        for i in range(1, self.num_gen):
            parents = self.__tournament(population)

            # Crossover
            new_parents = []
            for j in range(0, len(population) - 1, 2):
                child1, child2 = self.__uniform_crossover(parents[j], parents[j + 1])
                new_parents.append(child1)
                new_parents.append(child2)

            # Mutation
            offsprings = []
            for individual in new_parents:
                new_individual = self.__scramble_mutation(individual)
                offsprings.append(
                    [new_individual[0], self.__fitness(dataset, individual)])

            # Survivors selection - elitism
            elite_pop = population[:self.elite_num]
            population = self.__new_population(elite_pop, offsprings)

            best_individual = self.__best(population)
            logger.info(
                f'[PGA] Generation {i}/{self.num_gen} Best Individual: {[self.decoder[i] for i in best_individual[0]]}')

        return self.__phenotype(dataset, best_individual)



