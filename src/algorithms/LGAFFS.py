import copy
import itertools
import math
from random import sample

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict

from algorithms.Algorithm import Algorithm
from constants import NUM_DECIMALS
from datasets import Dataset, update_dataset
from evaluation import FairnessEvaluator
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import logger


class LGAFFS(Algorithm):

    def __init__(self, num_gen: int = 1, pop_size: int = 1, indiv_size: int = 1, tour_size: int = 2, elite_num: int = 2,
                 prob_mut: float = 0.05, prob_cross: float = 0.8, min_feature_prob: float = 0.0,
                 max_feature_prob: float = 1.0, n_splits: int = 5, epsilon: float = 0.01):
        """
        Initialize an Evolutionary Machine Learning object.

        @param num_gen: Number of generations
        @param pop_size: Size of population
        @param indiv_size: Size of an individual
        @param prob_mut: Mutation probability
        @param prob_cross: Crossover probability
        @param tour_size: Size of tournament
        @param elite_num: Size of elitism
        """
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.individual_size = indiv_size
        self.tour_size = tour_size
        self.elite_num = elite_num
        self.prob_mut = prob_mut
        self.prob_cross = prob_cross
        self.min_feature_prob = min_feature_prob
        self.max_feature_prob = max_feature_prob
        self.n_splits = n_splits
        self.epsilon = epsilon
        self.sensitive_attribute = None
        self.population = None

    def __gen_individual(self, feature_prob: float = 1.0):
        if feature_prob:
            features_prob = np.random.uniform(low=self.min_feature_prob, high=self.max_feature_prob,
                                              size=self.individual_size)
            return np.array([1 if features_prob[i] > feature_prob else 0 for i in range(self.individual_size)])
        return np.random.randint(2, size=self.individual_size)

    def __gen_ramped_population(self):
        pop_feature_probs = np.random.uniform(low=self.min_feature_prob, high=self.max_feature_prob, size=self.pop_size)
        return [[self.__gen_individual(pop_feature_probs[i]), []] for i in range(self.pop_size)]

    def __uniform_crossover(self, parent1, parent2):
        """
        Implements the uniform crossover operator.

        @param parent1: First parent to be used in the crossover
        @param parent2: Second parent to be used in the crossover
        @return: If crossover happened, returns the two children. Otherwise, returns the parents
        """

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

    def __mutation(self, individual):
        """
        Implements flip-mutation.

        @param individual: Individual to be mutated
        @return: Mutated individual
        """
        new_indiv = copy.deepcopy(individual)
        random_probs = np.random.rand(individual[0].shape[0])
        for i in range(random_probs.shape[0]):
            if random_probs[i] < self.prob_mut:
                new_indiv[0][i] = 1 - new_indiv[0][i]
        return new_indiv

    def __tournament(self, population):
        """
        Implements the tournament mechanism for parent selection.

        @param population: Population to participate in the tournament
        @return: List of parents
        """

        def one_tour(local_population):
            """
            Implements a single run of the tournament mechanism for parent selection.

            @param local_population: Population to participate in the tournament
            @return: Winner of tournament
            """
            pool = sample(local_population, self.tour_size)
            return self.__select_best(pool)

        mate_pool = []
        for _ in range(2):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool[0], mate_pool[1]

    def __select_best(self, population):

        def __sort_population(local_population, objective):
            local_population.sort(key=lambda x: x[1][objective], reverse=True)
            return local_population

        def __select_top(local_population, objective):
            local_population = __sort_population(local_population, objective)
            best_value = local_population[0][1][objective] - self.epsilon
            last_index = np.argmax(
                [individual[1][objective] < best_value - self.epsilon for individual in local_population])
            return local_population[:last_index + 1]

        # select best performing individuals
        metric = 'performance'
        population = __select_top(population, metric)

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

    def __fitness(self, data: Dataset, individual, folds: KFold):
        data = self.__phenotype(data, individual)

        # invalid individual
        if data.features.shape[1] == 0:
            return {
                'performance': -1.0,
                'ds': -1.0,
                # 'consistency': -1.0,
                'fperbs': -1.0,
                'fnerbs': -1.0
            }

        model = RandomForestClassifier()

        predictions = cross_val_predict(model, data.features.to_numpy(), data.targets.to_numpy().ravel(), cv=folds)
        predicted_data = update_dataset(data, targets=predictions)

        fairness_evaluator = FairnessEvaluator(data, predicted_data, self.sensitive_attribute)
        performance_evaluator = ModelEvaluator(data, predicted_data)
        performance_metric = np.round(
            math.sqrt(performance_evaluator.specificity() * performance_evaluator.sensitivity()), decimals=NUM_DECIMALS)

        ds = fairness_evaluator.discrimination_score()
        di = fairness_evaluator.disparate_impact()
        consistency = fairness_evaluator.consistency(k=3)
        false_positive_error_rate_balance_score = fairness_evaluator.false_positive_error_rate_balance_score()
        false_negative_error_rate_balance_score = fairness_evaluator.false_negative_error_rate_balance_score()

        return {
            'performance': performance_metric,
            'di': di,
            'ds': ds,
            'consistency': consistency,
            'false_positive_error_rate_balance_score': false_positive_error_rate_balance_score,
            'false_negative_error_rate_balance_score': false_negative_error_rate_balance_score
        }

    def __phenotype(self, data: Dataset, individual: list) -> Dataset:
        new_data = copy.deepcopy(data)
        features_to_drop = np.argwhere(individual[0] == 0).ravel()
        selected_features = data.features.drop(data.features.columns[features_to_drop], axis=1)
        new_data.features = selected_features

        return new_data

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.individual_size = data.features.shape[1]
        self.population = self.__gen_ramped_population()
        self.sensitive_attribute = sensitive_attribute

    def transform(self, data: Dataset) -> Dataset:
        folds = KFold(n_splits=self.n_splits)
        best_individual = []

        for i in range(self.num_gen):
            logger.info(f'[LGAFFS] Generation {i + 1}/{self.num_gen}')
            for individual in self.population:
                individual[1] = self.__fitness(data, individual, folds)

            best_individual = self.__select_best(self.population)
            new_population = [best_individual]

            while len(new_population) < len(self.population):
                individual_1, individual_2 = self.__tournament(self.population)
                new_individual_1, new_individual_2 = self.__uniform_crossover(individual_1, individual_2)
                new_individual_1 = self.__mutation(new_individual_1)
                new_individual_2 = self.__mutation(new_individual_2)
                new_population.append(new_individual_1)
                new_population.append(new_individual_2)

            self.population = new_population

        if not best_individual:
            return data

        print(f'[LGAFFS] Best individual: {best_individual[0]}')

        return self.__phenotype(data, best_individual)
