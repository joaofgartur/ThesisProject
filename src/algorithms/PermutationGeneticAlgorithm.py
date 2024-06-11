import copy
import numpy as np
from random import sample

import pandas as pd

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from constants import NUM_DECIMALS
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import logger, write_dataframe_to_csv, get_generator
from protocol.assessment import get_classifier_predictions, fairness_assessment


class PermutationGeneticAlgorithm(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters,
                 unbiasing_algorithms_pool: [Algorithm],
                 surrogate_models_pool: [object]):

        super().__init__()
        self.is_binary = False

        self.genetic_parameters = genetic_parameters
        self.validation_data = None

        self.unbiasing_algorithms_pool = unbiasing_algorithms_pool
        self.surrogate_models_pool = surrogate_models_pool
        self.sensitive_attribute = ''
        self.population = []
        self.decoder = {}

    def __generate_individual(self) -> list:
        """
        Function that generates a random individual.
        Individual has a genotype of the form [[v_i, a_j]...[v_n, a_m]] where v_i is the value of the
        sensitive attribute and a_j is the index of the unbiasing algorithm.

        Returns
        -------

        """
        n = self.genetic_parameters.individual_size
        m = len(self.unbiasing_algorithms_pool)

        rng = get_generator()

        return [[[value, rng.integers(0, m)] for value in rng.permutation(n)], {}, {}]

    def __delete_individual(self, population, individual):
        for i in range(len(population)):
            is_match = (all(population[i][0]) == all(individual[0])
                        and all(population[i][1]) == all(individual[1])
                        and all(population[i][2]) == all(individual[2]))
            if is_match:
                population.pop(i)
                break
        return population

    def __decode_individual(self, individual) -> pd.DataFrame:
        genome = [[self.decoder[val],
                   self.unbiasing_algorithms_pool[algo].__class__.__name__] for val, algo in individual[0]]

        surrogate_models = [model.__class__.__name__ for model in self.surrogate_models_pool]

        metrics = {}
        for model in surrogate_models:
            for metric in individual[1][model]:
                metrics.update({f'{model}_{metric}': individual[1][model][metric]})

            for metric in individual[2][model]:
                metrics.update({f'{model}_{metric}': individual[2][model][metric]})

        decoded_individual = pd.DataFrame()
        decoded_individual['Genotype'] = [genome]
        decoded_individual = pd.concat([decoded_individual,
                                        pd.DataFrame([metrics], columns=[list(metrics.keys())])], axis=1)

        return decoded_individual

    def __generate_population(self):
        return [self.__generate_individual() for _ in range(self.genetic_parameters.population_size)]

    def __crossover(self, parent1, parent2):

        def pmx_crossover(_parent1, _parent2, crossover_probability):
            if get_generator().random() < crossover_probability:
                offspring1 = [_parent1[0], {}, {}]
                offspring2 = [_parent1[0], {}, {}]

                n = len(_parent1[0])
                cp1 = get_generator().integers(0, n)
                cp2 = get_generator().integers(0, n - 1)

                if cp2 == cp1:
                    cp2 += 1
                elif cp2 < cp1:
                    cp1, cp2 = cp2, cp1

                for i in range(cp1, cp2):
                    gene1 = _parent1[0][i]
                    gene2 = _parent2[0][i]

                    _parent1[0][i], _parent1[0][offspring1[0][gene2[0]][0]] = gene2, gene1
                    _parent2[0][i], _parent2[0][offspring2[0][gene1[0]][0]] = gene1, gene2

                    # Position bookkeeping
                    offspring1[0][gene1[0]], offspring1[0][gene2[0]] = offspring1[0][gene2[0]], offspring1[0][gene1[0]]
                    offspring2[0][gene1[0]], offspring2[0][gene2[0]] = offspring2[0][gene2[0]], offspring2[0][gene1[0]]

                return offspring1, offspring2
            else:
                return _parent1, _parent2

        return pmx_crossover(parent1, parent2, self.genetic_parameters.probability_crossover)

    def __mutation(self, individual):

        def scramble_mutation(_individual: list, probability_mutation: float):
            _mutated_individual = copy.deepcopy(_individual)
            n = len(_individual[0])

            if get_generator().random() < probability_mutation:
                index_1, index_2 = get_generator().choice(n, 2, replace=False)
                segment = _mutated_individual[0][index_1:index_2]
                get_generator().shuffle(segment)
                _mutated_individual[0][index_1:index_2] = segment

            return _mutated_individual

        # attribute values mutation
        mutated_individual = scramble_mutation(individual, self.genetic_parameters.probability_mutation)

        # unbiasing algorithms mutation
        for i in range(len(mutated_individual[0])):
            if get_generator().random() < self.genetic_parameters.probability_mutation:
                mutated_individual[0][i][1] = get_generator().integers(0, len(self.unbiasing_algorithms_pool))

        return mutated_individual

    def __select_best(self, population):

        def sort_population(_population, objective: tuple):
            index, model, metric = objective
            _population.sort(key=lambda x: x[index][model][metric], reverse=True)
            return _population

        def select_top_individuals(_population, objective: tuple, epsilon):
            index, model, metric = objective

            sorted_population = sort_population(_population, objective)
            boundary_value = sorted_population[0][index][model][metric] - epsilon
            last_index = np.argmax(
                [individual[index][model][metric] < boundary_value - epsilon for individual in _population])

            return _population[:last_index + 1]

        def lexicographic_selection(_population, objective: tuple):
            index, model = objective
            metrics = rng.permutation([key for key in _population[0][index][model].keys()])

            for metric in metrics:
                if len(_population) == 1:
                    break
                _population = select_top_individuals(_population, (index, model, metric), 0.0)

            return _population

        rng = get_generator()

        surrogate_models = [model.__class__.__name__ for model in self.surrogate_models_pool]
        surrogate_models_order = rng.permutation(surrogate_models)

        for model in surrogate_models_order:
            if len(population) == 1:
                break
            population = lexicographic_selection(population, (1, model))
            population = lexicographic_selection(population, (2, model))

        return population[0]

    def __new_population(self, elite_pop, offsprings):
        offset = len(offsprings) - len(elite_pop)

        best_offspring = []
        for _ in range(len(offsprings)):
            best = self.__select_best(offsprings)
            best_offspring.append(best)
            offsprings = self.__delete_individual(offsprings, best)

        new_population = elite_pop + best_offspring[:offset]
        return new_population

    def __tournament(self, population):

        def one_tour(local_population):
            pool = sample(local_population, self.genetic_parameters.tournament_size)
            return self.__select_best(pool)

        mate_pool = []
        for _ in range(len(population)):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool

    def __performance_fitness(self, data: Dataset, predictions: Dataset):
        performance_evaluator = ModelEvaluator(data, predictions)

        return {
            'accuracy': performance_evaluator.accuracy(),
            'f1_score': performance_evaluator.f1_score(),
            'auc': performance_evaluator.auc()
        }

    def __fairness_fitness(self, data: Dataset, predictions: Dataset):
        metrics = fairness_assessment(data, predictions, self.sensitive_attribute)

        result = {}
        for metric in metrics.columns:
            if metrics[metric].dtype == 'object':
                continue
            sum_of_squares = np.round(np.sum((metrics[metric] - 1.0) ** 2), decimals=NUM_DECIMALS)
            result.update({metric: sum_of_squares})

        return result

    def __fitness(self, data: Dataset, individual):

        def _is_invalid(_individual):
            genes = set([gene[0] for gene in _individual[0]])
            return len(genes) != len(_individual[0])

        data = self.__phenotype(data, individual)

        if _is_invalid(individual) or data.error_flag:
            for model in self.surrogate_models_pool:
                individual[1].update({model.__class__.__name__: self.__performance_fitness(self.validation_data,
                                                                                            self.validation_data)})
                individual[2].update({model.__class__.__name__: self.__fairness_fitness(self.validation_data,
                                                                                        self.validation_data)})

            for model, metrics in individual[1].items():
                for metric in metrics:
                    individual[1][model][metric] = -1.0

            return individual

        for model in self.surrogate_models_pool:
            model_predictions = get_classifier_predictions(model, data, self.validation_data)
            individual[1].update({model.__class__.__name__: self.__performance_fitness(self.validation_data,
                                                                                       model_predictions)})
            individual[2].update({model.__class__.__name__: self.__fairness_fitness(self.validation_data,
                                                                                    model_predictions)})

        return individual

    def __phenotype(self, data: Dataset, individual):
        dummy_values = data.get_dummy_protected_feature(self.sensitive_attribute)
        transformed_data = copy.deepcopy(data)
        dimensions = transformed_data.features.shape

        for value, algorithm in individual[0]:

            if transformed_data.error_flag:
                break

            if self.sensitive_attribute not in transformed_data.features.columns:
                sensitive_attribute = pd.DataFrame(dummy_values[self.decoder[value]], columns=[self.sensitive_attribute])
                transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

            if transformed_data.features.shape[0] != dimensions[0]:
                sensitive_values = transformed_data.get_protected_attributes()
                sampled_values = transformed_data.protected_attributes.loc[transformed_data.sampled_indexes]

                new_sensitive_values = pd.concat([sensitive_values, sampled_values]).reset_index()

                transformed_data.protected_attributes = new_sensitive_values
                dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)

                dimensions = transformed_data.features.shape

            transformed_data.set_feature(self.sensitive_attribute, dummy_values[self.decoder[value]])
            self.unbiasing_algorithms_pool[algorithm].fit(transformed_data, self.sensitive_attribute)
            transformed_data = self.unbiasing_algorithms_pool[algorithm].transform(transformed_data)

        return transformed_data

    def set_validation_data(self, validation_data: Dataset):
        self.validation_data = validation_data

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.decoder = data.features_mapping[sensitive_attribute]
        self.genetic_parameters.individual_size = len(data.features_mapping[sensitive_attribute])
        self.population = self.__generate_population()
        self.sensitive_attribute = sensitive_attribute

    def transform(self, dataset: Dataset) -> Dataset:
        population = self.population

        population = [self.__fitness(dataset, individual) for individual in population]
        best_individual = self.__select_best(population)
        best_individuals = self.__decode_individual(best_individual)

        logger.info(f'[PGA] Generation {0}/{self.genetic_parameters.num_generations} '
                    f'Best Individual: {best_individuals.iloc[-1]["Genotype"]}')

        for i in range(1, self.genetic_parameters.num_generations):
            parents = self.__tournament(population)

            # Crossover
            new_parents = []
            for j in range(0, len(population) - 1, 2):
                child1, child2 = self.__crossover(parents[j], parents[j + 1])
                new_parents.append(child1)
                new_parents.append(child2)

            # Mutation
            offsprings = []
            for individual in parents:
                mutated_individual = self.__mutation(individual)
                offsprings.append(self.__fitness(dataset, mutated_individual))

            # Survivors selection - elitism
            elite_pop = population[:self.genetic_parameters.elite_size]
            population = self.__new_population(elite_pop, offsprings)

            best_individual = self.__select_best(population)
            best_individuals = pd.concat([best_individuals, self.__decode_individual(best_individual)])

            logger.info(
                f'[PGA] Generation {i}/{self.genetic_parameters.num_generations} '
                f'Best Individual: {best_individuals.iloc[-1]["Genotype"]}')

        write_dataframe_to_csv(best_individuals, f'pga_{self.sensitive_attribute}', 'best_individuals')

        return self.__phenotype(dataset, best_individual)
