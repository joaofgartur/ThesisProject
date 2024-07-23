import copy
import os
from itertools import product, combinations_with_replacement
from math import factorial

import numpy as np
from random import sample

import pandas as pd

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import write_dataframe_to_csv, get_generator, dict_to_dataframe, read_csv_to_dataframe, \
    delete_directory, get_seed
from protocol.assessment import get_classifier_predictions, fairness_assessment


class PermutationGeneticAlgorithm(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters,
                 unbiasing_algorithms_pool: [Algorithm],
                 surrogate_models_pool: [object],
                 threshold_k: int,
                 verbose: bool = False):

        super().__init__()
        self.genetic_search_flag = False
        self.is_binary = False
        self.needs_auxiliary_data = True
        self.algorithm_name = 'PermutationGeneticAlgorithm'
        self.cache_path = None

        self.genetic_parameters = genetic_parameters

        self.num_classes = 0
        self.unbiasing_algorithms_pool = unbiasing_algorithms_pool
        self.surrogate_models_pool = surrogate_models_pool
        self.sensitive_attribute = ''
        self.population = []
        self.decoder = {}

        self.problem_dimension = 0
        self.threshold_k = threshold_k

        self.verbose = verbose
        self.evaluated_individuals = {}
        self.valid_individuals = {}

    def __do_genetic_search(self):
        num_algorithms = len(self.unbiasing_algorithms_pool)
        max_length = 2 * self.num_classes
        n = self.num_classes * num_algorithms

        self.problem_dimension = 0
        for length in range(1, max_length + 1):
            self.problem_dimension += np.divide(factorial(n + length - 1), factorial(length) * factorial(n - 1))

        return self.problem_dimension >= factorial(self.threshold_k - 1)

    def __flatten_genotype(self, genotype: list[(int, int)]) -> str:
        return ''.join(f'{gene}{value}' for gene, value in genotype)

    def __generate_individual(self) -> list:
        """
        Function that generates a random individual.
        Individual has a genotype of the form [[v_i, a_j]...[v_n, a_m]] where v_i is the value of the
        sensitive attribute and a_j is the index of the unbiasing algorithm.

        Returns
        -------

        """
        rng = get_generator()

        num_attribute_classes = self.num_classes
        m = len(self.unbiasing_algorithms_pool)
        length = rng.integers(1,  2 * num_attribute_classes)

        genotype = [[rng.integers(0, num_attribute_classes), rng.integers(0, m)] for _ in range(length)]

        return [genotype, {}, {}]

    def __decode_individual(self, individual) -> pd.DataFrame:
        genome = [[self.decoder[val],
                   self.unbiasing_algorithms_pool[algo].__class__.__name__] for val, algo in individual[0]]

        metrics = {}
        models = individual[1].keys()
        for model in models:
            for metric, value in individual[1][model].items():
                metrics.update({f'{model}_{metric}': value})

            for metric, value in individual[2][model].items():
                metrics.update({f'{model}_{metric}': value})

        metrics = dict_to_dataframe(metrics)
        decoded_individual = pd.DataFrame()
        decoded_individual['Genotype'] = [genome]
        decoded_individual = pd.concat([decoded_individual, metrics], axis=1)

        return decoded_individual

    def __save_individual__(self, individual, transformed_dataset: Dataset):
        genome = self.__flatten_genotype(individual[0])

        features_file = f'features_{genome}'
        write_dataframe_to_csv(df=transformed_dataset.features, filename=features_file, path=self.cache_path)

        targets_file = f'targets_{genome}'
        write_dataframe_to_csv(df=transformed_dataset.targets, filename=targets_file, path=self.cache_path)

        protected_features_file = f'protected_features_{genome}'
        write_dataframe_to_csv(df=transformed_dataset.protected_features, filename=protected_features_file,
                               path=self.cache_path)

    def __fetch_individual__(self, individual: list[list], dataset: Dataset):
        genome = self.__flatten_genotype(individual[0])

        features_file = f'features_{genome}'
        dataset.features = read_csv_to_dataframe(features_file, self.cache_path)

        targets_file = f'targets_{genome}'
        dataset.targets = read_csv_to_dataframe(targets_file, self.cache_path)

        protected_features_file = f'protected_features_{genome}'
        dataset.protected_features = read_csv_to_dataframe(protected_features_file, self.cache_path)

        return dataset

    def __clean_cache__(self):
        delete_directory(self.cache_path)

    def __compute_population_average_fitness(self, population):

        decoded_population = pd.DataFrame()
        for individual in population:
            decoded_population = pd.concat([decoded_population, self.__decode_individual(individual)])

        metrics_columns = decoded_population.select_dtypes(include=[np.number]).columns
        decoded_population = decoded_population[metrics_columns]

        population_mean = decoded_population.mean().transpose().to_frame().transpose().add_suffix('mean')

        return population_mean

    def __generate_population(self):

        if self.genetic_search_flag:
            return [self.__generate_individual() for _ in range(self.genetic_parameters.population_size)]

        num_algorithms = len(self.unbiasing_algorithms_pool)
        genes_pool = list(product(range(self.num_classes), range(num_algorithms)))
        population = []
        max_individual_length = 2 * self.num_classes
        for length in range(1, max_individual_length + 1):
            for genotype in combinations_with_replacement(genes_pool, length):
                individual = [genotype, {}, {}]
                population.append(individual)

        return population

    def __crossover(self, parent1: list, parent2: list):
        value = get_generator().random()
        if value < self.genetic_parameters.probability_crossover:
            len1 = len(parent1[0])
            len2 = len(parent2[0])
            max_length = max(len1, len2)

            crossover_mask = get_generator().integers(0, 2, size=max_length)

            offspring1_genotype = [None] * max_length
            offspring2_genotype = [None] * max_length

            for i in range(max_length):
                if i < len1 and i < len2:
                    if crossover_mask[i] == 0:
                        offspring1_genotype[i] = parent1[0][i]
                        offspring2_genotype[i] = parent2[0][i]
                    else:
                        offspring1_genotype[i] = parent2[0][i]
                        offspring2_genotype[i] = parent1[0][i]
                elif i < len1:
                    offspring1_genotype[i] = parent1[0][i]
                elif i < len2:
                    offspring2_genotype[i] = parent2[0][i]

            offspring1_genotype = list(filter(lambda item: item is not None, offspring1_genotype))
            offspring2_genotype = list(filter(lambda item: item is not None, offspring2_genotype))

            return [(offspring1_genotype, {}), {}, {}], [(offspring2_genotype, {}), {}, {}]

        return parent1, parent2

    def __mutation(self, individual):

        def scramble_mutation(_individual: list, probability_mutation: float):
            genome = copy.deepcopy(_individual[0])
            n = len(genome)

            if n < 2:
                return _individual

            if get_generator().random() < probability_mutation:
                index_1, index_2 = get_generator().choice(n, 2, replace=False)
                segment = genome[index_1:index_2]
                get_generator().shuffle(segment)
                genome[index_1:index_2] = segment

            return [genome, {}, {}]

        # attribute values mutation
        mutated_individual = scramble_mutation(individual, self.genetic_parameters.probability_mutation)

        # unbiasing algorithms mutation
        mutation_probabilities = get_generator().random(len(mutated_individual[0]))
        mutation_indices = np.where(mutation_probabilities < self.genetic_parameters.probability_mutation)[0]
        for i in mutation_indices:
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
            population = lexicographic_selection(population, (1, model))  # performance
            population = lexicographic_selection(population, (2, model))  # fairness

        return population

    def __new_population(self, elite_pop, offsprings):
        offset = len(offsprings) - len(elite_pop)

        best_offspring = []
        for _ in range(len(offsprings)):
            sorted_offsprings = self.__select_best(offsprings)
            best = sorted_offsprings[0]
            best_offspring.append(best)
            sorted_offsprings.pop(0)

        new_population = elite_pop + best_offspring[:offset]
        return new_population

    def __tournament(self, population):

        def one_tour(local_population):
            pool = sample(local_population, self.genetic_parameters.tournament_size)
            return self.__select_best(pool)[0]

        mate_pool = []
        for _ in range(len(population)):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool

    def __find_longest_genotype_match(self, flattened_genotype) -> str | None:
        sorted_genotypes = sorted(self.evaluated_individuals.keys(), key=len, reverse=True)
        for key in sorted_genotypes:
            if flattened_genotype.startswith(key):
                return str(key)

        return None

    def __phenotype(self, data: Dataset, individual):

        flattened_genotype = self.__flatten_genotype(individual[0])
        longest_matching_genotype = self.__find_longest_genotype_match(flattened_genotype)

        transformed_data = copy.deepcopy(data)
        self.valid_individuals[flattened_genotype] = True

        if longest_matching_genotype is None:
            genotype_to_decode = individual[0]
        else:
            match_length = len(self.evaluated_individuals[longest_matching_genotype][0])

            if not self.valid_individuals[longest_matching_genotype]:
                self.valid_individuals[flattened_genotype] = False
                return data

            genotype_to_decode = individual[0][match_length:]
            transformed_data = self.__fetch_individual__(self.evaluated_individuals[longest_matching_genotype], data)

        dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
        dimensions = transformed_data.features.shape

        if self.sensitive_attribute not in transformed_data.features.columns:
            previous_value = self.evaluated_individuals[longest_matching_genotype][0][-1][0]
            sensitive_attribute = pd.DataFrame({self.sensitive_attribute: dummy_values[self.decoder[previous_value]]})
            transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

        for value, algorithm in genotype_to_decode:

            transformed_data.set_feature(self.sensitive_attribute, dummy_values[self.decoder[value]])
            unbiasing_algorithm = self.unbiasing_algorithms_pool[algorithm]
            unbiasing_algorithm.fit(transformed_data, self.sensitive_attribute)
            transformed_data = unbiasing_algorithm.transform(transformed_data)

            if transformed_data.features.shape[0] != dimensions[0]:
                dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
                dimensions = transformed_data.features.shape

            if self.sensitive_attribute not in transformed_data.features.columns:
                sensitive_attribute = pd.DataFrame({self.sensitive_attribute: dummy_values[self.decoder[value]]})
                transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

            if transformed_data.error_flag:
                self.valid_individuals[flattened_genotype] = False
                break

        self.__save_individual__(individual, transformed_data)

        return transformed_data

    def __performance_fitness(self, data: Dataset, predictions: Dataset):
        performance_evaluator = ModelEvaluator(data, predictions)

        return {
            'performance_accuracy': performance_evaluator.accuracy(),
            'performance_f1_score': performance_evaluator.f1_score(),
            'performance_auc': performance_evaluator.auc()
        }

    def __fairness_fitness(self, data: Dataset, predictions: Dataset):
        metrics = fairness_assessment(data, predictions, self.sensitive_attribute)

        result = {}
        numerical_columns = metrics.select_dtypes(include=[np.number]).columns
        for metric in numerical_columns:
            result[metric] = np.sum((metrics[metric] - 1.0) ** 2)
        return result

    def __fitness(self, data: Dataset, individual):

        flattened_genotype = self.__flatten_genotype(individual[0])
        if flattened_genotype in self.evaluated_individuals:
            return self.evaluated_individuals[flattened_genotype]

        data = self.__phenotype(data, individual)

        if len(individual[0]) > 2 * self.num_classes or not self.valid_individuals[flattened_genotype]:
            for model in self.surrogate_models_pool:
                individual[1][model.__class__.__name__] = {'performance_accuracy': -1.0,
                                                           'performance_f1_score': -1.0,
                                                           'performance_auc': -1.0}
                individual[2][model.__class__.__name__] = {'fairness_disparate_impact': -1.0,
                                                           'fairness_discrimination_score': -1.0,
                                                           'fairness_true_positive_rate_diff': -1.0,
                                                           'fairness_false_positive_rate_diff': -1.0,
                                                           'fairness_false_positive_error_rate_balance_score': -1.0,
                                                           'fairness_false_negative_error_rate_balance_score': -1.0,
                                                           'fairness_consistency': -1.0}

        else:
            for model in self.surrogate_models_pool:
                model_predictions = get_classifier_predictions(model, data, self.auxiliary_data)
                individual[1].update({model.__class__.__name__: self.__performance_fitness(self.auxiliary_data,
                                                                                           model_predictions)})
                individual[2].update({model.__class__.__name__: self.__fairness_fitness(self.auxiliary_data,
                                                                                        model_predictions)})

        self.evaluated_individuals[flattened_genotype] = individual

        return individual

    def __evaluate_population(self, dataset, population):
        population = sorted(population, key=lambda x: len(x[0]))
        for j, individual in enumerate(population):
            if self.verbose and not self.genetic_search_flag and j % 50 == 0:
                print(f'\t[PGA] Individual {j}/{len(population)}.', flush=True)
            population[j] = self.__fitness(dataset, individual)

        if self.verbose and not self.genetic_search_flag:
            print(f'\t[PGA] Individual {len(population)}/{len(population)}.', flush=True)

        return population

    def __genetic_search(self, dataset: Dataset) -> Dataset:

        if self.verbose:
            print(f'\t[PGA] Performing genetic search. Doing '
                  f'{self.genetic_parameters.population_size * self.genetic_parameters.num_generations} '
                  f'evaluations out of {self.problem_dimension} possible combinations.', flush=True)

        population = self.__evaluate_population(dataset, self.population)

        best_individual = self.__select_best(population)[0]
        decoded_best_individual = self.__decode_individual(best_individual)
        population_average = self.__compute_population_average_fitness(population)
        evolution = pd.concat([decoded_best_individual, population_average], axis=1)

        if self.verbose:
            print(f'\t[PGA] Generation {1}/{self.genetic_parameters.num_generations}.', flush=True)

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
                offsprings.append(mutated_individual)

            offsprings = self.__evaluate_population(dataset, offsprings)

            # Survivors selection - elitism
            elite_pop = population[:self.genetic_parameters.elite_size]
            population = self.__new_population(elite_pop, offsprings)

            best_individual = self.__select_best(population)[0]
            decoded_best_individual = self.__decode_individual(best_individual)
            population_average = self.__compute_population_average_fitness(population)
            generation = pd.concat([decoded_best_individual, population_average], axis=1)
            evolution = pd.concat([evolution, generation], axis=0)

            if self.verbose and i % 5 == 0:
                print(f'\t[PGA] Generation {i + 1}/{self.genetic_parameters.num_generations}.', flush=True)

        if self.verbose:
            print(f'\t[PGA] Generation {self.genetic_parameters.num_generations}'
                  f'/{self.genetic_parameters.num_generations}.', flush=True)

        self.__save_fitness_evolution(evolution, dataset.name)

        return self.__mount_individual(best_individual, dataset)

    def __extensive_search(self, dataset: Dataset) -> Dataset:

        if self.verbose:
            print(f'\t[PGA] Performing extensive search. Testing {self.problem_dimension} combinations.', flush=True)

        population = self.__evaluate_population(dataset, self.population)

        best_individual = self.__select_best(population)[0]
        decoded_best_individual = self.__decode_individual(best_individual)
        population_average = self.__compute_population_average_fitness(population)
        evolution = pd.concat([decoded_best_individual, population_average], axis=1)

        self.__save_fitness_evolution(evolution, dataset.name)

        return self.__mount_individual(best_individual, dataset)

    def __save_fitness_evolution(self, fitness_evolution: pd.DataFrame, dataset_name: str):
        save_path = os.path.join('best_individuals', f'{dataset_name}', f'{self.iteration_number}_iteration')
        filename = f'{self.algorithm_name}_{self.sensitive_attribute}_fitness_evolution'
        write_dataframe_to_csv(fitness_evolution, filename, save_path)

    def __mount_individual(self, individual: list[list], dataset: Dataset) -> Dataset:
        genotype = self.__flatten_genotype(individual[0])
        individual_dataset = self.__fetch_individual__(individual, dataset)
        individual_dataset.error_flag = not self.valid_individuals[genotype]

        return individual_dataset

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.decoder = data.features_mapping[sensitive_attribute]
        self.num_classes = len(data.features_mapping[sensitive_attribute])
        self.sensitive_attribute = sensitive_attribute
        self.genetic_search_flag = self.__do_genetic_search()
        self.population = self.__generate_population()
        self.evaluated_individuals = {}
        self.valid_individuals = {}
        self.cache_path = os.path.join(f'{self.algorithm_name}_{get_seed()}', data.name, self.sensitive_attribute)

    def transform(self, dataset: Dataset) -> Dataset:
        if self.genetic_search_flag:
            transformed_dataset = self.__genetic_search(dataset)
        else:
            transformed_dataset = self.__extensive_search(dataset)

        self.__clean_cache__()
        return transformed_dataset
