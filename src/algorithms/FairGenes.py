
import gc
import os
from itertools import product, combinations_with_replacement
from math import factorial

import numpy as np
from random import sample

import pandas as pd
from memory_profiler import profile

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import write_dataframe_to_csv, get_generator, dict_to_dataframe, logger, read_csv_to_dataframe, \
    delete_directory, get_seed, restore_dataset, backup_dataset
from protocol.assessment import get_classifier_predictions, fairness_assessment


class FairGenes(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters,
                 unbiasing_algorithms_pool: [Algorithm],
                 surrogate_models_pool: [object],
                 threshold_k: int,
                 verbose: bool = False):

        super().__init__()
        self.genetic_search_flag = False
        self.is_binary = False
        self.needs_auxiliary_data = True
        self.algorithm_name = 'FairGenes'

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
        self.cached = {}

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
        length = rng.integers(1, 2 * num_attribute_classes)

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
        seed = get_seed()

        features_file = f'{seed}_features_{genome}.csv'
        write_dataframe_to_csv(df=transformed_dataset.features, filename=features_file, path=self.cache_path)

        targets_file = f'{seed}_targets_{genome}.csv'
        write_dataframe_to_csv(df=transformed_dataset.targets, filename=targets_file, path=self.cache_path)

        protected_features_file = f'{seed}_protected_features_{genome}.csv'
        write_dataframe_to_csv(df=transformed_dataset.protected_features, filename=protected_features_file,
                               path=self.cache_path)

    def __fetch_individual__(self, individual: list[list], dataset: Dataset):
        genome = self.__flatten_genotype(individual[0])
        seed = get_seed()

        features_file = f'{seed}_features_{genome}.csv'
        dataset.features = read_csv_to_dataframe(features_file, self.cache_path)

        targets_file = f'{seed}_targets_{genome}.csv'
        dataset.targets = read_csv_to_dataframe(targets_file, self.cache_path)

        protected_features_file = f'{seed}_protected_features_{genome}.csv'
        dataset.protected_features = read_csv_to_dataframe(protected_features_file, self.cache_path)

        return dataset

    def __mount_individual(self, individual: list[list], dataset: Dataset) -> Dataset:
        genotype = self.__flatten_genotype(individual[0])
        individual_dataset = self.__fetch_individual__(individual, dataset)
        individual_dataset.error_flag = not self.valid_individuals[genotype]

        return individual_dataset

    def __clean_cache__(self):
        self.population = None
        self.evaluated_individuals = None
        self.valid_individuals = None
        self.decoder = None
        self.cached = None
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

            return [offspring1_genotype, {}, {}], [offspring2_genotype, {}, {}]

        return parent1, parent2

    def __mutation(self, individual):
        genotype = individual[0][:]

        mutation_indexes = np.where(get_generator().random(len(genotype))
                                    < self.genetic_parameters.probability_mutation)[0]

        generator = get_generator()
        for i in mutation_indexes:
            genotype[i][0] = generator.integers(0, self.num_classes)
            genotype[i][1] = generator.integers(0, len(self.unbiasing_algorithms_pool))

        return [genotype, {}, {}]

    def __select_best(self, population):

        def sort_population(_population, objective: tuple):
            index, model, metric = objective
            _population.sort(key=lambda x: x[index][model][metric], reverse=reverse)
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

            reverse = True
            population = lexicographic_selection(population, (1, model))  # performance
            reverse = False
            population = lexicographic_selection(population, (2, model))  # fairness

        return population

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
        sorted_genotypes = sorted(self.cached.keys(), key=len, reverse=True)
        for key in sorted_genotypes:
            if flattened_genotype.startswith(key) and len(key) < len(flattened_genotype):
                return str(key)

    def __phenotype(self, data: Dataset, individual):

        flattened_genotype = self.__flatten_genotype(individual[0])
        self.valid_individuals[flattened_genotype] = True

        longest_match = self.__find_longest_genotype_match(flattened_genotype)
        start_point = 0
        if longest_match:
            start_point = len(self.cached[longest_match])
            data = self.__fetch_individual__([self.cached[longest_match], {}, {}], data)

        transformed_data = data
        dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
        dimensions = transformed_data.features.shape

        if self.sensitive_attribute not in transformed_data.features.columns:
            previous_value = self.cached[longest_match][0][-1][0]
            sensitive_attribute = pd.DataFrame({self.sensitive_attribute: dummy_values[self.decoder[previous_value]]})
            transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

        for i in range(start_point, len(individual[0])):

            protected_group, algorithm = individual[0][i][0], individual[0][i][1]
            key = self.__flatten_genotype(individual[0][:i + 1])

            transformed_data.set_feature(self.sensitive_attribute, dummy_values[self.decoder[protected_group]])
            unbiasing_algorithm = self.unbiasing_algorithms_pool[algorithm]

            try:
                unbiasing_algorithm.fit(transformed_data, self.sensitive_attribute)
                transformed_data = unbiasing_algorithm.transform(transformed_data)
            except ValueError:
                self.valid_individuals[flattened_genotype] = False
                self.cached[key] = individual[0][:i + 1]
                raise ValueError(f'[FairGenes] Invalid individual: {individual[0]}.')

            if transformed_data.features.shape[0] != dimensions[0]:
                dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
                dimensions = transformed_data.features.shape

            if self.sensitive_attribute not in transformed_data.features.columns:
                sensitive_attribute = pd.DataFrame({self.sensitive_attribute: dummy_values[self.decoder[protected_group]]})
                transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

            if not self.cached.get(key, False):
                self.__save_individual__([individual[0][:i+1], {}, {}], transformed_data)
                self.cached[key] = individual[0][:i + 1]

        # self.__save_individual__(individual, transformed_data)

        return transformed_data

    def __performance_fitness(self, data: Dataset, predictions: pd.DataFrame):
        performance_evaluator = ModelEvaluator(data.targets, predictions)

        return {
            'performance_accuracy': performance_evaluator.accuracy(),
            'performance_f1_score': performance_evaluator.f1_score(),
            'performance_auc': performance_evaluator.auc()
        }

    def __fairness_fitness(self, data: Dataset, predictions: pd.DataFrame):
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

        try:
            if len(individual[0]) > 2 * self.num_classes:
                raise ValueError(f'[FairGenes] Invalid individual: {individual[0]}.')

            transformed_data = self.__phenotype(data, individual)

        except ValueError:
            for model in self.surrogate_models_pool:

                individual[1][model.__class__.__name__] = {'performance_accuracy': 0.0,
                                                           'performance_f1_score': 0.0,
                                                           'performance_auc': 0.0}
                individual[2][model.__class__.__name__] = {'fairness_disparate_impact': np.inf,
                                                           'fairness_discrimination_score': np.inf,
                                                           'fairness_true_positive_rate_diff': np.inf,
                                                           'fairness_false_positive_rate_diff': np.inf,
                                                           'fairness_false_positive_error_rate_balance_score': np.inf,
                                                           'fairness_false_negative_error_rate_balance_score': np.inf,
                                                           'fairness_consistency': np.inf}
        else:
            for model in self.surrogate_models_pool:
                model_predictions = get_classifier_predictions(model, transformed_data, self.auxiliary_data)
                individual[1].update({model.__class__.__name__: self.__performance_fitness(self.auxiliary_data,
                                                                                           model_predictions)})
                individual[2].update({model.__class__.__name__: self.__fairness_fitness(self.auxiliary_data,
                                                                                        model_predictions)})

        self.evaluated_individuals[flattened_genotype] = individual

        return individual

    def __evaluate_population(self, dataset, population):
        population = sorted(population, key=lambda x: len(x[0]))

        for j, individual in enumerate(population):

            if self.verbose and not self.genetic_search_flag:
                logger.info(f'\t[FairGenes] Individual {j + 1}/{len(population)}.')
            restore_dataset(dataset, self.cache_path)  # Restore dataset to original state
            population[j] = self.__fitness(dataset, individual)

            gc.collect()

        return population

    def __genetic_search(self, dataset: Dataset) -> Dataset:

        if self.verbose:
            logger.info(f'\t[FairGenes] Performing genetic search. Doing '
                        f'{self.genetic_parameters.population_size * self.genetic_parameters.num_generations} '
                        f'evaluations out of {self.problem_dimension} possible combinations.')
            logger.info(f'\t[FairGenes] Generation {1}/{self.genetic_parameters.num_generations}.')

        population = self.__evaluate_population(dataset, self.population)

        best_individual = self.__select_best(population)[0]

        decoded_best_individual = self.__decode_individual(best_individual)
        population_average = self.__compute_population_average_fitness(population)
        evolution = pd.concat([decoded_best_individual, population_average], axis=1)

        for i in range(1, self.genetic_parameters.num_generations):

            if self.verbose:
                logger.info(f'\t[FairGenes] Generation {i + 1}/{self.genetic_parameters.num_generations}.')

            parents = self.__tournament(population)

            # Crossover
            offspring = []
            for j in range(0, len(population) - 1, 2):
                child1, child2 = self.__crossover(parents[j], parents[j + 1])
                offspring.append(child1)
                offspring.append(child2)

            # Mutation
            offspring = [self.__mutation(individual) for individual in offspring]

            offspring = self.__evaluate_population(dataset, offspring)

            # Elitism
            elite_parents = []
            for j in range(self.genetic_parameters.elite_size):
                best = self.__select_best(parents)[0]
                parents.remove(best)
                elite_parents.append(best)

            best_offspring = []
            for j in range(len(population) - self.genetic_parameters.elite_size):
                best = self.__select_best(offspring)[0]
                offspring.remove(best)
                best_offspring.append(best)

            population = elite_parents + best_offspring

            best_individual = self.__select_best(population)[0]

            decoded_best_individual = self.__decode_individual(best_individual)
            population_average = self.__compute_population_average_fitness(population)
            generation = pd.concat([decoded_best_individual, population_average], axis=1)
            evolution = pd.concat([evolution, generation], axis=0)

        self.__save_fitness_evolution(evolution, dataset.name)

        return self.__mount_individual(best_individual, dataset)

    def __extensive_search(self, dataset: Dataset) -> Dataset:

        if self.verbose:
            logger.info(f'\t[FairGenes] Performing extensive search. Testing {self.problem_dimension} combinations.')

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

    def fit(self, data: Dataset, sensitive_attribute: str):
        self.decoder = data.features_mapping[sensitive_attribute]
        self.num_classes = len(data.features_mapping[sensitive_attribute])
        self.sensitive_attribute = sensitive_attribute
        self.genetic_search_flag = self.__do_genetic_search()
        self.population = self.__generate_population()
        self.evaluated_individuals = {}
        self.valid_individuals = {}
        self.cached = {}
        self.cache_path = f'{self.algorithm_name}_{data.name}_{get_seed()}_{self.sensitive_attribute}'

    def transform(self, dataset: Dataset) -> Dataset:

        backup_dataset(dataset, self.cache_path)

        if self.genetic_search_flag:
            transformed_dataset = self.__genetic_search(dataset)
        else:
            transformed_dataset = self.__extensive_search(dataset)

        self.__clean_cache__()

        return transformed_dataset
