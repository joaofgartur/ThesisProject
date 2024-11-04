"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum

import numpy as np
import pandas as pd
import multiprocessing as mp

from random import sample
from itertools import product, combinations_with_replacement
from math import factorial

from sklearnex import patch_sklearn, unpatch_sklearn

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from algorithms.utils import AlgorithmOptions, get_unbiasing_algorithm
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from utils import write_dataframe_to_csv, get_generator, dict_to_dataframe, logger, get_seed, enable_gpu_acceleration, \
    get_global_configs, set_global_configs, set_seed, set_gpu_device
from protocol.assessment import get_classifier_predictions, fairness_assessment


class FairGenes(Algorithm):
    """
        FairGenes algorithm for bias correction using genetic algorithms.

        Attributes
        ----------
        genetic_search_flag : bool
            Indicates if genetic search is enabled.
        is_binary : bool
            Indicates if the algorithm is binary.
        needs_auxiliary_data : bool
            Indicates if the algorithm needs auxiliary data.
        algorithm_name : str
            The name of the algorithm.
        cache_path : str or None
            The path to the cache.
        genetic_parameters : GeneticBasicParameters
            The genetic parameters for the algorithm.
        num_classes : int
            The number of classes in the dataset.
        unbiasing_algorithms_pool : dict
            The pool of unbiasing algorithms.
        surrogate_models_pool : list
            The pool of surrogate models.
        sensitive_attribute : str
            The sensitive attribute used by the algorithm.
        population : list
            The population of individuals.
        decoder : dict
            The decoder for the sensitive attribute.
        problem_dimension : int
            The dimension of the problem.
        threshold_k : int
            The threshold for the genetic search.
        verbose : bool
            Indicates if verbose output is enabled.
        evaluated_individuals : dict
            The evaluated individuals.
        valid_individuals : dict
            The valid individuals.

        Methods
        -------
        __init__(genetic_parameters: GeneticBasicParameters, unbiasing_algorithms_pool: dict, surrogate_models_pool: list, threshold_k: int, verbose: bool = False):
            Initializes the FairGenes object with the specified parameters.
        __do_genetic_search() -> bool:
            Determines if genetic search should be performed.
        __flatten_genotype(genotype: list) -> str:
            Flattens the genotype into a string.
        __generate_individual() -> list:
            Generates a random individual.
        __decode_individual(individual) -> pd.DataFrame:
            Decodes an individual into a DataFrame.
        __clean_cache__():
            Cleans the cache.
        __compute_population_average_fitness(population) -> pd.DataFrame:
            Computes the average fitness of the population.
        __generate_population() -> list:
            Generates the population.
        __crossover(parent1: list, parent2: list) -> tuple:
            Performs crossover between two parents.
        __mutation(individual) -> list:
            Mutates an individual.
        __select_best(population) -> list:
            Selects the best individuals from the population.
        __tournament(population) -> list:
            Performs tournament selection on the population.
        __phenotype(data: Dataset, individual) -> Dataset:
            Applies the phenotype to the data.
        _performance_fitness(data: Dataset, predictions: pd.DataFrame) -> dict:
            Computes the performance fitness of the predictions.
        _fairness_fitness(data: Dataset, predictions: pd.DataFrame) -> dict:
            Computes the fairness fitness of the predictions.
        _fitness(data: Dataset, individual, evaluated_individuals, valid_individuals, surrogate_models_pool, configs, seed) -> tuple:
            Computes the fitness of an individual.
        __evaluate_population(dataset: Dataset, population: list) -> list:
            Evaluates the population.
        __genetic_search(dataset: Dataset) -> Dataset:
            Performs genetic search on the dataset.
        __extensive_search(dataset: Dataset) -> Dataset:
            Performs extensive search on the dataset.
        __save_fitness_evolution(fitness_evolution: pd.DataFrame, dataset_name: str):
            Saves the fitness evolution to a file.
        fit(data: Dataset, sensitive_attribute: str):
            Fits the algorithm to the data.
        transform(dataset: Dataset) -> Dataset:
            Transforms the dataset using the fitted algorithm.
        """

    def __init__(self, genetic_parameters: GeneticBasicParameters,
                 unbiasing_algorithms_pool: dict,
                 surrogate_models_pool: [object],
                 threshold_k: int,
                 verbose: bool = False):
        """
        Initializes the FairGenes object with the specified parameters.

        Parameters
        ----------
        genetic_parameters : GeneticBasicParameters
            The genetic parameters for the algorithm.
        unbiasing_algorithms_pool : dict
            The pool of unbiasing algorithms.
        surrogate_models_pool : list
            The pool of surrogate models.
        threshold_k : int
            The threshold for the genetic search.
        verbose : bool, optional
            Indicates if verbose output is enabled (default is False).
        """

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

    def __do_genetic_search(self):
        """
        Determines if genetic search should be performed.

        Returns
        -------
        bool
            True if genetic search should be performed, False otherwise.
        """
        num_algorithms = len(self.unbiasing_algorithms_pool)
        max_length = 2 * self.num_classes
        n = self.num_classes * num_algorithms

        self.problem_dimension = 0
        for length in range(1, max_length + 1):
            self.problem_dimension += np.divide(factorial(n + length - 1), factorial(length) * factorial(n - 1))

        return self.problem_dimension >= factorial(self.threshold_k - 1)

    def __flatten_genotype(self, genotype: list[(int, int)]) -> str:
        """
        Flattens the genotype into a string.

        Parameters
        ----------
        genotype : list
            The genotype to flatten.

        Returns
        -------
        str
            The flattened genotype.
        """
        return ''.join(f'{gene}{value}' for gene, value in genotype)

    def __generate_individual(self) -> list:
        """
        Generates a random individual.

        Returns
        -------
        list
            The generated individual.
        """
        rng = get_generator()

        num_attribute_classes = self.num_classes
        m = len(self.unbiasing_algorithms_pool)
        length = rng.integers(1, 2 * num_attribute_classes)

        genotype = [[rng.integers(0, num_attribute_classes), rng.integers(0, m)] for _ in range(length)]

        return [genotype, {}, {}]

    def __decode_individual(self, individual) -> pd.DataFrame:
        """
        Decodes an individual into a DataFrame.

        Parameters
        ----------
        individual : list
            The individual to decode.

        Returns
        -------
        pd.DataFrame
            The decoded individual.
        """
        genome = [[self.decoder[val],
                   self.unbiasing_algorithms_pool[algo]] for val, algo in individual[0]]

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

    def __clean_cache__(self):
        """
        Cleans the cache.
        """
        self.population = None
        self.evaluated_individuals = None
        self.valid_individuals = None
        self.decoder = None

    def __compute_population_average_fitness(self, population):
        """
        Computes the average fitness of the population.

        Parameters
        ----------
        population : list
            The population to compute the average fitness for.

        Returns
        -------
        pd.DataFrame
            The average fitness of the population.
        """

        decoded_population = pd.DataFrame()
        for individual in population:
            decoded_population = pd.concat([decoded_population, self.__decode_individual(individual)])

        metrics_columns = decoded_population.select_dtypes(include=[np.number]).columns
        decoded_population = decoded_population[metrics_columns]

        population_mean = decoded_population.mean().transpose().to_frame().transpose().add_suffix('mean')

        return population_mean

    def __generate_population(self):
        """
        Computes the average fitness of the population.

        Parameters
        ----------
        population : list
            The population to compute the average fitness for.

        Returns
        -------
        pd.DataFrame
            The average fitness of the population.
        """

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
        genotype = individual[0][:]

        mutation_indexes = np.where(get_generator().random(len(genotype))
                                    < self.genetic_parameters.probability_mutation)[0]

        generator = get_generator()
        for i in mutation_indexes:
            genotype[i][0] = generator.integers(0, self.num_classes)
            genotype[i][1] = generator.integers(0, len(self.unbiasing_algorithms_pool))

        return [genotype, {}, {}]

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
        """
        Performs tournament selection on the population.

        Parameters
        ----------
        population : list
            The population to perform tournament selection on.

        Returns
        -------
        list
            The selected individuals.
        """

        def one_tour(local_population):
            pool = sample(local_population, self.genetic_parameters.tournament_size)
            return self.__select_best(pool)[0]

        mate_pool = []
        for _ in range(len(population)):
            winner = one_tour(population)
            mate_pool.append(winner)
        return mate_pool

    def __phenotype(self, data: Dataset, individual):
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
        Dataset
            The transformed dataset.
        """

        flattened_genotype = self.__flatten_genotype(individual[0])
        self.valid_individuals[flattened_genotype] = True

        transformed_data = data
        dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
        dimensions = transformed_data.features.shape

        for i in range(0, len(individual[0])):
            protected_group, algorithm = individual[0][i][0], individual[0][i][1]

            transformed_data.set_feature(self.sensitive_attribute, dummy_values[self.decoder[protected_group]])
            unbiasing_algorithm = decode_mapping(algorithm, self.unbiasing_algorithms_pool)
            unbiasing_algorithm = get_unbiasing_algorithm(unbiasing_algorithm, verbosity=False)

            try:
                unbiasing_algorithm.fit(transformed_data, self.sensitive_attribute)
                transformed_data = unbiasing_algorithm.transform(transformed_data)
            except ValueError:
                self.valid_individuals[flattened_genotype] = False
                raise ValueError(f'[FairGenes] Invalid individual: {individual[0]}.')

            if transformed_data.features.shape[0] != dimensions[0]:
                dummy_values = transformed_data.get_dummy_protected_feature(self.sensitive_attribute)
                dimensions = transformed_data.features.shape

            if self.sensitive_attribute not in transformed_data.features.columns:
                sensitive_attribute = pd.DataFrame({self.sensitive_attribute: dummy_values[self.decoder[protected_group]]})
                transformed_data.features = pd.concat([transformed_data.features, sensitive_attribute], axis=1)

        return transformed_data

    def _performance_fitness(self, data: Dataset, predictions: pd.DataFrame):
        """
        Computes the performance fitness of the predictions.

        Parameters
        ----------
        data : Dataset
            The dataset to compute the performance fitness for.
        predictions : pd.DataFrame
            The predictions to compute the performance fitness for.

        Returns
        -------
        dict
            The performance fitness.
        """
        performance_evaluator = ModelEvaluator(data.targets, predictions)

        return {
            'performance_accuracy': performance_evaluator.accuracy(),
            'performance_f1_score': performance_evaluator.f1_score(),
            'performance_auc': performance_evaluator.auc()
        }

    def _fairness_fitness(self, data: Dataset, predictions: pd.DataFrame):
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

        metrics = fairness_assessment(data, predictions, self.sensitive_attribute)

        result = {}
        numerical_columns = metrics.select_dtypes(include=[np.number]).columns
        for metric in numerical_columns:
            result[metric] = np.sum((metrics[metric] - 1.0) ** 2)
        return result

    def _fitness(self, data: Dataset, individual, evaluated_individuals, valid_individuals, surrogate_models_pool, configs, seed):
        """
        Computes the fitness of an individual.

        Parameters
        ----------
        data : Dataset
            The dataset to compute the fitness for.
        individual : list
            The individual to compute the fitness for.
        evaluated_individuals : dict
            The evaluated individuals.
        valid_individuals : dict
            The valid individuals.
        surrogate_models_pool : list
            The pool of surrogate models.
        configs : dict
            The global configurations.
        seed : int
            The seed for random number generation.

        Returns
        -------
        tuple
            The fitness of the individual and its validity.
        """

        set_global_configs(configs)
        set_seed(seed)

        patch_sklearn(global_patch=True)
        if configs.gpu_device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu_device_id)
            set_gpu_device(configs.gpu_device_id)
            enable_gpu_acceleration()

        flattened_genotype = self.__flatten_genotype(individual[0])
        if flattened_genotype in evaluated_individuals:
            return evaluated_individuals[flattened_genotype], valid_individuals[flattened_genotype]

        valid = True
        try:
            if len(individual[0]) > 2 * self.num_classes:
                raise ValueError(f'[FairGenes] Invalid individual: {individual[0]}.')

            transformed_data = copy.deepcopy(data)
            transformed_data = self.__phenotype(transformed_data, individual)

        except ValueError:
            valid = False
            for model in surrogate_models_pool:
                model_name = model.__class__.__name__
                individual[1][model_name] = {'performance_accuracy': 0.0,
                                                           'performance_f1_score': 0.0,
                                                           'performance_auc': 0.0}
                individual[2][model_name] = {'fairness_disparate_impact': np.inf,
                                                           'fairness_discrimination_score': np.inf,
                                                           'fairness_true_positive_rate_diff': np.inf,
                                                           'fairness_false_positive_rate_diff': np.inf,
                                                           'fairness_false_positive_error_rate_balance_score': np.inf,
                                                           'fairness_false_negative_error_rate_balance_score': np.inf,
                                                           'fairness_consistency': np.inf}
        else:
            for model in surrogate_models_pool:
                model_name = model.__class__.__name__
                model_predictions = get_classifier_predictions(model, transformed_data, self.auxiliary_data)
                individual[1][model_name] = self._performance_fitness(self.auxiliary_data, model_predictions)
                individual[2][model_name] = self._fairness_fitness(self.auxiliary_data, model_predictions)

        return individual, valid

    def __evaluate_population(self, dataset, population):
        """
        Evaluates the population.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the population on.
        population : list
            The population to evaluate.

        Returns
        -------
        list
            The evaluated population.
        """

        population = sorted(population, key=lambda x: len(x[0]))

        for j, individual in enumerate(population):

            with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as executor:
                if self.verbose:
                    logger.info(f'\t[FairGenes] Individual {j + 1}/{len(population)}: {population[j]}.')

                population[j], valid = executor.submit(self._fitness, dataset, individual, self.evaluated_individuals,
                                                       self.valid_individuals, self.surrogate_models_pool, get_global_configs(), get_seed()).result()

                genotype = self.__flatten_genotype(population[j][0])
                if genotype not in self.evaluated_individuals:
                    self.evaluated_individuals[genotype] = population[j]
                    self.valid_individuals[genotype] = valid

        return population

    def __genetic_search(self, dataset: Dataset) -> Dataset:
        """
        Performs genetic search on the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to perform genetic search on.

        Returns
        -------
        Dataset
            The transformed dataset.
        """

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

        return self.__phenotype(dataset, best_individual)

    def __extensive_search(self, dataset: Dataset) -> Dataset:
        """
        Performs extensive search on the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to perform extensive search on.

        Returns
        -------
        Dataset
            The transformed dataset.
        """

        if self.verbose:
            logger.info(f'\t[FairGenes] Performing extensive search. Testing {self.problem_dimension} combinations.')

        population = self.__evaluate_population(dataset, self.population)

        best_individual = self.__select_best(population)[0]
        decoded_best_individual = self.__decode_individual(best_individual)
        population_average = self.__compute_population_average_fitness(population)
        evolution = pd.concat([decoded_best_individual, population_average], axis=1)

        self.__save_fitness_evolution(evolution, dataset.name)

        return self.__phenotype(dataset, best_individual)

    def __save_fitness_evolution(self, fitness_evolution: pd.DataFrame, dataset_name: str):
        """
        Saves the fitness evolution to a file.

        Parameters
        ----------
        fitness_evolution : pd.DataFrame
            The fitness evolution to save.
        dataset_name : str
            The name of the dataset.
        """
        save_path = os.path.join('best_individuals', f'{dataset_name}', f'{self.iteration_number}_iteration')
        filename = f'{self.algorithm_name}_{self.sensitive_attribute}_fitness_evolution_seed_{get_seed()}.csv'
        write_dataframe_to_csv(fitness_evolution, filename, save_path)

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
        self.decoder = data.features_mapping[sensitive_attribute]
        self.num_classes = len(data.features_mapping[sensitive_attribute])
        self.sensitive_attribute = sensitive_attribute
        self.genetic_search_flag = self.__do_genetic_search()
        self.population = self.__generate_population()
        self.evaluated_individuals = {}
        self.valid_individuals = {}
        self.cache_path = f'{self.algorithm_name}_{data.name}_{get_seed()}_{self.sensitive_attribute}'

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

        if self.genetic_search_flag:
            transformed_dataset = self.__genetic_search(dataset)
        else:
            transformed_dataset = self.__extensive_search(dataset)

        self.__clean_cache__()

        return transformed_dataset


def decode_mapping(algorithm: int, unbiasing_algorithms: dict) -> Enum:

    match unbiasing_algorithms[algorithm]:
        case 'MSG':
            return AlgorithmOptions.Massaging
        case 'REW':
            return AlgorithmOptions.Reweighing
        case 'DIR':
            return AlgorithmOptions.DisparateImpactRemover
        case 'LGAFFS':
            return AlgorithmOptions.LGAFFS
        case _:
            raise ValueError('Algorithm option unknown!')