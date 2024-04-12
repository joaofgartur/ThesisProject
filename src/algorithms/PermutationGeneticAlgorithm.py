import copy
import numpy as np
from random import sample
from sklearn.ensemble import RandomForestClassifier

from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters, uniform_crossover, scramble_mutation, \
    select_best
from datasets import Dataset
from evaluation.ModelEvaluator import ModelEvaluator
from helpers import abs_diff, logger
from protocol.assessment import get_classifier_predictions, fairness_assessment


class PermutationGeneticAlgorithm(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters, base_algorithm: Algorithm):

        super().__init__()
        self.is_binary = False

        self.genetic_parameters = genetic_parameters

        self.base_algorithm = base_algorithm
        self.validation_data = None
        self.sensitive_attribute = ''
        self.population = []
        self.decoder = {}

    def __generate_individual(self) -> list:
        n = self.genetic_parameters.individual_size
        return [np.random.permutation(np.arange(0, n))[:n], {}, {}]

    def __delete_individual(self, population, individual):
        for i in range(len(population)):
            is_match = (all(population[i][0]) == all(individual[0])
                        and all(population[i][1]) == all(individual[1])
                        and all(population[i][2]) == all(individual[2]))
            if is_match:
                population.pop(i)
                break
        return population

    def __generate_population(self):
        return [self.__generate_individual() for _ in range(self.genetic_parameters.population_size)]

    def __uniform_crossover(self, parent1, parent2):
        return uniform_crossover(parent1, parent2, self.genetic_parameters.probability_crossover)

    def __mutation(self, individual):
        return scramble_mutation(individual, self.genetic_parameters.probability_mutation)

    def __select_best(self, population):
        return select_best(population)

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
        return {'performance': -1}

    def __fairness_fitness(self, data: Dataset, predictions: Dataset):
        metrics = fairness_assessment(data, predictions, self.sensitive_attribute)

        result = {}
        for metric in metrics.columns:
            if metrics[metric].dtype == 'object':
                continue
            result.update({metric: -abs_diff(metrics[metric].min(), metrics[metric].max())})

        return result

    def __fitness(self, data: Dataset, individual):

        def _is_invalid(_individual):
            unique_values = []
            for value in _individual[0]:
                if value in unique_values:
                    return True
                unique_values.append(value)
            return False

        if _is_invalid(individual):
            return [individual[0], {'performance': -1}, {}]

        data = self.__phenotype(data, individual)

        model = RandomForestClassifier()
        predictions = get_classifier_predictions(model, data, self.validation_data)

        return [individual[0], self.__performance_fitness(self.validation_data, predictions),
                self.__fairness_fitness(self.validation_data, predictions)]

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
        self.genetic_parameters.individual_size = len(data.features_mapping[sensitive_attribute])
        self.population = self.__generate_population()
        self.sensitive_attribute = sensitive_attribute

    def transform(self, dataset: Dataset) -> Dataset:
        population = self.population

        population = [self.__fitness(dataset, individual) for individual in population]
        best_individual = self.__select_best(population)
        logger.info(f'[PGA] Generation {0}/{self.genetic_parameters.num_generations} '
                    f'Best Individual: {[self.decoder[i] for i in best_individual[0]]}')

        for i in range(1, self.genetic_parameters.num_generations):
            parents = self.__tournament(population)

            """
            # Crossover
            new_parents = []
            for j in range(0, len(population) - 1, 2):
                child1, child2 = self.__uniform_crossover(parents[j], parents[j + 1])
                new_parents.append(child1)
                new_parents.append(child2)
            """

            # Mutation
            offsprings = []
            for individual in parents:
                mutated_individual = self.__mutation(individual)
                offsprings.append(self.__fitness(dataset, mutated_individual))

            # Survivors selection - elitism
            elite_pop = population[:self.genetic_parameters.elite_size]
            population = self.__new_population(elite_pop, offsprings)

            best_individual = self.__select_best(population)
            logger.info(
                f'[PGA] Generation {i}/{self.genetic_parameters.num_generations} '
                f'Best Individual: {[self.decoder[i] for i in best_individual[0]]}')

        return self.__phenotype(dataset, best_individual)
