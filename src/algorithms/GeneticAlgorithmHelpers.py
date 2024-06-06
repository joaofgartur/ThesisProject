import copy
import itertools

import numpy as np

from helpers import get_generator


class GeneticBasicParameters:

    def __init__(self, num_generations: int = 1,
                 population_size: int = 1,
                 individual_size: int = 1,
                 tournament_size: int = 2,
                 elite_size: int = 1,
                 probability_mutation: float = 0.05,
                 probability_crossover: float = 0.05):
        self.num_generations = num_generations
        self.population_size = population_size
        self.individual_size = individual_size
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover


def uniform_crossover(parent1: list, parent2: list, probability_crossover: float):
    value = get_generator().random()
    if value < probability_crossover:

        offspring1 = [np.zeros(parent1[0].shape, dtype=int), {}, {}]
        offspring2 = [np.zeros(parent1[0].shape, dtype=int), {}, {}]

        indexes_choice = get_generator().choice(parent1[0].shape[0])
        index_mask = indexes_choice < 0.5

        offspring1[0][index_mask] = parent1[0][index_mask]
        offspring2[0][index_mask] = parent2[0][index_mask]

        offspring1[0][~index_mask] = parent2[0][~index_mask]
        offspring2[0][~index_mask] = parent1[0][~index_mask]

        return offspring1, offspring2
    else:
        return parent1, parent2


def pmx_cromossover(parent1: list, parent2: list, probability_crossover: float):
    value = get_generator().random()
    if value < probability_crossover:
        offspring1 = [parent1[0], {}, {}]
        offspring2 = [parent1[0], {}, {}]

        n = len(parent1[0])
        cp1 = get_generator().integers(0, n)
        cp2 = get_generator().integers(0, n - 1)

        if cp2 == cp1:
            cp2 += 1
        elif cp2 < cp1:
            cp1, cp2 = cp2, cp1

        for i in range(cp1, cp2):
            gene1 = parent1[0][i]
            gene2 = parent2[0][i]

            parent1[0][i], parent1[0][offspring1[0][gene2]] = gene2, gene1
            parent2[0][i], parent2[0][offspring2[0][gene1]] = gene1, gene2

            # Position bookkeeping
            offspring1[0][gene1], offspring1[0][gene2] = offspring1[0][gene2], offspring1[0][gene1]
            offspring2[0][gene1], offspring2[0][gene2] = offspring2[0][gene2], offspring2[0][gene1]

        return offspring1, offspring2
    else:
        return parent1, parent2


def bit_flip_mutation(individual: list, probability_mutation: float):
    new_individual = copy.deepcopy(individual)
    random_probs = get_generator().random(individual[0].shape[0])
    for i in range(random_probs.shape[0]):
        if random_probs[i] < probability_mutation:
            new_individual[0][i] = 1 - new_individual[0][i]
    return new_individual


def scramble_mutation(individual: list, probability_mutation: float):
    mutated_individual = copy.deepcopy(individual)
    n = individual[0].shape[0]

    if get_generator().random() < probability_mutation:
        index_1, index_2 = get_generator().choice(n, 2, replace=False)
        segment = mutated_individual[0][index_1:index_2]
        get_generator().shuffle(segment)
        mutated_individual[0][index_1:index_2] = segment

    return mutated_individual


def sort_population(population: list, objective: str, index: int):
    population.sort(key=lambda x: x[index][objective], reverse=True)
    return population


def select_top_individuals(population, metric, epsilon: float = 0.0, fairness_metric=False):
    index = 1
    if fairness_metric:
        index = 2

    population = sort_population(population, metric, index)
    best_value = population[0][index][metric] - epsilon
    last_index = np.argmax([individual[index][metric] < best_value - epsilon for individual in population])
    return population[:last_index + 1]


def lexicographic_selection(population, metrics):
    for metric in metrics:
        if len(population) == 1:
            break
        population = select_top_individuals(population, metric)

    return population[0]


def select_best(population):
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


def print_population(population):
    for individual in population:
        print(individual)
