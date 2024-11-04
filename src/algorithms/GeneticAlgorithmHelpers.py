"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

class GeneticBasicParameters:
    """
    Class representing the basic parameters for a genetic algorithm.

    Attributes
    ----------
    num_generations : int
        The number of generations.
    population_size : int
        The size of the population.
    individual_size : int
        The size of an individual.
    tournament_size : int
        The size of the tournament.
    elite_size : int
        The size of the elite.
    probability_mutation : float
        The probability of mutation.
    probability_crossover : float
        The probability of crossover.

    Methods
    -------
    __init__(num_generations: int = 1, population_size: int = 1, individual_size: int = 1, tournament_size: int = 2, elite_size: int = 1, probability_mutation: float = 0.05, probability_crossover: float = 0.05):
        Initializes the GeneticBasicParameters object with the specified parameters.
    """

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
