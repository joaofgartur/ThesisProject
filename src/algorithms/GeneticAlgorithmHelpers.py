
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
