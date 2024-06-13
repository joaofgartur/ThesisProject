from algorithms import PermutationGeneticAlgorithm
from algorithms.Algorithm import Algorithm
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from datasets import Dataset


class PGALexicographicGeneticAlgorithmFairFeatureSelection(Algorithm):

    def __init__(self, genetic_parameters: GeneticBasicParameters,
                 unbiasing_algorithms_pool: [Algorithm],
                 surrogate_models_pool: [object],
                 verbose: bool = False):
        super().__init__()
        self.unbiasing_algorithm = PermutationGeneticAlgorithm(genetic_parameters,
                                                               unbiasing_algorithms_pool,
                                                               surrogate_models_pool,
                                                               verbose)
        self.unbiasing_algorithm.algorithm_name = 'PGALexicographicGAFFS'
        self.is_binary = False
        self.needs_auxiliary_data = True

    def fit(self, data: Dataset, sensitive_attribute: str):
        return self.unbiasing_algorithm.fit(data, sensitive_attribute)

    def transform(self, dataset: Dataset) -> Dataset:
        return self.unbiasing_algorithm.transform(dataset)
