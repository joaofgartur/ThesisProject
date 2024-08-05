from configparser import ConfigParser
from enum import Enum

from algorithms import GeneticBasicParameters
from algorithms.Algorithm import Algorithm
from utils import get_global_configs, get_surrogate_classifiers


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    LGAFFS = 3
    MulticlassLGAFFS = 4
    FairGenes = 5


def get_unbiasing_algorithm(unbiasing_algorithm: Enum, verbosity=True) -> Algorithm:

    match unbiasing_algorithm:
        case AlgorithmOptions.Massaging:
            from algorithms.Massaging import Massaging
            return Massaging()
        case AlgorithmOptions.Reweighing:
            from algorithms.Reweighting import Reweighing
            return Reweighing()
        case AlgorithmOptions.DisparateImpactRemover:
            from algorithms.DisparateImpactRemover import DisparateImpactRemover
            return DisparateImpactRemover(parse_dir_configs())
        case AlgorithmOptions.LGAFFS | AlgorithmOptions.MulticlassLGAFFS:
            (genetic_parameters, n_splits, min_feature_prob,
             max_feature_prob, verbose) = parse_fair_feature_selection_configs()

            if unbiasing_algorithm == AlgorithmOptions.LGAFFS:
                from algorithms.LexicographicGeneticAlgorithmFairFeatureSelection import (
                    LexicographicGeneticAlgorithmFairFeatureSelection)
                return LexicographicGeneticAlgorithmFairFeatureSelection(
                    genetic_parameters=genetic_parameters,
                    n_splits=n_splits,
                    min_feature_prob=min_feature_prob,
                    max_feature_prob=max_feature_prob,
                    verbose=verbosity and verbose
                )
            else:
                from algorithms.MulticlassLexicographicGeneticAlgorithmFairFeatureSelection import (
                    MulticlassLexicographicGeneticAlgorithmFairFeatureSelection)
                return MulticlassLexicographicGeneticAlgorithmFairFeatureSelection(
                    genetic_parameters=genetic_parameters,
                    n_splits=n_splits,
                    min_feature_prob=min_feature_prob,
                    max_feature_prob=max_feature_prob,
                    verbose=verbose,
                )
        case AlgorithmOptions.FairGenes:
            from algorithms.FairGenes import FairGenes
            verbose, threshold_k, genetic_parameters, algorithms_pool = parse_fair_genes()
            return FairGenes(
                genetic_parameters=genetic_parameters,
                unbiasing_algorithms_pool=algorithms_pool,
                surrogate_models_pool=get_surrogate_classifiers(),
                verbose=verbose,
                threshold_k=threshold_k
            )
        case _:
            raise ValueError('Algorithm option unknown!')


def parse_dir_configs():
    global_configs = get_global_configs()

    parser = ConfigParser()
    parser.read(global_configs.algorithms_configs_file)
    header = 'DisparateImpactRemover'

    return parser.getfloat(header, 'repair_level')


def parse_fair_feature_selection_configs():
    global_configs = get_global_configs()

    parser = ConfigParser()
    parser.read(global_configs.algorithms_configs_file)

    header = 'LexicographicGeneticAlgorithmFairFeatureSelection'
    genetic_parameters = GeneticBasicParameters(
        population_size=parser.getint(header, 'population_size'),
        num_generations=parser.getint(header, 'num_generations'),
        tournament_size=parser.getint(header, 'tournament_size'),
        probability_crossover=parser.getfloat(header, 'probability_crossover'),
        probability_mutation=parser.getfloat(header, 'probability_mutation'),
    )
    n_splits = parser.getint(header, 'n_splits')
    min_feature_prob = parser.getfloat(header, 'min_feature_prob')
    max_feature_prob = parser.getfloat(header, 'max_feature_prob')
    verbose = parser.getboolean(header, 'verbose')

    return genetic_parameters, n_splits, min_feature_prob, max_feature_prob, verbose


def parse_fair_genes():
    global_configs = get_global_configs()

    parser = ConfigParser()
    parser.read(global_configs.algorithms_configs_file)

    header = 'FairGenesParameters'
    verbose = parser.getboolean(header, 'verbose')
    threshold_k = parser.getint(header, 'threshold_k')
    genetic_parameters = GeneticBasicParameters(
        population_size=parser.getint(header, 'population_size'),
        num_generations=parser.getint(header, 'num_generations'),
        tournament_size=parser.getint(header, 'tournament_size'),
        elite_size=parser.getint(header, 'elite_size'),
        probability_crossover=parser.getfloat(header, 'probability_crossover'),
        probability_mutation=parser.getfloat(header, 'probability_mutation')
    )

    unbiasing_algorithms = {}
    header = 'FairGenesUnbiasingAlgorithms'
    options = parser[header]
    for algorithm in options:
        algorithm_id = int(algorithm.split('_')[-1]) - 1
        unbiasing_algorithms[algorithm_id] = options[algorithm]

    return verbose, threshold_k, genetic_parameters, unbiasing_algorithms
