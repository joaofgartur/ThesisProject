import configparser

from algorithms import (GeneticBasicParameters)
from utils import logger
from algorithms.utils import AlgorithmOptions


def get_algorithms_configs(configs_file: str, algorithm: AlgorithmOptions):
    configs = configparser.ConfigParser()
    configs.read(configs_file)

    match algorithm:
        case AlgorithmOptions.DisparateImpactRemover:
            return configs.getfloat('DisparateImpactRemover', 'repair_level')
        case AlgorithmOptions.LGAFFS:
            section = 'LexicographicGeneticAlgorithmFairFeatureSelection'
            genetic_parameters = GeneticBasicParameters(
                population_size=configs.getint(section, 'population_size'),
                num_generations=configs.getint(section, 'num_generations'),
                tournament_size=configs.getint(section, 'tournament_size'),
                probability_crossover=configs.getfloat(section, 'probability_crossover'),
                probability_mutation=configs.getfloat(section, 'probability_mutation'),
            )
            n_splits = configs.getint(section, 'n_splits')
            min_feature_prob = configs.getfloat(section, 'min_feature_prob')
            max_feature_prob = configs.getfloat(section, 'max_feature_prob')
            verbose = configs.getboolean(section, 'verbose')

            return genetic_parameters, n_splits, min_feature_prob, max_feature_prob, verbose
        case AlgorithmOptions.FairGenes:
            verbose = configs.getboolean('FairGenes', 'verbose')
            threshold_k = configs.getint('FairGenes', 'threshold_k')
            return verbose, threshold_k, GeneticBasicParameters(
                population_size=configs.getint('FairGenes', 'population_size'),
                num_generations=configs.getint('FairGenes', 'num_generations'),
                tournament_size=configs.getint('FairGenes', 'tournament_size'),
                elite_size=configs.getint('FairGenes', 'elite_size'),
                probability_crossover=configs.getfloat('FairGenes', 'probability_crossover'),
                probability_mutation=configs.getfloat('FairGenes', 'probability_mutation')
            )
        case _:
            logger.error(f'Algorithm {algorithm} not recognized.')
            raise ValueError


