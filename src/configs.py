import configparser
from enum import Enum

from algorithms import GeneticBasicParameters, Massaging, Reweighing, DisparateImpactRemover, \
    LearnedFairRepresentations, LexicographicGeneticAlgorithmFairFeatureSelection, PermutationGeneticAlgorithm
from algorithms.MulticlassLexicographicGeneticAlgorithmFairFeatureSelection import \
    MulticlassLexicographicGeneticAlgorithmFairFeatureSelection
from datasets import DatasetConfig, Dataset, GermanCredit, AdultIncome
from datasets.LawSchoolAdmissions import LawSchoolAdmissions
from helpers import logger, set_seed, get_seed, set_gpu_device, set_num_threads, set_gpu_allocated_memory, \
    get_surrogate_classifiers, enable_gpu_acceleration, disable_gpu_acceleration


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    # LearnedFairRepresentations = 3
    LGAFFS = 4
    MulticlassLGAFFS = 5
    PermutationGeneticAlgorithm = 6


def get_global_configs(configs_file: str) -> tuple[str, str, str, int]:
    global_configs = configparser.ConfigParser()
    global_configs.read(configs_file)

    try:
        dataset_configs = global_configs.get("GLOBAL", 'dataset_configs').strip('"')
        algorithms_configs = global_configs.get('GLOBAL', 'algorithms_configs').strip('"')
        num_iterations = global_configs.getint('GLOBAL', 'num_iterations')
        results_path = global_configs.get('GLOBAL', 'results_path')

        if get_seed() is None:
            seed = global_configs.getint('GLOBAL', 'seed')
            set_seed(seed)

        set_gpu_acceleration(global_configs)

        set_num_threads(global_configs.getint('MULTITHREADING', 'num_thread_workers'))

        return dataset_configs, algorithms_configs, results_path, num_iterations

    except configparser.NoSectionError as e:
        logger.error(f'Section [{e.section}] does not exist in the configuration file.')
        raise ValueError
    except configparser.NoOptionError as e:
        logger.error(f'Option [{e.option}] does not exist in the configuration file.')
        raise ValueError


def set_gpu_acceleration(configs: configparser.ConfigParser):
    enable_gpu = configs.getboolean('GPU', 'enable')

    if enable_gpu:
        enable_gpu_acceleration()
        gpu_device_id = configs.getint('GPU', 'device_id')
        set_gpu_device(gpu_device_id)
        gpu_allocated_memory = configs.getint('GPU', 'allocated_memory')
        set_gpu_allocated_memory(gpu_allocated_memory)
    else:
        disable_gpu_acceleration()


def get_dataset_configs(dataset: str, configs_file: str):
    configs = configparser.ConfigParser()
    configs.read(configs_file)

    return DatasetConfig(name=configs.get(dataset, 'name'),
                         target=configs.get(dataset, 'target'),
                         protected_features=configs.get(dataset, 'protected_attributes').split(','),
                         train_size=configs.getfloat(dataset, 'train_size'),
                         validation_size=configs.getfloat(dataset, 'validation_size'),
                         test_size=configs.getfloat(dataset, 'test_size'))


def get_algorithms_configs(configs_file: str, algorithm: AlgorithmOptions):
    configs = configparser.ConfigParser()
    configs.read(configs_file)

    match algorithm:
        case AlgorithmOptions.DisparateImpactRemover:
            return configs.getfloat('DisparateImpactRemover', 'repair_level')
        case AlgorithmOptions.LearnedFairRepresentations:
            k = configs.getint('LearnedFairRepresentations', 'k')
            ax = configs.getfloat('LearnedFairRepresentations', 'ax')
            ay = configs.getfloat('LearnedFairRepresentations', 'ay')
            az = configs.getfloat('LearnedFairRepresentations', 'az')
            verbose = configs.getboolean('LearnedFairRepresentations', 'verbose')

            return k, ax, ay, az, verbose

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
        case AlgorithmOptions.PermutationGeneticAlgorithm:
            verbose = configs.getboolean('PermutationGeneticAlgorithm', 'verbose')
            threshold_k = configs.getint('PermutationGeneticAlgorithm', 'threshold_k')
            return verbose, threshold_k, GeneticBasicParameters(
                population_size=configs.getint('PermutationGeneticAlgorithm', 'population_size'),
                num_generations=configs.getint('PermutationGeneticAlgorithm', 'num_generations'),
                tournament_size=configs.getint('PermutationGeneticAlgorithm', 'tournament_size'),
                elite_size=configs.getint('PermutationGeneticAlgorithm', 'elite_size'),
                probability_crossover=configs.getfloat('PermutationGeneticAlgorithm', 'probability_crossover'),
                probability_mutation=configs.getfloat('PermutationGeneticAlgorithm', 'probability_mutation')
            )
        case _:
            logger.error(f'Algorithm {algorithm} not recognized.')
            raise ValueError


def load_dataset(_dataset: str, configs_file: str) -> Dataset:
    match _dataset:
        case "GERMAN_CREDIT":
            return GermanCredit(get_dataset_configs('GERMAN_CREDIT', configs_file))
        case "ADULT_INCOME":
            return AdultIncome(get_dataset_configs('ADULT_INCOME', configs_file))
        case "LAW_SCHOOL_ADMISSION":
            return LawSchoolAdmissions(get_dataset_configs('LAW_SCHOOL_ADMISSION', configs_file))
        case _:
            logger.error('Dataset unknown! Currently supported datasets are: '
                         'GERMAN_CREDIT, ADULT_INCOME, LAW_SCHOOL_ADMISSION.')
            raise NotImplementedError


def load_algorithm(algorithm_configs_file: str, unbiasing_algorithm: Enum):
    match unbiasing_algorithm:
        case AlgorithmOptions.Massaging:
            return Massaging()
        case AlgorithmOptions.Reweighing:
            return Reweighing()
        case AlgorithmOptions.DisparateImpactRemover:
            repair_level = get_algorithms_configs(algorithm_configs_file, AlgorithmOptions.DisparateImpactRemover)
            return DisparateImpactRemover(repair_level=repair_level)
        case AlgorithmOptions.LearnedFairRepresentations:
            k, ax, ay, az, verbose = get_algorithms_configs(algorithm_configs_file,
                                                            AlgorithmOptions.LearnedFairRepresentations)
            return LearnedFairRepresentations(
                k=k,
                ax=ax,
                ay=ay,
                az=az,
                verbose=verbose
            )
        case AlgorithmOptions.LGAFFS | AlgorithmOptions.MulticlassLGAFFS:
            genetic_parameters, n_splits, min_feature_prob, max_feature_prob, verbose = (
                get_algorithms_configs(algorithm_configs_file, AlgorithmOptions.LGAFFS))

            if unbiasing_algorithm == AlgorithmOptions.LGAFFS:
                return LexicographicGeneticAlgorithmFairFeatureSelection(
                    genetic_parameters=genetic_parameters,
                    n_splits=n_splits,
                    min_feature_prob=min_feature_prob,
                    max_feature_prob=max_feature_prob,
                    verbose=verbose
                )
            else:
                return MulticlassLexicographicGeneticAlgorithmFairFeatureSelection(
                    genetic_parameters=genetic_parameters,
                    n_splits=n_splits,
                    min_feature_prob=min_feature_prob,
                    max_feature_prob=max_feature_prob,
                    verbose=verbose,
                )
        case AlgorithmOptions.PermutationGeneticAlgorithm:
            verbose, threshold_k, genetic_parameters = (
                get_algorithms_configs(algorithm_configs_file, AlgorithmOptions.PermutationGeneticAlgorithm))
            unbiasing_algorithms_pool = [
                load_algorithm(algorithm_configs_file, AlgorithmOptions.Massaging),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.Reweighing),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.DisparateImpactRemover),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.LGAFFS),
                # load_algorithm(algorithm_configs_file, AlgorithmOptions.LearnedFairRepresentations)
            ]
            unbiasing_algorithms_pool[3].verbose = False
            # unbiasing_algorithms_pool[4].verbose = False
            return PermutationGeneticAlgorithm(
                genetic_parameters=genetic_parameters,
                unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                surrogate_models_pool=get_surrogate_classifiers(),
                verbose=verbose,
                threshold_k=threshold_k
            )
        case _:
            logger.error('Algorithm option unknown!')
            raise ValueError
