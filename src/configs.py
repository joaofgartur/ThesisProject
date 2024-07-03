import configparser
from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from cuml.linear_model import LogisticRegression as LogisticRegression_GPU
from cuml.ensemble import RandomForestClassifier as RandomForestClassifier_GPU
from cuml.naive_bayes import GaussianNB as GaussianNB_GPU
from cuml.svm import SVC as SVC_GPU
from xgboost import XGBClassifier

from algorithms import GeneticBasicParameters, Massaging, Reweighing, DisparateImpactRemover, \
    LearnedFairRepresentations, LexicographicGeneticAlgorithmFairFeatureSelection, PermutationGeneticAlgorithm
from algorithms.MulticlassLexicographicGeneticAlgorithmFairFeatureSelection import \
    MulticlassLexicographicGeneticAlgorithmFairFeatureSelection
from datasets import DatasetConfig, Dataset, GermanCredit, AdultIncome
from datasets.LawSchoolAdmissions import LawSchoolAdmissions
from helpers import logger, set_seed, get_seed


GPU_LIMIT = 100


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    LearnedFairRepresentations = 4
    LGAFFS = 5
    MulticlassLGAFFS = 6
    PermutationGeneticAlgorithm = 7


def get_surrogate_classifiers() -> list:
    return [
        LogisticRegression(random_state=get_seed()),
        SVC(random_state=get_seed()),
        GaussianNB(),
        DecisionTreeClassifier(random_state=get_seed()),
        RandomForestClassifier(random_state=get_seed()),
    ]


def get_test_classifier(n) -> object:
    if n < GPU_LIMIT:
        return XGBClassifier(random_state=get_seed())
    return XGBClassifier(random_state=get_seed(), tree_method='gpu_hist')


def get_global_configs(configs_file: str) -> tuple[str, str, str, int]:
    global_configs = configparser.ConfigParser()
    global_configs.read(configs_file)

    try:
        dataset_configs = global_configs.get("GLOBAL", 'dataset_configs').strip('"')
        algorithms_configs = global_configs.get('GLOBAL', 'algorithms_configs').strip('"')

        results_path = global_configs.get('GLOBAL', 'results_path')

        seed = global_configs.getint('GLOBAL', 'seed')
        set_seed(seed)

        num_iterations = global_configs.getint('GLOBAL', 'num_iterations')

        return dataset_configs, algorithms_configs, results_path, num_iterations

    except configparser.NoSectionError as e:
        logger.error(f'Section [{e.section}] does not exist in the configuration file.')
        raise ValueError
    except configparser.NoOptionError as e:
        logger.error(f'Option [{e.option}] does not exist in the configuration file.')
        raise ValueError


def get_dataset_configs(dataset: str, configs_file: str):
    configs = configparser.ConfigParser()
    configs.read(configs_file)

    return DatasetConfig(name=configs.get(dataset, 'name'),
                         target=configs.get(dataset, 'target'),
                         protected_attributes=configs.get(dataset, 'protected_attributes').split(','),
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
                    data_size_limit=GPU_LIMIT,
                    verbose=verbose
                )
            else:
                return MulticlassLexicographicGeneticAlgorithmFairFeatureSelection(
                    genetic_parameters=genetic_parameters,
                    n_splits=n_splits,
                    min_feature_prob=min_feature_prob,
                    max_feature_prob=max_feature_prob,
                    verbose=verbose,
                    data_size_limit=GPU_LIMIT
                )
        case AlgorithmOptions.PermutationGeneticAlgorithm:
            verbose, threshold_k, genetic_parameters = (
                get_algorithms_configs(algorithm_configs_file, AlgorithmOptions.PermutationGeneticAlgorithm))
            unbiasing_algorithms_pool = [
                load_algorithm(algorithm_configs_file, AlgorithmOptions.Massaging),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.Reweighing),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.DisparateImpactRemover),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.LGAFFS),
                load_algorithm(algorithm_configs_file, AlgorithmOptions.LearnedFairRepresentations)
            ]
            unbiasing_algorithms_pool[3].verbose = False
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
