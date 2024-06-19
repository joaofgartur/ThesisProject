import configparser
from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from algorithms import GeneticBasicParameters, Massaging, Reweighing, DisparateImpactRemover, \
    LearnedFairRepresentations, LexicographicGeneticAlgorithmFairFeatureSelection, PermutationGeneticAlgorithm, \
    PGAMassaging, PGAReweighing, PGADisparateImpactRemover, PGALearnedFairRepresentations
from algorithms.MulticlassLexicographicGeneticAlgorithmFairFeatureSelection import \
    MulticlassLexicographicGeneticAlgorithmFairFeatureSelection
from algorithms.pga_variants.PGALexicographicGeneticAlgorithmFairFeatureSelection import \
    PGALexicographicGeneticAlgorithmFairFeatureSelection
from datasets import DatasetConfig, Dataset, GermanCredit, AdultIncome
from datasets.LawSchoolAdmissions import LawSchoolAdmissions
from helpers import logger, set_seed, get_seed


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    LearnedFairRepresentations = 4
    LGAFFS = 5
    MulticlassLGAFFS = 6
    PermutationGeneticAlgorithm = 7
    PGAMassaging = 8
    PGAReweighing = 9
    PGADisparateImpactRemover = 10
    PGALearnedFairRepresentations = 11
    PGALexicographicGeneticAlgorithmFairFeatureSelection = 12


test_classifier = XGBClassifier(random_state=get_seed())

surrogate_models = [
    LogisticRegression(),
    SVC(random_state=get_seed()),
    GaussianNB(),
    DecisionTreeClassifier(random_state=get_seed()),
    RandomForestClassifier(random_state=get_seed()),
]


def get_global_configs(configs_file: str) -> tuple[str, str, str]:
    global_configs = configparser.ConfigParser()
    global_configs.read(configs_file)

    try:
        dataset_configs = global_configs.get("GLOBAL", 'dataset_configs').strip('"')
        algorithms_configs = global_configs.get('GLOBAL', 'algorithms_configs').strip('"')

        results_path = global_configs.get('GLOBAL', 'results_path')

        seed = global_configs.getint('GLOBAL', 'seed')
        set_seed(seed)

        return dataset_configs, algorithms_configs, results_path

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
            return verbose, GeneticBasicParameters(
                population_size=configs.getint('PermutationGeneticAlgorithm', 'population_size'),
                num_generations=configs.getint('PermutationGeneticAlgorithm', 'num_generations'),
                tournament_size=configs.getint('PermutationGeneticAlgorithm', 'tournament_size'),
                elite_size=configs.getint('PermutationGeneticAlgorithm', 'elite_size'),
                probability_crossover=configs.getfloat('PermutationGeneticAlgorithm', 'probability_crossover'),
                probability_mutation=configs.getfloat('PermutationGeneticAlgorithm', 'probability_mutation')
            )
        case AlgorithmOptions.PGAMassaging | AlgorithmOptions.PGAReweighing | \
             AlgorithmOptions.PGADisparateImpactRemover | AlgorithmOptions.PGALearnedFairRepresentations | \
             AlgorithmOptions.PGALexicographicGeneticAlgorithmFairFeatureSelection:
            verbose = configs.getboolean('PermutationGeneticAlgorithm', 'verbose')
            return verbose, GeneticBasicParameters(
                population_size=configs.getint('PermutationGeneticAlgorithmVariant', 'population_size'),
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
            logger.error('Dataset unknown! Currently supported datasets are: GERMAN_CREDIT, ADULT_INCOME.')
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
                    verbose=verbose
                )
        case AlgorithmOptions.PermutationGeneticAlgorithm:
            verbose, genetic_parameters = get_algorithms_configs(algorithm_configs_file,
                                                                 AlgorithmOptions.PermutationGeneticAlgorithm)
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
                surrogate_models_pool=surrogate_models,
                verbose=verbose
            )
        case AlgorithmOptions.PGAMassaging | AlgorithmOptions.PGAReweighing | AlgorithmOptions.PGADisparateImpactRemover | AlgorithmOptions.PGALearnedFairRepresentations | AlgorithmOptions.PGALexicographicGeneticAlgorithmFairFeatureSelection:

            verbose, genetic_parameters = get_algorithms_configs(algorithm_configs_file, AlgorithmOptions.PGAMassaging)

            match unbiasing_algorithm:
                case AlgorithmOptions.PGAMassaging:
                    unbiasing_algorithms_pool = [load_algorithm(algorithm_configs_file, AlgorithmOptions.Massaging)]
                    return PGAMassaging(
                        genetic_parameters=genetic_parameters,
                        unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                        surrogate_models_pool=surrogate_models,
                        verbose=verbose
                    )
                case AlgorithmOptions.PGAReweighing:
                    unbiasing_algorithms_pool = [load_algorithm(algorithm_configs_file, AlgorithmOptions.Reweighing)]
                    return PGAReweighing(
                        genetic_parameters=genetic_parameters,
                        unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                        surrogate_models_pool=surrogate_models,
                        verbose=verbose
                    )
                case AlgorithmOptions.PGADisparateImpactRemover:
                    unbiasing_algorithms_pool = [load_algorithm(algorithm_configs_file,
                                                                AlgorithmOptions.DisparateImpactRemover)]
                    return PGADisparateImpactRemover(
                        genetic_parameters=genetic_parameters,
                        unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                        surrogate_models_pool=surrogate_models,
                        verbose=verbose
                    )
                case AlgorithmOptions.PGALearnedFairRepresentations:
                    unbiasing_algorithms_pool = [load_algorithm(algorithm_configs_file,
                                                                AlgorithmOptions.LearnedFairRepresentations)]
                    return PGALearnedFairRepresentations(
                        genetic_parameters=genetic_parameters,
                        unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                        surrogate_models_pool=surrogate_models,
                        verbose=verbose
                    )
                case AlgorithmOptions.PGALexicographicGeneticAlgorithmFairFeatureSelection:
                    unbiasing_algorithms_pool = [load_algorithm(algorithm_configs_file, AlgorithmOptions.LGAFFS)]
                    unbiasing_algorithms_pool[0].verbose = False
                    return PGALexicographicGeneticAlgorithmFairFeatureSelection(
                        genetic_parameters=genetic_parameters,
                        unbiasing_algorithms_pool=unbiasing_algorithms_pool,
                        surrogate_models_pool=surrogate_models,
                        verbose=verbose
                    )
                case _:
                    logger.error('Algorithm option unknown 2!')
                    raise ValueError

        case _:
            logger.error('Algorithm option unknown!')
            raise ValueError
