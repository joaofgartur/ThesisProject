"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""

from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from algorithms import (Massaging, Reweighing, DisparateImpactRemover, LGAFFS,
                        PermutationGeneticAlgorithm, AIF360LearningFairRepresentations)
from algorithms.GeneticAlgorithmHelpers import GeneticBasicParameters
from algorithms.MultivalueLGAFFS import MultivalueLGAFFS
from datasets.AIF360AdultIncome import AIF360AdultIncome
from protocol import Pipeline
from datasets import GermanCredit, AdultIncome, Compas
from helpers import logger, extract_filename, set_seed, get_seed
from xgboost import XGBClassifier


class DatasetOptions(Enum):
    COMPAS = 0
    ADULT = 1
    GERMAN = 2
    AIF360ADULT = 3


def load_dataset(option: Enum):
    match option:
        case DatasetOptions.COMPAS:
            compas_info = {
                'dataset_name': 'COMPAS',
                'target': 'two_year_recid',
                'protected_attributes': ['race', 'sex'],
                'explanatory_attributes': [],
                'privileged_classes': ['Caucasian', 'Male'],

            }
            return Compas(compas_info)
        case DatasetOptions.GERMAN:
            german_info = {
                'dataset_name': 'German Credit',
                'target': 'class',
                'protected_attributes': ['Attribute9', 'Attribute13'],
                'explanatory_attributes': [],
                'privileged_classes': ['Male', 'Aged'],

            }
            return GermanCredit(german_info)
        case DatasetOptions.ADULT:
            adult_info = {
                'dataset_name': 'Adult Income',
                'target': 'income',
                'protected_attributes': ['race', 'sex'],
                'explanatory_attributes': [],
                'privileged_classes': ['White', 'Male'],

            }
            return AdultIncome(adult_info)
        case DatasetOptions.AIF360ADULT:
            adult_info = {
                'dataset_name': 'Aif360 Adult Income',
                'target': 'income',
                'protected_attributes': ['race', 'sex'],
                'explanatory_attributes': [],
                'privileged_classes': ['White', 'Male'],

            }
            return AIF360AdultIncome(adult_info)
        case _:
            logger.error('Dataset option unknown!')
            raise NotImplementedError


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    AIF360LFR = 4
    LGAFFS = 5
    PGA = 6
    MLGAFFS = 7
    GPGA = 8


def load_algorithm(option: Enum):
    match option:
        case AlgorithmOptions.Massaging:
            return Massaging()
        case AlgorithmOptions.Reweighing:
            return Reweighing()
        case AlgorithmOptions.DisparateImpactRemover:
            return DisparateImpactRemover(repair_level=1.0)
        case AlgorithmOptions.AIF360LFR:
            return AIF360LearningFairRepresentations(
                k=10,
                ax=1e-4,
                ay=0.1,
                az=1000,
                verbose=False
            )
        case AlgorithmOptions.LGAFFS:
            genetic_parameters = GeneticBasicParameters(
                population_size=101,
                num_generations=2,
                tournament_size=2,
                probability_crossover=0.9,
                probability_mutation=0.05
            )
            return LGAFFS(
                genetic_parameters=genetic_parameters,
                n_splits=3,
                min_feature_prob=0.1,
                max_feature_prob=0.5,
            )
        case AlgorithmOptions.PGA:
            genetic_parameters = GeneticBasicParameters(
                population_size=20,
                num_generations=2,
                tournament_size=2,
                elite_size=2,
                probability_crossover=0.9,
                probability_mutation=0.05
            )
            return PermutationGeneticAlgorithm(
                genetic_parameters=genetic_parameters,
                unbiasing_algorithms_pool=[Massaging()],
                surrogate_models_pool=[SVC(random_state=get_seed()),
                                       GaussianNB(),
                                       RandomForestClassifier(random_state=get_seed())]
            )
        case AlgorithmOptions.MLGAFFS:
            genetic_parameters = GeneticBasicParameters(
                population_size=101,
                num_generations=2,
                tournament_size=2,
                probability_crossover=0.9,
                probability_mutation=0.05
            )
            return MultivalueLGAFFS(
                genetic_parameters=genetic_parameters,
                n_splits=3,
                min_feature_prob=0.1,
                max_feature_prob=0.5,
            )
        case AlgorithmOptions.GPGA:
            genetic_parameters = GeneticBasicParameters(
                population_size=3,
                num_generations=2,
                tournament_size=2,
                elite_size=2,
                probability_crossover=0.9,
                probability_mutation=0.05
            )
            return PermutationGeneticAlgorithm(
                genetic_parameters=genetic_parameters,
                unbiasing_algorithms_pool=[
                    load_algorithm(AlgorithmOptions.Massaging),
                    load_algorithm(AlgorithmOptions.Reweighing),
                    load_algorithm(AlgorithmOptions.DisparateImpactRemover),
                    load_algorithm(AlgorithmOptions.LGAFFS),
                    load_algorithm(AlgorithmOptions.AIF360LFR)
                ],
                surrogate_models_pool=[SVC(random_state=get_seed()),
                                       GaussianNB(),
                                       RandomForestClassifier(random_state=get_seed())]
            )
        case _:
            logger.error('Algorithm option unknown!')
            raise NotImplementedError


if __name__ == '__main__':

    results_path = 'results'
    settings = {
        'seed': 42,
        'train_size': 0.7,
        'validation_size': 0.1,
        "test_size": 0.2,
        "num_repetitions": 1
    }

    set_seed(settings['seed'])

    test_classifier = XGBClassifier(random_state=get_seed())
    surrogate_models = {
        #'LR': LogisticRegression(),
        'SVC': SVC(),
        'GNB': GaussianNB(),
        #"DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
    }

    logger.info(f'[{extract_filename(__file__)}] Initializing.')

    run_all = False
    run_all_dataset = False
    num_runs = 1

    if run_all:
        for i in DatasetOptions:
            dataset = load_dataset(i)

            for j in AlgorithmOptions:
                unbiasing_algorithms = load_algorithm(j)

                pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_models, test_classifier, settings)
                pipeline.run_and_save()
    elif run_all_dataset:
        for i in range(max(1, num_runs)):
            dataset = load_dataset(DatasetOptions.GERMAN)
            unbiasing_algorithms = [load_algorithm(j) for j in AlgorithmOptions]
            pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_models, test_classifier, settings)
            pipeline.run_and_save()
    else:
        for i in range(max(1, num_runs)):
            dataset = load_dataset(DatasetOptions.ADULT)
            unbiasing_algorithms = [load_algorithm(AlgorithmOptions.GPGA)]
            pipeline = Pipeline(dataset, unbiasing_algorithms, surrogate_models, test_classifier, settings)
            pipeline.run_and_save(results_path)

    logger.info(f'[{extract_filename(__file__)}] End of program.')
