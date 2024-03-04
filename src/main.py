"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
from enum import Enum

from algorithms import (Massaging, Reweighing, DisparateImpactRemover, LearningFairRepresentations)
from protocol import Pipeline
from datasets import GermanCredit, AdultIncome, Compas
from helpers import logger, logger_levels, config_logger
from xgboost import XGBClassifier


class DatasetOptions(Enum):
    COMPAS = 0
    ADULT = 1
    GERMAN = 2


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
                'protected_attributes': ['Attribute9'],
                'explanatory_attributes': [],
                'privileged_classes': ['Male'],

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
        case _:
            logger.error('Dataset option unknown!')
            raise NotImplementedError


class AlgorithmOptions(Enum):
    Massaging = 0
    Reweighing = 1
    DisparateImpactRemover = 2
    LearningFairRepresentations = 3


def load_algorithm(option: Enum):
    match option:
        case AlgorithmOptions.Massaging:
            return Massaging(learning_settings={'train_size': 0.7, 'test_size': 0.3})
        case AlgorithmOptions.Reweighing:
            return Reweighing()
        case AlgorithmOptions.DisparateImpactRemover:
            return DisparateImpactRemover(repair_level=1.0)
        case AlgorithmOptions.LearningFairRepresentations:
            return LearningFairRepresentations()
        case _:
            logger.error('Algorithm option unknown!')
            raise NotImplementedError


if __name__ == '__main__':
    config_logger(level=logger_levels.INFO.value)
    results_path = 'results'
    settings = {
        'seed': 125,
        'train_size': 0.5,
        'validation_size': 0.2,
        "test_size": 0.3,
    }
    model = XGBClassifier(random_state=settings['seed'])

    logger.info("Initializing...")

    run_all = False
    run_all_dataset = True

    if run_all:
        for i in DatasetOptions:
            dataset = load_dataset(i)

            for j in AlgorithmOptions:
                algorithm = load_algorithm(j)

                pipeline = Pipeline(dataset, algorithm, model, settings)
                pipeline.run_and_save()
    elif run_all_dataset:
        dataset = load_dataset(DatasetOptions.COMPAS)

        for j in AlgorithmOptions:
            algorithm = load_algorithm(j)

            pipeline = Pipeline(dataset, algorithm, model, settings)
            pipeline.run_and_save()
    else:
        dataset = load_dataset(DatasetOptions.COMPAS)
        algorithm = load_algorithm(AlgorithmOptions.Massaging)

        pipeline = Pipeline(dataset, algorithm, model, settings)
        pipeline.run_and_save()

    logger.info("End of program.")
