"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""
from enum import Enum

from algorithms import (Massaging, Reweighing, DisparateImpactRemover, LearningFairRepresentations)
from algorithms import bias_correction
from datasets import GermanCredit, AdultIncome, Compas
from helpers import logger, logger_levels, config_logger, write_dataframe_to_csv


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
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125, 'cross_validation': 5}

    logger.info("Initializing...")

    for dataset in DatasetOptions:

        dataset = load_dataset(dataset)

        algorithms = [load_algorithm(AlgorithmOptions.DisparateImpactRemover)]

        results = bias_correction(dataset, _learning_settings, algorithms)

        write_dataframe_to_csv(df=results, dataset_name=dataset.name,
                               path=results_path)

    logger.info("End of program.")
