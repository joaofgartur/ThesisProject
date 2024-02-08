"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""

from algorithms import (Massaging, Reweighing, DisparateImpactRemover, LearningFairRepresentations)
from datasets import GermanCredit, AdultIncome, Compas
from algorithms import bias_correction
from helpers import logger, logger_levels, config_logger
from enum import Enum


class DatasetOptions(Enum):
    COMPAS = 0
    ADULT = 1
    GERMAN = 2


def load_dataset(dataset_option: Enum):
    match dataset_option:
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


def load_algorithm(option: Enum, learning_settings: dict):
    match option:
        case AlgorithmOptions.Massaging:
            return Massaging(learning_settings=learning_settings)
        case AlgorithmOptions.Reweighing:
            return Reweighing()
        case AlgorithmOptions.DisparateImpactRemover:
            return DisparateImpactRemover(repair_level=1.0, learning_settings=learning_settings)
        case AlgorithmOptions.LearningFairRepresentations:
            return LearningFairRepresentations(learning_settings=learning_settings)
        case _:
            logger.error('Algorithm option unknown!')
            raise NotImplementedError


if __name__ == '__main__':
    config_logger(level=logger_levels.INFO.value)
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}

    logger.info("Initializing...")

    dataset = load_dataset(DatasetOptions.COMPAS)

    algorithm = load_algorithm(AlgorithmOptions.Massaging, _learning_settings)

    logger.info(f"Applying bias correction method {algorithm.__module__} to dataset {dataset.name}.")

    bias_correction(dataset, _learning_settings, algorithm)

    logger.info("End of program.")
