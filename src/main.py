"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""

from algorithms import (Massaging, Reweighing, DisparateImpactRemover, LearningFairRepresentations,
                        AttributeRemoval, DoNothing)
from datasets import GermanCredit, AdultIncome, Compas
from algorithms import bias_correction_algorithm
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


if __name__ == '__main__':
    config_logger(level=logger_levels.INFO.value)
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}

    logger.info("Initializing...")

    dataset = load_dataset(DatasetOptions.COMPAS)

    algorithms = [
        # Massaging(learning_settings=_learning_settings),
        # Reweighing(),
        # DisparateImpactRemover(repair_level=1.0, learning_settings=_learning_settings),
        # LearningFairRepresentations(learning_settings=_learning_settings),
        # AttributeRemoval(),
        # DoNothing(),
    ]

    for algorithm in algorithms:
        logger.info(f"Applying bias correction method {algorithm.__module__} to dataset {dataset.name}.")
        bias_correction_algorithm(dataset, _learning_settings, algorithm)

    logger.info("End of program.")
