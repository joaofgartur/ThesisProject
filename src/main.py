"""
Author: João Artur
Project: Master's Thesis
Last edited: 30-11-2023
"""

from algorithms import Massaging, Reweighing, DisparateImpactRemover, LearningFairRepresentations
from datasets import GermanCredit, AdultIncome, Compas
from algorithms import bias_correction_algorithm
from helpers import logger, logger_levels, config_logger


if __name__ == '__main__':
    config_logger(level=logger_levels.INFO.value)
    _learning_settings = {"train_size": 0.7, "test_size": 0.3, "seed": 125}

    logger.info("Initializing...")

    compas_info = {
        "dataset_name": "COMPAS",
        "sensitive_attributes": {"race": "Caucasian", "sex": "Male"}
    }
    # dataset = Compas(compas_info)

    german_info = {
        "dataset_name": "German Credit",
        "sensitive_attributes": {"Attribute9": "Male"},
    }
    dataset = GermanCredit(german_info)

    adult_info = {
        "dataset_name": "Adult Income",
        "sensitive_attributes": {"race": "White", "sex": "Male"}
    }
    # dataset = AdultIncome(adult_info)

    algorithms = [
        Massaging(learning_settings=_learning_settings),
        Reweighing(),
        DisparateImpactRemover(repair_level=1.0, learning_settings=_learning_settings),
        LearningFairRepresentations(learning_settings=_learning_settings),
    ]

    for algorithm in algorithms:
        logger.info(f"Applying bias correction method {algorithm.__module__} to dataset {dataset.name}.")
        bias_correction_algorithm(dataset, _learning_settings, algorithm)

    logger.info("End of program.")