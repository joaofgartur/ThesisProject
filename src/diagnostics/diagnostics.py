"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

from datasets import Dataset
from classifiers.classifiers import train_all_classifiers
from metrics import compute_metrics_suite
from errors import error_check_dataset, error_check_sensitive_attribute
from helpers import logger


def diagnostics(dataset: Dataset, learning_settings: dict) -> dict:
    """
    Conduct diagnostics on a dataset, including fairness metrics and classifier accuracies.

    Parameters
    ----------
    dataset : Dataset
        The dataset object containing features, targets, and sensitive attributes.
    learning_settings : dict
        Dictionary containing learning settings.

    Returns
    -------
    dict
        A dictionary containing fairness metrics and classifier accuracies.

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If there are missing sensitive attributes in the dataset.
        - If learning settings are missing required keys.
    """
    error_check_dataset(dataset)

    metrics_results = {}
    for sensitive_attribute in dataset.protected_features:
        logger.info(f"Computing fairness metrics for attribute \'{sensitive_attribute}\'...")

        error_check_sensitive_attribute(dataset, sensitive_attribute)

        metrics = compute_metrics_suite(dataset, sensitive_attribute)
        metrics_results.update({sensitive_attribute: metrics})

        logger.info(f"Fairness metrics for attribute \'{sensitive_attribute}\' computed.")

    logger.info("Training classifiers...")

    classifiers_results = train_all_classifiers(dataset, learning_settings)

    logger.info("Classifiers trained.")

    return {"metrics": metrics_results, "classifiers": classifiers_results}
