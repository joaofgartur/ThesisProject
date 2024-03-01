"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
import pandas as pd
from aif360.datasets import StandardDataset

from algorithms.Algorithm import Algorithm
from datasets import Dataset
from errors import error_check_dataset
from helpers import logger
from .assessment import assess_all_surrogates


def scale_dataset(scaler, dataset: StandardDataset) -> StandardDataset:
    dataset.features = scaler.fit_transform(dataset.features)
    return dataset


def bias_correction(dataset: Dataset, learning_settings: dict, algorithms: [Algorithm]) -> pd.DataFrame:
    """
    Apply a bias protocol algorithm to a dataset and display pre- and post-protocol assessment.

    Parameters
    ----------
    dataset :
        Original dataset object containing features, targets, and sensitive attributes.
    learning_settings :
        Dictionary containing learning settings.
    algorithm :
        Bias protocol algorithm to be applied. It should accept 'dataset', 'sensitive_attribute',
        and 'learning_settings'
        as parameters.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
        - If the sensitive attribute is not present in the dataset.
        - If the provided algorithm is not a callable function.
        - If the provided algorithm does not receive the correct parameters.
    """
    try:
        error_check_dataset(dataset)

        # pre-protocol assessment stage
        logger.info("Computing pre-protocol assessment stage...")
        correction_results = assess_all_surrogates(dataset, learning_settings)
        logger.info("Pre-protocol assessment computed.")

        for algorithm in algorithms:

            # protocol stage
            for feature in dataset.protected_features:
                logger.info(f"Applying bias protocol for attribute {feature}...")

                new_dataset = algorithm.repair(dataset, feature)

                logger.info(f"Finished correcting bias. Computing post-protocol assessment "
                            f"for attribute {feature}...")

                results = assess_all_surrogates(new_dataset, learning_settings, feature, algorithm.__class__.__name__)
                correction_results = pd.concat([correction_results, results])

                logger.info("Post-protocol assessment computed.")

        return correction_results

        # Rest of the code
    except Exception as e:
        logger.error(f"An error occurred during bias protocol algorithm execution: {e}")
        raise
