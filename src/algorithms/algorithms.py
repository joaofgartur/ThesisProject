"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""
from aif360.datasets import StandardDataset

from datasets import Dataset
from assessment import bias_assessment
from errors import error_check_dataset
from helpers import logger


def scale_dataset(scaler, dataset: StandardDataset) -> StandardDataset:
    dataset.features = scaler.fit_transform(dataset.features)
    return dataset


def bias_correction(dataset: Dataset, learning_settings: dict, algorithm) -> None:
    """
    Apply a bias correction algorithm to a dataset and display pre- and post-correction assessment.

    Parameters
    ----------
    dataset :
        Original dataset object containing features, targets, and sensitive attributes.
    learning_settings :
        Dictionary containing learning settings.
    algorithm :
        Bias correction algorithm to be applied. It should accept 'dataset', 'sensitive_attribute',
        and 'learning_settings'
        as parameters.

    Returns
    -------
    None

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

        # pre-correction assessment stage
        logger.info("Computing pre-correction assessment stage...")
        print(bias_assessment(dataset, learning_settings).to_string())
        logger.info("Pre-correction assessment computed.")

        # correction stage
        for feature in dataset.protected_features:
            logger.info(f"Applying bias correction for attribute {feature}...")

            new_dataset = algorithm.repair(dataset, feature)

            logger.info(f"Finished correcting bias. Computing post-correction assessment "
                        f"for attribute {feature}...")

            results = bias_assessment(new_dataset, learning_settings, feature)
            print(results.to_string())

            logger.info("Post-correction assessment computed.")

        # Rest of the code
    except Exception as e:
        logger.error(f"An error occurred during bias correction algorithm execution: {e}")
        raise
