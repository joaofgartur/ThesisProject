"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

from datasets import Dataset
from diagnostics import diagnostics
from errors import error_check_dataset
from helpers import logger


def bias_correction_algorithm(dataset: Dataset, learning_settings: dict, algorithm) -> None:
    """
    Apply a bias correction algorithm to a dataset and display pre- and post-correction diagnostics.

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
    _DATASET = "dataset"
    _SENSITIVE_ATTRIBUTE = "sensitive_attribute"

    try:
        error_check_dataset(dataset)

        # pre-correction diagnostics stage
        logger.info("Computing pre-correction diagnostics stage...")
        print(diagnostics(dataset, learning_settings))
        logger.info("Pre-correction diagnostics computed.")

        # correction stage
        post_results = {}
        for sensitive_attribute in dataset.sensitive_attributes_info.keys():
            logger.info(f"Applying bias correction for attribute {sensitive_attribute}...")

            new_dataset = algorithm.repair(dataset, sensitive_attribute)

            logger.info(f"Finished correcting bias. Computing post-correction diagnostics "
                        f"for attribute{sensitive_attribute}...")

            results = diagnostics(new_dataset, learning_settings)

            logger.info("Post-correction diagnostics computed.")

            print(results)
            post_results.update({sensitive_attribute: results})

        # Rest of the code
    except Exception as e:
        logger.error(f"An error occurred during bias correction algorithm execution: {e}")
        raise
