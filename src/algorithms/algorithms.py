"""
Author: João Artur
Project: Master's Thesis
Last edited: 20-11-2023
"""

import inspect
import logging
from typing import Callable

from datasets import Dataset
from diagnostics import diagnostics
from errors import error_check_dataset, error_check_parameters


def bias_correction_algorithm(dataset: Dataset, learning_settings: dict, algorithm: Callable, **kwargs) -> None:
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
    **kwargs :
        Additional keyword arguments to be passed to the bias correction algorithm.

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

        if not callable(algorithm):
            raise ValueError("The provided 'algorithm' is not a callable function.")

        # pre-correction diagnostics stage
        results = diagnostics(dataset, learning_settings)
        print(results)

        # correction stage
        post_results = {}
        for sensitive_attribute in dataset.sensitive_attributes_info.keys():

            algorithm_signature = inspect.signature(algorithm)
            if 'learning_settings' in algorithm_signature.parameters:
                kwargs['learning_settings'] = learning_settings

            error_check_parameters(algorithm, [_DATASET, _SENSITIVE_ATTRIBUTE])

            new_dataset = algorithm(dataset=dataset, sensitive_attribute=sensitive_attribute, **kwargs)

            results = diagnostics(new_dataset, learning_settings)
            post_results.update({sensitive_attribute: results})

        # post-correction diagnostics stage
        print(post_results)

        # Rest of the code
    except Exception as e:
        logging.error(f"An error occurred during bias correction algorithm execution: {e}")
        raise
