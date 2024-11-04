"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import inspect
from typing import Callable

from datasets import Dataset


def error_check_dataset(dataset: Dataset) -> None:
    """
    Check if the provided dataset is valid.

    Parameters
    ----------
    dataset :
        Dataset object to be checked.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        - If an invalid dataset is provided.
        - If the dataset does not contain both features and targets.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("Invalid dataset provided.")

    if not dataset.features.columns.tolist() or not dataset.targets.columns.tolist():
        raise ValueError("Dataset should contain both features and targets.")


def error_check_sensitive_attribute(dataset: Dataset, sensitive_attribute) -> None:
    """
    Check if the sensitive attribute is present in the dataset.

    Parameters
    ----------
    dataset :
        Dataset object to be checked.
    sensitive_attribute :
        Name of the sensitive attribute.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the sensitive attribute is not present in the dataset.
    """
    if sensitive_attribute not in dataset.features.columns:
        raise ValueError(f"The sensitive attribute '{sensitive_attribute}' is not present in the dataset.")


def error_check_dictionary_keys(dictionary: dict, keys: list) -> None:
    """
    Check if the specified keys are present in the dictionary.

    Parameters
    ----------
    dictionary :
        Dictionary to be checked.
    keys :
        List of keys to be checked.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the specified keys are missing in the dictionary.
    """
    missing_keys = [key for key in keys if key not in dictionary.keys()]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(f"The dictionary should contain the following keys: {missing_keys_str}.")


def error_check_callable(function: Callable) -> None:
    """
    Check if the provided object is callable.

    Parameters
    ----------
    function :
        Object to be checked.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the provided object is not callable.
    """

    if not callable(function):
        raise ValueError("The provided 'algorithm' is not a callable function.")


def error_check_parameters(function: Callable, parameters: list) -> None:
    """
    Check if the specified parameters are present in the function's signature.

    Parameters
    ----------
    function :
        Function to be checked.
    parameters :
        List of parameter names.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the function does not accept the specified parameters.
    """
    function_signature = inspect.signature(function)
    missing_params = [param for param in parameters if param not in function_signature.parameters]
    if missing_params:
        missing_params_str = ", ".join(missing_params)
        raise ValueError(f"The function should accept the following parameters: {missing_params_str}.")
