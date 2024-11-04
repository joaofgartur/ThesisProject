"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import abc

from datasets import Dataset


class Algorithm(metaclass=abc.ABCMeta):
    """
    Abstract base class for an Algorithm.

    Attributes
    ----------
    is_binary : bool
        Indicates if the algorithm is binary.
    sensitive_attribute : str or None
        The sensitive attribute used by the algorithm.
    needs_auxiliary_data : bool
        Indicates if the algorithm needs auxiliary data.
    auxiliary_data : any
        The auxiliary data used by the algorithm.
    iteration_number : int
        The current iteration number of the algorithm.

    Methods
    -------
    __init__():
        Initializes the Algorithm object.
    fit(data: Dataset, sensitive_attribute: str):
        Abstract method to fit the algorithm to the data.
    transform(dataset: Dataset) -> Dataset:
        Abstract method to transform the dataset using the algorithm.
    set_validation_data(validation_data: Dataset):
        Sets the validation data for the algorithm.
    """


    def __init__(self):
        """
        Initializes the Algorithm object with default values.
        """
        self.is_binary = True
        self.sensitive_attribute = None
        self.needs_auxiliary_data = False
        self.auxiliary_data = None
        self.iteration_number = 0

    @abc.abstractmethod
    def fit(self, data: Dataset, sensitive_attribute: str):
        """
        Abstract method to fit the algorithm to the data.

        Parameters
        ----------
        data : Dataset
            The dataset to fit the algorithm to.
        sensitive_attribute : str
            The sensitive attribute used by the algorithm.
        """
        pass


    @abc.abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Abstract method to transform the dataset using the algorithm.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        pass

    def set_validation_data(self, validation_data: Dataset):
        """
        Sets the validation data for the algorithm.

        Parameters
        ----------
        validation_data : Dataset
            The validation dataset.
        """
        pass
