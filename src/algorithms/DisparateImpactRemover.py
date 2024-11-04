"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""
from aif360.algorithms.preprocessing import DisparateImpactRemover as DIR

from algorithms.Algorithm import Algorithm
from datasets import Dataset, convert_to_standard_dataset


class DisparateImpactRemover(Algorithm):
    """
    Disparate Impact Remover algorithm for bias correction.

    Attributes
    ----------
    repair_level : float
        The level of repair to apply to the dataset.
    transformer : DIR or None
        The transformer used to apply the Disparate Impact Remover.
    sensitive_attribute : str or None
        The sensitive attribute used by the algorithm.

    Methods
    -------
    __init__(repair_level: float):
        Initializes the DisparateImpactRemover object with the specified repair level.
    fit(data: Dataset, sensitive_attribute: str):
        Fits the Disparate Impact Remover to the data.
    transform(data: Dataset) -> Dataset:
        Transforms the dataset using the fitted Disparate Impact Remover.
    """

    def __init__(self, repair_level: float):
        """
        Initializes the DisparateImpactRemover object with the specified repair level.

        Parameters
        ----------
        repair_level : float
            The level of repair to apply to the dataset.
        """
        super().__init__()
        self.repair_level = repair_level
        self.transformer = None
        self.sensitive_attribute = None

    def fit(self, data: Dataset, sensitive_attribute: str):
        """
        Fits the Disparate Impact Remover to the data.

        Parameters
        ----------
        data : Dataset
            The dataset to fit the algorithm to.
        sensitive_attribute : str
            The sensitive attribute used by the algorithm.
        """
        self.sensitive_attribute = sensitive_attribute
        self.transformer = DIR(repair_level=self.repair_level)

    def transform(self, data: Dataset) -> Dataset:
        """
        Transforms the dataset using the fitted Disparate Impact Remover.

        Parameters
        ----------
        data : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        standard_data = convert_to_standard_dataset(data, self.sensitive_attribute)

        transformed_data = self.transformer.fit_transform(standard_data)
        data.update(transformed_data.features, transformed_data.labels)

        return data
