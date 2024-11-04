"""
Project Name: Bias Correction in Datasets
Author: João Artur
Date of Modification: 2024-04-11
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

from constants import NEGATIVE_OUTCOME, POSITIVE_OUTCOME
from datasets import Dataset
from utils.DatasetConfigs import DatasetConfig
from utils import logger, extract_filename


class AdultIncome(Dataset):
    """
    Representation of the Adult Income dataset [1].

    Attributes
    ----------
    _LOCAL_DATA_FILE : str
        The local path to the dataset file.

    Methods
    -------
    __init__(self, config: DatasetConfig):
        Initializes the AdultIncome object with the provided dataset information.
    _load_dataset(self):
        Loads the dataset from an online source or local file, filters the data, and separates features and targets.
    _transform_protected_attributes(self):
        Transforms the protected attributes in the dataset.

    References
    ----------
    [1] Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996.
    DOI: https://doi.org/10.24432/C5XW20 URL https://archive.ics.uci.edu/ml/datasets/adult
    """

    _LOCAL_DATA_FILE = "datasets/local_storage/adult_income/adult.data"

    def __init__(self, config: DatasetConfig):
        """
        Initializes the AdultIncome object with the provided dataset information.

        Parameters
        ----------
        config : DatasetConfig
            The configuration information for the dataset.
        """
        logger.info(f'[{extract_filename(__file__)}] Loading...')
        Dataset.__init__(self, config)

    def _load_dataset(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads the dataset from an online source or local file, filters the data, and separates features and targets.

        Returns
        -------
        features : pd.DataFrame
            The features from the dataset.
        targets : pd.DataFrame
            The target values from the dataset.
        """
        try:
            dataset = fetch_ucirepo(id=2)
            return dataset.data.features, dataset.data.targets
        except ConnectionError:
            logger.error(f'[{extract_filename(__file__)}] Failed to load from online source.'
                         f' Loading from local storage.')

            dataset = pd.read_csv(self._LOCAL_DATA_FILE, header=None)
            labels = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            dataset = dataset.set_axis(labels=labels, axis="columns")

            targets = pd.DataFrame(dataset["income"])
            features = dataset.drop(columns=["income"])

            return features, targets

    def _transform_dataset(self):
        """
        Transforms the dataset.

        The transformations include replacing the income values with the NEGATIVE_OUTCOME and POSITIVE_OUTCOME
        constants and converting the income column to int64. The transformation was derived from the dataset's
        documentation [1].
        """

        def derive_age(x, lower_limit=25, upper_limit=60):
            if lower_limit < x < upper_limit:
                return 'Adult'
            elif x >= upper_limit:
                return 'Aged'
            return 'Young'

        age_column = 'age'
        self.features[age_column] = self.features[age_column].apply(lambda x: derive_age(x))
        self.targets = (self.targets.replace("<=", NEGATIVE_OUTCOME, regex=True)
                        .replace(">", POSITIVE_OUTCOME, regex=True)
                        .astype('int64'))
