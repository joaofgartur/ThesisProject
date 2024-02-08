import abc

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from datasets import drop_invalid_instances, convert_categorical_into_numerical
from helpers import logger, extract_value


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, dataset_info: dict):
        """Initializes class instance"""
        self.features_mapping = None
        self.targets_mapping = None

        self.__parse_dataset_info(dataset_info)

        # load raw data into dataframe
        self.features, self.targets = self._load_dataset()

        # preprocess dataset
        self._preprocessing()

    def __parse_dataset_info(self, dataset_info: dict):
        dataset_name = 'dataset_name'
        protected_attributes = 'protected_attributes'
        explanatory_attributes = 'explanatory_attributes'
        privileged_classes = 'privileged_classes'
        target = 'target'

        self.name = extract_value(dataset_name, dataset_info)
        self.protected_features = extract_value(protected_attributes, dataset_info)
        self.explanatory_features = extract_value(explanatory_attributes, dataset_info)
        self.privileged_classes = extract_value(privileged_classes, dataset_info)
        self.target = extract_value(target, dataset_info)

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _transform_protected_attributes(self):
        """"""

    def _preprocessing(self):
        logger.info("Pre-processing dataset...")

        # drop invalid instances
        self.features, removed_indexes = drop_invalid_instances(self.features)
        self.targets, _ = drop_invalid_instances(self.targets, removed_indexes)

        # rearrange indexes
        new_indexes = np.arange(0, len(self.features), 1, dtype=int)
        self.features.index = new_indexes
        self.targets.index = new_indexes

        # transform protected attributes
        self._transform_protected_attributes()

        # convert categorical into numerical
        self.features, self.features_mapping = convert_categorical_into_numerical(self.features)
        self.targets, self.target_mapping = convert_categorical_into_numerical(self.targets)

        # normalize dataset
        scaler = MinMaxScaler(copy=False)
        normalized_features = scaler.fit_transform(self.features)
        self.features = pd.DataFrame(normalized_features, columns=self.features.columns)

    def get_protected_features(self) -> pd.DataFrame:
        return self.features.loc[:, self.protected_features]

    def merge_features_and_targets(self) -> (pd.DataFrame, str):
        data = pd.concat([self.features, self.targets], axis='columns')
        outcome = self.targets.columns[0]

        return data, outcome

    def print_dataset(self):
        """Prints the dataset"""
        print(f'---- Features ----\n{self.features}')
        print(f'---- targets ----\n{self.targets}')
