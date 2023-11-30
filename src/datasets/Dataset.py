import abc
import numpy as np
import pandas as pd

from datasets import remove_invalid_columns, convert_categorical_into_numerical
from constants import PRIVILEGED, UNPRIVILEGED


class Dataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset_info: dict):
        """Initializes class instance"""
        self.__parse_dataset_info__(dataset_info)
        self.features = None
        self.targets = None
        self.features_mapping = None
        self.target_mapping = None

    def __parse_dataset_info__(self, dataset_info: dict):
        dataset_name_key = "dataset_name"
        sensitive_attributes_key = "sensitive_attributes"

        if dataset_name_key in dataset_info.keys():
            self.dataset_name = dataset_info[dataset_name_key]
        else:
            raise ValueError("Missing dataset name.")

        if sensitive_attributes_key in dataset_info.keys():
            self.sensitive_attributes_info = dataset_info[sensitive_attributes_key]
        else:
            raise ValueError("Missing dataset sensitive attributes information.")

    def __define_privileged_and_unprivileged__(self):
        for sensitive_attribute in self.sensitive_attributes_info.keys():
            unprivileged_value = self.sensitive_attributes_info[sensitive_attribute]["unprivileged_value"]
            self.features[sensitive_attribute] = np.where(self.features[sensitive_attribute] == unprivileged_value,
                                                          UNPRIVILEGED, PRIVILEGED)

    @abc.abstractmethod
    def _load_dataset(self):
        """Loads the dataset"""

    @abc.abstractmethod
    def _transform_dataset(self):
        """"""

    def _prepare_dataset(self):
        # drop any instances with empty values
        self.features, removed_indexes = remove_invalid_columns(self.features, [])
        self.targets, _ = remove_invalid_columns(self.targets, removed_indexes)

        # transform attributes
        self._transform_dataset()

        # convert categorical into numerical
        self.features, self.features_mapping = convert_categorical_into_numerical(self.features)
        self.targets, self.target_mapping = convert_categorical_into_numerical(self.targets)

        # transform indexes
        new_indexes = np.arange(0, len(self.features), 1, dtype=int)
        self.features.index = new_indexes
        self.targets.index = new_indexes

    def get_privileged_and_unprivileged_value_for_attribute(self, attribute: str) -> int:
        unprivileged_label = self.sensitive_attributes_info[attribute]["unprivileged_value"]
        return self.features_mapping[attribute][unprivileged_label]

    def get_sensitive_attributes(self) -> pd.DataFrame:
        columns_names = list(self.sensitive_attributes_info.keys())
        return self.features.loc[:, columns_names]

    def merge_features_and_targets(self) -> (pd.DataFrame, str):
        data = pd.concat([self.features, self.targets], axis="columns")
        outcome_column = self.targets.columns[0]

        return data, outcome_column



    def print_dataset(self):
        """Prints the dataset"""
        print(self.features)
        print(self.targets)
